package io.github.olleoltorp.mobcapture.label;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.platform.NativeImage;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.GpuTexture;
import com.mojang.logging.LogUtils;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import net.minecraft.client.Camera;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.entity.state.EntityRenderState;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.util.ARGB;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.decoration.Mannequin;
import net.minecraft.world.phys.AABB;
import org.slf4j.Logger;

/**
 * Takes a picture and its label image from the same frame, on request over a local socket.
 *
 * <p>On a label frame every visible mob is drawn a second time, into the glowing-outline target,
 * in its own colour (its index in this frame) with depth testing against the world, so each pixel
 * of the label image says which mob, if any, is seen there. Other entities are drawn in
 * {@link #OTHER} so they hide mobs behind them. The picture is the world as drawn in that same
 * frame, before the HUD. Protocol: protocol/label-capture.md.
 */
public final class LabelCapture {
	/** Label colour for entities that are not mobs; they hide mobs but get no label. */
	public static final int OTHER = 0xFFFFFF;

	private static final Logger LOGGER = LogUtils.getLogger();
	private static final Pattern NAME = Pattern.compile("[A-Za-z0-9_.-]+");
	private static final long TIMEOUT_SECONDS = 30;
	private static final BlockingQueue<Request> PENDING = new LinkedBlockingQueue<>();
	// Writing PNGs takes longer than a frame; several writers keep up with one shot per frame.
	private static final ExecutorService WRITER = Executors.newFixedThreadPool(3, r -> {
		Thread t = new Thread(r, "mobcapture-label-writer");
		t.setDaemon(true);
		return t;
	});

	// Render thread only.
	private static Request current;
	private static boolean readThisFrame;
	private static final List<Labelled> LABELLED = new ArrayList<>();

	private record Request(Path dir, String name, CompletableFuture<String> done) {
	}

	private LabelCapture() {
	}

	/** Starts the request server on the given loopback port, in a background thread. */
	public static void startServer(int port) {
		Thread t = new Thread(() -> serve(port), "mobcapture-label-server");
		t.setDaemon(true);
		t.start();
	}

	private static void serve(int port) {
		try (ServerSocket server = new ServerSocket(port, 50, InetAddress.getLoopbackAddress())) {
			LOGGER.info("mobcapture label capture listening on port {}", port);
			while (true) {
				Socket socket = server.accept();
				Thread t = new Thread(() -> handle(socket), "mobcapture-label-client");
				t.setDaemon(true);
				t.start();
			}
		} catch (IOException e) {
			LOGGER.error("mobcapture label capture server stopped", e);
		}
	}

	private static void handle(Socket socket) {
		try (socket;
			BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
			Writer out = new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8)) {
			String line;
			while ((line = in.readLine()) != null) {
				out.write(answer(line.strip()) + "\n");
				out.flush();
			}
		} catch (IOException e) {
			LOGGER.warn("mobcapture label capture connection closed: {}", e.toString());
		}
	}

	private static String answer(String line) {
		if (line.equals("PING")) {
			return "PONG";
		}
		if (line.equals("STATUS") || line.equals("REFRESH") || line.equals("REBUILD") || line.startsWith("PROBE ") || line.startsWith("ROOMS ")) {
			return onClientThread(line);
		}
		String[] parts = line.split(" ", 3);
		if (parts.length != 3 || !parts[0].equals("CAPTURE")) {
			return "ERR expected: CAPTURE <name> <dir>";
		}
		if (!NAME.matcher(parts[1]).matches()) {
			return "ERR bad name: " + parts[1];
		}
		Request request = new Request(Path.of(parts[2]), parts[1], new CompletableFuture<>());
		PENDING.add(request);
		try {
			return request.done().get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
		} catch (Exception e) {
			// Leave no half-done request behind to be served later with a stale reply.
			PENDING.remove(request);
			Throwable cause = e.getCause() != null ? e.getCause() : e;
			return "ERR " + cause.toString().replace('\n', ' ');
		}
	}

	private static String onClientThread(String line) {
		Minecraft minecraft = Minecraft.getInstance();
		try {
			if (line.equals("STATUS")) {
				return "STATUS " + minecraft.submit(() -> Probe.status(minecraft)).get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
			}
			if (line.equals("REFRESH")) {
				// After a teleport the game's list of sections in view can miss chunks that
				// arrived while it was being worked out; far terrain then stays undrawn (and
				// the mobs on it with it) until the camera moves. This works the list out again.
				minecraft.submit(() -> minecraft.levelRenderer.needsUpdate()).get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
				return "OK REFRESH";
			}
			if (line.equals("REBUILD")) {
				// Last resort when terrain stays missing: drop and rebuild every section (F3+A).
				minecraft.submit(() -> minecraft.levelRenderer.allChanged()).get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
				return "OK REBUILD";
			}
			String[] parts = line.split(" ");
			if (parts[0].equals("ROOMS")) {
				if (parts.length != 4) {
					return "ERR expected: ROOMS <radius> <count> <seed>";
				}
				int radius = Integer.parseInt(parts[1]);
				int count = Integer.parseInt(parts[2]);
				long seed = Long.parseLong(parts[3]);
				return "ROOMS " + minecraft.submit(() -> Probe.rooms(minecraft, radius, count, seed)).get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
			}
			if (parts.length != 4) {
				return "ERR expected: PROBE <samples> <max_distance> <seed>";
			}
			int samples = Integer.parseInt(parts[1]);
			double maxDistance = Double.parseDouble(parts[2]);
			long seed = Long.parseLong(parts[3]);
			return "PROBE " + minecraft.submit(() -> Probe.probe(minecraft, samples, maxDistance, seed)).get(TIMEOUT_SECONDS, TimeUnit.SECONDS);
		} catch (Exception e) {
			Throwable cause = e.getCause() != null ? e.getCause() : e;
			return "ERR " + cause.toString().replace('\n', ' ');
		}
	}

	/** Frame start: picks up a waiting request, which makes this a label frame. */
	public static void beginFrame() {
		if (current == null) {
			current = PENDING.poll();
		}
		LABELLED.clear();
	}

	/** Whether the frame being drawn is a label frame. */
	public static boolean isLabelFrame() {
		return current != null;
	}

	/** Frame end: a request whose frame was read back is done with. */
	public static void endFrame() {
		if (readThisFrame) {
			current = null;
			readThisFrame = false;
		}
	}

	/**
	 * Label colour for an entity about to be drawn: its 1-based index among this frame's mobs,
	 * {@link #OTHER} for other entities. Invisible mobs are labelled too: the camera is a
	 * spectator, which sees them as faint ghosts, and their armour fully.
	 *
	 * @param state the entity's render state, whose position is where it is drawn this frame
	 */
	public static int colourFor(Entity entity, EntityRenderState state) {
		if (!isLabelled(entity) || LABELLED.size() + 1 >= OTHER) {
			return ARGB.opaque(OTHER);
		}
		LABELLED.add(new Labelled(entity, state.x, state.y, state.z));
		return ARGB.opaque(LABELLED.size());
	}

	/**
	 * Whether an entity gets its own label: mobs, and mannequins (player-shaped, standing in
	 * for players). Armour stands and other entities count as background ({@link #OTHER}).
	 */
	private static boolean isLabelled(Entity entity) {
		return entity instanceof Mob || entity instanceof Mannequin;
	}

	/** A mob drawn in a label frame, with where it was drawn (between game ticks). */
	private record Labelled(Entity entity, double x, double y, double z) {
	}

	/**
	 * Reads back the picture and the label image of the frame just drawn, then writes them.
	 *
	 * @param labelTarget the glowing-outline target the labels were drawn into
	 */
	public static void readBack(Minecraft minecraft, RenderTarget labelTarget) {
		Request request = current;
		if (request == null || readThisFrame) {
			return;
		}
		readThisFrame = true;
		if (labelTarget == null || labelTarget.getColorTexture() == null) {
			request.done().completeExceptionally(new IllegalStateException("no label target"));
			return;
		}
		String info = describe(minecraft);
		AtomicReference<Pixels> picture = new AtomicReference<>();
		AtomicReference<Pixels> labels = new AtomicReference<>();
		Runnable whenBoth = () -> {
			if (picture.get() != null && labels.get() != null) {
				// The frame is safe in memory: answer now, write in the background.
				request.done().complete("OK " + request.name());
				WRITER.execute(() -> write(request, picture.get(), labels.get(), info));
			}
		};
		copyOut(minecraft.getMainRenderTarget().getColorTexture(), pixels -> {
			picture.set(pixels);
			whenBoth.run();
		});
		copyOut(labelTarget.getColorTexture(), pixels -> {
			labels.set(pixels);
			whenBoth.run();
		});
	}

	/** A copy of a texture's pixels as the GPU gave them: RGBA bytes, rows bottom-up. */
	private record Pixels(ByteBuffer data, int width, int height) {
		NativeImage toImage() {
			NativeImage image = new NativeImage(width, height, false);
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int abgr = data.getInt((x + y * width) * 4);
					image.setPixelABGR(x, height - y - 1, abgr | 0xFF000000);
				}
			}
			return image;
		}
	}

	private static void copyOut(GpuTexture texture, Consumer<Pixels> sink) {
		int width = texture.getWidth(0);
		int height = texture.getHeight(0);
		int pixelSize = texture.getFormat().pixelSize();
		if (pixelSize != 4) {
			throw new IllegalStateException("expected an RGBA8 texture, got " + texture.getFormat());
		}
		CommandEncoder encoder = RenderSystem.getDevice().createCommandEncoder();
		GpuBuffer buffer = RenderSystem.getDevice()
			.createBuffer(() -> "mobcapture readback", GpuBuffer.USAGE_MAP_READ | GpuBuffer.USAGE_COPY_DST, (long) width * height * pixelSize);
		encoder.copyTextureToBuffer(texture, buffer, 0L, () -> {
			ByteBuffer copy = ByteBuffer.allocateDirect(width * height * pixelSize).order(ByteOrder.LITTLE_ENDIAN);
			try (GpuBuffer.MappedView view = encoder.mapBuffer(buffer, true, false)) {
				copy.put(view.data().duplicate().order(ByteOrder.LITTLE_ENDIAN));
			}
			buffer.close();
			sink.accept(new Pixels(copy.clear(), width, height));
		}, 0);
	}

	private static String describe(Minecraft minecraft) {
		JsonObject root = new JsonObject();
		Camera camera = minecraft.gameRenderer.getMainCamera();
		JsonObject cam = new JsonObject();
		cam.addProperty("x", camera.position().x);
		cam.addProperty("y", camera.position().y);
		cam.addProperty("z", camera.position().z);
		cam.addProperty("yaw", camera.yRot());
		cam.addProperty("pitch", camera.xRot());
		root.add("camera", cam);
		root.addProperty("width", minecraft.getMainRenderTarget().width);
		root.addProperty("height", minecraft.getMainRenderTarget().height);
		if (minecraft.level != null) {
			root.addProperty("dimension", minecraft.level.dimension().identifier().toString());
			root.addProperty("game_time", minecraft.level.getGameTime());
			root.addProperty("day_time", minecraft.level.getDayTime());
			root.addProperty("raining", minecraft.level.isRaining());
			root.addProperty("thundering", minecraft.level.isThundering());
		}
		root.addProperty("other_colour", OTHER);
		JsonArray mobs = new JsonArray();
		for (int i = 0; i < LABELLED.size(); i++) {
			Labelled drawn = LABELLED.get(i);
			Entity entity = drawn.entity();
			JsonObject mob = new JsonObject();
			mob.addProperty("colour", i + 1);
			mob.addProperty("type", BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString());
			mob.addProperty("id", entity.getId());
			mob.addProperty("uuid", entity.getUUID().toString());
			mob.addProperty("x", drawn.x());
			mob.addProperty("y", drawn.y());
			mob.addProperty("z", drawn.z());
			mob.addProperty("yaw", entity.getYRot());
			mob.addProperty("invisible", entity.isInvisible());
			// The game's box is at the last tick; shift it to where the mob was drawn.
			AABB box = entity.getBoundingBox().move(drawn.x() - entity.getX(), drawn.y() - entity.getY(), drawn.z() - entity.getZ());
			JsonArray aabb = new JsonArray();
			for (double v : new double[] {box.minX, box.minY, box.minZ, box.maxX, box.maxY, box.maxZ}) {
				aabb.add(v);
			}
			mob.add("aabb", aabb);
			mobs.add(mob);
		}
		root.add("mobs", mobs);
		// Mobs the window knows about but did not draw this frame (out of view, or the game
		// skipped them), for finding why a summoned mob is missing.
		JsonArray skipped = new JsonArray();
		if (minecraft.level != null) {
			java.util.Set<Entity> drawn = new java.util.HashSet<>();
			LABELLED.forEach(l -> drawn.add(l.entity()));
			for (Entity entity : minecraft.level.entitiesForRendering()) {
				if (isLabelled(entity) && !drawn.contains(entity)) {
					JsonObject mob = new JsonObject();
					mob.addProperty("type", BuiltInRegistries.ENTITY_TYPE.getKey(entity.getType()).toString());
					mob.addProperty("x", entity.getX());
					mob.addProperty("y", entity.getY());
					mob.addProperty("z", entity.getZ());
					mob.addProperty("section_visible", minecraft.levelRenderer.isSectionCompiledAndVisible(entity.blockPosition()));
					skipped.add(mob);
				}
			}
		}
		root.add("not_drawn", skipped);
		root.add("terrain", Probe.holes(minecraft, 48, 27));
		return new Gson().toJson(root);
	}

	// Each file appears under its final name only when complete; the JSON comes last, so
	// readers wait for it.
	private static void write(Request request, Pixels picture, Pixels labels, String info) {
		Path dir = request.dir();
		String name = request.name();
		try (NativeImage pictureImage = picture.toImage(); NativeImage labelImage = labels.toImage()) {
			Files.createDirectories(dir);
			// Temporary names of their own, so two shots of one name never mix files.
			String tmp = ".tmp" + System.nanoTime();
			pictureImage.writeToFile(dir.resolve(name + ".png" + tmp));
			labelImage.writeToFile(dir.resolve(name + ".label.png" + tmp));
			Files.writeString(dir.resolve(name + ".json" + tmp), info);
			for (String suffix : new String[] {".png", ".label.png", ".json"}) {
				Files.move(dir.resolve(name + suffix + tmp), dir.resolve(name + suffix), StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
			}
		} catch (Exception e) {
			LOGGER.error("mobcapture could not write shot {} in {}", name, dir, e);
		}
	}
}

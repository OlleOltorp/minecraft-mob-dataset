package io.github.olleoltorp.mobcapture.label;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import io.github.olleoltorp.mobcapture.mixin.ClientLevelAccessor;
import io.github.olleoltorp.mobcapture.mixin.LevelRendererAccessor;
import io.github.olleoltorp.mobcapture.mixin.SectionOcclusionGraphAccessor;
import io.github.olleoltorp.mobcapture.mixin.ViewAreaInvoker;
import net.minecraft.client.renderer.chunk.CompiledSectionMesh;
import net.minecraft.core.SectionPos;
import java.util.Random;
import java.util.concurrent.Future;
import net.minecraft.client.Camera;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.chunk.SectionRenderDispatcher;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.tags.FluidTags;
import net.minecraft.world.level.ClipContext;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.BlockHitResult;
import net.minecraft.world.phys.HitResult;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.CollisionContext;
import org.joml.Vector3f;

/**
 * Looks at the world the window sees, for placing mobs where they fit and can be seen.
 *
 * <p>Rays go out through random points of the picture. Where a ray meets the top of a block
 * that is a ground spot; where it enters water, a water spot below the surface; lava gives a
 * lava-surface spot; and a point part-way along the ray gives an air spot. Each spot says how
 * much free room it has, so the caller can pick spots a mob of a given size fits in.
 * Must run on the client thread.
 */
public final class Probe {
	private static final int MAX_HEIGHT = 8;
	private static final int MAX_RADIUS = 2;
	private static final int MAX_DROP = 64;
	/** Most free section build buffers seen: all of them, when nothing is being built. */
	private static int buffersIdle;

	private Probe() {
	}

	/** Terrain loading state: chunks the window has, sections still waiting to be built. */
	public static String status(Minecraft minecraft) {
		JsonObject root = new JsonObject();
		root.addProperty("in_world", minecraft.level != null);
		if (minecraft.level != null) {
			root.addProperty("loaded_chunks", minecraft.level.getChunkSource().getLoadedChunksCount());
			SectionRenderDispatcher sections = minecraft.levelRenderer.getSectionRenderDispatcher();
			root.addProperty("compile_queue", sections == null ? -1 : sections.getCompileQueueSize());
			root.addProperty("to_upload", sections == null ? -1 : sections.getToUpload());
			root.addProperty("rendered_sections", minecraft.levelRenderer.countRenderedSections());
			// The compile queue drops a task once a worker takes it, and says nothing about
			// sections not yet known to be in view, so it reads 0 while terrain is still being
			// built. These say what is really left.
			LevelRendererAccessor renderer = (LevelRendererAccessor) minecraft.levelRenderer;
			int unbuilt = 0;
			for (SectionRenderDispatcher.RenderSection section : renderer.mobcapture$visibleSections()) {
				// Sections at the edge of view lack neighbours and are never built.
				if (section.isDirty() && section.hasAllNeighbors()) {
					unbuilt++;
				}
			}
			root.addProperty("unbuilt_in_view", unbuilt);
			root.addProperty(
				"light_pending",
				((ClientLevelAccessor) minecraft.level).mobcapture$lightUpdateQueue().size()
					+ (minecraft.level.getLightEngine().hasLightWork() ? 1 : 0)
			);
			int free = minecraft.renderBuffers().sectionBufferPool().getFreeBufferCount();
			buffersIdle = Math.max(buffersIdle, free);
			root.addProperty("building", buffersIdle - free);
			SectionOcclusionGraphAccessor graph = (SectionOcclusionGraphAccessor) renderer.mobcapture$occlusionGraph();
			Future<?> task = graph.mobcapture$fullUpdateTask();
			root.addProperty(
				"view_list_busy",
				graph.mobcapture$needsFullUpdate() || (task != null && !task.isDone()) || graph.mobcapture$needsFrustumUpdate().get()
			);
			// Share of the chunks within view distance that the window has: new terrain arrives
			// from the server in bursts, so "nothing changed for a while" is not enough.
			int radius = minecraft.options.getEffectiveRenderDistance();
			BlockPos at = BlockPos.containing(minecraft.gameRenderer.getMainCamera().position());
			int cx = at.getX() >> 4;
			int cz = at.getZ() >> 4;
			int want = 0;
			int have = 0;
			for (int dx = -radius; dx <= radius; dx++) {
				for (int dz = -radius; dz <= radius; dz++) {
					if (dx * dx + dz * dz <= radius * radius) {
						want++;
						if (ready(minecraft.level, cx + dx, cz + dz)) {
							have++;
						}
					}
				}
			}
			root.addProperty("view_chunks", want);
			root.addProperty("view_chunks_loaded", have);
			root.add("camera", camera(minecraft));
		}
		return new Gson().toJson(root);
	}

	/**
	 * Spots in view for placing mobs.
	 *
	 * @param samples rays to cast
	 * @param maxDistance longest ray, in blocks
	 * @param seed seed for the ray directions
	 */
	public static String probe(Minecraft minecraft, int samples, double maxDistance, long seed) {
		ClientLevel level = minecraft.level;
		if (level == null) {
			throw new IllegalStateException("not in a world");
		}
		Camera camera = minecraft.gameRenderer.getMainCamera();
		Vec3 eye = camera.position();
		Vector3f forward = new Vector3f(camera.forwardVector());
		Vector3f up = new Vector3f(camera.upVector());
		Vector3f right = new Vector3f(camera.leftVector()).negate();
		double tanV = Math.tan(Math.toRadians(minecraft.options.fov().get() / 2.0));
		double aspect = (double) minecraft.getWindow().getWidth() / minecraft.getWindow().getHeight();
		double tanH = tanV * aspect;
		Random random = new Random(seed);
		JsonArray spots = new JsonArray();
		for (int i = 0; i < samples; i++) {
			double sx = random.nextDouble() * 2 - 1;
			double sy = random.nextDouble() * 2 - 1;
			Vec3 dir = new Vec3(
				forward.x() + sx * tanH * right.x() + sy * tanV * up.x(),
				forward.y() + sx * tanH * right.y() + sy * tanV * up.y(),
				forward.z() + sx * tanH * right.z() + sy * tanV * up.z()
			).normalize();
			Vec3 end = eye.add(dir.scale(maxDistance));
			BlockHitResult solid = level.clip(new ClipContext(eye, end, ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, CollisionContext.empty()));
			BlockHitResult fluid = level.clip(new ClipContext(eye, end, ClipContext.Block.COLLIDER, ClipContext.Fluid.ANY, CollisionContext.empty()));
			boolean hitSolid = solid.getType() != HitResult.Type.MISS;
			double solidDistance = hitSolid ? solid.getLocation().distanceTo(eye) : maxDistance;
			if (hitSolid && !loaded(level, solid.getBlockPos())) {
				continue;
			}
			if (fluid.getType() != HitResult.Type.MISS) {
				double fluidDistance = fluid.getLocation().distanceTo(eye);
				BlockPos fluidPos = fluid.getBlockPos();
				if (fluidDistance < solidDistance - 0.01 && loaded(level, fluidPos)) {
					BlockState state = level.getBlockState(fluidPos);
					if (state.getFluidState().is(FluidTags.WATER)) {
						// A point under the surface along the same ray.
						double reach = Math.min(solidDistance, fluidDistance + 12) - 0.3;
						if (reach > fluidDistance + 0.5) {
							Vec3 p = eye.add(dir.scale(fluidDistance + 0.5 + random.nextDouble() * (reach - fluidDistance - 0.5)));
							BlockPos pos = BlockPos.containing(p);
							if (isWater(level, pos)) {
								spots.add(spot(level, eye, "water", p, sx, sy));
							}
						}
					} else if (state.getFluidState().is(FluidTags.LAVA)) {
						Vec3 p = new Vec3(fluid.getLocation().x, fluidPos.getY() + state.getFluidState().getHeight(level, fluidPos), fluid.getLocation().z);
						spots.add(spot(level, eye, "lava", p, sx, sy));
					}
				}
			}
			if (hitSolid && solid.getDirection() == Direction.UP) {
				Vec3 p = solid.getLocation();
				BlockPos feet = BlockPos.containing(p.x, p.y + 0.01, p.z);
				String kind = isWater(level, feet) ? "water_floor" : "ground";
				spots.add(spot(level, eye, kind, p, sx, sy));
			}
			// A point in the open part-way along the ray, for flying mobs.
			Vec3 p = eye.add(dir.scale(solidDistance * (0.15 + 0.75 * random.nextDouble())));
			BlockPos pos = BlockPos.containing(p);
			if (loaded(level, pos) && isFree(level, pos, false)) {
				spots.add(spot(level, eye, "air", p, sx, sy));
			}
		}
		JsonObject root = new JsonObject();
		root.add("camera", camera(minecraft));
		BlockPos at = BlockPos.containing(eye);
		root.addProperty("biome", biome(level, at));
		root.addProperty("dimension", level.dimension().identifier().toString());
		root.addProperty("camera_in_fluid", camera.getFluidInCamera().toString());
		root.addProperty("precipitation", level.getBiome(at).value().getPrecipitationAt(at, level.getSeaLevel()).toString());
		root.add("spots", spots);
		return new Gson().toJson(root);
	}

	/**
	 * Open places near the camera to put it, for caves and the Nether: three free blocks
	 * above a floor, scored by how far one sees from there.
	 *
	 * @param radius search half-size in blocks (vertically half of it)
	 * @param count places to return, best views first among a random sample
	 * @param seed seed for the sample
	 */
	public static String rooms(Minecraft minecraft, int radius, int count, long seed) {
		ClientLevel level = minecraft.level;
		if (level == null) {
			throw new IllegalStateException("not in a world");
		}
		BlockPos centre = BlockPos.containing(minecraft.gameRenderer.getMainCamera().position());
		Random random = new Random(seed);
		java.util.List<BlockPos> found = new java.util.ArrayList<>();
		int vertical = Math.max(4, radius / 2);
		for (int dy = -vertical; dy <= vertical; dy++) {
			for (int dx = -radius; dx <= radius; dx++) {
				for (int dz = -radius; dz <= radius; dz++) {
					BlockPos feet = centre.offset(dx, dy, dz);
					if (isFree(level, feet, false) && isFree(level, feet.above(), false) && isFree(level, feet.above(2), false)
						&& !isFree(level, feet.below(), false) && !level.getBlockState(feet.below()).getCollisionShape(level, feet.below()).isEmpty()) {
						found.add(feet);
					}
				}
			}
		}
		java.util.Collections.shuffle(found, random);
		java.util.List<JsonObject> scored = new java.util.ArrayList<>();
		for (BlockPos feet : found.subList(0, Math.min(found.size(), 300))) {
			Vec3 eye = new Vec3(feet.getX() + 0.5, feet.getY() + 1.62, feet.getZ() + 0.5);
			double sum = 0;
			for (int k = 0; k < 16; k++) {
				double a = Math.PI * 2 * k / 16;
				Vec3 end = eye.add(Math.cos(a) * 48, (k % 3 - 1) * 8, Math.sin(a) * 48);
				BlockHitResult hit = level.clip(new ClipContext(eye, end, ClipContext.Block.COLLIDER, ClipContext.Fluid.ANY, CollisionContext.empty()));
				sum += hit.getType() == HitResult.Type.MISS ? 48 : hit.getLocation().distanceTo(eye);
			}
			JsonObject room = new JsonObject();
			room.addProperty("x", feet.getX() + 0.5);
			room.addProperty("y", feet.getY());
			room.addProperty("z", feet.getZ() + 0.5);
			room.addProperty("view", sum / 16);
			room.addProperty("sky_light", level.getBrightness(LightLayer.SKY, feet));
			room.addProperty("block_light", level.getBrightness(LightLayer.BLOCK, feet));
			room.addProperty("biome", biome(level, feet));
			scored.add(room);
		}
		scored.sort((p, q) -> Double.compare(q.get("view").getAsDouble(), p.get("view").getAsDouble()));
		JsonArray rooms = new JsonArray();
		scored.stream().limit(count).forEach(rooms::add);
		JsonObject root = new JsonObject();
		root.addProperty("candidates", found.size());
		root.add("rooms", rooms);
		return new Gson().toJson(root);
	}

	private static JsonObject camera(Minecraft minecraft) {
		Camera camera = minecraft.gameRenderer.getMainCamera();
		JsonObject cam = new JsonObject();
		cam.addProperty("x", camera.position().x);
		cam.addProperty("y", camera.position().y);
		cam.addProperty("z", camera.position().z);
		cam.addProperty("yaw", camera.yRot());
		cam.addProperty("pitch", camera.xRot());
		return cam;
	}

	/**
	 * Holes in the drawn terrain: rays through a grid of points of the picture that meet a
	 * block the window has but whose section it does not draw. Such a spot shows sky or
	 * black instead of ground (and hides any mob standing there).
	 */
	public static JsonObject holes(Minecraft minecraft, int columns, int rows) {
		JsonObject out = new JsonObject();
		ClientLevel level = minecraft.level;
		if (level == null) {
			return out;
		}
		Camera camera = minecraft.gameRenderer.getMainCamera();
		Vec3 eye = camera.position();
		Vector3f forward = new Vector3f(camera.forwardVector());
		Vector3f up = new Vector3f(camera.upVector());
		Vector3f right = new Vector3f(camera.leftVector()).negate();
		double tanV = Math.tan(Math.toRadians(minecraft.options.fov().get() / 2.0));
		double tanH = tanV * minecraft.getWindow().getWidth() / minecraft.getWindow().getHeight();
		double reach = minecraft.options.getEffectiveRenderDistance() * 16.0;
		// A built section the game has dropped from its list of sections in view is not drawn either.
		java.util.Set<SectionRenderDispatcher.RenderSection> inView =
			new java.util.HashSet<>(((LevelRendererAccessor) minecraft.levelRenderer).mobcapture$visibleSections());
		var area = ((LevelRendererAccessor) minecraft.levelRenderer).mobcapture$viewArea();
		int hits = 0;
		int holes = 0;
		double nearest = -1;
		for (int i = 0; i < columns; i++) {
			for (int j = 0; j < rows; j++) {
				double sx = (i + 0.5) / columns * 2 - 1;
				double sy = (j + 0.5) / rows * 2 - 1;
				Vec3 dir = new Vec3(
					forward.x() + sx * tanH * right.x() + sy * tanV * up.x(),
					forward.y() + sx * tanH * right.y() + sy * tanV * up.y(),
					forward.z() + sx * tanH * right.z() + sy * tanV * up.z()
				).normalize();
				BlockHitResult hit = level.clip(new ClipContext(eye, eye.add(dir.scale(reach)), ClipContext.Block.COLLIDER, ClipContext.Fluid.NONE, CollisionContext.empty()));
				if (hit.getType() == HitResult.Type.MISS || !loaded(level, hit.getBlockPos())) {
					continue;
				}
				hits++;
				SectionRenderDispatcher.RenderSection section = area == null ? null : ((ViewAreaInvoker) area).mobcapture$sectionAt(hit.getBlockPos());
				if (!minecraft.levelRenderer.isSectionCompiledAndVisible(hit.getBlockPos()) || !inView.contains(section)) {
					if (holes == 0) {
						out.add("first_hole", whyUndrawn(minecraft, hit.getBlockPos(), eye));
					}
					holes++;
					double d = hit.getLocation().distanceTo(eye);
					nearest = nearest < 0 ? d : Math.min(nearest, d);
				}
			}
		}
		out.addProperty("rays", columns * rows);
		out.addProperty("hits", hits);
		out.addProperty("holes", holes);
		out.addProperty("nearest_hole", nearest);
		return out;
	}

	/**
	 * Whether the window has a chunk and its light, as the game needs before it builds the
	 * chunk's terrain. (The client world's own hasChunk answers yes for every chunk.)
	 */
	private static boolean ready(ClientLevel level, int x, int z) {
		return level.getChunkSource().hasChunk(x, z) && level.getLightEngine().lightOnInColumn(SectionPos.asLong(x, 0, z));
	}

	/** Whether the window has the chunk of a block; a missing chunk reads as all air. */
	private static boolean loaded(ClientLevel level, BlockPos pos) {
		return level.getChunkSource().hasChunk(pos.getX() >> 4, pos.getZ() >> 4);
	}

	/** What keeps the window from drawing the terrain section of a block. */
	private static JsonObject whyUndrawn(Minecraft minecraft, BlockPos pos, Vec3 eye) {
		JsonObject why = new JsonObject();
		why.addProperty("distance", Math.sqrt(pos.distToCenterSqr(eye)));
		why.addProperty("chunk_dx", (pos.getX() >> 4) - (BlockPos.containing(eye).getX() >> 4));
		why.addProperty("chunk_dz", (pos.getZ() >> 4) - (BlockPos.containing(eye).getZ() >> 4));
		why.addProperty("light_on", minecraft.level.getLightEngine().lightOnInColumn(SectionPos.getZeroNode(SectionPos.asLong(pos))));
		LevelRendererAccessor renderer = (LevelRendererAccessor) minecraft.levelRenderer;
		var area = renderer.mobcapture$viewArea();
		SectionRenderDispatcher.RenderSection section = area == null ? null : ((ViewAreaInvoker) area).mobcapture$sectionAt(pos);
		why.addProperty("has_section", section != null);
		if (section != null) {
			var mesh = section.getSectionMesh();
			why.addProperty("mesh", mesh == CompiledSectionMesh.UNCOMPILED ? "uncompiled" : mesh == CompiledSectionMesh.EMPTY ? "empty" : "built");
			why.addProperty("dirty", section.isDirty());
			why.addProperty("all_neighbours", section.hasAllNeighbors());
			why.addProperty("in_view_list", renderer.mobcapture$visibleSections().contains(section));
		}
		// The neighbour chunks a section needs before it is built: present, and lit.
		JsonArray missing = new JsonArray();
		int cx = pos.getX() >> 4;
		int cz = pos.getZ() >> 4;
		for (int dx = -1; dx <= 1; dx++) {
			for (int dz = -1; dz <= 1; dz++) {
				boolean present = minecraft.level.getChunkSource().getChunk(cx + dx, cz + dz, false) != null;
				boolean lit = minecraft.level.getLightEngine().lightOnInColumn(SectionPos.asLong(cx + dx, 0, cz + dz));
				if (!present || !lit) {
					JsonObject n = new JsonObject();
					n.addProperty("dx", dx);
					n.addProperty("dz", dz);
					n.addProperty("present", present);
					n.addProperty("lit", lit);
					missing.add(n);
				}
			}
		}
		why.add("neighbours_not_ready", missing);
		return why;
	}

	private static JsonObject spot(ClientLevel level, Vec3 eye, String kind, Vec3 p, double sx, double sy) {
		BlockPos pos = BlockPos.containing(p.x, p.y + 0.01, p.z);
		boolean water = !kind.equals("ground") && !kind.equals("air") && !kind.equals("lava");
		JsonObject spot = new JsonObject();
		spot.addProperty("kind", kind);
		spot.addProperty("x", p.x);
		spot.addProperty("y", p.y);
		spot.addProperty("z", p.z);
		spot.addProperty("distance", p.distanceTo(eye));
		spot.addProperty("sx", sx);
		spot.addProperty("sy", sy);
		spot.addProperty("biome", biome(level, pos));
		spot.addProperty("sky_light", level.getBrightness(LightLayer.SKY, pos));
		spot.addProperty("block_light", level.getBrightness(LightLayer.BLOCK, pos));
		BlockState below = level.getBlockState(pos.below());
		spot.addProperty("below", BuiltInRegistries.BLOCK.getKey(below.getBlock()).toString());
		// Free height for a column of each half-width 0..MAX_RADIUS, centred on the spot.
		JsonArray clear = new JsonArray();
		for (int r = 0; r <= MAX_RADIUS; r++) {
			clear.add(freeHeight(level, pos, r, water));
		}
		spot.add("clear", clear);
		if (kind.equals("air")) {
			int drop = 0;
			while (drop < MAX_DROP && isFree(level, pos.below(drop + 1), false)) {
				drop++;
			}
			spot.addProperty("above_ground", drop);
		}
		if (water) {
			int depth = 0;
			while (depth < 32 && isWater(level, pos.above(depth + 1))) {
				depth++;
			}
			spot.addProperty("water_above", depth);
		}
		return spot;
	}

	private static int freeHeight(ClientLevel level, BlockPos base, int radius, boolean water) {
		for (int h = 0; h < MAX_HEIGHT; h++) {
			for (int dx = -radius; dx <= radius; dx++) {
				for (int dz = -radius; dz <= radius; dz++) {
					if (!isFree(level, base.offset(dx, h, dz), water)) {
						return h;
					}
				}
			}
		}
		return MAX_HEIGHT;
	}

	/** No collision; and either no fluid, or (when water is allowed) only water. */
	private static boolean isFree(ClientLevel level, BlockPos pos, boolean water) {
		if (!loaded(level, pos)) {
			return false;
		}
		BlockState state = level.getBlockState(pos);
		if (!state.getCollisionShape(level, pos).isEmpty()) {
			return false;
		}
		return state.getFluidState().isEmpty() || (water && state.getFluidState().is(FluidTags.WATER));
	}

	private static boolean isWater(ClientLevel level, BlockPos pos) {
		return loaded(level, pos) && level.getBlockState(pos).getFluidState().is(FluidTags.WATER)
			&& level.getBlockState(pos).getCollisionShape(level, pos).isEmpty();
	}

	private static String biome(ClientLevel level, BlockPos pos) {
		return level.getBiome(pos).unwrapKey().map(key -> key.identifier().toString()).orElse("unknown");
	}
}

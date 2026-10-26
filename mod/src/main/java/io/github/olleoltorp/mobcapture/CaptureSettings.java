package io.github.olleoltorp.mobcapture;

/**
 * Capture settings, read once from JVM system properties set by the launcher
 * ({@code game/client.py} builds them from {@code config/client.toml}).
 */
public final class CaptureSettings {
	/** Draw every entity up to this many blocks away, whatever its size; 0 keeps vanilla. */
	public static final double ENTITY_RENDER_DISTANCE =
		Double.parseDouble(System.getProperty("mobcapture.entityRenderDistance", "0"));

	/** Hide pop-ups (toasts) such as tutorial hints and advancement notices. */
	public static final boolean HIDE_TOASTS = Boolean.getBoolean("mobcapture.hideToasts");

	/**
	 * Skip the game's guess that far terrain sections are hidden (it drops sections in plain
	 * view, and the mobs on them, depending on the order it looks at them).
	 */
	public static final boolean EXACT_CULLING = Boolean.getBoolean("mobcapture.exactCulling");

	/** Local TCP port for capture requests (protocol/label-capture.md); 0 turns it off. */
	public static final int CONTROL_PORT = Integer.getInteger("mobcapture.controlPort", 0);

	private CaptureSettings() {
	}
}

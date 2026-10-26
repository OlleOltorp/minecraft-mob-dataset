package io.github.olleoltorp.mobcapture.mixin;

import io.github.olleoltorp.mobcapture.CaptureSettings;
import net.minecraft.client.renderer.SectionOcclusionGraph;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.Constant;
import org.spongepowered.asm.mixin.injection.ModifyConstant;

/**
 * Turns off the game's shortcut for terrain sections more than 60 blocks away: it keeps such a
 * section only if a line from it towards the camera passes through sections already looked
 * at, so whether a far section in plain view is drawn depends on the order of the search.
 * That leaves holes in far terrain, and mobs standing there are not drawn. The exact check
 * (can one face of a section be seen from another) stays.
 */
@Mixin(SectionOcclusionGraph.class)
public abstract class ExactCullingMixin {
	// The shortcut walks from the section towards the camera while farther than 60 blocks
	// (60^2 = 3600); an endless limit means the walk never starts and the section is kept.
	@ModifyConstant(method = "runUpdates", constant = @Constant(doubleValue = 3600.0))
	private double mobcapture$noFarShortcut(double sqrDistance) {
		return CaptureSettings.EXACT_CULLING ? Double.POSITIVE_INFINITY : sqrDistance;
	}
}

package io.github.olleoltorp.mobcapture.mixin;

import io.github.olleoltorp.mobcapture.CaptureSettings;
import net.minecraft.world.entity.Entity;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Draws every mob up to a fixed distance. The game hides an entity beyond bounding-box size x 64
 * x view scale, where view scale = clamp(render distance / 8, 1, 2.5) x entity distance setting.
 * Even at the maximum (12.5) a silverfish, endermite or tadpole vanishes at about 290 blocks,
 * and at lower render distances far sooner; far-distance captures need the same reach for every
 * mob. Phantoms override this to always draw; PhantomRenderDistanceMixin limits them too.
 */
@Mixin(Entity.class)
public abstract class EntityRenderDistanceMixin {
	@Inject(method = "shouldRenderAtSqrDistance", at = @At("HEAD"), cancellable = true)
	private void mobcapture$fixedRenderDistance(
			double sqrDistance, CallbackInfoReturnable<Boolean> cir) {
		double reach = CaptureSettings.ENTITY_RENDER_DISTANCE;
		if (reach > 0) {
			cir.setReturnValue(sqrDistance < reach * reach);
		}
	}
}

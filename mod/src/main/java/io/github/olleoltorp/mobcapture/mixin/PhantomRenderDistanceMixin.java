package io.github.olleoltorp.mobcapture.mixin;

import io.github.olleoltorp.mobcapture.CaptureSettings;
import net.minecraft.world.entity.monster.Phantom;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Gives phantoms the same fixed reach as every other mob. Phantoms override the render
 * distance check to always draw, so EntityRenderDistanceMixin does not reach them; without
 * this, phantoms cleared into the void far below would be drawn into the label image where
 * the picture shows only fog.
 */
@Mixin(Phantom.class)
public abstract class PhantomRenderDistanceMixin {
	@Inject(method = "shouldRenderAtSqrDistance", at = @At("HEAD"), cancellable = true)
	private void mobcapture$fixedRenderDistance(
			double sqrDistance, CallbackInfoReturnable<Boolean> cir) {
		double reach = CaptureSettings.ENTITY_RENDER_DISTANCE;
		if (reach > 0) {
			cir.setReturnValue(sqrDistance < reach * reach);
		}
	}
}

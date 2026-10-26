package io.github.olleoltorp.mobcapture.mixin;

import io.github.olleoltorp.mobcapture.label.LabelCapture;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Frame bounds for label frames, and the read-back right after the world is drawn. */
@Mixin(GameRenderer.class)
abstract class LabelGameRendererMixin {
	@Shadow
	@Final
	private Minecraft minecraft;

	@Inject(method = "render", at = @At("HEAD"))
	private void mobcapture$beginFrame(DeltaTracker deltaTracker, boolean renderLevel, CallbackInfo ci) {
		LabelCapture.beginFrame();
	}

	// Before the HUD is drawn and before the outlines are blended onto the picture.
	@Inject(
		method = "render",
		at = @At(value = "INVOKE", target = "Lnet/minecraft/client/renderer/GameRenderer;tryTakeScreenshotIfNeeded()V")
	)
	private void mobcapture$readBack(DeltaTracker deltaTracker, boolean renderLevel, CallbackInfo ci) {
		if (LabelCapture.isLabelFrame()) {
			LabelCapture.readBack(minecraft, ((LevelRendererAccessor) minecraft.levelRenderer).mobcapture$outlineTarget());
		}
	}

	@Inject(method = "render", at = @At("RETURN"))
	private void mobcapture$endFrame(DeltaTracker deltaTracker, boolean renderLevel, CallbackInfo ci) {
		LabelCapture.endFrame();
	}
}

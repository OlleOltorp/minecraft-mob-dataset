package io.github.olleoltorp.mobcapture.mixin;

import io.github.olleoltorp.mobcapture.CaptureSettings;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.components.BossHealthOverlay;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Draws no boss health bars (wither, ender dragon, raids). Pictures are taken before the HUD,
 * so the bars never reach the dataset, but they would still cover the capture windows; with
 * the same setting as the toasts, nothing overlays the windows either.
 */
@Mixin(BossHealthOverlay.class)
public abstract class HideBossBarMixin {
	@Inject(method = "render", at = @At("HEAD"), cancellable = true)
	private void mobcapture$hideBossBars(GuiGraphics graphics, CallbackInfo ci) {
		if (CaptureSettings.HIDE_TOASTS) {
			ci.cancel();
		}
	}
}

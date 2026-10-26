package io.github.olleoltorp.mobcapture.mixin;

import io.github.olleoltorp.mobcapture.CaptureSettings;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.client.gui.components.toasts.ToastManager;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Draws no pop-ups (toasts) in the top-right corner. Capture windows otherwise show the tutorial,
 * "chat messages can't be verified" on offline-mode servers, advancements and recipes, all of
 * which would land in the pictures. Skipping the drawing catches every kind, however added.
 */
@Mixin(ToastManager.class)
public abstract class HideToastsMixin {
	@Inject(method = "render", at = @At("HEAD"), cancellable = true)
	private void mobcapture$hideToasts(GuiGraphics graphics, CallbackInfo ci) {
		if (CaptureSettings.HIDE_TOASTS) {
			ci.cancel();
		}
	}
}

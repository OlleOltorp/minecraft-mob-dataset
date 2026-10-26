package io.github.olleoltorp.mobcapture.mixin;

import com.mojang.blaze3d.vertex.PoseStack;
import io.github.olleoltorp.mobcapture.label.LabelCapture;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.MultiBufferSource;
import net.minecraft.client.renderer.OutlineBufferSource;
import net.minecraft.client.renderer.SubmitNodeCollection;
import net.minecraft.client.renderer.SubmitNodeStorage;
import net.minecraft.client.renderer.entity.state.EntityRenderState;
import net.minecraft.client.renderer.feature.FlameFeatureRenderer;
import net.minecraft.client.resources.model.AtlasManager;
import org.joml.Quaternionf;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Labels the flames drawn around burning entities (a charged blaze, mobs on fire). The game
 * draws them apart from the entity's model and never into the outline target, so a burning
 * blaze would be labelled without the flame sheet that covers most of it. On a label frame each
 * entity's flames are drawn once more into the label image, in its label colour. They are
 * drawn first, from copies of the poses, because drawing flames changes the pose.
 */
@Mixin(FlameFeatureRenderer.class)
abstract class LabelFlameMixin {
	@Shadow
	private void renderFlame(
		PoseStack.Pose pose, MultiBufferSource buffers, EntityRenderState state, Quaternionf rotation, AtlasManager atlas
	) {
	}

	@Inject(method = "render", at = @At("HEAD"))
	private void mobcapture$labelFlames(
		SubmitNodeCollection submits, MultiBufferSource.BufferSource buffers, AtlasManager atlas, CallbackInfo ci
	) {
		if (!LabelCapture.isLabelFrame()) {
			return;
		}
		OutlineBufferSource outline = Minecraft.getInstance().renderBuffers().outlineBufferSource();
		for (SubmitNodeStorage.FlameSubmit flame : submits.getFlameSubmits()) {
			int colour = flame.entityRenderState().outlineColor;
			if (colour != 0) {
				outline.setColor(colour);
				renderFlame(flame.pose().copy(), outline, flame.entityRenderState(), flame.rotation(), atlas);
			}
		}
	}
}

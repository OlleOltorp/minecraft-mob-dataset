package io.github.olleoltorp.mobcapture.mixin;

import com.mojang.blaze3d.framegraph.FrameGraphBuilder;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.vertex.PoseStack;
import io.github.olleoltorp.mobcapture.label.LabelCapture;
import net.minecraft.client.Camera;
import net.minecraft.client.DeltaTracker;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.PostChain;
import net.minecraft.client.renderer.SubmitNodeCollector;
import net.minecraft.client.renderer.culling.Frustum;
import net.minecraft.client.renderer.entity.state.EntityRenderState;
import net.minecraft.client.renderer.state.LevelRenderState;
import net.minecraft.world.entity.Entity;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** Turns the glowing-outline pass into the label pass on a label frame. */
@Mixin(LevelRenderer.class)
abstract class LabelLevelRendererMixin {
	@Shadow
	@Final
	private Minecraft minecraft;

	@Shadow
	private RenderTarget entityOutlineTarget;

	@Inject(method = "extractEntity", at = @At("RETURN"))
	private void mobcapture$labelColour(Entity entity, float partialTick, CallbackInfoReturnable<EntityRenderState> cir) {
		if (LabelCapture.isLabelFrame()) {
			EntityRenderState state = cir.getReturnValue();
			state.outlineColor = LabelCapture.colourFor(entity, state);
		}
	}

	// Even with no mob in view, so the label image is cleared rather than left from before.
	@Inject(method = "extractVisibleEntities", at = @At("TAIL"))
	private void mobcapture$alwaysOutline(
		Camera camera, Frustum frustum, DeltaTracker deltaTracker, LevelRenderState state, CallbackInfo ci
	) {
		if (LabelCapture.isLabelFrame()) {
			state.haveGlowingEntities = true;
		}
	}

	// The world's depth so far, so terrain in front of a mob hides it in the label image too.
	@Inject(method = "submitEntities", at = @At("HEAD"))
	private void mobcapture$copyWorldDepth(
		PoseStack poseStack, LevelRenderState state, SubmitNodeCollector collector, CallbackInfo ci
	) {
		if (LabelCapture.isLabelFrame() && entityOutlineTarget != null) {
			entityOutlineTarget.copyDepthFrom(minecraft.getMainRenderTarget());
		}
	}

	// The outline post effect would turn the label image into blurred edges.
	@Redirect(
		method = "renderLevel",
		at = @At(
			value = "INVOKE",
			target = "Lnet/minecraft/client/renderer/PostChain;addToFrame(Lcom/mojang/blaze3d/framegraph/FrameGraphBuilder;IILnet/minecraft/client/renderer/PostChain$TargetBundle;)V",
			ordinal = 0
		)
	)
	private void mobcapture$skipOutlineEffect(
		PostChain chain, FrameGraphBuilder builder, int width, int height, PostChain.TargetBundle targets
	) {
		if (!LabelCapture.isLabelFrame()) {
			chain.addToFrame(builder, width, height, targets);
		}
	}

	// Keeps the id colours off the game window.
	@Inject(method = "doEntityOutline", at = @At("HEAD"), cancellable = true)
	private void mobcapture$hideLabels(CallbackInfo ci) {
		if (LabelCapture.isLabelFrame()) {
			ci.cancel();
		}
	}
}

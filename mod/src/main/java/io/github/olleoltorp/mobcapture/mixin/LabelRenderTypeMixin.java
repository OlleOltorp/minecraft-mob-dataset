package io.github.olleoltorp.mobcapture.mixin;

import com.mojang.blaze3d.pipeline.RenderPipeline;
import io.github.olleoltorp.mobcapture.label.LabelCapture;
import io.github.olleoltorp.mobcapture.label.LabelPipelines;
import net.minecraft.client.renderer.rendertype.RenderType;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

/** Draws outlines with the depth-tested label pipelines on a label frame. */
@Mixin(RenderType.class)
abstract class LabelRenderTypeMixin {
	@ModifyArg(
		method = "draw",
		at = @At(value = "INVOKE", target = "Lcom/mojang/blaze3d/systems/RenderPass;setPipeline(Lcom/mojang/blaze3d/pipeline/RenderPipeline;)V")
	)
	private RenderPipeline mobcapture$labelPipeline(RenderPipeline pipeline) {
		return LabelCapture.isLabelFrame() ? LabelPipelines.swap(pipeline) : pipeline;
	}
}

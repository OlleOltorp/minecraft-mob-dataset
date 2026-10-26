package io.github.olleoltorp.mobcapture.label;

import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.platform.DepthTestFunction;
import com.mojang.blaze3d.shaders.UniformType;
import com.mojang.blaze3d.vertex.DefaultVertexFormat;
import com.mojang.blaze3d.vertex.VertexFormat;
import net.minecraft.client.renderer.RenderPipelines;
import net.minecraft.resources.Identifier;

/**
 * Pipelines that replace the glowing-outline ones on a label frame.
 *
 * <p>Vanilla outlines ignore depth, so a glowing mob shows through walls. These test and write
 * depth against the world's depth (copied in before entities draw), so each pixel keeps only the
 * nearest mob, and they write the mob's id colour exactly, with no blending. Texels with
 * alpha 0 are cut away (the game's entity cutout drops alpha below 0.1; mob textures are
 * almost all fully opaque or fully clear, so the two agree).
 */
public final class LabelPipelines {
	public static final RenderPipeline LABEL_CULL = build("pipeline/label_cull", true);
	public static final RenderPipeline LABEL_NO_CULL = build("pipeline/label_no_cull", false);

	private LabelPipelines() {
	}

	/** The label pipeline standing in for a vanilla outline pipeline, or the pipeline itself. */
	public static RenderPipeline swap(RenderPipeline pipeline) {
		if (pipeline == RenderPipelines.OUTLINE_CULL) {
			return LABEL_CULL;
		}
		if (pipeline == RenderPipelines.OUTLINE_NO_CULL) {
			return LABEL_NO_CULL;
		}
		return pipeline;
	}

	private static RenderPipeline build(String location, boolean cull) {
		return RenderPipeline.builder()
			.withLocation(Identifier.fromNamespaceAndPath("mobcapture", location))
			.withUniform("DynamicTransforms", UniformType.UNIFORM_BUFFER)
			.withUniform("Projection", UniformType.UNIFORM_BUFFER)
			.withVertexShader("core/rendertype_outline")
			// The game's own outline shader writes the vertex colour exactly. A shader of our
			// own would not load: without Fabric API, mod assets are not in the resources.
			.withFragmentShader("core/rendertype_outline")
			.withSampler("Sampler0")
			.withVertexFormat(DefaultVertexFormat.POSITION_TEX_COLOR, VertexFormat.Mode.QUADS)
			.withDepthTestFunction(DepthTestFunction.LEQUAL_DEPTH_TEST)
			.withDepthWrite(true)
			.withCull(cull)
			.build();
	}
}

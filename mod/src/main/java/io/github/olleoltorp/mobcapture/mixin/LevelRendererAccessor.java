package io.github.olleoltorp.mobcapture.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import it.unimi.dsi.fastutil.objects.ObjectArrayList;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.SectionOcclusionGraph;
import net.minecraft.client.renderer.ViewArea;
import net.minecraft.client.renderer.chunk.SectionRenderDispatcher;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/**
 * The outline target itself (the frame's handle to it is gone once the frame is drawn), and
 * the sections in view with the graph that picks them, for telling when terrain is done.
 */
@Mixin(LevelRenderer.class)
public interface LevelRendererAccessor {
	@Accessor("entityOutlineTarget")
	RenderTarget mobcapture$outlineTarget();

	@Accessor("visibleSections")
	ObjectArrayList<SectionRenderDispatcher.RenderSection> mobcapture$visibleSections();

	@Accessor("sectionOcclusionGraph")
	SectionOcclusionGraph mobcapture$occlusionGraph();

	@Accessor("viewArea")
	ViewArea mobcapture$viewArea();
}

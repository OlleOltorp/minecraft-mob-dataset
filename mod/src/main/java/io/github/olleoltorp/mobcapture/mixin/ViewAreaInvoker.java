package io.github.olleoltorp.mobcapture.mixin;

import net.minecraft.client.renderer.ViewArea;
import net.minecraft.client.renderer.chunk.SectionRenderDispatcher;
import net.minecraft.core.BlockPos;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

/** The terrain section the window keeps for a block, for finding why it is not drawn. */
@Mixin(ViewArea.class)
public interface ViewAreaInvoker {
	@Invoker("getRenderSectionAt")
	SectionRenderDispatcher.RenderSection mobcapture$sectionAt(BlockPos pos);
}

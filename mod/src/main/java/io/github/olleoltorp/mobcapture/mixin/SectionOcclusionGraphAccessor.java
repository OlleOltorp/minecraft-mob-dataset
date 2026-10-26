package io.github.olleoltorp.mobcapture.mixin;

import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import net.minecraft.client.renderer.SectionOcclusionGraph;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Whether the game is still working out which terrain sections are in view. */
@Mixin(SectionOcclusionGraph.class)
public interface SectionOcclusionGraphAccessor {
	@Accessor("needsFullUpdate")
	boolean mobcapture$needsFullUpdate();

	@Accessor("fullUpdateTask")
	Future<?> mobcapture$fullUpdateTask();

	@Accessor("needsFrustumUpdate")
	AtomicBoolean mobcapture$needsFrustumUpdate();
}

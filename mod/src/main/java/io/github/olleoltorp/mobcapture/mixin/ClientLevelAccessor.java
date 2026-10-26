package io.github.olleoltorp.mobcapture.mixin;

import java.util.Deque;
import net.minecraft.client.multiplayer.ClientLevel;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Light updates the window still has to apply: chunks without light are not built yet. */
@Mixin(ClientLevel.class)
public interface ClientLevelAccessor {
	@Accessor("lightUpdateQueue")
	Deque<Runnable> mobcapture$lightUpdateQueue();
}

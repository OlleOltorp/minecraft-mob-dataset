package io.github.olleoltorp.mobcapture.mixin;

import net.minecraft.world.entity.monster.Blaze;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Invoker;

/** The blaze's charged flag, which is what draws its flames (Blaze#isOnFire). */
@Mixin(Blaze.class)
public interface BlazeInvoker {
	@Invoker("setCharged")
	void mobcapture$setCharged(boolean charged);
}

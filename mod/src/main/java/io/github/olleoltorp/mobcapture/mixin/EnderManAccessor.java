package io.github.olleoltorp.mobcapture.mixin;

import net.minecraft.network.syncher.EntityDataAccessor;
import net.minecraft.world.entity.monster.EnderMan;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** The enderman's open-mouth flag, otherwise only set when it picks a target. */
@Mixin(EnderMan.class)
public interface EnderManAccessor {
	@Accessor("DATA_CREEPY")
	static EntityDataAccessor<Boolean> mobcapture$creepy() {
		throw new AssertionError();
	}
}

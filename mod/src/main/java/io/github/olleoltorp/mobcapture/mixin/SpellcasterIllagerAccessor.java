package io.github.olleoltorp.mobcapture.mixin;

import net.minecraft.network.syncher.EntityDataAccessor;
import net.minecraft.world.entity.monster.illager.SpellcasterIllager;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/**
 * The synced spell id, which clients use for the casting pose. Set directly because the spell
 * enum is protected; the server's own casting timer stays at zero, which only the AI reads.
 */
@Mixin(SpellcasterIllager.class)
public interface SpellcasterIllagerAccessor {
	@Accessor("DATA_SPELL_CASTING_ID")
	static EntityDataAccessor<Byte> mobcapture$spellCasting() {
		throw new AssertionError();
	}
}

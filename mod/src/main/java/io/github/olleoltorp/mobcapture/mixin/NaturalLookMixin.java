package io.github.olleoltorp.mobcapture.mixin;

import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.Mob;
import net.minecraft.world.entity.monster.Blaze;
import net.minecraft.world.entity.monster.EnderMan;
import net.minecraft.world.entity.monster.Ghast;
import net.minecraft.world.entity.monster.Vex;
import net.minecraft.world.entity.monster.illager.Pillager;
import net.minecraft.world.entity.monster.illager.SpellcasterIllager;
import net.minecraft.world.level.storage.ValueInput;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Mid-fight looks for summoned mobs (server side). A blaze's flames, a ghast's shooting face or a
 * zombie's raised arms are synced flags that only AI goals set and that are never saved, so
 * summon NBT can't give them and frozen NoAI mobs never get them. This reads extra summon keys
 * (vanilla ignores unknown ones) after the whole entity has loaded and sets the flag, so the
 * look is in the spawn packet: no ticks needed. The keys are listed in
 * {@code config/mobs/states-1.21.11.toml}.
 */
@Mixin(Entity.class)
public abstract class NaturalLookMixin {
	@Inject(method = "load", at = @At("TAIL"))
	private void mobcapture$naturalLook(ValueInput input, CallbackInfo ci) {
		if (!((Object) this instanceof Mob mob)) {
			return;
		}
		if (input.getBooleanOr("mobcapture_aggressive", false)) {
			mob.setAggressive(true);
		}
		byte spell = input.getByteOr("mobcapture_spell", (byte) 0);
		if (spell > 0 && mob instanceof SpellcasterIllager) {
			mob.getEntityData().set(SpellcasterIllagerAccessor.mobcapture$spellCasting(), spell);
		}
		if (!input.getBooleanOr("mobcapture_charging", false)) {
			return;
		}
		switch (mob) {
			case Blaze blaze -> ((BlazeInvoker) blaze).mobcapture$setCharged(true);
			case Ghast ghast -> ghast.setCharging(true);
			case Vex vex -> vex.setIsCharging(true);
			case Pillager pillager -> pillager.setChargingCrossbow(true);
			case EnderMan enderMan -> enderMan.getEntityData().set(EnderManAccessor.mobcapture$creepy(), true);
			default -> throw new IllegalArgumentException("mobcapture_charging has no look for " + mob.getType());
		}
	}
}

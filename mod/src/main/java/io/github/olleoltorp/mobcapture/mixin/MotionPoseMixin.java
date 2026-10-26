package io.github.olleoltorp.mobcapture.mixin;

import net.minecraft.client.renderer.entity.LivingEntityRenderer;
import net.minecraft.client.renderer.entity.state.LivingEntityRenderState;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.monster.Creeper;
import net.minecraft.world.entity.monster.breeze.Breeze;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Walking, swimming and flying poses for frozen mobs. Limb swing, tail wiggle, wing flap and
 * tentacle sway all come from the walk animation and the entity's age, which only move while
 * the world ticks, so frozen mobs would all stand still in their first frame. The capture
 * picks a pose per mob and writes it into the last 32 bits of the mob's UUID (the
 * client knows nothing else about a summon): a marker byte, the walk speed, the walk phase
 * and the age phase. Here those set the values the model is drawn with.
 */
@Mixin(LivingEntityRenderer.class)
public abstract class MotionPoseMixin {
	private static final int MARKER = 0xA7;
	/** Limb swing repeats every 2 pi / 0.6662 blocks of walking (HumanoidModel and others). */
	private static final float WALK_PERIOD = (float) (2 * Math.PI / 0.6662);
	/** Ages spread over this many ticks: longer than every idle animation's cycle. */
	private static final float AGE_SPAN = 256.0F;
	/**
	 * Ages for mobs with a scrolling layer (breeze wind, charged creeper swirl) are whole
	 * multiples of this, where the scroll offset (age x 0.02 or 0.01) is zero and the layer's
	 * label lines up with the picture (ScrollingLayerOutlineMixin).
	 */
	private static final float SCROLL_AGE_STEP = 100.0F;

	// Full descriptor: the class also has a bridge extractRenderState(Entity, EntityRenderState, float).
	@Inject(
		method = "extractRenderState(Lnet/minecraft/world/entity/LivingEntity;"
			+ "Lnet/minecraft/client/renderer/entity/state/LivingEntityRenderState;F)V",
		at = @At("TAIL")
	)
	private void mobcapture$motionPose(LivingEntity entity, LivingEntityRenderState state, float partialTick, CallbackInfo ci) {
		int code = (int) entity.getUUID().getLeastSignificantBits();
		if ((code >>> 24) != MARKER) {
			return;
		}
		float speed = ((code >>> 20) & 0xF) / 15.0F;
		if (speed > 0 && !entity.isPassenger()) {
			state.walkAnimationSpeed = speed;
			state.walkAnimationPos = ((code >>> 12) & 0xFF) / 256.0F * WALK_PERIOD;
		}
		int agePhase = (code >>> 4) & 0xFF;
		state.ageInTicks = entity instanceof Breeze || entity instanceof Creeper
			? agePhase * SCROLL_AGE_STEP
			: agePhase / 256.0F * AGE_SPAN;
	}
}

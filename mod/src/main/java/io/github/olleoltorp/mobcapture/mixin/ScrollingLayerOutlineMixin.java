package io.github.olleoltorp.mobcapture.mixin;

import net.minecraft.client.renderer.rendertype.RenderSetup;
import net.minecraft.client.renderer.rendertype.RenderTypes;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

/**
 * Labels the breeze's wind and the energy swirl (charged creeper, wither armour). Their render
 * types have no outline, so the label pass would skip them and a breeze would be labelled
 * without the whirlwind that is most of its body. With an outline they are drawn into the label
 * image like any other layer. The outline ignores their scrolling texture offset, so
 * MotionPoseMixin gives captured mobs with these layers an age at which the offset is zero and
 * the two line up.
 */
@Mixin(RenderTypes.class)
abstract class ScrollingLayerOutlineMixin {
	@Redirect(
		method = {"breezeWind", "energySwirl"},
		at = @At(
			value = "INVOKE",
			target = "Lnet/minecraft/client/renderer/rendertype/RenderSetup$RenderSetupBuilder;createRenderSetup()Lnet/minecraft/client/renderer/rendertype/RenderSetup;"
		)
	)
	private static RenderSetup mobcapture$withOutline(RenderSetup.RenderSetupBuilder builder) {
		return builder.setOutline(RenderSetup.OutlineProperty.AFFECTS_OUTLINE).createRenderSetup();
	}
}

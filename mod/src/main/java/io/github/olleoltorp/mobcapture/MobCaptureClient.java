package io.github.olleoltorp.mobcapture;

import io.github.olleoltorp.mobcapture.label.LabelCapture;
import net.fabricmc.api.ClientModInitializer;

/** Client entry point: starts the label capture server when a control port is set. */
public final class MobCaptureClient implements ClientModInitializer {
	@Override
	public void onInitializeClient() {
		if (CaptureSettings.CONTROL_PORT > 0) {
			LabelCapture.startServer(CaptureSettings.CONTROL_PORT);
		}
	}
}

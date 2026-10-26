"""Run a fine-tuned RF-DETR on pictures the way its test loader prepares them.

`mobdataset train` keeps the pictures' shape: short side at the model's resolution, then each
batch padded right and down to a multiple of 32, with the padding masked out. The model's boxes
are relative to the unpadded part. RF-DETR's predict() instead resizes to one fixed shape with
no padding; on 16:9 pictures that puts boxes about 1 % too far right at the right edge, enough
to miss most small (far-away) mobs and halve the test mAP. Predicting here, the same way the
training run scored itself, gives the same mAP it reported.
"""

from typing import Any

import numpy as np
import supervision as sv
import torch
import torchvision.transforms.functional as F
from PIL import Image
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

MAX_SIZE = 1333  # longest side RF-DETR's test resize allows


def resized(image: Image.Image, short: int) -> Image.Image:
    """The picture with its short side at `short`, rounded as RF-DETR's test resize rounds."""
    w, h = image.size
    if max(w, h) / min(w, h) * short > MAX_SIZE:
        short = round(MAX_SIZE * min(w, h) / max(w, h))
    size = (round(short * w / h), short) if w >= h else (short, round(short * h / w))
    return image.resize(size, Image.Resampling.BILINEAR)


@torch.inference_mode()
def predict(model: Any, images: list[Image.Image], threshold: float) -> list[sv.Detections]:
    """Boxes for each picture, in its own pixels.

    Args:
        model: A fine-tuned RF-DETR (RFDETRLarge(pretrain_weights=...) and the like).
        images: RGB pictures; sizes may differ.
        threshold: Lowest score kept.

    Returns:
        One Detections per picture; class_id is the position in model.class_names.
    """
    config = model.model_config
    # RF-DETR loads the weights on the CPU and moves them only inside its own predict().
    device = model.model.device
    net = model.model.model.to(device).eval()
    tensors = [
        F.normalize(F.to_tensor(resized(im, config.resolution)), model.means, model.stds)
        for im in images
    ]
    batch = nested_tensor_from_tensor_list(
        tensors, block_size=config.patch_size * config.num_windows
    )
    outputs = net(batch.to(device))
    sizes = torch.tensor([im.size[::-1] for im in images], device=device)
    out = []
    for result in model.model.postprocess(outputs, target_sizes=sizes):
        keep = result["scores"] > threshold
        out.append(
            sv.Detections(
                xyxy=result["boxes"][keep].float().cpu().numpy(),
                confidence=result["scores"][keep].float().cpu().numpy(),
                class_id=result["labels"][keep].cpu().numpy().astype(np.int64),
            )
        )
    return out

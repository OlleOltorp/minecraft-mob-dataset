"""The test-time resize must match RF-DETR's own, or boxes land in the wrong place."""

import pytest

pytest.importorskip("rfdetr")

from PIL import Image  # noqa: E402

from mobdataset.model.infer import resized  # noqa: E402


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((1280, 720), (1252, 704)),  # the dataset's pictures: short side 704, 16:9 kept
        ((720, 1280), (704, 1252)),
        ((704, 704), (704, 704)),
        ((3000, 704), (1334, 313)),  # long side capped near 1333, rounded as RF-DETR does
    ],
)
def test_resized_matches_rfdetr(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert resized(Image.new("RGB", size), 704).size == expected

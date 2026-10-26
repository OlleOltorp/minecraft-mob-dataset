"""Known-answer tests for the scoring used by the notebooks."""

from typing import Any

import pandas as pd
import pytest

from mobdataset.analysis import coco_scores, match


def label(ann_id: int, image_id: int, category_id: int, bbox: list[float]) -> dict[str, Any]:
    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
    }


GT: dict[str, Any] = {
    "images": [{"id": 1, "width": 100, "height": 100}, {"id": 2, "width": 100, "height": 100}],
    "categories": [{"id": 1, "name": "zombie"}, {"id": 2, "name": "cow"}],
    "annotations": [
        label(10, 1, 1, [10, 10, 20, 30]),
        label(11, 1, 2, [50, 50, 30, 20]),
        label(12, 2, 1, [5, 5, 40, 40]),
    ],
}


def boxes(rows: list[tuple[int, str, float, list[float]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"image_id": i, "category": c, "score": s, "x": b[0], "y": b[1], "w": b[2], "h": b[3]}
         for i, c, s, b in rows]
    )  # fmt: skip


def test_perfect_predictions_score_one() -> None:
    perfect = boxes([(a["image_id"], ["zombie", "cow"][a["category_id"] - 1], 0.9, a["bbox"])
                     for a in GT["annotations"]])  # fmt: skip
    scores = coco_scores(GT, perfect, per_class=True)
    assert scores["mAP50-95"] > 0.99 and scores["labels"] == 3
    assert scores["classes"] == pytest.approx({"zombie": 1.0, "cow": 1.0})


def test_missing_mob_lowers_recall_and_ignored_label_does_not() -> None:
    no_cow = boxes([(1, "zombie", 0.9, [10, 10, 20, 30]), (2, "zombie", 0.9, [5, 5, 40, 40])])
    assert coco_scores(GT, no_cow)["AR100"] < 0.99
    # The cow ignored: nothing is missed, and the two zombies are found.
    only_zombies = coco_scores(GT, no_cow, keep=[10, 12])
    assert only_zombies["mAP50-95"] > 0.99 and only_zombies["labels"] == 2


def test_subset_of_pictures() -> None:
    one = boxes([(2, "zombie", 0.9, [5, 5, 40, 40])])
    assert coco_scores(GT, one, image_ids=[2])["mAP50-95"] > 0.99
    assert coco_scores(GT, one, image_ids=[1])["labels"] == 2


def test_match_finds_hits_wrong_classes_misses_and_false_alarms() -> None:
    names = {1: "zombie", 2: "cow"}
    labels = pd.DataFrame(
        [
            {"image_id": a["image_id"], "id": a["id"], "category": names[a["category_id"]]}
            | {"bbox": a["bbox"]}
            for a in GT["annotations"]
        ]
    )
    preds = boxes(
        [
            (1, "zombie", 0.9, [10, 10, 20, 30]),  # hit
            (1, "pig", 0.8, [50, 50, 30, 20]),  # right place, wrong class
            (2, "zombie", 0.7, [80, 80, 10, 10]),  # nothing there; the zombie at 5,5 is missed
        ]
    )
    pairs = match(labels, preds).set_index("label_id", drop=False)
    assert pairs.loc[10, "predicted"] == "zombie"
    assert pairs.loc[11, "predicted"] == "pig"
    assert pd.isna(pairs.loc[12, "predicted"])
    alarms = pairs[pairs.label_id.isna()]
    assert list(alarms.predicted) == ["zombie"] and list(alarms.score) == [0.7]

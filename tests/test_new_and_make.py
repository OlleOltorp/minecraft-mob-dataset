"""Tests for `mobdataset new` files and `mobdataset make` step keys."""

import tomllib
from pathlib import Path

import pytest

from mobdataset.datasets import DatasetError, load_dataset, merge
from mobdataset.make import STEPS, needed, step_keys
from mobdataset.new import dataset_file


def test_new_file_loads(tmp_path: Path) -> None:
    text = dataset_file("my-set", "full", "42", ["plains", "desert"], ["hostile"], 600)
    path = tmp_path / "my-set.toml"
    path.write_text(text, encoding="utf-8")
    cfg = load_dataset(path)
    assert cfg["world"] == "my-set" and cfg["world_seed"] == "42"
    assert cfg["areas"]["land"]["biomes"] == ["plains", "desert"]
    assert cfg["areas"]["ocean"]["pictures"] == 0
    assert cfg["areas"]["structures"]["pictures"] == 0
    assert cfg["scene"]["mobs"] == ["hostile"]
    assert 500 <= sum(a["pictures"] for a in cfg["areas"].values()) <= 700


def test_new_refuses_unknown_biome() -> None:
    with pytest.raises(DatasetError, match="unknown biomes"):
        dataset_file("x", "full", "1", ["mars"], None, None)


def keys_after(change: dict) -> dict[str, str]:
    cfg = merge(load_dataset("tiny"), change)
    return step_keys(cfg, windows=1)


def test_a_change_reruns_its_step_and_every_later_one() -> None:
    before = step_keys(load_dataset("tiny"), windows=1)
    after = keys_after({"time": {"night": {"ticks": [15000, 21000]}}})  # capture only
    first = STEPS.index("capture")
    assert all(before[s] == after[s] for s in STEPS[:first])
    assert all(before[s] != after[s] for s in STEPS[first:])
    seed = keys_after({"survey": {"half_size": 512}})
    assert all(before[s] != seed[s] for s in STEPS)


def test_tiny_needs_no_structure_steps() -> None:
    assert needed(load_dataset("tiny")) == set(STEPS) - {"structures", "features", "replan"}
    assert needed(load_dataset("full")) == set(STEPS)


def test_presets_are_valid_toml() -> None:
    for path in Path("datasets").glob("*.toml"):
        tomllib.loads(path.read_text(encoding="utf-8"))


def test_new_scales_structure_spots_too(tmp_path: Path) -> None:
    path = tmp_path / "small.toml"
    path.write_text(dataset_file("small", "full", "1", None, None, 500), encoding="utf-8")
    cfg = load_dataset(path)
    spots = sum(t["spots"] for t in cfg["areas"]["structures"]["types"].values())
    per = cfg["spots"]["pictures_per_spot"]
    assert spots * per < 1000  # the full dataset has 583 structure spots

"""Tests for loading and checking dataset files."""

import tomllib
from pathlib import Path

import pytest

from mobdataset.datasets import (
    DATASETS,
    ENGINE,
    DatasetError,
    load_dataset,
    merge,
    mob_ids,
    world_properties,
)


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "test-set.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", sorted(p.stem for p in DATASETS.glob("*.toml")))
def test_every_preset_loads(name: str) -> None:
    cfg = load_dataset(name)
    assert cfg["world"]
    assert world_properties(cfg["world"], cfg)["level-name"]


def test_full_is_dataset_file_plus_engine() -> None:
    raw = tomllib.loads((DATASETS / "full.toml").read_text(encoding="utf-8"))
    engine = tomllib.loads(ENGINE.read_text(encoding="utf-8"))
    cfg = load_dataset("full")
    assert cfg == merge(engine, raw, over_first=True)
    assert list(cfg["areas"]) == list(raw["areas"])  # steps walk the areas in file order


def test_extends_changes_only_what_it_sets(tmp_path: Path) -> None:
    path = write(tmp_path, 'extends = "full"\n[areas.land]\npictures = 10\n')
    cfg, full = load_dataset(path), load_dataset("full")
    assert cfg["areas"]["land"]["pictures"] == 10
    assert cfg["areas"]["land"]["biomes"] == full["areas"]["land"]["biomes"]
    assert cfg["areas"]["ocean"] == full["areas"]["ocean"]


def test_unknown_key_names_the_nearest(tmp_path: Path) -> None:
    path = write(tmp_path, 'extends = "full"\n[scene]\nmob_max = 5\n')
    with pytest.raises(DatasetError, match="did you mean 'mobs_max'"):
        load_dataset(path)


def test_wrong_kind_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, 'extends = "full"\n[scene]\nmobs_max = "many"\n')
    with pytest.raises(DatasetError, match="should be number"):
        load_dataset(path)


def test_unknown_biome_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, 'extends = "full"\n[areas.land]\nbiomes = ["plains", "mars"]\n')
    with pytest.raises(DatasetError, match="unknown \\['mars'\\]"):
        load_dataset(path)


def test_shares_must_add_up(tmp_path: Path) -> None:
    path = write(tmp_path, 'extends = "full"\n[weather]\nclear = 0.5\n')
    with pytest.raises(DatasetError, match="weather shares add up to 0.7"):
        load_dataset(path)


def test_engine_settings_stay_in_the_engine_file(tmp_path: Path) -> None:
    path = write(tmp_path, 'extends = "full"\n[scene]\nterrain_timeout = 5.0\n')
    with pytest.raises(DatasetError, match="engine setting"):
        load_dataset(path)


def test_mob_groups() -> None:
    hostile = mob_ids(["hostile"])
    assert "zombie" in hostile and "cow" not in hostile
    assert mob_ids(["all"]) >= hostile | mob_ids(["passive"])
    with pytest.raises(DatasetError, match="did you mean 'zombie'"):
        mob_ids(["zombi"])


def test_own_world_from_seed() -> None:
    cfg = {"world": "my-set", "world_seed": "42"}
    props = world_properties("my-set", cfg)
    # The seed is in the folder name: a seed only applies when a world is created.
    assert props == {
        "level-name": "my-set-42",
        "level-type": "minecraft:normal",
        "level-seed": "42",
    }
    with pytest.raises(DatasetError):
        world_properties("my-set", {"world": "my-set", "world_seed": ""})


def test_typo_inside_an_area_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, 'extends = "full"\n[areas.land]\npicturs = 5\n')
    with pytest.raises(DatasetError, match="did you mean 'pictures'"):
        load_dataset(path)


def test_new_structure_type_is_allowed(tmp_path: Path) -> None:
    text = 'extends = "full"\n[areas.structures.types.my_ruin]\nspots = 3\ncamera = "surface"\n'
    assert load_dataset(write(tmp_path, text))["areas"]["structures"]["types"]["my_ruin"]

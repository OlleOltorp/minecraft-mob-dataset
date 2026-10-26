"""Random visible mob states (baby, armour, colours, riders...) for summons.

Reads config/mobs/states-<version>.toml (its header explains the format) and turns a roll
into summon NBT plus follow-up commands. A summon with NBT skips the game's own spawn
set-up, so every natural look (a skeleton's bow, a sheep's colour) has to come from here.
"""

import random
import re
import tomllib
from dataclasses import dataclass, field
from typing import Any

from mobdataset.paths import mob_config

STATES_FILE = mob_config("states")
SECTIONS = {"meta", "templates", "general", "riders"}


@dataclass
class MobState:
    """One rolled mob.

    Attributes:
        mob: Mob id without "minecraft:".
        nbt: Snippets that go inside the summon's {...}.
        equipment: Slot to item SNBT.
        commands: Run after the summon; "{target}" stands for a selector of this mob.
        features: What was applied, e.g. ["baby", "armour=iron"], for the labels.
        needs_ticks: Game ticks needed before the look shows (0 for none).
        passengers: Riders, first in front.
    """

    mob: str
    nbt: list[str] = field(default_factory=list)
    equipment: dict[str, str] = field(default_factory=dict)
    commands: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    needs_ticks: int = 0
    passengers: list["MobState"] = field(default_factory=list)

    def compound(self, base: str) -> str:
        """The summon NBT compound, with base snippets (NoAI, tags...) on every mob.

        Args:
            base: Snippets for every mob in the stack, e.g. 'NoAI:1b,Tags:["x"]'.

        Returns:
            "{...}" for the summon command.
        """
        parts = [p for p in (base, *self.nbt) if p]
        if self.equipment:
            parts.append(
                "equipment:{" + ",".join(f"{k}:{v}" for k, v in self.equipment.items()) + "}"
            )
        if self.passengers:
            # Riders share the base but not its UUID: two entities can't have the same one.
            rider_base = re.sub(r",?UUID:\[I;[-\d,]+\]", "", base)
            riders = ",".join(
                p.compound(rider_base)[:-1] + f',id:"minecraft:{p.mob}"}}' for p in self.passengers
            )
            parts.append(f"Passengers:[{riders}]")
        return "{" + ",".join(parts) + "}"

    def all_features(self) -> list[str]:
        """Features of this mob and its riders, riders prefixed with their id."""
        out = list(self.features)
        for p in self.passengers:
            out += [f"{p.mob}:{f}" for f in p.all_features()]
        return out


class StateTable:
    """The state table, ready to roll.

    Args:
        allow_unverified: Also use features marked verify (not yet confirmed in game).
        allow_ticks: Also use features that need game ticks to show.

    Raises:
        ValueError: If a feature refers to an unknown feature or template.
    """

    def __init__(self, allow_unverified: bool = True, allow_ticks: bool = True) -> None:
        self.table = tomllib.loads(STATES_FILE.read_text(encoding="utf-8"))
        self.allow_unverified = allow_unverified
        self.allow_ticks = allow_ticks
        self.mobs = {k: v for k, v in self.table.items() if k not in SECTIONS}
        for mob, feats in self.mobs.items():
            for name, feat in feats.items():
                for ref in feat.get("requires", []) + feat.get("unless", []):
                    if ref not in feats:
                        raise ValueError(f"{mob}.{name} refers to unknown feature {ref!r}")
                if "template" in feat and feat["template"] not in self.table["templates"]:
                    raise ValueError(f"{mob}.{name}: unknown template {feat['template']!r}")

    def features(self, mob: str) -> dict[str, dict[str, Any]]:
        """The features that can apply to a mob, general ones first, in file order."""
        out = {
            f"general.{name}": feat
            for name, feat in self.table.get("general", {}).items()
            if mob not in feat.get("except", [])
        }
        out.update(self.mobs.get(mob, {}))
        return {k: v for k, v in out.items() if self._usable(v)}

    def roll(
        self,
        mob: str,
        rng: random.Random,
        player_uuid: str = "",
        force: dict[str, int] | None = None,
        only_forced: bool = False,
        riders: bool = True,
    ) -> MobState:
        """Roll a random state for one mob.

        Args:
            mob: Mob id without "minecraft:".
            rng: Random source (seeded by the caller).
            player_uuid: The camera player's UUID as an SNBT int array, for features that
                need it (an angry enderman's target); those are skipped without it.
            force: Feature name to alternative index, applied whatever the chance.
            only_forced: Apply only forced features and those with chance 1 (their first
                alternative unless forced), for showing one state at a time.
            riders: Also roll riders.

        Returns:
            The rolled state.
        """
        force = self._with_required(mob, force or {})
        state = MobState(mob)
        skip: set[str] = set()
        if riders and not only_forced:
            self._roll_riders(state, rng, player_uuid, skip)
        applied: set[str] = set()
        for name, feat in self.features(mob).items():
            short = name.removeprefix("general.")
            if short in skip:
                continue
            if name not in force and any(u in applied for u in feat.get("unless", [])):
                continue
            if not all(r in applied for r in feat.get("requires", [])):
                continue
            if name in force:
                pick = force[name]
            elif only_forced:
                if feat.get("chance", 0) < 1.0:
                    continue
                pick = 0
            elif rng.random() >= feat.get("chance", 0):
                continue
            else:
                pick = -1
            if not self._apply(state, short, feat, rng, pick, player_uuid):
                continue
            applied.add(short)
        return state

    def _with_required(self, mob: str, force: dict[str, int]) -> dict[str, int]:
        """Forced features plus the ones they require (first alternative), recursively."""
        feats = self.features(mob)
        out = dict(force)
        todo = list(force)
        while todo:
            for req in feats.get(todo.pop(), {}).get("requires", []):
                if req in feats and req not in out:
                    out[req] = 0
                    todo.append(req)
        return out

    def alternatives(self, mob: str, name: str) -> int:
        """How many alternatives a feature has (1 for templates and commands)."""
        feat = self.features(mob)[name]
        return len(feat.get("nbt") or feat.get("equip") or [None])

    def _usable(self, feat: dict[str, Any]) -> bool:
        if feat.get("verify") and not self.allow_unverified:
            return False
        return not (feat.get("needs_ticks") and not self.allow_ticks)

    def _apply(
        self,
        state: MobState,
        name: str,
        feat: dict[str, Any],
        rng: random.Random,
        pick: int,
        player_uuid: str,
    ) -> bool:
        """Apply one feature; pick -1 draws an alternative by weight. False if it can't apply."""
        if "template" in feat:
            label = self._armour(state, self.table["templates"][feat["template"]], rng)
            state.features.append(f"{name}={label}")
        elif "command" in feat:
            state.commands.append(feat["command"])
            state.features.append(name)
        else:
            options = feat.get("nbt") or feat.get("equip")
            if not options:
                raise ValueError(f"{state.mob}.{name} has no nbt, equip, template or command")
            if pick < 0:
                pick = rng.choices(range(len(options)), weights=feat.get("weights"))[0]
            value = options[pick]
            if isinstance(value, str) and "{player_uuid}" in value:
                if not player_uuid:
                    return False
                value = value.replace("{player_uuid}", player_uuid)
            if "merge_into" in feat:
                _merge(state, feat["merge_into"], value)
            elif "equip" in feat:
                state.equipment.update(value)
            else:
                state.nbt.append(value)
            state.features.append(f"{name}={pick}" if len(options) > 1 else name)
        if feat.get("also") and feat["also"] not in state.nbt:
            state.nbt.append(feat["also"])
        state.needs_ticks = max(state.needs_ticks, int(feat.get("needs_ticks", 0)))
        return True

    def _armour(self, state: MobState, tpl: dict[str, Any], rng: random.Random) -> str:
        material = rng.choices(tpl["materials"], weights=tpl["material_weights"])[0]
        for i, slot in enumerate(tpl["slots"]):
            if i > 0 and rng.random() < tpl["stop_chance"]:
                break
            state.equipment[slot] = f'{{id:"minecraft:{material}_{tpl["items"][slot]}"}}'
        return material

    def _roll_riders(
        self, state: MobState, rng: random.Random, player_uuid: str, skip: set[str]
    ) -> None:
        for name, combo in self.table.get("riders", {}).items():
            if combo["vehicle"] != state.mob or rng.random() >= combo["chance"]:
                continue
            group = rng.choices(combo["alternatives"], weights=combo.get("weights"))[0]
            for p in group:
                rider = (
                    self.roll(p["id"], rng, player_uuid, riders=False)
                    if p.get("roll_states")
                    else MobState(p["id"])
                )
                if p.get("nbt"):
                    rider.nbt.append(p["nbt"])
                rider.equipment.update(p.get("equip", {}))
                state.passengers.append(rider)
            if combo.get("vehicle_nbt"):
                state.nbt.append(combo["vehicle_nbt"])
            state.equipment.update(combo.get("vehicle_equip", {}))
            state.features.append(f"riders={name}")
            skip.update(combo.get("skip", []))
            return  # one rider combo per vehicle


def _merge(state: MobState, target: str, value: str) -> None:
    """Write a picked value into a field of an earlier snippet, e.g. VillagerData.level."""
    compound, key = target.split(".")
    for i, snippet in enumerate(state.nbt):
        if snippet.startswith(compound + ":"):
            old = re.search(rf"(?<=[{{,]){key}:(\"[^\"]*\"|[^,}}]+)", snippet)
            if old is None:
                raise ValueError(f"{key} not in {snippet}")
            new = f'"minecraft:{value}"' if old.group(1).startswith('"') else value
            state.nbt[i] = snippet[: old.start(1)] + new + snippet[old.end(1) :]
            return
    raise ValueError(f"{state.mob}: no {compound} snippet to put {key}={value} in")

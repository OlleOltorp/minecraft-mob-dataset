"""Command-line entry point: `mobdataset <command> [options]`.

Each command is one module with its own `main(argv)`; it is imported only when run, so a
command never needs the libraries of another (training needs the `train` extra, window
checks the `window` extra).
"""

import importlib
import sys
from collections.abc import Sequence

from mobdataset import __version__

# command: (module, one-line help), in pipeline order.
COMMANDS: dict[str, tuple[str, str]] = {
    "doctor": ("doctor", "check this machine can make datasets; say how to fix what is missing"),
    "new": ("new", "start a dataset file from a preset"),
    "make": ("make", "make a dataset in one go: every step below, in order, resumable"),
    "session": ("game.session", "start the server and game windows, run a job, shut down"),
    "stop": ("game.stop", "stop the server and every game window"),
    "survey": ("world.survey", "map the world's biomes without generating terrain"),
    "structures": ("world.structures", "find generated structures with the game's locate"),
    "features": ("world.features", "find dungeons, geodes, fossils and wells in region files"),
    "plan": ("world.spots", "pick capture spots and split them into train, valid and test"),
    "pregen": ("world.pregen", "generate the terrain around every spot with Chunky"),
    "capture": ("capture.main", "take the pictures and their label images"),
    "export": ("export.build", "turn captures into COCO and Parquet"),
    "report": ("export.report", "counts and contact sheets of a capture"),
    "train": ("model.train", "fine-tune RF-DETR on an exported dataset"),
    "evaluate": ("model.evaluate", "run a trained model on a split and save its predictions"),
    "watch": ("model.watch", "live page of a training run"),
    "plot": ("model.plot", "plot a finished training run"),
    "detect": ("model.detect", "run a trained model on a video"),
    "showcase": ("model.showcase", "record a video of every mob for judging a model"),
    "flythrough": ("model.flythrough", "fly past mobs on real terrain: true and model boxes"),
    "windows": ("game.view_windows", "watch the game windows live"),
}
CHECKS: dict[str, tuple[str, str]] = {
    "setup": ("checks.setup", "camera, field of view and window frames line up"),
    "label-image": ("checks.label_image", "label images against known scenes"),
    "mob-states": ("checks.mob_states", "every random mob state shows in the picture"),
    "far-render": ("checks.far_render", "mobs are drawn at every distance"),
    "spread": ("checks.spread", "a capture's spread over places, times and mobs"),
    "clean-plates": ("checks.clean_plates", "labels against the same shot without the mobs"),
}


def usage() -> str:
    """The help text listing every command.

    Returns:
        Usage and command list.
    """
    width = max(len(name) for name in [*COMMANDS, *CHECKS]) + 8
    lines = [
        "usage: mobdataset <command> [options]",
        "       mobdataset check <check> [options]",
        "",
        "Generate labelled Minecraft mob datasets and train detectors on them.",
        "",
        "commands:",
    ]
    lines += [f"  {name:<{width}}{text}" for name, (_, text) in COMMANDS.items()]
    lines += ["", "checks:"]
    lines += [f"  check {name:<{width - 6}}{text}" for name, (_, text) in CHECKS.items()]
    lines += ["", "Run `mobdataset <command> --help` for a command's options."]
    return "\n".join(lines)


def run(module: str, prog: str, argv: Sequence[str]) -> int:
    """Import a command's module and run its main.

    Args:
        module: Module path inside the package.
        prog: Program name its help shows.
        argv: The command's own arguments.

    Returns:
        The exit code.
    """
    main = importlib.import_module(f"mobdataset.{module}").main
    sys.argv = [prog, *argv]
    result = main(list(argv))
    return result if isinstance(result, int) else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run `mobdataset` with the given arguments.

    Args:
        argv: Arguments without the program name; the process's own when None.

    Returns:
        The exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(usage())
        return 0
    if args[0] == "--version":
        print(f"mobdataset {__version__}")
        return 0
    name, rest = args[0], args[1:]
    if name == "check":
        if not rest or rest[0] not in CHECKS:
            print(usage(), file=sys.stderr)
            return 2
        return run(CHECKS[rest[0]][0], f"mobdataset check {rest[0]}", rest[1:])
    if name not in COMMANDS:
        print(f"mobdataset: unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    return run(COMMANDS[name][0], f"mobdataset {name}", rest)

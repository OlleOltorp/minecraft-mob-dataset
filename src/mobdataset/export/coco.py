"""The COCO file layout of the published dataset; `build` fills it in."""


def coco_skeleton(class_names: list[str]) -> dict[str, list[dict[str, object]]]:
    """Make an empty COCO file with categories laid out the way Roboflow exports them.

    Args:
        class_names: Class names; they get category ids 1..N, id 0 is the umbrella class.

    Returns:
        COCO dict with categories and empty images/annotations.
    """
    categories: list[dict[str, object]] = [{"id": 0, "name": "mobs", "supercategory": "none"}]
    categories += [
        {"id": i + 1, "name": name, "supercategory": "mobs"} for i, name in enumerate(class_names)
    ]
    return {"images": [], "annotations": [], "categories": categories}

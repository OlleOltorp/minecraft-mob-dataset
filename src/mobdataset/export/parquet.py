"""The Parquet layout of the published dataset and a writer for its shards.

The fields are filled in `labels`; `build` writes the rows.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROWS_PER_GROUP = 64  # Parquet row group size; small groups keep image reads cheap


IMAGE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


OBJECT = pa.struct(
    [
        ("id", pa.int64()),
        ("category", pa.string()),
        ("category_id", pa.int64()),
        ("bbox", pa.list_(pa.float64())),
        ("segmentation", pa.struct([("counts", pa.string()), ("size", pa.list_(pa.int32()))])),
        ("area", pa.int64()),
        ("iscrowd", pa.int64()),
        ("distance", pa.float64()),
        ("placed", pa.bool_()),
        ("native", pa.bool_()),
        ("features", pa.list_(pa.string())),
        ("motion", pa.string()),
        ("visibility", pa.float64()),
        ("too_dark", pa.bool_()),
        ("brightness", pa.float64()),
        ("contrast", pa.float64()),
        ("invisible", pa.bool_()),
        ("box_full", pa.list_(pa.float64())),
        ("truncated", pa.float64()),
        ("visible_share", pa.float64()),
    ]
)


# Picture fields, shared by the COCO image entries and the Parquet rows.
PICTURE = [
    ("image_id", pa.int64()),
    ("file_name", pa.string()),
    ("width", pa.int64()),
    ("height", pa.int64()),
    ("capture", pa.string()),
    ("stem", pa.string()),
    ("scene", pa.int64()),
    ("split", pa.string()),
    ("spot_id", pa.int64()),
    ("area", pa.string()),
    ("structure", pa.string()),
    ("camera_kind", pa.string()),
    ("dimension", pa.string()),
    ("biome", pa.string()),
    ("spot_biome", pa.string()),
    ("time", pa.string()),
    ("day_time", pa.int64()),
    ("weather", pa.string()),
    ("precipitation", pa.string()),
    ("raining", pa.bool_()),
    ("thundering", pa.bool_()),
    ("camera_x", pa.float64()),
    ("camera_y", pa.float64()),
    ("camera_z", pa.float64()),
    ("camera_height", pa.float64()),
    ("yaw", pa.float64()),
    ("pitch", pa.float64()),
    ("fov", pa.float64()),
    ("underwater", pa.bool_()),
    ("aerial", pa.bool_()),
    ("empty", pa.bool_()),
    ("window", pa.string()),
    ("brightness", pa.float64()),
    ("has_clean_plate", pa.bool_()),
    ("clean_file_name", pa.string()),
    ("clean_placed_left", pa.int64()),
]


SCHEMA = pa.schema(
    [("image", IMAGE), ("clean_plate", IMAGE), *PICTURE, ("objects", pa.list_(OBJECT))],
    # Tells Hugging Face datasets to decode the two image columns as pictures; the other
    # columns' types come from the Parquet schema itself.
    metadata={
        "huggingface": json.dumps(
            {"info": {"features": {c: {"_type": "Image"} for c in ("image", "clean_plate")}}}
        )
    },
)


class ShardWriter:
    """Parquet shards of one split, about `target` bytes of pictures each.

    Shards are written under temporary names and renamed to
    <split>-NNNNN-of-NNNNN.parquet once their number is known.

    Args:
        folder: Folder for the shards.
        split: Split name.
        target: Bytes of pictures per shard.
    """

    def __init__(self, folder: Path, split: str, target: int) -> None:
        self.folder, self.split, self.target = folder, split, target
        self.paths: list[Path] = []
        self.writer: pq.ParquetWriter | None = None
        self.rows: list[dict] = []
        self.size = 0

    def add(self, row: dict, size: int) -> None:
        """Queue a row.

        Args:
            row: Picture fields, images and objects.
            size: Bytes its pictures take.
        """
        if self.writer is None:
            self.paths.append(self.folder / f".{self.split}-{len(self.paths)}.tmp.parquet")
            self.writer = pq.ParquetWriter(self.paths[-1], SCHEMA)
        self.rows.append(row)
        self.size += size
        if len(self.rows) >= ROWS_PER_GROUP or self.size >= self.target:
            self._flush()
        if self.size >= self.target:
            self._close_shard()

    def _flush(self) -> None:
        if self.rows and self.writer is not None:
            self.writer.write_table(pa.Table.from_pylist(self.rows, schema=SCHEMA))
            self.rows = []

    def _close_shard(self) -> None:
        self._flush()
        if self.writer is not None:
            self.writer.close()
        self.writer, self.size = None, 0

    def close(self) -> list[Path]:
        """Finish the last shard and give the shards their final names.

        Returns:
            The shards' paths.
        """
        self._close_shard()
        n = len(self.paths)
        final = [self.folder / f"{self.split}-{i:05d}-of-{n:05d}.parquet" for i in range(n)]
        for tmp, path in zip(self.paths, final, strict=True):
            tmp.replace(path)
        return final

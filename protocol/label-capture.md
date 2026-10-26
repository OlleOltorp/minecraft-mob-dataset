# Label capture (mod ↔ Python)

Each game window with the `mobcapture` mod listens on `127.0.0.1:<port>`.
Port = `[mod] control_port_base` in `config/client.toml`, plus k for window
`camk` (plus 0 for a single signed-in window). Text lines, UTF-8, `\n`-terminated; one reply per
request, in order.

| Request | Reply |
|---|---|
| `PING` | `PONG` |
| `CAPTURE <name> <dir>` | `OK <name>` once the frame is copied, or `ERR <reason>` |

`<name>`: `[A-Za-z0-9_.-]+`. `<dir>`: absolute folder, rest of the line (may contain spaces).
The next frame the window draws is used; the mod gives up after 30 s. The files are written
after the reply, each under a temporary name and then renamed; the JSON comes last, so a
reader waits for `<name>.json`.

Files written, all from that one frame:

- `<name>.png` — the picture: the world only, no HUD, same as the window shows.
- `<name>.label.png` — label image, same size. Pixel colour as a 24-bit number
  `r << 16 | g << 8 | b`:
  - `0` nothing;
  - `1..n` the mob listed with that `colour` in the JSON;
  - `0xFFFFFF` an entity that is not a mob (item, boat, arrow…); it hides what is behind it.
  Terrain in front of a mob hides it; the nearer of two mobs wins. Invisible mobs are labelled
  (the spectator camera sees them faintly).
- `<name>.json`:

```json
{
  "camera": {"x": 0.5, "y": -58.38, "z": 0.5, "yaw": 0.0, "pitch": 0.0},
  "width": 1280, "height": 720,
  "dimension": "minecraft:overworld", "game_time": 1234, "day_time": 6000,
  "raining": false, "thundering": false,
  "other_colour": 16777215,
  "mobs": [
    {"colour": 1, "type": "minecraft:zombie", "id": 42, "uuid": "…",
     "x": 0.5, "y": -60.0, "z": 12.5, "yaw": 180.0, "invisible": false,
     "aabb": [0.2, -60.0, 12.2, 0.8, -58.05, 12.8]}
  ]
}
```

Mob `x/y/z` and `aabb` are where the mob was drawn in that frame (between game ticks).

Known limits: block entities (chests, signs, beds) and see-through terrain (water, stained
glass) do not hide mobs in the label image.

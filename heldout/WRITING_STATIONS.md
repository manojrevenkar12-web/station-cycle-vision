# Writing a held-out station

You are writing one test station for a system that watches manual assembly stations through an
overhead camera and recognises work steps, cycle times and deviations. You must not look at the
system's code, its other stations or its results; this brief is all you get. The station is used
once: its first result is recorded and never rewritten, and nobody changes the system to do better
on it.

## The world

The camera looks down at a 96 x 96 bench (x to the right, y downwards; the worker stands below the
bottom edge, so arms reach in from y = 96). On the bench there are:

- `base_bin`: where base plates are picked from
- `screw_bin`: where screws are picked from
- `fixture`: where the plate is placed and screwed (the plate is about 18 x 12)
- `output`: where finished plates are put
- `tool_rest`: where the screwdriver lies when not in use
- `rest_left`, `rest_right`: where the left and right hand wait

Each is an `[x, y]` point that must lie between 4 and 92 in both coordinates. Bins and the output
tray are about 16 x 14; the fixture about 24 x 18; keep things from overlapping heavily.

## The work

Steps: `pick_base`, `place_base`, `pick_screw`, `fasten`, `inspect`, `unload`. A station's
standard work is the step sequence of one cycle. Rules: it starts with `pick_base`, ends with
`unload`, and every `fasten` comes directly after a `pick_screw`. Between 1 and 4 fastenings is
realistic. `inspect` may come anywhere after `place_base` and before `unload`.

## The recording conditions

Ranges `[low, high]`, drawn once per recording (camera, light) or as rates:

| key | meaning | typical bench recording |
|---|---|---|
| rotation_deg | camera rotation | [-4, 4] |
| scale | zoom | [0.95, 1.05] |
| shift | camera offset in bench units | [-2, 2] |
| tilt | perspective (0 = straight down; up to 0.3) | [0.0, 0.03] |
| gain | brightness multiplier | [0.9, 1.1] |
| tint | per-colour-channel multiplier | [0.95, 1.05] |
| vignette | darkening towards one side (0 to 0.6) | [0.0, 0.1] |
| noise | sensor noise std (0 to 0.1) | [0.01, 0.02] |
| occlusions_per_minute | people walking through the view (0 to 8) | [0, 0] |
| frame_drop | share of frames the camera repeats (0 to 0.3) | [0, 0] |

Any key you leave out keeps the typical bench value.

## What to write

Describe in a sentence or two (as a YAML comment) a plausible station on a real line that differs
from a tidy bench: a different layout, a different product (number of screws, where inspection
happens) and the conditions of where its camera is mounted. Keep it realistic, neither trivial nor
impossible. Write `heldout/station.yaml`:

```yaml
station:
  layout: {base_bin: [x, y], screw_bin: [x, y], fixture: [x, y], output: [x, y], tool_rest: [x, y], rest_left: [x, y], rest_right: [x, y]}
  standard_work: [pick_base, place_base, ...]
  bench: [r, g, b]        # bench colour, each 0..1
conditions:
  rotation_deg: [low, high]
  # ... any keys from the table
```

Then write `heldout/STATUS` containing the single word `blind`.

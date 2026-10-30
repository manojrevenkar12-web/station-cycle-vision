# 1. Simulated stations, rendered from a work schedule

Status: accepted

Real overhead video of assembly stations, with frame-level step labels, cycle boundaries and labelled
deviations, is not public, and a camera moved by 20 degrees cannot be replayed on a recording. The
questions here (does it survive a moved camera, how many labels does a new station cost, can the box
count cycles from a live stream) need the shift to be a dial and the truth to be exact.

Each recording is generated from a schedule: a station's standard work, a worker with a speed and a
per-step jitter, deviations (a fastening missing, two steps swapped, a fastening repeated, a pause),
and idle time between cycles. The renderer draws the bench, bins, fixture, plate, screws, the tool,
the worker's arms and hands and anyone walking past, then applies the camera (a homography), lighting,
sensor noise and dropped frames. The truth comes from the same schedule: step per frame, cycle
boundaries, deviations, hand positions projected through the camera, the person mask.

Consequences: accuracy figures do not transfer to a real line; the method, the evaluation and the
plumbing do. The first step with real recordings is to label a few cycles per station and run the
same evaluation.

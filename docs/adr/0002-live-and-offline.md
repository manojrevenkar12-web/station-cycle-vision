# 2. A causal model on the box, an offline model for re-segmentation

Status: accepted

Cycle times must be available while the line runs, so the model on the camera box cannot look ahead.
A causal dilated TCN (MS-TCN layout, two stages, receptive field 511 frames) runs one frame at a time
with explicit state: each dilated layer keeps a ring buffer of the last 2d hidden vectors, and a zero
state is exactly the zero padding of the offline model, so streaming equals batch (tested).

An offline bidirectional transformer is trained on the same features as the comparison: what waiting
for the whole recording would buy. The report shows both; the box runs the causal one.

Labels are turned into events by a causal debounce (a change is committed after it holds for 0.4 s)
and the cycle rules: a cycle starts at pick_base and ends with unload, and its step sequence is aligned
with the station's standard work (Damerau-Levenshtein) to name what was missing, extra or out of
order. `scv.process.live.run_live` is the reference; the service must emit exactly the same events
whatever the request chunking (tested).

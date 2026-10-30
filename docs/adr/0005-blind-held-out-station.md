# 5. A held-out station written blind, run once

Status: accepted

`heldout/station.yaml` (layout, product, standard work, camera and light) was written from
`heldout/WRITING_STATIONS.md` alone by someone who had not seen the code, the other stations or any
result. Its sha256 is in `heldout/station.sha256`; its first run is `reports/heldout/first_run.json`
and is never rewritten. Nothing is changed to do better on it; if anyone studies its failures, write
`regression` to `heldout/STATUS` and have a new one written from the brief.

The first run's file reports a cycle-time error of 0.00 s for models that found no cycle at all; that
was a bug (no match counted as zero error), fixed afterwards so later reports say "n/a". The first
run is left as recorded.

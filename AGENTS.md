# Working in this repository

Fast feedback: `pytest -m "not slow"` (about two minutes: simulator, process layer, metrics, streaming
equivalence, ONNX parity, the service, DDP, protocol), `cd web && npm test`. Everything:
`scv all --profile quick` (a minute) or `scv all` (the committed run, about 20 minutes on two cores).
`ruff check src tests scripts && ruff format --check src tests scripts`; `cd web && npm run lint`.

Invariants (each has a test):

- Settings live in `config/` (stations, domains, experiment, service, release), not in code.
- Nothing trains or tunes on a test set: encoder, temporal models and detector train on `data["train"]`;
  the detector threshold is chosen on `data["val"]`; the new station contributes one onboarding
  recording that is not among its test recordings.
- The streaming TCN equals the offline TCN; the ONNX edge path equals PyTorch; the service emits exactly
  what `scv.process.live.run_live` emits.
- The edge image has no PyTorch. Video stays on the box; clips are anonymised before they are written.
- `reports/`, `docs/REPORT.md` and the dashboard's numbers are generated; the dashboard computes nothing.
- Release targets in `config/release.yaml` are never lowered; floors are regression guards.

Held-out rule: never change anything to do better on `heldout/station.yaml`. Its first run in
`reports/heldout/` is never rewritten. If you study its failures, write `regression` to
`heldout/STATUS` and have someone who has not seen the code write a new station from
`heldout/WRITING_STATIONS.md` only.

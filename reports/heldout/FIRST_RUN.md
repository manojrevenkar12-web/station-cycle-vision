# Held-out station: first run

Recorded 2026-10-30T20:34:10Z; station file sha256 `4cfd1be52329354360da4decfbba93b6ca4226f5d71a73567f228d4d1fc7b08b`. Never rewritten. Written from `heldout/WRITING_STATIONS.md` by someone who had not seen the code, the other stations or any result.

| Model | Frame accuracy | Edit | F1@50 | Cycles found | Cycle-time error |
|---|---:|---:|---:|---:|---:|
| frame-only | 68.3% | 44.6 | 40.0 | 22.7% | 3.72 s |
| causal-tcn (no augmentation) | 43.1% | 32.9 | 15.9 | 0.0% | 0.00 s |
| causal-tcn | 75.6% | 60.4 | 53.9 | 54.5% | 1.72 s |
| offline-transformer | 69.5% | 53.3 | 42.9 | 31.8% | 2.85 s |

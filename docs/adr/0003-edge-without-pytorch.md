# 3. The edge image runs ONNX Runtime only

Status: accepted

The camera box image contains ONNX Runtime, FastAPI and NumPy, not PyTorch (CI checks that `torch`
is not importable in it). The encoder is exported to ONNX and statically quantised to int8 (QDQ,
per-channel weights, calibrated on bench validation frames); the streaming TCN is exported as a
one-frame step whose state is explicit inputs and outputs; the detector is exported for the
anonymiser. Every export is checked against PyTorch (float path < 1e-3).

Events and anonymised clips stay on the box in SQLite and on its volume; the NetworkPolicy allows no
egress. Training and evaluation run as a Kubernetes Job on a training cluster (`deploy/k8s/jobs`),
tracked in MLflow, with a data-parallel path (DDP, gloo or nccl) whose gradient equivalence is tested.

TensorRT is the obvious next step on NVIDIA camera boxes; it is not used here because nothing in this
environment has a GPU, and an untested TensorRT path would be a claim, not a feature.

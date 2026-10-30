"""`scv run | edge | report | gate | all | serve`."""

from __future__ import annotations

import argparse
import sys

from . import config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scv")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, help_ in [
        ("run", "render data, train, evaluate"),
        ("edge", "export ONNX/int8, evaluate the live edge path"),
        ("report", "write reports/results.json and docs/REPORT.md"),
        ("gate", "check config/release.yaml"),
        ("all", "run, edge, report, gate"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--profile", default="full", choices=sorted(config.load("experiment.yaml")["profiles"]))
    s = sub.add_parser("serve", help="the edge service")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn

        from .service.app import create_app

        uvicorn.run(create_app(), host=a.host, port=a.port, log_level="info")
        return 0
    steps = ["run", "edge", "report", "gate"] if a.cmd == "all" else [a.cmd]
    for step in steps:
        if step == "run":
            from .evaluate.experiment import run

            run(a.profile)
        elif step == "edge":
            from .edge.pipeline import run_edge

            run_edge(a.profile)
        elif step == "report":
            from .evaluate.report import write_reports

            write_reports(a.profile)
        elif step == "gate":
            from .evaluate.gate import gate

            ok, lines = gate(a.profile)
            print("\n".join(lines))
            if not ok:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
from __future__ import annotations

import json


def main() -> None:
    report = {
        "import_esm": False,
        "has_pretrained": False,
        "has_esmfold_v1": False,
        "missing_modules": [],
        "install_hint": "Install fair-esm with ESMFold support and OpenFold dependencies, then retry.",
    }
    try:
        import esm  # type: ignore

        report["import_esm"] = True
        pretrained = getattr(esm, "pretrained", None)
        report["has_pretrained"] = pretrained is not None
        report["has_esmfold_v1"] = bool(pretrained is not None and hasattr(pretrained, "esmfold_v1"))
    except ModuleNotFoundError as exc:
        report["missing_modules"].append(exc.name or "esm")
        report["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    if report["import_esm"] and report["has_esmfold_v1"]:
        for module in ["openfold"]:
            try:
                __import__(module)
            except ModuleNotFoundError:
                report["missing_modules"].append(module)
        report["ok"] = not report["missing_modules"]
    else:
        if "esm" not in report["missing_modules"]:
            report["missing_modules"].append("fair-esm")
        report["ok"] = False

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

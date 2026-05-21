from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "16_download_or_check_progen3.py"
    spec = importlib.util.spec_from_file_location("progen3_preflight_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_hf_repo_without_remote_or_official_code_fails(monkeypatch, tmp_path, capsys):
    script = _load_script()
    config_path = tmp_path / "rl.yaml"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        script,
        "load_yaml",
        lambda _path: {
            "oracle": {
                "progen3": {
                    "mode": "real",
                    "model_name_or_path": "Profluent-Bio/progen3-3b",
                    "device": "cpu",
                }
            }
        },
    )
    monkeypatch.setattr(script, "code_import_report", lambda _path: {"checked_paths": [], "importable": False, "error": "cannot import progen3"})
    monkeypatch.setattr(script, "remote_code_report", lambda _repo: {"checked": True, "has_remote_code": False, "files": [], "error": None})
    monkeypatch.setattr(sys, "argv", ["16_download_or_check_progen3.py", "--config", str(config_path)])

    try:
        script.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("preflight unexpectedly passed")

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["model"]["ok"] is False
    assert "Missing ProGen3 official Python package/code; cannot real score" in report["model"]["error"]


def test_mock_mode_preflight_does_not_probe_hf(monkeypatch, tmp_path, capsys):
    script = _load_script()
    config_path = tmp_path / "rl.yaml"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(script, "load_yaml", lambda _path: {"oracle": {"progen3": {"mode": "mock"}}})
    monkeypatch.setattr(script, "remote_code_report", lambda _repo: (_ for _ in ()).throw(AssertionError("HF should not be probed")))
    monkeypatch.setattr(sys, "argv", ["16_download_or_check_progen3.py", "--config", str(config_path)])

    script.main()

    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["model"]["type"] == "mock"


def test_missing_code_path_reports_clear_error(tmp_path):
    script = _load_script()
    missing = tmp_path / "missing_progen3" / "src"

    report = script.code_import_report(str(missing))

    assert report["importable"] is False
    assert "code_path does not exist" in report["error"]
    assert "Missing ProGen3 official Python package/code; cannot real score" in report["error"]

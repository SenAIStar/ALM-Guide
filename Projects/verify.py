from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parent
python_files = sorted(root.rglob("*.py"))
for path in python_files:
    compile(path.read_text(encoding="utf-8"), str(path), "exec")

project_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name[:2].isdigit())
if len(project_dirs) != 7:
    raise AssertionError(f"expected 7 projects, got {len(project_dirs)}")

manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
if Path(manifest.get("root", "")).is_absolute() or manifest.get("root") != ".":
    raise AssertionError("manifest root must be the portable relative path '.'")
if manifest.get("projects") != len(project_dirs):
    raise AssertionError("manifest project count is stale")
file_count = sum(1 for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
if manifest.get("files") != file_count:
    raise AssertionError(f"manifest file count is stale: expected {file_count}")

required_readme_sections = ("## 项目定位", "## 数据协议", "## 运行", "## 评测", "## 常见故障")
for project in project_dirs:
    config = json.loads((project / "config.json").read_text(encoding="utf-8"))
    if config.get("metrics_status") != "not_measured":
        raise AssertionError(f"{project.name}: metrics_status must remain not_measured")
    readme = (project / "README.md").read_text(encoding="utf-8")
    missing = [section for section in required_readme_sections if section not in readme]
    if missing:
        raise AssertionError(f"{project.name}: README missing {missing}")
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        cwd=project,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )

blocked_markers = ("model_not_loaded", "Wire the validated manifest", "Connect the validated batch collator")
for path in python_files:
    if path.resolve() == Path(__file__).resolve():
        continue
    text = path.read_text(encoding="utf-8")
    marker = next((value for value in blocked_markers if value in text), None)
    if marker:
        raise AssertionError(f"{path}: unfinished marker {marker!r}")

print(json.dumps({"projects": len(project_dirs), "python_files": len(python_files), "status": "passed"}, indent=2))

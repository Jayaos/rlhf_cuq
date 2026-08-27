#!/usr/bin/env python3
"""Validate that legacy editable packages resolve to their pinned Git sources."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "artifacts" / "source_manifest.json"
PACKAGE_SOURCES = (
    ("model_training", "open_assistant"),
    ("oasst_data", "open_assistant"),
    ("trlx", "trlx"),
    ("alpaca_farm", "coste_alpaca_farm_fork"),
)


class LegacySourceError(RuntimeError):
    """Raised when an imported legacy package is not the pinned source."""


def _module_path(module: ModuleType, package_name: str) -> Path:
    value = getattr(module, "__file__", None)
    if not value:
        locations = list(getattr(module, "__path__", ()))
        collision_hint = ""
        project_collision = (ROOT / "src" / package_name).resolve()
        if any(Path(location).resolve() == project_collision for location in locations):
            collision_hint = (
                f"; {project_collision} is an editable checkout placed inside "
                "the project package tree. Move it outside project/src and "
                "reinstall with --src $CONDA_PREFIX/legacy-src"
            )
        raise LegacySourceError(
            f"{package_name} resolved as a namespace without __file__; "
            f"locations={locations!r}{collision_hint}"
        )
    return Path(value).resolve()


def _git_output(module_path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(module_path.parent), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise LegacySourceError(
            f"cannot verify editable Git source for {module_path}: {detail}"
        )
    return result.stdout.strip()


def _expected_revisions() -> dict[str, str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        entry["name"]: entry["revision"] for entry in manifest["repositories"]
    }


def _validate_trlx_api(module: ModuleType, module_path: Path) -> None:
    exported_train = getattr(module, "train", None)
    if not callable(exported_train):
        raise LegacySourceError(
            "trlx does not expose callable trlx.train; "
            f"imported module={module_path}. The pinned source exports this API."
        )
    implementation = importlib.import_module("trlx.trlx")
    if exported_train is not getattr(implementation, "train", None):
        raise LegacySourceError(
            "trlx.train is not the function exported by trlx.trlx; "
            f"imported module={module_path}"
        )


def main() -> int:
    expected = _expected_revisions()
    failures: list[str] = []

    for package_name, source_name in PACKAGE_SOURCES:
        try:
            module = importlib.import_module(package_name)
            module_path = _module_path(module, package_name)
            checkout_root = Path(
                _git_output(module_path, "rev-parse", "--show-toplevel")
            ).resolve()
            actual_revision = _git_output(module_path, "rev-parse", "HEAD")
            expected_revision = expected[source_name]
            if actual_revision != expected_revision:
                raise LegacySourceError(
                    f"{package_name} revision mismatch: expected "
                    f"{expected_revision}, found {actual_revision} at {checkout_root}"
                )
            if package_name == "trlx":
                _validate_trlx_api(module, module_path)
            print(
                f"PASS {package_name}: {actual_revision} "
                f"({module_path})"
            )
        except Exception as error:  # Keep reporting every invalid source.
            failures.append(f"{package_name}: {error}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}", file=sys.stderr)
        print(
            "ERROR: legacy source validation failed. Repair the editable "
            "install before submitting a GPU job.",
            file=sys.stderr,
        )
        return 1

    print("PASS pinned legacy source revisions and APIs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

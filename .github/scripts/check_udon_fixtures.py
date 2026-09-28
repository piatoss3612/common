#!/usr/bin/env python3
"""Check standalone Udon and Bento fixtures with Common's nightly formatter."""

import difflib
import json
from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[2]
metadata = json.loads(subprocess.check_output(
    ["cargo", "+nightly", "metadata", "--locked", "--offline", "--no-deps", "--format-version", "1"],
    cwd=root,
))
for package in metadata["packages"]:
    if package["name"] not in {"zakura-udon", "zakura-bento", "zakura-bento-core", "zakura-bento-macros"}:
        continue
    tests = Path(package["manifest_path"]).parent / "tests"
    for source in sorted(tests.rglob("*.rs")):
        if "fixtures" not in source.relative_to(tests).parts:
            continue
        # stdin avoids resolving modules that the consumer harness assembles.
        original = source.read_text()
        result = subprocess.run(
            ["rustup", "run", "nightly", "rustfmt", "--edition", package["edition"]],
            input=original,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=root,
        )
        if result.returncode:
            print(f"Formatting failed: {source.relative_to(root)}", file=sys.stderr)
            sys.stderr.write(result.stderr)
            sys.exit(result.returncode)
        # --check does not report stdin differences on the pinned rustfmt.
        if result.stdout != original:
            name = str(source.relative_to(root))
            sys.stderr.writelines(difflib.unified_diff(
                original.splitlines(keepends=True),
                result.stdout.splitlines(keepends=True),
                fromfile=name,
                tofile=f"{name} (formatted)",
            ))
            sys.exit(1)

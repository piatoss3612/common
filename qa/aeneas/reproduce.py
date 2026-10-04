#!/usr/bin/env python3
"""Extract Udon arithmetic and check its pinned Aeneas/Lean proofs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


HERE = Path(__file__).resolve().parent
PIN = json.loads((HERE / "provenance.json").read_text())
CATALOG = json.loads((HERE / "catalog.json").read_text())


def read_command(arguments, cwd):
    return subprocess.check_output(arguments, cwd=cwd, text=True).strip()


def make_slice(repo, destination):
    source = repo / "crates/udon/src/field/pasta"
    parameters = (source / "parameters.rs").read_text()
    primes = re.findall(
        r'    (PallasBase|PallasScalar),\n    modulus: "(0x[0-9a-f]+)"', parameters
    )
    if len(primes) != 2:
        raise RuntimeError("Could not identify both pinned Pasta moduli")
    lines = [
        "#![no_std]",
        "#![forbid(unsafe_code)]",
        "pub mod field {",
        "pub mod pasta {",
        "pub trait PrimeModulus {",
        "    const MODULUS: [u64; 4];",
        "    const TWICE_MODULUS: [u64; 4];",
        "    const MONTGOMERY_INV: u64;",
        "}",
    ]
    radix = 1 << 64

    def limbs(value):
        return "[" + ", ".join(hex((value >> (64 * i)) % radix) for i in range(4)) + "]"

    for marker, literal in primes:
        modulus = int(literal, 16)
        coefficient = -pow(modulus % radix, -1, radix) % radix
        assert ((modulus % radix) * coefficient + 1) % radix == 0
        assert modulus >> 128 == 1 << 126
        assert 3 * modulus < 1 << 256
        lines.extend([
            f"#[derive(Clone, Copy)] pub enum {marker} {{}}",
            f"impl PrimeModulus for {marker} {{",
            f"    const MODULUS: [u64; 4] = {limbs(modulus)};",
            f"    const TWICE_MODULUS: [u64; 4] = {limbs(2 * modulus)};",
            f"    const MONTGOMERY_INV: u64 = {hex(coefficient)};",
            "}",
        ])
    for name in ["word", "montgomery"]:
        lines.append(f"#[path = {json.dumps(str(source / (name + '.rs')))}] pub mod {name};")
    lines.extend(["}", "}"])
    destination.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[1])
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--offline", action="store_true", help="Require cached Cargo dependencies")
    args = parser.parse_args()
    repo = args.repo.resolve()
    tools = args.tools.resolve()
    head = read_command(["git", "rev-parse", "HEAD"], repo)
    subprocess.run(["git", "merge-base", "--is-ancestor", PIN["udon_revision"], head],
                   cwd=repo, check=True)
    for relative, expected in PIN["source_hashes"].items():
        actual = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source differs from the pinned experiment: {relative}")
    aeneas = tools / "aeneas"
    charon = tools / "charon"
    aeneas_version = read_command([str(aeneas), "-version"], repo)
    charon_version = read_command([str(charon), "version"], repo)
    if aeneas_version != "aeneas " + PIN["aeneas_release"]:
        raise RuntimeError("Unexpected Aeneas version: " + aeneas_version)
    if PIN["charon_revision"] not in charon_version:
        raise RuntimeError("Unexpected Charon version: " + charon_version)
    backend = tools / "backends/lean"
    lean_toolchain = (backend / "lean-toolchain").read_text().strip()
    if lean_toolchain != "leanprover/lean4:v4.31.0":
        raise RuntimeError("Unexpected Lean toolchain: " + lean_toolchain)
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = Path(tempfile.mkdtemp(prefix="udon-aeneas-repro-"))
    print("Output:", output, flush=True)
    for directory in ["logs", "llbc", "slice", "lean"]:
        (output / directory).mkdir()
    environment = os.environ.copy()
    environment["CARGO_TARGET_DIR"] = str(output / "cargo-target")
    environment["RUST_MIN_STACK"] = "67108864"
    results = []

    def run(name, arguments, cwd):
        log = output / "logs" / (name + ".log")
        started = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.run(arguments, cwd=cwd, env=environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
        seconds = round(time.monotonic() - started, 3)
        results.append({"name": name, "arguments": arguments, "cwd": str(cwd),
                        "exit_code": process.returncode, "seconds": seconds, "log": str(log)})
        print(f"{name}: exit {process.returncode} ({seconds}s)", flush=True)
        if process.returncode:
            print("\n".join(log.read_text().splitlines()[-35:]))
            raise RuntimeError("Stage failed; inspect " + str(log))

    checked = ["--rustc-arg=-Coverflow-checks=on", "--rustc-arg=-Cdebug-assertions=on"]
    cargo = ["--package", "zakura-udon", "--lib", "--locked"]
    if args.offline:
        cargo.append("--offline")
    run("word-charon", [str(charon), "cargo", "--preset=aeneas", "--sysroot=default",
        "--start-from", "crate::field::pasta::word", *checked, "--dest-file",
        str(output / "llbc/word.llbc"), "--", *cargo], repo)
    slice_file = output / "slice/lib.rs"
    make_slice(repo, slice_file)
    run("slice-charon", [str(charon), "rustc", "--preset=aeneas", "--sysroot=default",
        *checked, "--dest-file", str(output / "llbc/kernel-slice.llbc"), "--",
        "--crate-type", "lib", "--edition", "2024", "--crate-name", "udon_kernel_slice",
        str(slice_file)], output / "slice")
    for llbc_name, module in [("word", "Word"), ("kernel-slice", "KernelSlice")]:
        data = json.loads((output / "llbc" / (llbc_name + ".llbc")).read_text())
        if data["has_errors"]:
            raise RuntimeError("Charon exported a partial model: " + llbc_name)
        run(module.lower() + "-aeneas", [str(aeneas), "-backend", "lean", "-split-files",
            "-subdir", module, "-no-progress-bar", "-dest", str(output / "lean"),
            str(output / "llbc" / (llbc_name + ".llbc"))], output)
    lean = output / "lean"
    shutil.copyfile(backend / "lean-toolchain", lean / "lean-toolchain")
    for module in CATALOG["modules"]:
        shutil.copyfile(HERE / "proofs" / (module + ".lean"), lean / (module + ".lean"))
    configuration = ["import Lake", "open Lake DSL",
                     "require aeneas from " + json.dumps(str(backend)),
                     "package udonFeasibility"]
    for name in ["Word", "KernelSlice", *CATALOG["modules"]]:
        configuration.append("@[default_target] lean_lib " + name)
    (lean / "lakefile.lean").write_text("\n".join(configuration) + "\n")
    for name in ["Word", "KernelSlice"]:
        (lean / (name + ".lean")).write_text("import " + name + ".Funs\n")
    (lean / ".lake").mkdir()
    cached_packages = backend / ".lake/packages"
    if cached_packages.is_dir():
        (lean / ".lake/packages").symlink_to(cached_packages, target_is_directory=True)
    source_files = [file for file in lean.rglob("*.lean") if ".lake" not in file.parts]
    for file in source_files:
        if "External" in file.name or re.search(r"\b(sorry|admit|axiom|opaque)\b", file.read_text()):
            raise RuntimeError("A model has holes or requires external definitions: " + str(file))
    run("lean-build", ["lake", "build"], lean)
    census = "\n".join("import " + module for module in CATALOG["modules"]) + "\n"
    census += "\n".join("#print axioms " + theorem for theorem in CATALOG["proved"]) + "\n"
    (lean / "Census.lean").write_text(census)
    run("proof-census", ["lake", "env", "lean", "Census.lean"], lean)
    proof_log = (output / "logs/proof-census.log").read_text()
    if "sorryAx" in proof_log:
        raise RuntimeError("Proof trust census contains sorryAx")
    for theorem in CATALOG["proved"]:
        if f"'{theorem}' depends on axioms: [propext, Classical.choice, Quot.sound]" not in proof_log:
            raise RuntimeError("Unexpected proof trust census for " + theorem)
    summary = {"udon_revision": PIN["udon_revision"], "aeneas_version": aeneas_version,
               "charon_version": charon_version, "lean_toolchain": lean_toolchain,
               "source_hashes": PIN["source_hashes"], "stages": results,
               "checkout_revision": head, "proved": CATALOG["proved"],
               "scope": CATALOG["scope"]}
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Passed: extraction, Lean typechecking, {len(CATALOG['proved'])} proofs, and proof trust census", flush=True)


if __name__ == "__main__":
    main()

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
import time
import tomllib

from build_sqrt_charon import verify_manifest

if not __debug__:
    raise RuntimeError("The extraction checks require Python assertions")


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


def check_native_lock(repo, wrapper, package_name="udon-native-field-proof"):
    original = tomllib.loads((repo / "Cargo.lock").read_text())
    actual = tomllib.loads((wrapper / "Cargo.lock").read_text())
    fields = ["name", "version", "source", "checksum"]
    pinned = {tuple(package.get(key) for key in fields) for package in original["package"]}
    for package in actual["package"]:
        if package["name"] == package_name:
            continue
        if tuple(package.get(key) for key in fields) not in pinned:
            raise RuntimeError("Unpinned native extraction dependency: " + package["name"])


def check_ordering(llbc):
    name = [{"Ident": [part, 0]} for part in ["core", "cmp", "Ordering"]]
    declaration = next(item for item in llbc["translated"]["type_decls"]
                       if item and item["item_meta"]["name"] == name)
    variants = declaration["kind"]["Enum"]
    actual = [(item["name"], item["discriminant"]) for item in variants]
    expected = [("Less", {"Signed": ["I8", "-1"]}),
                ("Equal", {"Signed": ["I8", "0"]}),
                ("Greater", {"Signed": ["I8", "1"]})]
    if actual != expected:
        raise RuntimeError("Unexpected Rust Ordering discriminants: " + repr(actual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[1])
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--sqrt-tools", type=Path, required=True,
                        help="Pinned source build from build_sqrt_charon.py")
    parser.add_argument("--output", type=Path, help="New output directory; must not already exist")
    parser.add_argument("--offline", action="store_true", help="Require cached Cargo dependencies")
    args = parser.parse_args()
    repo = args.repo.resolve()
    tools = args.tools.resolve()
    sqrt_tools = args.sqrt_tools.resolve()
    sqrt_manifest = verify_manifest(sqrt_tools)
    sqrt_charon = sqrt_tools / "build/release/charon"
    sqrt_charon_version = read_command([str(sqrt_charon), "version"], repo)
    if PIN["charon_sqrt"]["revision_label"] not in sqrt_charon_version:
        raise RuntimeError("Unexpected square-root Charon version: " + sqrt_charon_version)
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
    for relative, expected in PIN.get("tool_hashes", {}).items():
        actual = hashlib.sha256((tools / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("Tool differs from the pinned release: " + relative)
    backend = tools / "backends/lean"
    lean_toolchain = (backend / "lean-toolchain").read_text().strip()
    if lean_toolchain != "leanprover/lean4:v4.31.0":
        raise RuntimeError("Unexpected Lean toolchain: " + lean_toolchain)
    backend_manifest = json.loads((backend / "lake-manifest.json").read_text())
    mathlib = next(package for package in backend_manifest["packages"] if package["name"] == "mathlib")
    if mathlib["rev"] != PIN["mathlib_revision"]:
        raise RuntimeError("Unexpected Mathlib revision: " + mathlib["rev"])
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        output = repo / "target/aeneas" / time.strftime("repro-%Y%m%d-%H%M%S")
        output.mkdir(parents=True, exist_ok=False)
    print("Output:", output, flush=True)
    for directory in ["logs", "llbc", "slice", "native/src", "sqrt/src", "lean"]:
        (output / directory).mkdir(parents=True)
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
        (output / "stages.json").write_text(json.dumps(results, indent=2) + "\n")
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
    algorithms = output / "llbc/sqrt-algorithms.llbc"
    run("sqrt-algorithms-charon", [str(charon), "cargo", "--preset=aeneas", "--sysroot=default",
        "--include", "zakura_udon", "--include", "core::tuple", "--include", "core::cmp",
        "--start-from", "zakura_udon::field::pasta::algorithms::tonelli_shanks_with_roots",
        "--start-from", "zakura_udon::field::pasta::algorithms::tonelli_shanks_alt_with_roots",
        *checked, "--dest-file", str(algorithms), "--", *cargo], repo)
    if json.loads(algorithms.read_text())["has_errors"]:
        raise RuntimeError("Charon exported a partial square-root algorithm model")
    run("sqrt-algorithms-aeneas", [str(aeneas), "-backend", "lean", "-namespace", "SqrtAlgorithms",
        "-subdir", "SqrtAlgorithms", "-split-files", "-filter-trait-methods", "-no-progress-bar",
        "-abort-on-error", "-dest", str(output / "lean"), str(algorithms)], output)
    native = output / "native"
    (native / "Cargo.toml").write_text(
        '[package]\nname = "udon-native-field-proof"\nversion = "0.0.0"\n'
        'edition = "2024"\n\n[workspace]\n\n[dependencies]\n'
        'udon = { package = "zakura-udon", path = ' +
        json.dumps(str(repo / "crates/udon")) + ' }\n')
    shutil.copyfile(HERE / "native_wrapper.rs", native / "src/lib.rs")
    shutil.copyfile(repo / "Cargo.lock", native / "Cargo.lock")
    run("native-lock", ["cargo", "+" + PIN["rust_toolchain"], "metadata",
                         "--format-version=1", "--offline"], native)
    check_native_lock(repo, native)
    raw = output / "llbc/native-raw.llbc"
    run("native-charon", [str(charon), "cargo", "--preset=aeneas", "--sysroot=default",
        "--consts=values", "--remove-adt-clauses", "--lift-associated-types=*",
        "--remove-unused-clauses", "--include", "zakura_udon",
        "--include", "core::cmp::*::is_lt", "--include", "core::num::*::unsigned_abs",
        "--include", "core::num::*::wrapping_neg", "--include", "core::num::*::wrapping_abs",
        "--include", "core::num::*::is_negative",
        "--include", "zakura_bento_core::addchain", "--include", "core::mem::drop",
        "--start-from", "udon_native_field_proof",
        "--start-from", "zakura_udon::field::pasta::PastaField::half",
        "--no-dedup-serialized-ast", *checked, "--dest-file", str(raw), "--", "--lib",
        "--locked", *( ["--offline"] if args.offline else [])], native)
    data = json.loads(raw.read_text())
    if data["has_errors"]:
        raise RuntimeError("Charon exported a partial native model")
    check_ordering(data)
    previous = raw
    for name in ["normalize_constants", "project_parameters", "mark_constant_effects",
                 "select_private_roots"]:
        adapted = output / "llbc" / (name + ".llbc")
        flags = ["--retain-callback-functions"] if name == "project_parameters" else []
        run(name, ["python3", str(HERE / "adapters" / (name + ".py")),
                   str(previous), str(adapted), *flags], output)
        previous = adapted
    run("native-aeneas", [str(aeneas), "-backend", "lean", "-namespace", "NativeField",
        "-subdir", "NativeField", "-filter-trait-methods", "-split-files", "-no-progress-bar",
        "-abort-on-error", "-dest", str(output / "lean"), str(previous)], output)
    lean = output / "lean"
    native_funs = lean / "NativeField/Funs.lean"
    generated = native_funs.read_text()
    anchor = "public import Aeneas\n"
    if generated.count(anchor) != 1:
        raise RuntimeError("Could not locate the native model imports")
    native_funs.write_text(generated.replace(anchor, anchor + "public import NativeField.Prelude\n"))
    for name in ["Prelude.lean", "FunsExternal.lean"]:
        shutil.copyfile(HERE / "native_support" / name, lean / "NativeField" / name)
    (lean / "NativeField/FunsExternal_Template.lean").unlink(missing_ok=True)
    sqrt = output / "sqrt"
    (sqrt / "Cargo.toml").write_text(
        '[package]\nname = "udon-sqrt-proof"\nversion = "0.0.0"\n'
        'edition = "2024"\n\n[workspace]\n\n[dependencies]\n'
        'udon = { package = "zakura-udon", path = ' +
        json.dumps(str(repo / "crates/udon")) + ' }\n')
    shutil.copyfile(HERE / "sqrt_wrapper.rs", sqrt / "src/lib.rs")
    shutil.copyfile(repo / "Cargo.lock", sqrt / "Cargo.lock")
    run("sqrt-lock", ["cargo", "+" + PIN["rust_toolchain"], "metadata",
                      "--format-version=1", "--offline"], sqrt)
    check_native_lock(repo, sqrt, "udon-sqrt-proof")
    sqrt_raw = output / "llbc/sqrt-raw.llbc"
    run("sqrt-charon", [str(sqrt_charon), "cargo", "--preset=aeneas", "--sysroot=default",
        "--consts=values", "--remove-adt-clauses", "--lift-associated-types=*",
        "--remove-unused-clauses", "--include", "zakura_udon",
        "--include", "core::cmp::*::is_lt", "--include", "core::num::*::unsigned_abs",
        "--include", "core::num::*::wrapping_neg", "--include", "core::num::*::wrapping_abs",
        "--include", "core::num::*::is_negative",
        "--opaque", "zakura_udon::field::pasta::parameters::{impl zakura_udon::field::pasta::parameters::sealed::Parameters<_> for _}::pow_sqrt_exponent",
        "--start-from", "udon_sqrt_proof", "--no-dedup-serialized-ast", *checked,
        "--dest-file", str(sqrt_raw), "--", "--lib", "--locked",
        *(["--offline"] if args.offline else [])], sqrt)
    sqrt_data = json.loads(sqrt_raw.read_text())
    if sqrt_data["has_errors"]:
        raise RuntimeError("Charon exported a partial concrete square-root model")
    check_ordering(sqrt_data)
    previous = sqrt_raw
    for name in ["normalize_constants", "mark_constant_effects", "project_unused_parent"]:
        adapted = output / "llbc" / ("sqrt-" + name + ".llbc")
        run("sqrt-" + name, ["python3", str(HERE / "adapters" / (name + ".py")),
                            str(previous), str(adapted)], output)
        previous = adapted
    run("sqrt-aeneas", [str(aeneas), "-backend", "lean", "-namespace", "SqrtNative",
        "-subdir", "SqrtNative", "-filter-trait-methods", "-split-files", "-no-progress-bar",
        "-abort-on-error", "-dest", str(lean), str(previous)], output)
    sqrt_funs = lean / "SqrtNative/Funs.lean"
    generated = sqrt_funs.read_text()
    if generated.count(anchor) != 1:
        raise RuntimeError("Could not locate the concrete square-root model imports")
    sqrt_funs.write_text(generated.replace(anchor, anchor + "public import NativeField.Prelude\n"))
    external = lean / "SqrtNative/FunsExternal_Template.lean"
    declarations = set(re.findall(r"\baxiom\s+([A-Za-z0-9_.]+)", external.read_text()))
    expected_external = {
        "core.marker.PhantomData.Insts.CoreCloneClone.clone", "core.option.Option.map",
        "Pair.Insts.CoreCmpPartialEqPair.eq",
        "zakura_udon.field.pasta.parameters.PallasBase.Insts.Zakura_udonFieldPastaParametersSealedParametersPallasBase.pow_sqrt_exponent",
        "zakura_udon.field.pasta.parameters.PallasScalar.Insts.Zakura_udonFieldPastaParametersSealedParametersPallasScalar.pow_sqrt_exponent",
    }
    if declarations != expected_external:
        raise RuntimeError("Unexpected concrete square-root external declarations: " + repr(declarations))
    shutil.copyfile(HERE / "sqrt_support/FunsExternal.lean", lean / "SqrtNative/FunsExternal.lean")
    external.unlink()
    shutil.copyfile(backend / "lean-toolchain", lean / "lean-toolchain")
    for module in CATALOG["modules"]:
        shutil.copyfile(HERE / "proofs" / (module + ".lean"), lean / (module + ".lean"))
    configuration = ["import Lake", "open Lake DSL",
                     "require aeneas from " + json.dumps(str(backend)),
                     "package udonFeasibility"]
    for name in ["Word", "KernelSlice", "SqrtAlgorithms", *CATALOG["modules"]]:
        configuration.append("@[default_target] lean_lib " + name)
    (lean / "lakefile.lean").write_text("\n".join(configuration) + "\n")
    for name in ["Word", "KernelSlice", "SqrtAlgorithms"]:
        (lean / (name + ".lean")).write_text("import " + name + ".Funs\n")
    (lean / ".lake").mkdir()
    cached_packages = backend / ".lake/packages"
    if cached_packages.is_dir():
        (lean / ".lake/packages").symlink_to(cached_packages, target_is_directory=True)
    source_files = [file for file in lean.rglob("*.lean") if ".lake" not in file.parts]
    for file in source_files:
        permitted = file == lean / "NativeField/FunsExternal.lean" and \
            file.read_bytes() == (HERE / "native_support/FunsExternal.lean").read_bytes()
        permitted |= file == lean / "SqrtNative/FunsExternal.lean" and \
            file.read_bytes() == (HERE / "sqrt_support/FunsExternal.lean").read_bytes()
        if ("External" in file.name and not permitted) or re.search(
                r"\b(sorry|admit|axiom|opaque)\b", file.read_text()):
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
        pattern = re.escape("'" + theorem + "' depends on axioms:") + r"\s*\[([^]]*)\]"
        match = re.search(pattern, proof_log)
        if not match:
            raise RuntimeError("Missing proof trust census for " + theorem)
        actual = {item.strip() for item in match.group(1).split(",") if item.strip()}
        allowed = {"propext", "Classical.choice", "Quot.sound"}
        allowed.update(CATALOG.get("type_only_axioms", {}).get(theorem, []))
        if not actual <= allowed:
            raise RuntimeError(f"Unexpected proof trust census for {theorem}: {actual}")
    summary = {"udon_revision": PIN["udon_revision"], "aeneas_version": aeneas_version,
               "charon_version": charon_version, "lean_toolchain": lean_toolchain,
               "sqrt_charon_version": sqrt_charon_version, "sqrt_charon_build": sqrt_manifest,
               "source_hashes": PIN["source_hashes"], "stages": results,
               "checkout_revision": head, "proved": CATALOG["proved"],
               "scope": CATALOG["scope"]}
    (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Passed: extraction, Lean typechecking, {len(CATALOG['proved'])} proofs, and proof trust census", flush=True)


if __name__ == "__main__":
    main()

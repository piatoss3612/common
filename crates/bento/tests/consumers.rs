//! Cargo consumers check dependency paths, const evaluation, and diagnostics.
//!
//! Tests inside the facade already see its direct core dependency, so
//! separate consumer manifests exercise the dependencies available to callers.

use std::{fs, path::Path};

mod support;
use support::{cargo, diagnostics};

#[test]
fn downstream_paths_no_std_doctests_and_diagnostics() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..").canonicalize().unwrap();
    let temporary = support::workspace("bento-macros-");
    let root = temporary.path();
    let fixtures = manifest.join("tests/fixtures");
    let facade = workspace.join("crates/bento");
    let core = workspace.join("crates/bento-core");
    let macros = workspace.join("crates/bento-macros");
    fs::write(
        root.join("Cargo.toml"),
        format!(
            r#"[workspace]
members = ["facade-default", "facade-renamed", "direct-core", "reexport", "missing-support", "inactive-optional", "inactive-dev", "inactive-target", "build-only"]
resolver = "3"

[workspace.dependencies]
support = {{ package = "zakura-bento", path = {facade:?} }}
support-core = {{ package = "zakura-bento-core", path = {core:?} }}
macros = {{ package = "zakura-bento-macros", path = {macros:?} }}
"#
        ),
    )
    .unwrap();
    fs::copy(workspace.join("Cargo.lock"), root.join("Cargo.lock")).unwrap();

    for (name, source, dependencies, features) in [
        (
            "facade-default",
            "consumers/facade.rs",
            format!("zakura-bento = {{ path = {facade:?} }}"),
            "renamed = []",
        ),
        (
            "facade-renamed",
            "consumers/facade.rs",
            "support.workspace = true".into(),
            "default = [\"renamed\"]\nrenamed = []",
        ),
        (
            "direct-core",
            "consumers/direct_core.rs",
            "support-core.workspace = true\nmacros.workspace = true\nsupport = { workspace = true, optional = true }".into(),
            "with-facade = [\"dep:support\"]",
        ),
        (
            "reexport",
            "pod/reexport.rs",
            "bridge = { package = \"facade-default\", path = \"../facade-default\" }".into(),
            "",
        ),
        (
            "missing-support",
            "consumers/missing_support.rs",
            "macros.workspace = true".into(),
            "",
        ),
    ] {
        let package = root.join(name);
        fs::create_dir_all(package.join("src")).unwrap();
        fs::write(
            package.join("Cargo.toml"),
            format!(
                "[package]\nname = {name:?}\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\nautobins = false\nautoexamples = false\n\n[dependencies]\n{dependencies}\n\n[features]\n{features}\n"
            ),
        )
        .unwrap();
        fs::copy(fixtures.join(source), package.join("src/lib.rs")).unwrap();
        fs::copy(fixtures.join("pod/consumer.rs"), package.join("src/pod.rs")).unwrap();
        fs::write(package.join("src/record.bin"), [1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    }

    for (name, extra) in [
        (
            "inactive-optional",
            "[dependencies.support-core]\nworkspace = true\noptional = true",
        ),
        (
            "inactive-dev",
            "[dev-dependencies]\nsupport-core.workspace = true",
        ),
        (
            "inactive-target",
            "[target.'cfg(any())'.dependencies]\nsupport-core.workspace = true",
        ),
        ("build-only", ""),
    ] {
        let package = root.join(name);
        fs::create_dir_all(package.join("src")).unwrap();
        let table = if name == "build-only" {
            "build-dependencies"
        } else {
            "dependencies"
        };
        fs::write(package.join("Cargo.toml"), format!(
            "[package]\nname = {name:?}\nversion = \"0.0.0\"\nedition = \"2024\"\n[{table}]\nsupport.workspace = true\n{extra}\n"
        )).unwrap();
        if name == "build-only" {
            fs::write(package.join("src/lib.rs"), "#![no_std]\n").unwrap();
            fs::copy(
                fixtures.join("consumers/build_only.rs"),
                package.join("build.rs"),
            )
            .unwrap();
        } else {
            fs::copy(
                fixtures.join("consumers/facade_usage.rs"),
                package.join("src/lib.rs"),
            )
            .unwrap();
        }
    }

    // Build library targets before tests can activate their dev dependencies.
    let output = cargo(
        root,
        &[
            "build",
            "--release",
            "--workspace",
            "--exclude",
            "missing-support",
        ],
    );
    assert!(output.status.success(), "{}", diagnostics(&output));

    let output = cargo(
        root,
        &[
            "test",
            "--workspace",
            "--exclude",
            "missing-support",
            "--release",
        ],
    );
    assert!(output.status.success(), "{}", diagnostics(&output));
    let output = cargo(
        root,
        &[
            "test",
            "--release",
            "-p",
            "direct-core",
            "--features",
            "with-facade",
        ],
    );
    assert!(output.status.success(), "{}", diagnostics(&output));

    let output = cargo(root, &["check", "-p", "missing-support"]);
    let diagnostic = diagnostics(&output);
    assert!(
        !output.status.success(),
        "missing-support unexpectedly compiled"
    );
    assert!(
        diagnostic.contains(
            "cannot discover zakura-bento; use #[pod(crate = path)] to name the support path"
        ),
        "{diagnostic}"
    );
    assert!(
        diagnostic.contains("missing-support/src/lib.rs:"),
        "{diagnostic}"
    );

    // Only register failing binaries after the successful consumer build.
    // Check stable messages and source locations, not whole rustc renderings.
    let package = root.join("facade-default");
    fs::create_dir_all(package.join("src/bin")).unwrap();
    let manifest_path = package.join("Cargo.toml");
    let mut manifest_text = fs::read_to_string(&manifest_path).unwrap();
    for (fixture, expected) in [
        (
            "addition_chain/zero",
            "addition_chain! scalar must be nonzero; the trait has no identity operation",
        ),
        (
            "addition_chain/suffix",
            "addition_chain! scalar must be an unsuffixed integer literal",
        ),
        ("addition_chain/constant", "expected integer literal"),
        (
            "addition_chain/negative",
            "addition_chain! scalar must be positive",
        ),
        (
            "addition_chain/forwarded_negative",
            "addition_chain! scalar must be positive",
        ),
        ("addition_chain/missing_trait", "Value: AdditionChain"),
        ("addition_chain/missing_clone", "Value: Clone"),
        ("addition_chain/moved_value", "use of moved value: `value`"),
        ("const_arithmetic/invalid_modulus", "modulus must be odd"),
        (
            "const_arithmetic/invalid_two_adicity",
            "two_adicity exceeds the trailing zeros",
        ),
        ("const_arithmetic/unreduced_base", "base must be reduced"),
        (
            "const_arithmetic/ratio_overflow",
            "quotient exceeds five limbs",
        ),
    ] {
        let name = Path::new(fixture).file_name().unwrap().to_str().unwrap();
        fs::copy(
            fixtures.join(format!("{fixture}.rs")),
            package.join(format!("src/bin/{name}.rs")),
        )
        .unwrap();
        manifest_text.push_str(&format!("\n[[bin]]\nname = {name:?}\n"));
        fs::write(&manifest_path, &manifest_text).unwrap();
        let output = cargo(
            root,
            &["build", "--release", "-p", "facade-default", "--bin", name],
        );
        let diagnostic = diagnostics(&output);
        assert!(!output.status.success(), "{name} unexpectedly compiled");
        assert!(diagnostic.contains(expected), "{name}: {diagnostic}");
        assert!(
            diagnostic.contains(&format!("src/bin/{name}.rs:")),
            "{diagnostic}"
        );
    }
}

//! Cargo consumers check dependency aliases, support paths, and re-exports.
//!
//! Tests inside the facade already see its direct core dependency, so
//! separate consumer manifests exercise the dependencies available to callers.

use std::{fs, path::Path};

use crate::harness::{self, cargo, diagnostics};

#[test]
fn downstream_dependency_paths_and_doctests() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..").canonicalize().unwrap();
    let temporary = harness::workspace("bento-dependencies-");
    let root = temporary.path();
    let fixtures = manifest.join("tests/api/fixtures/dependencies");
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
            "facade.rs",
            format!("zakura-bento = {{ path = {facade:?} }}"),
            "renamed = []",
        ),
        (
            "facade-renamed",
            "facade.rs",
            "support.workspace = true".into(),
            "default = [\"renamed\"]\nrenamed = []",
        ),
        (
            "direct-core",
            "direct_core.rs",
            "support-core.workspace = true\nmacros.workspace = true\nsupport = { workspace = true, optional = true }".into(),
            "with-facade = [\"dep:support\"]",
        ),
        (
            "reexport",
            "reexport.rs",
            "bridge = { package = \"facade-default\", path = \"../facade-default\" }".into(),
            "",
        ),
        (
            "missing-support",
            "missing_support.rs",
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
        fs::copy(
            fixtures.join("arithmetic.rs"),
            package.join("src/arithmetic.rs"),
        )
        .unwrap();
        fs::copy(fixtures.join("pod.rs"), package.join("src/pod.rs")).unwrap();
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
            fs::copy(fixtures.join("build_only.rs"), package.join("build.rs")).unwrap();
        } else {
            fs::copy(fixtures.join("facade_usage.rs"), package.join("src/lib.rs")).unwrap();
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

    let output = cargo(root, &["check", "--release", "-p", "missing-support"]);
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
}

//! Cargo consumer tests for dependency path resolution and diagnostics.
//!
//! Tests inside the macro crate already see its direct core dependency, so
//! separate consumer manifests are needed to prove the facade fallback works.
//! Keep these builds offline and seed their resolution from the workspace lock;
//! the parent workspace build fetches the dependencies the consumers need.

use std::{fs, path::Path};

mod support;
use support::{cargo, diagnostics};

#[test]
fn downstream_paths_no_std_doctests_and_diagnostics() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..").canonicalize().unwrap();
    let root = workspace.join("target/macro-consumers");
    let fixtures = manifest.join("tests/fixtures");
    let facade = workspace.join("crates/bento");
    let core = workspace.join("crates/bento-core");
    let macros = workspace.join("crates/bento-macros");
    fs::create_dir_all(&root).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        format!(
            r#"[workspace]
members = ["facade-default", "facade-renamed", "direct-core", "reexport", "missing-support"]
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
                "[package]\nname = {name:?}\nversion = \"0.0.0\"\nedition = \"2024\"\npublish = false\nautoexamples = false\n\n[dependencies]\n{dependencies}\n\n[features]\n{features}\n"
            ),
        )
        .unwrap();
        fs::copy(fixtures.join(source), package.join("src/lib.rs")).unwrap();
        fs::copy(fixtures.join("pod/consumer.rs"), package.join("src/pod.rs")).unwrap();
        fs::write(package.join("src/record.bin"), [1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
    }

    let output = cargo(
        &root,
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
        &root,
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

    let output = cargo(&root, &["check", "-p", "missing-support"]);
    let diagnostic = diagnostics(&output);
    assert!(
        !output.status.success(),
        "missing-support unexpectedly compiled"
    );
    assert!(
        diagnostic.contains(
            "failed to find zakura-bento or zakura-bento-core; add zakura-bento to your Cargo.toml dependencies"
        ),
        "{diagnostic}"
    );
    assert!(
        diagnostic.contains("missing-support/src/lib.rs:"),
        "{diagnostic}"
    );

    // Only register failing examples after the successful consumer build.
    // Check stable messages and source locations, not whole rustc renderings.
    let package = root.join("facade-default");
    fs::create_dir_all(package.join("examples")).unwrap();
    let manifest_path = package.join("Cargo.toml");
    let mut manifest_text = fs::read_to_string(&manifest_path).unwrap();
    for (name, expected) in [
        (
            "zero",
            "addition_chain! scalar must be nonzero; the trait has no identity operation",
        ),
        (
            "suffix",
            "addition_chain! scalar must be an unsuffixed integer literal",
        ),
        ("constant", "expected integer literal"),
        ("negative", "addition_chain! scalar must be positive"),
        (
            "forwarded_negative",
            "addition_chain! scalar must be positive",
        ),
        ("missing_trait", "Value: AdditionChain"),
        ("missing_clone", "Value: Clone"),
        ("moved_value", "use of moved value: `value`"),
    ] {
        fs::copy(
            fixtures.join(format!("addition_chain/{name}.rs")),
            package.join(format!("examples/{name}.rs")),
        )
        .unwrap();
        manifest_text.push_str(&format!("\n[[example]]\nname = {name:?}\n"));
        fs::write(&manifest_path, &manifest_text).unwrap();
        let output = cargo(&root, &["check", "-p", "facade-default", "--example", name]);
        let diagnostic = diagnostics(&output);
        assert!(!output.status.success(), "{name} unexpectedly compiled");
        assert!(diagnostic.contains(expected), "{name}: {diagnostic}");
        assert!(
            diagnostic.contains(&format!("examples/{name}.rs:")),
            "{diagnostic}"
        );
    }
}

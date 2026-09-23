//! Full-build rejection checks for fixture programs in a single consumer.

use std::{fs, path::Path};

use crate::harness::{cargo, diagnostics, workspace};

pub fn check_rejections(package: &str, fixtures: &Path, cases: &[(&str, &str)]) {
    let facade = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository = facade.join("../..").canonicalize().unwrap();
    let temporary = workspace(&format!("bento-{package}-"));
    let root = temporary.path();
    fs::create_dir_all(root.join("src/bin")).unwrap();
    let manifest_path = root.join("Cargo.toml");
    let mut manifest = format!(
        r#"[package]
name = {package:?}
version = "0.0.0"
edition = "2024"
publish = false
autobins = false

[workspace]

[dependencies]
zakura-bento = {{ path = {facade:?} }}
"#
    );
    fs::copy(repository.join("Cargo.lock"), root.join("Cargo.lock")).unwrap();
    if fixtures.join("lib.rs").is_file() {
        fs::copy(fixtures.join("lib.rs"), root.join("src/lib.rs")).unwrap();
    }

    // Reuse this workspace for the matrix, targeting one binary at a time so
    // earlier rejection cases do not interfere with later builds.
    for &(name, expected) in cases {
        let source = fs::read_to_string(fixtures.join(format!("{name}.rs"))).unwrap();
        fs::write(root.join(format!("src/bin/{name}.rs")), &source).unwrap();
        manifest.push_str(&format!("\n[[bin]]\nname = {name:?}\n"));
        fs::write(&manifest_path, &manifest).unwrap();
        let output = cargo(root, &["build", "--release", "--bin", name]);
        let diagnostic = diagnostics(&output);
        assert!(!output.status.success(), "{name} unexpectedly compiled");
        assert!(diagnostic.contains(expected), "{name}: {diagnostic}");
        assert!(
            diagnostic.contains(&format!("src/bin/{name}.rs:")),
            "{diagnostic}"
        );
        for (line, _) in source
            .lines()
            .enumerate()
            .filter(|(_, line)| line.ends_with("// rejected"))
        {
            assert!(
                diagnostic.contains(&format!("src/bin/{name}.rs:{}:", line + 1)),
                "missing diagnostic for line {}: {diagnostic}",
                line + 1,
            );
        }
    }
}

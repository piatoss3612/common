//! A downstream build script generates field records for typed embedding.

use std::{collections::BTreeMap, fs, path::Path, process::Command};

#[test]
fn generated_fields_embed_in_a_downstream_consumer() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..").canonicalize().unwrap();
    let temporary = tempfile::Builder::new()
        .prefix("udon-embedding-")
        .tempdir()
        .unwrap();
    let directory = temporary.path();
    let bento = workspace.join("crates/bento");
    let udon = workspace.join("crates/udon");
    fs::create_dir(directory.join("src")).unwrap();
    fs::write(
        directory.join("Cargo.toml"),
        format!(
            r#"[package]
name = "field-embedding-consumer"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[features]
sqrt-table-large = ["udon/sqrt-table-large"]

[dependencies]
bento = {{ package = "zakura-bento", path = {bento:?} }}
udon = {{ package = "zakura-udon", path = {udon:?} }}

[build-dependencies]
bento = {{ package = "zakura-bento", path = {bento:?} }}
udon = {{ package = "zakura-udon", path = {udon:?} }}
"#
        ),
    )
    .unwrap();
    fs::copy(workspace.join("Cargo.lock"), directory.join("Cargo.lock")).unwrap();
    for path in ["build.rs", "src/record.rs", "src/main.rs"] {
        fs::copy(
            manifest.join("tests/fixtures/embedding").join(path),
            directory.join(path),
        )
        .unwrap();
    }
    // Isolate the nested build from the parent Cargo lock and concurrent tests.
    // The square-root configuration must not change the stored representation,
    // so both feature sets must generate byte-identical artifacts.
    let mut stored = None;
    for features in ["", "sqrt-table-large"] {
        let output = Command::new(env!("CARGO"))
            .current_dir(directory)
            .args([
                "run",
                "--release",
                "--quiet",
                "--offline",
                "--features",
                features,
            ])
            .env("CARGO_TARGET_DIR", directory.join("target"))
            .env("CARGO_TERM_COLOR", "never")
            .output()
            .expect("run the field embedding consumer");
        assert!(
            output.status.success(),
            "consumer with features {features:?} failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let artifacts = generated_artifacts(&directory.join("target"));
        assert_eq!(
            artifacts.keys().map(String::as_str).collect::<Vec<_>>(),
            ["field-values-mont-u64x4.bin", "fp-values-mont-u64x4.bin"]
        );
        match &stored {
            None => stored = Some(artifacts),
            Some(first) => assert_eq!(
                first, &artifacts,
                "stored bytes must not depend on the square-root configuration"
            ),
        }
    }
}

// Collects the generated files across the consumer's build script out
// directories. Cargo may keep one directory per feature set; entries sharing a
// filename must already agree before the configurations are compared.
fn generated_artifacts(target: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut artifacts = BTreeMap::new();
    let build = target.join("release/build");
    for entry in fs::read_dir(&build).expect("read the nested build directory") {
        let path = entry.unwrap().path();
        let out = path.join("out");
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !name.starts_with("field-embedding-consumer-") || !out.is_dir() {
            continue;
        }
        for file in fs::read_dir(&out).unwrap() {
            let file = file.unwrap().path();
            let name = file.file_name().unwrap().to_string_lossy().into_owned();
            let bytes = fs::read(&file).unwrap();
            if let Some(previous) = artifacts.insert(name.clone(), bytes) {
                assert_eq!(
                    previous, artifacts[&name],
                    "artifact {name} differs between out directories"
                );
            }
        }
    }
    artifacts
}

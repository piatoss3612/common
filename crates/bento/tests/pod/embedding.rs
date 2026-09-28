//! Generates artifact files, then compiles and runs a consumer that embeds them.
//!
//! The build script and consumer share the record definition, as they would in
//! an artifact-owning crate. Cargo runs the generator as a build script before
//! compiling the consumer's file inclusions.

use std::{fs, path::Path};

use crate::harness::{self, cargo, diagnostics};

#[test]
fn generated_records_round_trip_through_file_embedding() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..").canonicalize().unwrap();
    let temporary = harness::workspace("bento-embedding-");
    let directory = temporary.path();
    let fixtures = manifest.join("tests/pod/fixtures/embedding");
    let facade = workspace.join("crates/bento");
    fs::create_dir_all(directory.join("src")).unwrap();
    fs::write(
        directory.join("Cargo.toml"),
        format!(
            r#"[package]
name = "embedding-consumer"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
bento = {{ package = "zakura-bento", path = {facade:?} }}

[build-dependencies]
bento = {{ package = "zakura-bento", path = {facade:?} }}
"#
        ),
    )
    .unwrap();
    fs::copy(workspace.join("Cargo.lock"), directory.join("Cargo.lock")).unwrap();
    for path in ["build.rs", "src/record.rs", "src/main.rs"] {
        fs::copy(fixtures.join(path), directory.join(path)).unwrap();
    }

    let output = cargo(directory, &["run", "--release", "--quiet"]);
    assert!(
        output.status.success(),
        "consumer failed:\n{}",
        diagnostics(&output),
    );
}

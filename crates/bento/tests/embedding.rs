//! Generates artifact files, then compiles and runs a consumer that embeds them.
//!
//! The build script and consumer share the record definition, as they would in
//! an artifact-owning crate. Separate Cargo builds let the generator run before
//! the consumer expands its file inclusions.

use std::{fs, path::Path};

mod support;
use support::{cargo, diagnostics};

#[test]
fn generated_records_round_trip_through_file_embedding() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..").canonicalize().unwrap();
    let temporary = support::workspace("bento-embedding-");
    let directory = temporary.path();
    let fixtures = manifest.join("tests/fixtures/pod/embedding");
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

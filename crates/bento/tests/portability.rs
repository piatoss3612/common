//! Full library builds check storage and arithmetic without a target linker.

use std::{fs, path::Path};

mod support;
use support::{cargo, diagnostics};

#[test]
#[ignore = "requires thumbv7em-none-eabi and s390x-unknown-linux-gnu target libraries"]
fn target_portability_contracts() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository = manifest.join("../..").canonicalize().unwrap();
    let temporary = support::workspace("bento-portability-");
    let root = temporary.path();
    fs::create_dir(root.join("src")).unwrap();
    let facade = repository.join("crates/bento");
    let udon = repository.join("crates/udon");
    fs::write(
        root.join("Cargo.toml"),
        format!(
            r#"[package]
name = "portability-consumer"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
bento = {{ package = "zakura-bento", path = {facade:?} }}
udon = {{ package = "zakura-udon", path = {udon:?} }}
[features]
record = []
empty-record = []
primitive = []
zero-array = []
"#
        ),
    )
    .unwrap();
    fs::copy(repository.join("Cargo.lock"), root.join("Cargo.lock")).unwrap();
    fs::copy(
        manifest.join("tests/fixtures/consumers/portability.rs"),
        root.join("src/lib.rs"),
    )
    .unwrap();
    fs::write(root.join("src/record.bin"), [0; 8]).unwrap();

    for (target, features) in [
        (
            "thumbv7em-none-eabi",
            "record,empty-record,primitive,zero-array",
        ),
        ("s390x-unknown-linux-gnu", ""),
    ] {
        let output = cargo(
            root,
            &[
                "build",
                "--release",
                "--lib",
                "--target",
                target,
                "--features",
                features,
            ],
        );
        assert!(
            output.status.success(),
            "{target}: {}",
            diagnostics(&output)
        );
    }

    for feature in ["record", "empty-record", "primitive", "zero-array"] {
        let output = cargo(
            root,
            &[
                "build",
                "--release",
                "--lib",
                "--target",
                "s390x-unknown-linux-gnu",
                "--features",
                feature,
            ],
        );
        let diagnostic = diagnostics(&output);
        assert!(
            !output.status.success(),
            "big-endian {feature} unexpectedly compiled"
        );
        assert!(
            diagnostic.contains("embedded static data requires little-endian in-memory layout"),
            "{feature}: {diagnostic}"
        );
    }
}

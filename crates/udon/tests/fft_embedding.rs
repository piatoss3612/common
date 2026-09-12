//! A downstream owner prepares FFT tables and embeds them in a `no_std` library.

use std::{fs, path::Path, process::Command};

#[test]
fn generated_fft_tables_embed_in_a_downstream_consumer() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..").canonicalize().unwrap();
    let temporary = tempfile::Builder::new()
        .prefix("udon-fft-embedding-")
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
name = "fft-embedding-consumer"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[features]
alloc = ["udon/alloc"]
sqrt-table-large = ["udon/sqrt-table-large"]

[dependencies]
bento = {{ package = "zakura-bento", path = {bento:?} }}
udon = {{ package = "zakura-udon", path = {udon:?}, default-features = false }}

[build-dependencies]
bento = {{ package = "zakura-bento", path = {bento:?} }}
udon = {{ package = "zakura-udon", path = {udon:?}, default-features = false }}
"#
        ),
    )
    .unwrap();
    fs::copy(workspace.join("Cargo.lock"), directory.join("Cargo.lock")).unwrap();
    for path in ["build.rs", "src/record.rs", "src/lib.rs", "src/main.rs"] {
        fs::copy(
            manifest.join("tests/fixtures/fft_embedding").join(path),
            directory.join(path),
        )
        .unwrap();
    }
    for features in ["", "alloc,sqrt-table-large"] {
        for (damage, diagnostic) in [
            ("", None),
            (
                "permutation",
                Some("embedded FFT tables must match the domain: InvalidTables"),
            ),
            (
                "field",
                Some("embedded FFT tables must match the domain: InvalidTables"),
            ),
            (
                "scales",
                Some("embedded residue scales must match the domain: InvalidTables"),
            ),
            (
                "metadata",
                Some("embedded metadata must match the domain: InvalidTables"),
            ),
            (
                "factored",
                Some("embedded factored twiddles must match the domain: InvalidTables"),
            ),
            (
                "truncate",
                Some("embedded byte length must equal the requested type's size"),
            ),
        ] {
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
                .env("FFT_ARTIFACT_DAMAGE", damage)
                .output()
                .expect("run the FFT embedding consumer");
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(diagnostic) = diagnostic {
                assert!(
                    !output.status.success(),
                    "damaged artifact {damage} was accepted"
                );
                assert!(
                    stderr.contains(diagnostic),
                    "wrong rejection for {damage} with features {features:?}:\n{stdout}{stderr}"
                );
                assert!(
                    stderr.contains("src/lib.rs"),
                    "rejection must come from the embedding consumer:\n{stderr}"
                );
            } else {
                assert!(
                    output.status.success(),
                    "consumer with features {features:?} failed:\n{stdout}{stderr}"
                );
            }
        }
    }
}

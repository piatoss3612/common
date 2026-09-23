//! Nested Cargo execution and temporary workspaces shared by Bento consumers.

use std::{
    path::Path,
    process::{Command, Output},
};

pub fn cargo(root: &Path, args: &[&str]) -> Output {
    // A separate target directory avoids locking the parent Cargo build.
    Command::new(env!("CARGO"))
        .current_dir(root)
        .args(args)
        .arg("--offline")
        .env("CARGO_TARGET_DIR", root.join("target"))
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .expect("run Cargo for consumer fixture")
}

pub fn diagnostics(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Creates a unique fixture workspace that is removed when the handle is dropped.
///
/// Keep the handle alive until all nested Cargo processes have finished.
pub fn workspace(prefix: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(prefix)
        .tempdir()
        .expect("create consumer workspace")
}

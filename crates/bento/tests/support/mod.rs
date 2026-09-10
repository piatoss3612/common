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

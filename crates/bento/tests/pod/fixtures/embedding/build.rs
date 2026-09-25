//! Writes artifacts before the consumer's embedding macros run.
#![forbid(unsafe_code)]

use std::{env, fs, path::PathBuf};

#[path = "src/record.rs"]
mod record;

fn main() {
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(
        directory.join("record.bin"),
        bento::bytes_of(&record::RECORD),
    )
    .unwrap();
    fs::write(
        directory.join("records.bin"),
        bento::bytes_of_slice(&record::RECORDS),
    )
    .unwrap();
}

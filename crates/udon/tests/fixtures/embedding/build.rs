//! Generates actual field values with the same Udon types the consumer uses.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::{env, fs, path::PathBuf};
use udon::{
    STORED_FORM,
    field::{Fp, Fq},
};

#[path = "src/record.rs"]
mod record;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/record.rs");
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let values = record::FieldValues {
        fp: [0, 1, 7, u64::MAX].map(|n| Fp::from_u64(n).square()),
        fq: [0, 1, 7, u64::MAX].map(|n| Fq::from_u64(n).square().add(&Fq::ONE)),
    };
    fs::write(
        directory.join(format!("field-values-{STORED_FORM}.bin")),
        bento::bytes_of(&values),
    )
    .unwrap();
    let values = [Fp::ZERO, Fp::ONE, Fp::from_u64(7), Fp::ONE.neg()];
    fs::write(
        directory.join(format!("fp-values-{STORED_FORM}.bin")),
        bento::bytes_of_slice(&values),
    )
    .unwrap();
}

//! Generates actual field values with the same Udon types the consumer uses.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::{env, fs, path::PathBuf};
use udon::{
    STORED_FORM,
    field::{Fp, PallasBase, PallasScalar},
};

#[path = "src/record.rs"]
mod record;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/record.rs");
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let fp = record::samples::<PallasBase>();
    let fq = record::samples::<PallasScalar>();
    let values = record::FieldValues {
        fp,
        fq,
        fp_reduced: fp.map(|value| value.reduce()),
        fq_reduced: fq.map(|value| value.reduce()),
    };
    fs::write(
        directory.join(format!("field-values-{STORED_FORM}.bin")),
        bento::bytes_of(&values),
    )
    .unwrap();
    let values = [<Fp>::ZERO, <Fp>::ONE, <Fp>::from_u64(7), <Fp>::ONE.neg()];
    fs::write(
        directory.join(format!("fp-values-{STORED_FORM}.bin")),
        bento::bytes_of_slice(&values),
    )
    .unwrap();
}

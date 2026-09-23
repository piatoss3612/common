//! Generates tables with Udon and stores the owner's records through Bento POD.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::{env, fs, path::PathBuf};
use udon::{
    STORED_FORM,
    fft::{Domain, ExpansionScaleNormalization, ExpansionScales, TwiddleTable},
};

#[path = "src/record.rs"]
pub mod record;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/record.rs");
    println!("cargo::rerun-if-env-changed=FFT_ARTIFACT_DAMAGE");
    let damage = env::var("FFT_ARTIFACT_DAMAGE").unwrap_or_default();
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    macro_rules! generate {
        ($name:literal, $record:ty) => {{
            let domain = Domain::for_size(record::SIZE).unwrap().subgroup();
            let extended = Domain::for_size(record::EXTENDED_SIZE).unwrap().coset();
            let mut record = <$record>::empty();
            record.destinations().prepare(domain);
            TwiddleTable::prepare(record::TWIDDLES, &mut record.packed).unwrap();
            ExpansionScales::prepare(
                record::SIZE,
                extended,
                ExpansionScaleNormalization::UnscaledInverse,
                &mut record.residues,
            )
            .unwrap();
            let mut bytes = bento::bytes_of(&record).to_vec();
            if $name == "fp-fft" && damage == "truncate" {
                bytes.pop();
            }
            fs::write(
                directory.join(format!("{}-{STORED_FORM}.bin", $name)),
                bytes,
            )
            .unwrap();
        }};
    }
    generate!("fp-fft", record::FpTables);
    generate!("fq-fft", record::FqTables);
}

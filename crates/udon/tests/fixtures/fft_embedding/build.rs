//! Generates tables with Udon and stores the owner's records through Bento POD.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::{env, fs, path::PathBuf};
use udon::{
    STORED_FORM,
    fft::{Domain, ExpansionScaleNormalization, ExpansionScales, TwiddleTable},
    field::{Fp, Fq},
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
        ($name:literal, $record:ty, $field:ty) => {{
            let domain = Domain::for_size(record::SIZE).unwrap().subgroup();
            let extended = Domain::for_size(record::EXTENDED_SIZE)
                .unwrap()
                .coset(<$field>::ZETA)
                .unwrap();
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
            record.header.validate(extended).unwrap();
            // Inject damage after generation so the consumer's import checks
            // must reject an otherwise valid artifact.
            if $name == "fp-fft" {
                match damage.as_str() {
                    "field" => record.forward[0] = *bento::AlignedBytes([0xff; 32]).as_value(),
                    "scales" => record.residues[1] = <$field>::ZERO,
                    "metadata" => record.header.normalization = 0,
                    "schema" => record.header.schema_version = 1,
                    "kind" => record.header.twiddle_kind = 1,
                    "packed" => record.packed[1] = <$field>::ZERO,
                    "" | "truncate" => {}
                    _ => panic!("unknown artifact damage"),
                }
            }
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
    generate!("fp-fft", record::FpTables, Fp);
    generate!("fq-fft", record::FqTables, Fq);
}

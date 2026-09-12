//! Generates both curves' expanded tables using the public runtime APIs.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::{env, fs, path::PathBuf};
use udon::{
    StoredForm,
    curve::{FixedBaseTable, Pallas, PastaCurve, ProjectivePoint, Vesta},
    field::PastaField,
};

#[path = "src/record.rs"]
pub mod record;

fn generate<C: PastaCurve>(name: &str, damage: &str) {
    let mut record = record::Record::<C>::empty();
    let mut projective = [ProjectivePoint::IDENTITY; record::REQUIREMENTS.projective_scratch];
    let mut field = [PastaField::ZERO; record::REQUIREMENTS.field_scratch];
    FixedBaseTable::prepare(
        record::DESCRIPTION,
        &record.base,
        &mut record.entries,
        &mut projective,
        &mut field,
    )
    .unwrap();
    // Damage is introduced after preparation, so rejection must come from
    // the consumer's layout and mathematical checks.
    match damage {
        "coordinate" => record.entries[1] = *bento::AlignedBytes([0xff; 64]).as_value(),
        "point" => record.entries[1] = *bento::AlignedBytes([0; 64]).as_value(),
        "order" => record.entries.swap(0, 1),
        "carry" => record.entries[record::REQUIREMENTS.affine_points - 1] = record.base,
        "base" => record.base = record.base.neg(),
        "schema" => record.schema = 2,
        "curve" => record.base_modulus = record.scalar_modulus,
        "window" => record.window_bits = 8,
        "" | "truncate" => {}
        _ => panic!("unknown artifact damage"),
    }
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let form = StoredForm::for_target(&env::var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap());
    let mut bytes = bento::bytes_of(&record).to_vec();
    if damage == "truncate" {
        bytes.pop();
    }
    fs::write(
        directory.join(format!("{name}-fixed-base-{}.bin", form.descriptor())),
        bytes,
    )
    .unwrap();
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/record.rs");
    println!("cargo::rerun-if-env-changed=CURVE_ARTIFACT_DAMAGE");
    let damage = env::var("CURVE_ARTIFACT_DAMAGE").unwrap_or_default();
    generate::<Pallas>("pallas", &damage);
    generate::<Vesta>("vesta", "");
}

//! Generates fixed-base tables and commitment bases through public APIs.
#![forbid(unsafe_code)]
#![deny(warnings)]

use std::{env, fs, path::PathBuf};
use udon::{
    STORED_FORM,
    curve::{
        EisensteinTable, FixedBaseTable, Pallas, PastaCurve, Point, PreparedAffinePoint,
        ProjectivePoint, Vesta, batch_normalize,
    },
    fft::reference,
    field::PastaField,
};

#[path = "src/record.rs"]
pub mod record;

fn generate<C: PastaCurve>(name: &str, truncate: bool) {
    let mut record = record::Record::<C>::empty();
    let mut projective = [ProjectivePoint::IDENTITY; record::REQUIREMENTS.table_entries];
    let mut field = [PastaField::ZERO; record::REQUIREMENTS.table_entries];
    FixedBaseTable::prepare_with(
        record::DESCRIPTION,
        &record.base,
        &mut record.entries,
        &mut projective,
        &mut field,
    )
    .unwrap();
    FixedBaseTable::prepare_with(
        record::DESCRIPTION,
        &record.base,
        &mut record.cached,
        &mut projective,
        &mut field,
    )
    .unwrap();
    EisensteinTable::prepare(
        &record.base,
        &mut record.compact,
        &mut projective,
        &mut field,
    );
    EisensteinTable::prepare(
        &record.base,
        &mut record.compact_cached,
        &mut projective,
        &mut field,
    );
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut bytes = bento::bytes_of(&record).to_vec();
    if truncate {
        bytes.pop();
    }
    fs::write(
        directory.join(format!("{name}-fixed-base-{STORED_FORM}.bin")),
        bytes,
    )
    .unwrap();
}

fn generate_srs<C: PastaCurve>(name: &str) {
    let mut record = record::SrsRecord::<C>::empty();
    let domain = record.domain();
    // Known generator multiples make this fixture reproducible. The scalar 7
    // provides no setup secrecy and is used solely for test data.
    let mut power = PastaField::ONE;
    let mut projective = core::array::from_fn::<_, { record::SRS_SIZE }, _>(|_| {
        let point = ProjectivePoint::<C>::GENERATOR.mul(&power);
        power = power.mul(&PastaField::<_>::from_u64(7));
        point
    });
    let mut points = [Point::IDENTITY; record::SRS_SIZE];
    let mut scratch = [PastaField::ZERO; record::SRS_SIZE];
    batch_normalize(&projective, &mut points, &mut scratch);
    record.coefficient =
        points.map(|point| PreparedAffinePoint::from_affine(point.as_affine().unwrap()));
    reference::inverse_transform(
        &mut projective,
        &domain.inverse_root(),
        &domain.size_inverse(),
    );
    batch_normalize(&projective, &mut points, &mut scratch);
    record.lagrange =
        points.map(|point| PreparedAffinePoint::from_affine(point.as_affine().unwrap()));
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    fs::write(
        directory.join(format!("{name}-srs-{STORED_FORM}.bin")),
        bento::bytes_of(&record),
    )
    .unwrap();
}

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/record.rs");
    println!("cargo::rerun-if-env-changed=CURVE_ARTIFACT_DAMAGE");
    let truncate = env::var("CURVE_ARTIFACT_DAMAGE").is_ok_and(|value| value == "truncate");
    generate::<Pallas>("pallas", truncate);
    generate::<Vesta>("vesta", false);
    generate_srs::<Pallas>("pallas");
    generate_srs::<Vesta>("vesta");
}

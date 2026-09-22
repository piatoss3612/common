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

fn generate<C: PastaCurve>(name: &str, damage: &str) {
    let mut record = record::Record::<C>::empty();
    let mut projective = [ProjectivePoint::IDENTITY; record::REQUIREMENTS.projective_scratch];
    let mut field = [PastaField::ZERO; record::REQUIREMENTS.field_scratch];
    FixedBaseTable::prepare(
        &record.base,
        &mut record.entries,
        &mut projective,
        &mut field,
    )
    .unwrap();
    FixedBaseTable::prepare(
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
    )
    .unwrap();
    EisensteinTable::prepare(
        &record.base,
        &mut record.compact_cached,
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
        "carry" => record.entries[record::REQUIREMENTS.table_entries - 1] = record.base,
        "base" => record.base = record.base.neg(),
        "schema" => record.schema = 1,
        "table-kind" => record.table_kinds[0] = 2,
        "entry-layout" => record.entry_bytes[0] = 96,
        "compact-order" => record.compact.swap(0, 1),
        "compact-cached-order" => record.compact_cached.swap(0, 1),
        "cache" | "compact-cache" => {
            let mut bytes = bento::bytes_of(&record.cached[0]).to_vec();
            bytes[32..64].fill(0xff);
            let raw = Box::leak(Box::new(bento::AlignedBytes::<96>(
                bytes.try_into().unwrap(),
            )));
            let damaged: PreparedAffinePoint<C> = *raw.as_value();
            if damage == "cache" {
                record.cached[0] = damaged;
            } else {
                record.compact_cached[0] = damaged;
            }
        }
        "curve" => record.base_modulus = record.scalar_modulus,
        "window" => record.window_bits = 8,
        "" | "truncate" => {}
        _ => panic!("unknown artifact damage"),
    }
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let mut bytes = bento::bytes_of(&record).to_vec();
    if damage == "truncate" {
        bytes.pop();
    }
    fs::write(
        directory.join(format!("{name}-fixed-base-{STORED_FORM}.bin")),
        bytes,
    )
    .unwrap();
}

fn generate_srs<C: PastaCurve>(name: &str, damage: &str) {
    let mut record = record::SrsRecord::<C>::empty();
    let domain = record.domain();
    // Known generator multiples make this fixture reproducible. The scalar 7
    // provides no setup secrecy and is used solely for test data.
    let mut power = PastaField::ONE;
    let mut projective = core::array::from_fn::<_, { record::SRS_SIZE }, _>(|_| {
        let point = ProjectivePoint::<C>::GENERATOR.mul(&power);
        power = power.mul(&PastaField::from_u64(7));
        point
    });
    let mut points = [Point::IDENTITY; record::SRS_SIZE];
    let mut scratch = [PastaField::ZERO; record::SRS_SIZE];
    batch_normalize(&projective, &mut points, &mut scratch).unwrap();
    record.coefficient =
        points.map(|point| PreparedAffinePoint::from_affine(point.as_affine().unwrap()));
    reference::inverse_transform(
        &mut projective,
        &domain.inverse_root(),
        &domain.size_inverse(),
    );
    batch_normalize(&projective, &mut points, &mut scratch).unwrap();
    record.lagrange =
        points.map(|point| PreparedAffinePoint::from_affine(point.as_affine().unwrap()));
    match damage {
        "srs-root" => record.root = record.root.neg(),
        "srs-order" => record.lagrange.swap(0, 1),
        "" => {}
        _ => panic!("unknown SRS artifact damage"),
    }
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
    let damage = env::var("CURVE_ARTIFACT_DAMAGE").unwrap_or_default();
    let (table_damage, srs_damage) = if damage.starts_with("srs-") {
        ("", damage.as_str())
    } else {
        (damage.as_str(), "")
    };
    generate::<Pallas>("pallas", table_damage);
    generate::<Vesta>("vesta", "");
    generate_srs::<Pallas>("pallas", srs_damage);
    generate_srs::<Vesta>("vesta", "");
}

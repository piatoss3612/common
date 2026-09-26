//! Full builds force coordinate assertions, including in runtime expressions.

use super::harness::Consumer;

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn curve_constants_enforce_types_and_values_through_a_dependency_alias() {
    let cases = [
        ("invalid-pallas", "point must be on the curve"),
        ("invalid-vesta", "point must be on the curve"),
        ("runtime-pallas", "non-constant value in a constant"),
        ("runtime-vesta", "non-constant value in a constant"),
        ("wrong-pallas-field", "mismatched types"),
        ("wrong-vesta-field", "mismatched types"),
        ("wrong-curve", "mismatched types"),
        ("wrong-scalar", "mismatched types"),
    ];
    let consumer = Consumer::new(
        "curve-constants-consumer",
        "curve/fixtures/constants.rs",
        "arithmetic",
        &cases.map(|(name, _)| name),
    );
    for (features, diagnostic) in [("", None), ("sqrt-table-large", None)]
        .into_iter()
        .chain(cases.map(|(feature, diagnostic)| (feature, Some(diagnostic))))
    {
        consumer.check(
            if diagnostic.is_some() { "build" } else { "run" },
            features,
            &[],
            diagnostic,
            &["src/main.rs"],
        );
    }
}

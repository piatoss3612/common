//! Consumer APIs require an explicit opt-in; native arithmetic does not.

use super::harness::Consumer;

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn consumer_interfaces_require_the_traits_feature() {
    let cases = [
        ("field", "cannot find trait `Field`"),
        ("curve", "cannot find trait `Affine`"),
        ("domain", "no method named `transform`"),
        ("poly", "could not find `poly`"),
        ("cycle", "could not find `cycle`"),
        ("poseidon", "cannot find trait `PoseidonPermutation`"),
    ];
    let features: Vec<_> = cases.iter().map(|&(feature, _)| feature).collect();
    let consumer = Consumer::new(
        "traits-feature-consumer",
        "api/fixtures/traits.rs",
        "arithmetic",
        &features,
    );

    for tables in ["", "sqrt-table-large"] {
        consumer.check("run", tables, &[], None, &["src/main.rs"]);
        for (feature, diagnostic) in cases {
            consumer.check(
                "build",
                &format!("{tables},{feature}"),
                &[],
                Some(diagnostic),
                &["src/main.rs"],
            );
        }
        consumer.check(
            "run",
            &format!("{tables},traits,{}", features.join(",")),
            &[],
            None,
            &["src/main.rs"],
        );
    }
}

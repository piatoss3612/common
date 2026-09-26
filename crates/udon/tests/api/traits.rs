//! Consumer APIs require an explicit opt-in; native arithmetic does not.

use super::harness::Consumer;

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn consumer_interfaces_require_the_traits_feature() {
    let cases = [
        ("field", "cannot find trait `Field`"),
        ("curve", "cannot find trait `Affine`"),
        ("domain", "unresolved import"),
        ("polynomial", "cannot find function `evaluate_iter`"),
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

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn native_types_do_not_acquire_arithmetic_operators() {
    let cases = [
        ("field-add", "error[E0369]"),
        ("field-sub", "error[E0369]"),
        ("field-mul", "error[E0369]"),
        ("field-neg", "error[E0600]"),
        ("field-add-assign", "error[E0368]"),
        ("field-sub-assign", "error[E0368]"),
        ("field-mul-assign", "error[E0368]"),
        ("field-reduced-add", "error[E0369]"),
        ("field-sum", "error[E0277]"),
        ("field-product", "error[E0277]"),
        ("point-neg", "error[E0600]"),
        ("point-mul", "error[E0369]"),
        ("affine-neg", "error[E0600]"),
        ("affine-mul", "error[E0369]"),
        ("projective-add", "error[E0369]"),
        ("projective-sub", "error[E0369]"),
        ("projective-neg", "error[E0600]"),
        ("projective-mul", "error[E0369]"),
        ("projective-add-assign", "error[E0368]"),
        ("projective-sub-assign", "error[E0368]"),
        ("projective-sum", "error[E0277]"),
    ];
    let features: Vec<_> = cases.iter().map(|&(feature, _)| feature).collect();
    let consumer = Consumer::new(
        "explicit-native-arithmetic",
        "api/fixtures/operators.rs",
        "arithmetic",
        &features,
    );
    for configuration in ["", "traits", "sqrt-table-large", "traits,sqrt-table-large"] {
        consumer.check("run", configuration, &[], None, &["src/main.rs"]);
        for (feature, diagnostic) in cases {
            consumer.check(
                "build",
                &format!("{configuration},{feature}"),
                &[],
                Some(diagnostic),
                &["src/main.rs"],
            );
        }
    }
}

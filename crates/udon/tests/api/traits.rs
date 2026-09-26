//! Consumer APIs and the complete Poseidon module have explicit opt-ins.

use super::harness::Consumer;

const INTERFACE_FEATURES: &[&str] = &[
    "field",
    "curve",
    "domain",
    "polynomial",
    "poseidon-parameters",
    "poseidon-interface",
    "cycle",
];

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn consumer_interfaces_require_the_traits_feature() {
    let cases = [
        ("field", "cannot find trait `Field`"),
        ("curve", "cannot find trait `Affine`"),
        ("domain", "unresolved import"),
        ("polynomial", "cannot find function `evaluate_iter`"),
    ];
    let features: Vec<_> = cases.iter().map(|&(feature, _)| feature).collect();
    let consumer = Consumer::new(
        "traits-feature-consumer",
        "api/fixtures/traits.rs",
        "arithmetic",
        INTERFACE_FEATURES,
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
fn poseidon_apis_require_the_poseidon_feature() {
    let consumer = Consumer::new(
        "poseidon-feature-consumer",
        "api/fixtures/traits.rs",
        "arithmetic",
        INTERFACE_FEATURES,
    );

    for tables in ["", "sqrt-table-large"] {
        for traits in ["", "traits"] {
            let configuration = format!("{tables},{traits}");
            for (feature, diagnostic) in [
                ("poseidon-parameters", "could not find `poseidon`"),
                ("poseidon-interface", "could not find `poseidon`"),
                ("cycle", "could not find `cycle`"),
            ] {
                consumer.check(
                    "build",
                    &format!("{configuration},{feature}"),
                    &[],
                    Some(diagnostic),
                    &["src/main.rs"],
                );
            }
            // Poseidon enables the field interfaces its consumer views require.
            consumer.check(
                "run",
                &format!("{configuration},poseidon,poseidon-parameters,poseidon-interface,cycle"),
                &[],
                None,
                &["src/main.rs"],
            );
        }
    }
}

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn native_arithmetic_and_conversions_stay_explicit() {
    let cases = [
        ("field-from-u64", "error[E0277]"),
        ("field-reduced-from-u64", "error[E0277]"),
        ("point-from-projective", "error[E0277]"),
        ("projective-from-point", "error[E0277]"),
        ("field-eq", "error[E0369]"),
        ("field-ne", "error[E0369]"),
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
    let adapter_cases = [
        "field-from-adapter",
        "point-from-adapter",
        "projective-from-adapter",
    ];
    let features: Vec<_> = cases
        .iter()
        .map(|&(feature, _)| feature)
        .chain(adapter_cases)
        .collect();
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
    for configuration in ["traits", "traits,sqrt-table-large"] {
        for feature in adapter_cases {
            consumer.check(
                "build",
                &format!("{configuration},{feature}"),
                &[],
                Some("error[E0277]"),
                &["src/main.rs"],
            );
        }
    }
}

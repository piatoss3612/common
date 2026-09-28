//! Public arithmetic boundaries enforce type contracts and hide implementation choices.

use super::harness::Consumer;

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn public_boundaries_hide_parameters_and_implementation_choices() {
    let constants = [
        ("roots", "ROOTS"),
        ("inverse-roots", "INVERSE_ROOTS"),
        ("montgomery-inv", "MONTGOMERY_INV"),
        ("r", "R"),
        ("r2", "R2"),
        ("r3", "R3"),
        ("b448", "B448"),
        ("sqrt-exponent", "SQRT_EXPONENT"),
        ("two-inverse", "TWO_INVERSE"),
        ("delta", "DELTA"),
        ("zeta", "ZETA"),
        ("zeta-inverse", "ZETA_INVERSE"),
        ("modulus-signed62", "MODULUS_SIGNED62"),
        ("safegcd-corrections", "SAFEGCD_CORRECTIONS"),
        ("power-of-two-inverses", "POWER_OF_TWO_INVERSES"),
        ("glv-a", "GLV_A"),
        ("glv-b", "GLV_B"),
    ];
    let features: Vec<_> = constants
        .iter()
        .map(|&(feature, _)| feature)
        .chain([
            "pow-sqrt-exponent",
            "sqrt-large",
            "sqrt-finish-large",
            "foreign-modulus",
            "foreign-curve",
            "foreign-reduction",
            "foreign-reduction-flag",
            "loose-order",
            "loose-sqrt",
            "loose-sqrt-alt",
            "loose-sqrt-ratio",
            "loose-sqrt-denominator",
            "loose-coordinates",
            "invalid-reduced-limbs",
            "invalid-loose-limbs",
            "msm-arithmetic",
            "msm-kernel",
            "msm-accumulation",
            "matrix-bases-mutation",
            "matrix-scalars-mutation",
            "matrix-curve-mismatch",
            "suffix-output-mutation",
            "suffix-differences-mutation",
            "suffix-curve-mismatch",
            "suffix-field-mismatch",
            "sum-original-mutation",
            "sum-indices-mutation",
            "sum-differences-mutation",
            "sum-curve-mismatch",
            "sum-field-mismatch",
            "support-field-mismatch",
            "support-bases-mutation",
            "support-mapping-mutation",
            "support-indices-mutation",
            "support-scalars-mutation",
            "coalesce-original-mutation",
            "coalesce-keys-mutation",
            "coalesce-points-mutation",
            "coalesce-field-mismatch",
            "coalesce-curve-mismatch",
            "coalesce-indices-mutation",
            "coalesce-order-mutation",
            "coalesce-output-mutation",
            "fft-codelet",
            "fft-strategy",
            "cache-options",
            "eisenstein-length",
            "empty-msm-slots",
            "empty-interpolation",
            "evaluation-powers-mutation",
            "evaluation-field-mismatch",
            "division-field-mismatch",
            "division-reduced-storage",
            "monic-field-mismatch",
            "monic-reduced-storage",
            "vanishing-field-mismatch",
            "vanishing-reduced-storage",
            "lagrange-field-mismatch",
            "lagrange-reduced-storage",
            "lagrange-completion-field-mismatch",
            "interpolation-points-mutation",
            "interpolation-weights-mutation",
            "interpolation-field-mismatch",
            "interpolation-reduced-storage",
            "interpolation-completion-field-mismatch",
            "vanishing-factors-mutation",
            "vanishing-finish-field-mismatch",
            "vanishing-finish-reduced-storage",
            "tail-samples-mutation",
            "tail-input-mutation",
            "tail-field-mismatch",
            "tail-reduced-storage",
        ])
        .collect();
    let consumer = Consumer::new(
        "api-boundaries-consumer",
        "api/fixtures/boundaries",
        "arithmetic",
        &features,
    );
    for configuration in ["", "sqrt-table-large", "traits", "traits,sqrt-table-large"] {
        consumer.check("run", configuration, &[], None, &["src/main.rs"]);
        // Build one failing access at a time: an earlier rejection must not
        // conceal a later member that is still reachable.
        for (feature, member) in constants {
            let diagnostic = if member == "SQRT_EXPONENT" {
                // The ordinary exponent is not a field parameter: production
                // uses its generated multiplication schedule, and the parameter
                // tests derive the exponent independently.
                // Rust versions differ in how they describe the missing item.
                format!("named `{member}`")
            } else {
                format!("associated constant `{member}` is private")
            };
            consumer.check(
                "build",
                &format!("{configuration},{feature}"),
                &[],
                Some(&diagnostic),
                &[rejection_source(feature)],
            );
        }
        for (feature, diagnostic) in [
            (
                "pow-sqrt-exponent",
                "associated function `pow_sqrt_exponent` is private",
            ),
            (
                "sqrt-large",
                if configuration
                    .split(',')
                    .any(|feature| feature == "sqrt-table-large")
                {
                    "associated function `sqrt_large` is private"
                } else {
                    "named `sqrt_large`"
                },
            ),
            (
                "sqrt-finish-large",
                if configuration
                    .split(',')
                    .any(|feature| feature == "sqrt-table-large")
                {
                    "associated function `sqrt_finish_large` is private"
                } else {
                    "named `sqrt_finish_large`"
                },
            ),
            ("foreign-modulus", "Sealed` is not satisfied"),
            ("foreign-curve", "Sealed` is not satisfied"),
            ("foreign-reduction", "Sealed` is not satisfied"),
            (
                "foreign-reduction-flag",
                "associated constant `REDUCED` is private",
            ),
            ("loose-order", "PastaField<M>: Ord` is not satisfied"),
            ("loose-sqrt", "no method named `sqrt`"),
            ("loose-sqrt-alt", "no method named `sqrt_alt`"),
            ("loose-sqrt-ratio", "no method named `sqrt_ratio`"),
            ("loose-sqrt-denominator", "mismatched types"),
            ("loose-coordinates", "expected `PastaField"),
            (
                "invalid-reduced-limbs",
                "Montgomery limbs exceed the representation bound",
            ),
            (
                "invalid-loose-limbs",
                "Montgomery limbs exceed the representation bound",
            ),
            ("msm-arithmetic", "struct `ArithmeticOptions` is private"),
            ("msm-kernel", "enum `Algorithm` is private"),
            ("msm-accumulation", "enum `Accumulation` is private"),
            (
                "matrix-bases-mutation",
                "cannot assign to `matrix_bases[_]` because it is borrowed",
            ),
            (
                "matrix-scalars-mutation",
                "cannot assign to `records[_]` because it is borrowed",
            ),
            ("matrix-curve-mismatch", "mismatched types"),
            (
                "suffix-output-mutation",
                "cannot assign to `sums[_]` because it is borrowed",
            ),
            (
                "suffix-differences-mutation",
                "cannot assign to `differences[_]` because it is borrowed",
            ),
            ("suffix-curve-mismatch", "mismatched types"),
            (
                "sum-original-mutation",
                "cannot assign to `original[_]` because it is borrowed",
            ),
            (
                "sum-indices-mutation",
                "cannot assign to `indices[_]` because it is borrowed",
            ),
            (
                "sum-differences-mutation",
                "cannot assign to `differences[_]` because it is borrowed",
            ),
            ("sum-curve-mismatch", "mismatched types"),
            ("sum-field-mismatch", "mismatched types"),
            (
                "coalesce-original-mutation",
                "cannot assign to `original[_]` because it is borrowed",
            ),
            (
                "coalesce-keys-mutation",
                "cannot assign to `keys[_]` because it is borrowed",
            ),
            (
                "coalesce-points-mutation",
                "cannot assign to `points[_]` because it is borrowed",
            ),
            ("support-field-mismatch", "mismatched types"),
            (
                "support-scalars-mutation",
                "cannot assign to `scalars[_]` because it is borrowed",
            ),
            (
                "support-bases-mutation",
                "cannot assign to `original[_]` because it is borrowed",
            ),
            (
                "support-mapping-mutation",
                "cannot assign to `mapping[_]` because it is borrowed",
            ),
            (
                "support-indices-mutation",
                "cannot assign to `indices[_]` because it is borrowed",
            ),
            ("coalesce-field-mismatch", "mismatched types"),
            ("coalesce-curve-mismatch", "mismatched types"),
            (
                "coalesce-indices-mutation",
                "cannot assign to `indices[_]` because it is borrowed",
            ),
            (
                "coalesce-order-mutation",
                "cannot assign to `order[_]` because it is borrowed",
            ),
            (
                "coalesce-output-mutation",
                "cannot assign to `output[_]` because it is borrowed",
            ),
            (
                "suffix-field-mismatch",
                "arguments to this method are incorrect",
            ),
            ("fft-codelet", "enum `Codelet` is private"),
            ("fft-strategy", "struct `Strategy` is private"),
            (
                "cache-options",
                "expected `&MsmPlan<C>`, found `ExecutionOptions`",
            ),
            ("eisenstein-length", "expected an array with a size of 8"),
            (
                "evaluation-powers-mutation",
                "cannot assign to `powers[_]` because it is borrowed",
            ),
            ("evaluation-field-mismatch", "mismatched types"),
            ("division-field-mismatch", "mismatched types"),
            ("division-reduced-storage", "mismatched types"),
            ("monic-field-mismatch", "mismatched types"),
            ("monic-reduced-storage", "mismatched types"),
            ("vanishing-field-mismatch", "mismatched types"),
            ("vanishing-reduced-storage", "mismatched types"),
            ("lagrange-field-mismatch", "mismatched types"),
            ("lagrange-reduced-storage", "mismatched types"),
            ("lagrange-completion-field-mismatch", "mismatched types"),
            (
                "interpolation-points-mutation",
                "cannot assign to `points[_]` because it is borrowed",
            ),
            (
                "interpolation-weights-mutation",
                "cannot assign to `weights[_]` because it is borrowed",
            ),
            ("interpolation-field-mismatch", "mismatched types"),
            ("interpolation-reduced-storage", "mismatched types"),
            (
                "interpolation-completion-field-mismatch",
                "mismatched types",
            ),
            (
                "vanishing-factors-mutation",
                "cannot assign to `storage[_]` because it is borrowed",
            ),
            ("vanishing-finish-field-mismatch", "mismatched types"),
            ("vanishing-finish-reduced-storage", "mismatched types"),
            (
                "tail-samples-mutation",
                "cannot assign to `samples[_]` because it is borrowed",
            ),
            (
                "tail-input-mutation",
                "cannot assign to `tail[_]` because it is borrowed",
            ),
            ("tail-field-mismatch", "mismatched types"),
            ("tail-reduced-storage", "mismatched types"),
        ] {
            consumer.check(
                "build",
                &format!("{configuration},{feature}"),
                &[],
                Some(diagnostic),
                &[rejection_source(feature)],
            );
        }
        // Monomorphization reports inline-const contracts at their definitions.
        for (feature, diagnostic, source) in [
            (
                "empty-msm-slots",
                "parallel slots must be nonzero",
                "src/msm/execution/chunks.rs",
            ),
            (
                "empty-interpolation",
                "interpolation needs an output class",
                "src/fft/execution/interpolation.rs",
            ),
        ] {
            consumer.check(
                "build",
                &format!("{configuration},{feature}"),
                &[],
                Some(diagnostic),
                &[source],
            );
        }
    }
}

fn rejection_source(feature: &str) -> &'static str {
    match feature {
        "cache-options" | "eisenstein-length" | "empty-msm-slots" | "foreign-curve" | "glv-a"
        | "glv-b" | "loose-coordinates" | "msm-accumulation" | "msm-arithmetic" | "msm-kernel" => {
            "src/reject_curve.rs"
        }
        "empty-interpolation" | "fft-codelet" | "fft-strategy" => "src/reject_fft.rs",
        feature
            if ["matrix-", "suffix-", "sum-", "support-", "coalesce-"]
                .iter()
                .any(|prefix| feature.starts_with(prefix)) =>
        {
            "src/reject_curve.rs"
        }
        feature
            if [
                "lagrange-",
                "tail-",
                "vanishing-factors-",
                "vanishing-finish-",
            ]
            .iter()
            .any(|prefix| feature.starts_with(prefix)) =>
        {
            "src/reject_fft.rs"
        }
        _ => "src/reject_field.rs",
    }
}

//! Public arithmetic boundaries reject private parameters and misplaced policy.

#[path = "support/consumer.rs"]
mod consumer;

use consumer::Consumer;

#[test]
#[ignore = "slow nested Cargo builds; run explicitly with --ignored"]
fn public_boundaries_hide_parameters_and_separate_msm_policy() {
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
            "foreign-modulus",
            "foreign-curve",
            "arithmetic-budget",
            "arithmetic-limit",
            "plan-batch-options",
            "cache-batch-options",
            "batch-arithmetic-options",
        ])
        .collect();
    let consumer = Consumer::new(
        "api-boundaries-consumer",
        "api_boundaries.rs",
        "arithmetic",
        &features,
    );
    for configuration in ["", "sqrt-table-large"] {
        consumer.check("run", configuration, &[], None, &["src/main.rs"]);
        // Build one failing access at a time: an earlier rejection must not
        // conceal a later member that is still reachable.
        for (feature, member) in constants {
            let diagnostic = if member == "SQRT_EXPONENT" {
                // The ordinary exponent is only retained for internal tests;
                // production uses its generated multiplication schedule.
                format!("no associated item named `{member}`")
            } else {
                format!("associated constant `{member}` is private")
            };
            consumer.check(
                "build",
                &format!("{configuration},{feature}"),
                &[],
                Some(&diagnostic),
                &["src/main.rs"],
            );
        }
        for (feature, diagnostic) in [
            (
                "pow-sqrt-exponent",
                "associated function `pow_sqrt_exponent` is private",
            ),
            (
                "sqrt-large",
                if configuration.is_empty() {
                    "no function or associated item named `sqrt_large`"
                } else {
                    "associated function `sqrt_large` is private"
                },
            ),
            ("foreign-modulus", "Sealed` is not satisfied"),
            ("foreign-curve", "Sealed` is not satisfied"),
            ("arithmetic-budget", "no method named `with_task_budget`"),
            ("arithmetic-limit", "no method named `with_memory_limit`"),
            (
                "plan-batch-options",
                "expected `ArithmeticOptions`, found `BatchOptions`",
            ),
            (
                "cache-batch-options",
                "expected `ArithmeticOptions`, found `BatchOptions`",
            ),
            (
                "batch-arithmetic-options",
                "expected `BatchOptions`, found `ArithmeticOptions`",
            ),
        ] {
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

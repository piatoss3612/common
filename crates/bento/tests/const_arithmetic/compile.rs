//! The facade enforces constant inputs and arithmetic preconditions.

use std::path::Path;

#[test]
fn rejects_runtime_inputs_and_invalid_parameters() {
    crate::compiler::check_rejections(
        "const-arithmetic-consumer",
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/const_arithmetic/fixtures"),
        &[
            ("invalid_modulus", "modulus must be odd"),
            (
                "invalid_two_adicity",
                "two_adicity exceeds the trailing zeros",
            ),
            ("unreduced_base", "base must be reduced"),
            ("unreduced_power_table", "base must be reduced"),
            ("ratio_overflow", "quotient exceeds five limbs"),
            (
                "runtime_arguments",
                "attempt to use a non-constant value in a constant",
            ),
            (
                "const_fn_arguments",
                "attempt to use a non-constant value in a constant",
            ),
            (
                "const_local",
                "attempt to use a non-constant value in a constant",
            ),
            ("direct_functions", "expected value, found macro"),
            (
                "direct_context",
                "could not find `MontgomeryContext` in `m255`",
            ),
            ("invalid_runtime_expression", "modulus must be odd"),
        ],
    );
}

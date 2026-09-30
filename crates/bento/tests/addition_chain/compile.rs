//! Invalid scalars and incompatible value types must fail in callers.

use std::path::Path;

#[test]
fn rejects_invalid_scalars_and_value_types() {
    crate::compiler::check_rejections(
        "addition-chain-consumer",
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/addition_chain/fixtures"),
        &[
            (
                "zero",
                "addition_chain! scalar must be nonzero; the trait has no identity operation",
            ),
            (
                "suffix",
                "addition_chain! scalar must be an unsuffixed integer literal",
            ),
            ("constant", "expected an integer literal or tonelli_shanks"),
            ("negative", "addition_chain! scalar must be positive"),
            (
                "forwarded_negative",
                "addition_chain! scalar must be positive",
            ),
            ("missing_trait", "Value: AdditionChain"),
            ("missing_clone", "Value: Clone"),
            ("moved_value", "use of moved value: `value`"),
        ],
    );
}

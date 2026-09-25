//! Length and capacity assertions shared by arithmetic modules.

pub(crate) fn assert_length(buffer: &str, expected: usize, actual: usize) {
    assert_eq!(actual, expected, "{buffer} length");
}

pub(crate) fn assert_scratch(buffer: &str, required: usize, provided: usize) {
    assert!(
        provided >= required,
        "{buffer} scratch requires {required} elements, got {provided}"
    );
}

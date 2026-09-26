#![no_std]
#![deny(warnings)]
#![forbid(unsafe_code)]

mod bento {
    pub use macros::Pod;
    pub use support_core::*;
}

pub mod pod;

#[derive(Clone)]
pub struct Value(pub u64);

impl support_core::addchain::AdditionChain for Value {
    fn double(&self) -> Self {
        Self(self.0 * 2)
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

pub fn scale(value: Value) -> Value {
    macros::addition_chain!(crate = support_core; value, 181)
}

#[test]
fn direct_core_consumer() {
    assert_eq!(scale(Value(7)).0, 1267);
}

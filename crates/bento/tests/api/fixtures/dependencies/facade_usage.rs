#![no_std]
#![deny(warnings)]
#![forbid(unsafe_code)]

#[repr(transparent)]
#[derive(Clone, Copy, support::Pod)]
pub struct Value(pub u32);

const _: () = <Value as support::Pod>::ASSERT_LAYOUT;

impl support::addchain::AdditionChain for Value {
    fn double(&self) -> Self {
        Self(self.0 * 2)
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

pub fn scale(value: Value) -> Value {
    support::addition_chain!(value, 181)
}

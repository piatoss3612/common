#![deny(warnings)]
#![forbid(unsafe_code)]

#[repr(transparent)]
#[derive(Clone, Copy, support::Pod)]
#[pod(crate = support)]
struct Value(u32);

impl support::addchain::AdditionChain for Value {
    fn double(&self) -> Self {
        Self(self.0 * 2)
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

fn main() {
    let value = support::addition_chain!(Value(7), 181);
    assert_eq!(support::bytes_of(&value), 1267_u32.to_le_bytes());
}

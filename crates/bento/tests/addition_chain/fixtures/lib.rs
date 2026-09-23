#![no_std]
#![forbid(unsafe_code)]

#[derive(Clone)]
pub struct Value(pub u64);

impl zakura_bento::addchain::AdditionChain for Value {
    fn double(&self) -> Self {
        Self(self.0 * 2)
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

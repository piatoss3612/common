#![no_std]
#![deny(warnings)]
#![forbid(unsafe_code)]

// The harness compiles this with both an inherited alias and the registry name.
#[cfg(feature = "renamed")]
use support as bento;
#[cfg(not(feature = "renamed"))]
use zakura_bento as bento;

pub mod arithmetic;
pub mod pod;

// Consumers can expose the support API through their own facade.
pub use bento::const_arithmetic::u256::shr as shift;
pub use bento::*;

// Exercise automatic derive discovery independently of the shared core fixture.
#[repr(transparent)]
#[derive(Clone, Copy, bento::Pod)]
pub struct Automatic(pub u32);
const _: () = <Automatic as bento::Pod>::ASSERT_LAYOUT;

#[derive(Clone)]
pub struct Value(pub u64);

impl bento::addchain::AdditionChain for Value {
    fn double(&self) -> Self {
        Self(self.0 * 2)
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

/// Scales a value by 181 through the facade's macro re-export.
///
/// ```
/// # #[cfg(feature = "renamed")]
/// # use facade_renamed::{scale, Value};
/// # #[cfg(not(feature = "renamed"))]
/// # use facade_default::{scale, Value};
/// assert_eq!(scale(Value(7)).0, 1267);
/// ```
pub fn scale(value: Value) -> Value {
    bento::addition_chain!(value, 181)
}

#[test]
fn facade_only_consumer() {
    arithmetic::check();
    assert_eq!(scale(Value(7)).0, 1267);
    assert_eq!(bento::addition_chain!(Value(7), 1).0, 7);
    assert_eq!(bento::addition_chain!(Value(7), 3).0, 21);
}

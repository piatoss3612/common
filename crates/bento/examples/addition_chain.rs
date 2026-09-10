//! Two interpretations of the same fixed-scalar chain; no Copy requirement.

use zakura_bento as bento;

#[derive(Clone, Debug, PartialEq)]
struct Vector(i64, i64);

impl bento::addchain::AdditionChain for Vector {
    fn double(&self) -> Self {
        Self(self.0 * 2, self.1 * 2)
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Power(u64);

impl bento::addchain::AdditionChain for Power {
    fn double(&self) -> Self {
        Self(self.0 * self.0 % 65_521)
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0 * rhs.0 % 65_521)
    }
}

// The macro works inside a function generic over the internal support trait.
fn scale_181<T: bento::addchain::AdditionChain>(value: T) -> T {
    bento::addition_chain!(value, 0xb5)
}

fn main() {
    let vector = scale_181(Vector(2, -3));
    assert_eq!(vector, Vector(362, -543));
    println!("181 * (2, -3) = {vector:?}");

    let power = scale_181(Power(3));
    let expected = (0..181).fold(1, |acc, _| acc * 3 % 65_521);
    assert_eq!(power, Power(expected));
    println!("3^181 mod 65521 = {}", power.0);
}

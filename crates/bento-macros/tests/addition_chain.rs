//! Behavioral checks through the public facade, separate from token snapshots.

use std::{cell::Cell, rc::Rc};

#[derive(Debug, PartialEq)]
struct Value(u128);

impl Clone for Value {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl bento::addchain::AdditionChain for Value {
    fn double(&self) -> Self {
        Self(self.0.checked_mul(2).unwrap())
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0.checked_add(rhs.0).unwrap())
    }
}

#[test]
fn scales_non_copy_values_at_boundaries_and_in_all_radices() {
    assert_eq!(bento::addition_chain!(Value(7), 1), Value(7));
    assert_eq!(bento::addition_chain!(Value(7), 2), Value(14));
    assert_eq!(bento::addition_chain!(Value(7), 181), Value(1267));
    assert_eq!(bento::addition_chain!(Value(7), 0xb5,), Value(1267));
    assert_eq!(bento::addition_chain!(Value(7), 0o265), Value(1267));
    assert_eq!(bento::addition_chain!(Value(7), 0b1011_0101), Value(1267));
    assert_eq!(
        bento::addition_chain!(Value(1), 0x1_0000000000000000),
        Value(1 << 64)
    );
    assert_eq!(
        bento::addition_chain!(Value(1), 0xffffffffffffffff_ffffffffffffffff),
        Value(u128::MAX)
    );
}

#[test]
fn a_256_bit_scalar_is_preserved_exactly() {
    #[derive(Clone, Debug, PartialEq)]
    struct Wide([u64; 4]);

    impl bento::addchain::AdditionChain for Wide {
        fn double(&self) -> Self {
            self.add(self)
        }

        fn add(&self, rhs: &Self) -> Self {
            let mut result = [0; 4];
            let mut carry = 0u128;
            for ((out, a), b) in result.iter_mut().zip(self.0).zip(rhs.0) {
                let sum = u128::from(a) + u128::from(b) + carry;
                *out = sum as u64;
                carry = sum >> 64;
            }
            assert_eq!(carry, 0);
            Self(result)
        }
    }

    assert_eq!(
        bento::addition_chain!(
            Wide([1, 0, 0, 0]),
            0xfedcba9876543210_0123456789abcdef_1122334455667788_99aabbccddeeff00
        ),
        Wide([
            0x99aabbccddeeff00,
            0x1122334455667788,
            0x0123456789abcdef,
            0xfedcba9876543210,
        ])
    );
}

#[test]
fn multiplication_interpretation_matches_reference_exponentiation() {
    #[derive(Clone, Debug, PartialEq)]
    struct Power(u64);

    impl bento::addchain::AdditionChain for Power {
        fn double(&self) -> Self {
            self.add(self)
        }

        fn add(&self, rhs: &Self) -> Self {
            Self(self.0 * rhs.0 % 65_521)
        }
    }

    for base in [0, 1, 2, 3, 65_520] {
        let expected = (0..65_537).fold(1, |acc, _| acc * base % 65_521);
        assert_eq!(
            bento::addition_chain!(Power(base), 0x1_0001),
            Power(expected)
        );
    }
}

#[test]
fn evaluates_once_clones_once_and_dispatches_only_to_the_trait() {
    #[derive(Default)]
    struct Counts {
        evaluations: Cell<usize>,
        clones: Cell<usize>,
        doubles: Cell<usize>,
        adds: Cell<usize>,
    }

    struct Tracked {
        value: u64,
        counts: Rc<Counts>,
    }

    impl Clone for Tracked {
        fn clone(&self) -> Self {
            self.counts.clones.set(self.counts.clones.get() + 1);
            Self {
                value: self.value,
                counts: Rc::clone(&self.counts),
            }
        }
    }

    impl bento::addchain::AdditionChain for Tracked {
        fn double(&self) -> Self {
            self.counts.doubles.set(self.counts.doubles.get() + 1);
            Self {
                value: self.value * 2,
                counts: Rc::clone(&self.counts),
            }
        }

        fn add(&self, rhs: &Self) -> Self {
            self.counts.adds.set(self.counts.adds.get() + 1);
            Self {
                value: self.value + rhs.value,
                counts: Rc::clone(&self.counts),
            }
        }
    }

    // Inherent names deliberately conflict with the trait and Clone.
    #[allow(dead_code)]
    impl Tracked {
        fn double(&self) -> Self {
            panic!("inherent double")
        }
        fn add(&self, _: &Self) -> Self {
            panic!("inherent add")
        }
        fn clone(&self) -> Self {
            panic!("inherent clone")
        }
    }

    let counts = Rc::new(Counts::default());
    let make_value = || {
        counts.evaluations.set(counts.evaluations.get() + 1);
        Tracked {
            value: 7,
            counts: Rc::clone(&counts),
        }
    };
    let result = bento::addition_chain!(make_value(), 0xb5);
    assert_eq!(result.value, 1267);
    assert_eq!(counts.evaluations.get(), 1);
    assert_eq!(counts.clones.get(), 1);
    assert_eq!(counts.doubles.get(), 6);
    assert_eq!(counts.adds.get(), 4);

    let result = bento::addition_chain!(make_value(), 1);
    assert_eq!(result.value, 7);
    assert_eq!(counts.evaluations.get(), 2);
    assert_eq!(counts.clones.get(), 2);
    assert_eq!(counts.doubles.get(), 6);
    assert_eq!(counts.adds.get(), 4);
}

#[test]
fn superseded_accumulators_drop_between_steps() {
    #[derive(Default)]
    struct Counts {
        live: Cell<usize>,
        peak: Cell<usize>,
    }

    struct Tracked {
        value: u64,
        counts: Rc<Counts>,
    }

    impl Tracked {
        fn new(value: u64, counts: &Rc<Counts>) -> Self {
            let live = counts.live.get() + 1;
            counts.live.set(live);
            counts.peak.set(counts.peak.get().max(live));
            Self {
                value,
                counts: Rc::clone(counts),
            }
        }
    }

    impl Clone for Tracked {
        fn clone(&self) -> Self {
            Self::new(self.value, &self.counts)
        }
    }

    impl bento::addchain::AdditionChain for Tracked {
        fn double(&self) -> Self {
            Self::new(self.value.wrapping_mul(2), &self.counts)
        }

        fn add(&self, rhs: &Self) -> Self {
            Self::new(self.value.wrapping_add(rhs.value), &self.counts)
        }
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.counts.live.set(self.counts.live.get() - 1);
        }
    }

    fn check(scale: impl FnOnce(Tracked) -> Tracked, expected: u64) {
        let counts = Rc::new(Counts::default());
        let result = scale(Tracked::new(1, &counts));
        assert_eq!(result.value, expected);
        // These sparse scalars need no odd table: only the input, accumulator,
        // and its replacement should coexist, regardless of chain length.
        assert!(counts.peak.get() <= 3, "{} live values", counts.peak.get());
        assert_eq!(counts.live.get(), 1);
        drop(result);
        assert_eq!(counts.live.get(), 0);
    }

    check(|value| bento::addition_chain!(value, 1), 1);
    check(
        |value| bento::addition_chain!(value, 0x1_0000000000000000),
        0,
    );
    check(
        |value| {
            bento::addition_chain!(
                value,
                0x4000000000000000_0000000000000000_0000000000000000_0000000000000002
            )
        },
        2,
    );
}

#[test]
fn caller_bindings_and_expression_syntax_survive_expansion() {
    macro_rules! scale {
        ($value:expr, $scalar:literal) => {
            bento::addition_chain!($value, $scalar)
        };
    }

    fn make<A, B>(_: A, _: B) -> Value {
        Value(7)
    }

    let __bento_odd_0 = Value(7);
    let __bento_accumulator = 3;
    let __bento_clone = 4;
    assert_eq!(
        bento::addition_chain!(
            {
                assert_eq!(__bento_accumulator + __bento_clone, 7);
                __bento_odd_0
            },
            181
        ),
        Value(1267)
    );
    assert_eq!(bento::addition_chain!(make::<u8, u16>(1, 2), 3), Value(21));
    assert_eq!(scale!(Value(7), 181), Value(1267));
    assert_eq!(
        bento::addition_chain!((|a: u128, b: u128| Value(a + b))(3, 4), 3),
        Value(21)
    );
}

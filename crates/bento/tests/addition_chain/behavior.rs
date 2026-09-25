//! Behavioral checks through the public facade, separate from token snapshots.

use zakura_bento as bento;

use std::{cell::Cell, rc::Rc};

#[test]
#[allow(non_upper_case_globals)]
fn caller_constants_cannot_become_generated_patterns() {
    mod names {
        pub const __bento_odd_0: u128 = 1;
        pub const __bento_odd_1: u128 = 2;
        pub const __bento_odd_2: u128 = 3;
        pub const __bento_doubled: u128 = 4;
        pub const __bento_accumulator: u128 = 5;
        pub const __bento_clone: u128 = 6;
        pub const __bento_chain: u128 = 7;
        pub const __bento_value_0: u128 = 8;
    }
    use names::*;
    assert_eq!(
        bento::addition_chain!(
            Value(__bento_value_0),
            9,
            chain = |x| {
                let y = double(x, 3);
                let result = add(y, x);
                result
            },
            emission = batched
        ),
        Value(72)
    );
    assert_eq!(bento::addition_chain!(Value(__bento_odd_0), 1), Value(1));
    assert_eq!(
        bento::addition_chain!(
            Value(
                __bento_odd_0
                    + __bento_odd_1
                    + __bento_odd_2
                    + __bento_doubled
                    + __bento_accumulator
                    + __bento_clone
                    + __bento_chain
            ),
            181
        ),
        Value(28 * 181)
    );
}

#[test]
fn input_control_flow_stays_in_the_caller() {
    fn fallible(input: Result<Value, ()>) -> Result<Value, ()> {
        Ok(bento::addition_chain!(input?, 3))
    }
    fn early_return(stop: bool) -> Value {
        bento::addition_chain!(
            {
                if stop {
                    return Value(11);
                }
                Value(7)
            },
            3
        )
    }
    assert_eq!(fallible(Ok(Value(7))), Ok(Value(21)));
    assert_eq!(fallible(Err(())), Err(()));
    assert_eq!(early_return(true), Value(11));
    assert_eq!(early_return(false), Value(21));
    let mut stop = false;
    let result = loop {
        let value = bento::addition_chain!(
            {
                if stop {
                    break Value(11);
                }
                Value(7)
            },
            3
        );
        assert_eq!(value, Value(21));
        stop = true;
    };
    assert_eq!(result, Value(11));
}

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
fn evaluates_once_moves_result_and_dispatches_only_to_the_trait() {
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

    // Inherent names deliberately conflict with the trait and `Clone`.
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
    assert_eq!(counts.clones.get(), 0);
    assert_eq!(counts.doubles.get(), 6);
    assert_eq!(counts.adds.get(), 4);

    let result = bento::addition_chain!(make_value(), 1);
    assert_eq!(result.value, 7);
    assert_eq!(counts.evaluations.get(), 2);
    assert_eq!(counts.clones.get(), 0);
    assert_eq!(counts.doubles.get(), 6);
    assert_eq!(counts.adds.get(), 4);
}

#[test]
#[ignore = "owned-result timing experiment; run alone with --nocapture"]
fn owned_result_timing() {
    use std::{hint::black_box, time::Instant};
    #[derive(Clone)]
    struct Row(Vec<u64>);
    impl bento::addchain::AdditionChain for Row {
        fn double(&self) -> Self {
            Self(self.0.iter().map(|x| x.wrapping_mul(2)).collect())
        }
        fn add(&self, rhs: &Self) -> Self {
            Self(
                self.0
                    .iter()
                    .zip(&rhs.0)
                    .map(|(a, b)| a.wrapping_add(*b))
                    .collect(),
            )
        }
    }
    fn time(size: usize, operation: impl Fn(Row) -> Row) -> f64 {
        let mut samples = [0.0f64; 9];
        for sample in &mut samples {
            let inputs: Vec<_> = (0..1024).map(|i| Row(vec![i; size])).collect();
            let start = Instant::now();
            for row in inputs {
                black_box(operation(black_box(row)));
            }
            *sample = start.elapsed().as_nanos() as f64 / 1024.0;
        }
        samples.sort_by(f64::total_cmp);
        samples[4]
    }
    for size in [64, 1024] {
        for scalar in [1, 181] {
            let operation = |row| match scalar {
                1 => bento::addition_chain!(row, 1),
                _ => bento::addition_chain!(row, 181),
            };
            // Reproduce the old emitter's final clone after the same chain.
            let cloned = time(size, |row| black_box(operation(row)).clone());
            let moved = time(size, operation);
            std::println!(
                "words={size}, scalar={scalar}: clone {cloned:.1}, move {moved:.1} ns/row"
            );
        }
    }
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
        |value| {
            bento::addition_chain!(
                value,
                9,
                chain = |x| {
                    let unused = double(x, 1);
                    let eight = double(x, 3);
                    let nine = add(eight, x);
                    nine
                },
                emission = batched
            )
        },
        9,
    );
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
// The immediate closure call exercises commas inside the macro's expression.
#[allow(clippy::redundant_closure_call)]
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

#[test]
fn supplied_chains_and_all_emission_modes_support_non_copy_values() {
    assert_eq!(
        bento::addition_chain!(
            Value(7),
            9,
            chain = |x| {
                let x8 = double(x, 3);
                let x9 = add(x8, x);
                x9
            }
        ),
        Value(63)
    );
    assert_eq!(
        bento::addition_chain!(
            Value(7),
            9,
            chain = |x| {
                let x8 = double(x, 3);
                let x9 = add(x8, x);
                x9
            },
            emission = unrolled
        ),
        Value(63)
    );
    assert_eq!(
        bento::addition_chain!(
            Value(7),
            9,
            chain = |x| {
                let x8 = double(x, 3);
                let x9 = add(x8, x);
                x9
            },
            emission = batched
        ),
        Value(63)
    );
    assert_eq!(
        bento::addition_chain!(
            Value(7),
            tonelli_shanks(
                "0x0000000000000000000000000000000000000000000000000000000000000061",
                5
            )
        ),
        Value(7)
    );
}

#[test]
fn batched_emission_dispatches_to_overrides() {
    #[derive(Clone)]
    struct Batch(u64);
    impl bento::addchain::AdditionChain for Batch {
        fn double(&self) -> Self {
            panic!("run should use override")
        }
        fn add(&self, _: &Self) -> Self {
            panic!("add should be fused")
        }
        fn double_n(&self, n: usize) -> Self {
            Self(self.0 << n)
        }
        fn double_n_add(&self, n: usize, rhs: &Self) -> Self {
            Self((self.0 << n) + rhs.0)
        }
    }
    assert_eq!(
        bento::addition_chain!(Batch(3), 0x101, emission = batched).0,
        771
    );
    assert_eq!(
        bento::addition_chain!(Batch(3), 0x100, emission = batched).0,
        768
    );
}

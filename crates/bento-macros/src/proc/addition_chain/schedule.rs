//! Sliding-window planning for addition chains on the build host.
//!
//! [`plan`] chooses a [`Schedule`] independently of Rust syntax and target
//! types. Expansion consumes that schedule to generate calls to
//! [`bento_core::addchain::AdditionChain`].
//!
//! # Background
//!
//! A binary chain starts from the input and processes each bit after the leading
//! bit with a doubling and, if the bit is set, an addition of the input. Sliding
//! windows group nearby set bits so one addition can incorporate several bits at
//! once. This requires precomputing odd multiples of the input.
//!
//! # Design
//!
//! Each table entry represents `(2 * index + 1)` copies of the input value.
//! A [`Schedule`] starts from one entry and applies a sequence of [`Step`]
//! operations. Only entries through the highest referenced index are prepared.
//! [`plan`] compares candidate schedules using [`Schedule::cost`], which includes
//! preparation so its cost can be weighed against the saved additions.

/// An operation that updates the accumulator using the prepared odd multiples.
#[derive(Clone, Copy, Debug)]
pub(super) enum Step {
    /// Doubles the accumulator.
    Double,

    /// Adds the odd multiple at the given table index to the accumulator.
    AddOdd(usize),
}

/// A sequence of operations and the odd multiples needed to scale an input.
pub(super) struct Schedule {
    /// The table index used to initialize the accumulator.
    pub(super) first: usize,

    /// The operations applied after initialization, in execution order.
    pub(super) steps: Vec<Step>,

    /// The highest required table index, or zero when only the input is needed.
    pub(super) max_odd_index: usize,
}

impl Schedule {
    /// Counts additions and doublings equally, including table preparation.
    ///
    /// Cloning, storage, and the implementation's relative costs are excluded.
    fn cost(&self) -> usize {
        let table = if self.max_odd_index == 0 {
            0
        } else {
            1 + self.max_odd_index
        };
        table + self.steps.len()
    }
}

/// Selects a sliding-window schedule for a positive scalar.
///
/// The scalar's limbs are little-endian; leading zero limbs are allowed.
/// Returns `None` for zero. The heuristic does not guarantee a shortest chain.
pub(super) fn plan(limbs: &[u64]) -> Option<Schedule> {
    let top_index = limbs.iter().rposition(|limb| *limb != 0)?;
    let bits = top_index * 64 + (64 - limbs[top_index].leading_zeros() as usize);

    // Compare widths six through one. The first minimum wins, so ties prefer
    // wider windows.
    (1..=6)
        .rev()
        .map(|width| plan_with_width(limbs, bits, width))
        .min_by_key(Schedule::cost)
}

fn plan_with_width(limbs: &[u64], bits: usize, width: usize) -> Schedule {
    let bit_at = |bit: usize| limbs[bit / 64] >> (bit % 64) & 1 == 1;
    let mut steps = Vec::new();
    let mut max_odd_index = 0;
    let mut first = None;
    let mut remaining = bits;
    while remaining > 0 {
        let high = remaining - 1;
        if !bit_at(high) {
            steps.push(Step::Double);
            remaining -= 1;
            continue;
        }
        let mut low = remaining.saturating_sub(width);
        while !bit_at(low) {
            low += 1;
        }
        let mut window = 0;
        for position in (low..=high).rev() {
            window = window << 1 | usize::from(bit_at(position));
        }
        let index = (window - 1) / 2;
        max_odd_index = max_odd_index.max(index);
        if first.is_none() {
            first = Some(index);
        } else {
            steps.extend((low..=high).map(|_| Step::Double));
            steps.push(Step::AddOdd(index));
        }
        remaining = low;
    }
    Schedule {
        first: first.expect("a nonzero scalar has a leading window"),
        steps,
        max_odd_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(scalar: u128) {
        let limbs = [scalar as u64, (scalar >> 64) as u64];
        let planned = plan(&limbs).unwrap();
        let binary = plan_with_width(&limbs, 128 - scalar.leading_zeros() as usize, 1);
        assert!(planned.cost() <= binary.cost());
        let mut value = (2 * planned.first + 1) as u128;
        for step in planned.steps {
            value = match step {
                Step::Double => value.checked_mul(2).unwrap(),
                Step::AddOdd(index) => {
                    assert!(index <= planned.max_odd_index);
                    value.checked_add((2 * index + 1) as u128).unwrap()
                }
            };
        }
        assert_eq!(value, scalar);
    }

    #[test]
    fn schedules_replay_to_the_scalar_and_never_cost_more_than_binary() {
        for scalar in 1..=4096 {
            check(scalar);
        }
        for bit in 1..128 {
            let power = 1u128 << bit;
            check(power - 1);
            check(power);
            check(power + 1);
        }
        check(u128::MAX);
    }

    #[test]
    fn windowing_improves_dense_and_pasta_sized_scalars() {
        for literal in [
            "0x2000000000000000000000000000000011234c7e04a67c8dcc969876",
            "0x23456789abcdef0123456789abcdef0123456789abcdef0123456789",
        ] {
            let scalar: syn::LitInt = syn::parse_str(literal).unwrap();
            let limbs = super::super::limbs_from_decimal(scalar.base10_digits());
            let bits = (limbs.len() - 1) * 64 + 64 - limbs.last().unwrap().leading_zeros() as usize;
            assert!(plan(&limbs).unwrap().cost() < plan_with_width(&limbs, bits, 1).cost());
        }
    }

    #[test]
    fn every_window_replays_wide_scalars_independently() {
        use num_bigint::BigUint;

        let mut scalars = Vec::new();
        for bits in [1usize, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 1024] {
            let power = BigUint::from(1u8) << bits;
            scalars.extend([&power - 1u8, power.clone(), &power + 1u8]);
        }
        // Fixed recurrence gives reproducible mixed patterns without a RNG dependency.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for length in [3, 4, 8, 16] {
            let mut bytes = Vec::new();
            for _ in 0..length {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                bytes.extend_from_slice(&state.to_le_bytes());
            }
            scalars.push(BigUint::from_bytes_le(&bytes));
        }
        for scalar in scalars {
            let limbs = scalar.to_u64_digits();
            for width in 1..=6 {
                let schedule = plan_with_width(&limbs, scalar.bits() as usize, width);
                let mut value = BigUint::from(2 * schedule.first + 1);
                assert!(schedule.first <= schedule.max_odd_index);
                for step in schedule.steps {
                    match step {
                        Step::Double => value <<= 1usize,
                        Step::AddOdd(index) => {
                            assert!(index <= schedule.max_odd_index);
                            value += BigUint::from(2 * index + 1);
                        }
                    }
                }
                assert_eq!(value, scalar, "window width {width}");
            }
        }
    }

    #[test]
    fn zero_has_no_chain() {
        assert!(plan(&[]).is_none());
        assert!(plan(&[0, 0]).is_none());
    }
}

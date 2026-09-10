//! Host-only sliding-window planning, independent of Rust syntax and types.
//!
//! An odd-table entry represents `(2 * index + 1)` copies of the input value.

#[derive(Clone, Copy, Debug)]
pub(super) enum Step {
    Double,
    AddOdd(usize),
}

pub(super) struct Schedule {
    pub(super) first: usize,
    pub(super) steps: Vec<Step>,
    pub(super) max_odd_index: usize,
}

impl Schedule {
    /// Count additions and doublings equally, including table preparation.
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

/// Compare widths six through one; the first minimum wins, so ties prefer
/// wider windows. This is a heuristic, not a shortest-chain algorithm.
pub(super) fn plan(limbs: &[u64]) -> Option<Schedule> {
    let top_index = limbs.iter().rposition(|limb| *limb != 0)?;
    let bits = top_index * 64 + (64 - limbs[top_index].leading_zeros() as usize);
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
    fn zero_has_no_chain() {
        assert!(plan(&[]).is_none());
        assert!(plan(&[0, 0]).is_none());
    }
}

use super::{PastaField, PrimeModulus};

pub(crate) fn fill_powers<M: PrimeModulus>(
    first: PastaField<M>,
    step: PastaField<M>,
    values: &mut [PastaField<M>],
) {
    if values.len() < 8 {
        if let Some((head, tail)) = values.split_first_mut() {
            let mut power = first;
            *head = power;
            for value in tail {
                power = power.mul(&step);
                *value = power;
            }
        }
        return;
    }

    // Even and odd powers advance independently by step^2, so one product
    // need not wait for the preceding entry's multiplication to finish.
    let stride = step.square();
    let mut even = first;
    let mut odd = first.mul(&step);
    let (head, tail) = values.split_at_mut(2);
    head.copy_from_slice(&[even, odd]);
    let mut pairs = tail.chunks_exact_mut(2);
    for pair in pairs.by_ref() {
        even = even.mul(&stride);
        odd = odd.mul(&stride);
        pair.copy_from_slice(&[even, odd]);
    }
    if let [last] = pairs.into_remainder() {
        *last = even.mul(&stride);
    }
}

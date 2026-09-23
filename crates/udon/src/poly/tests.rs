//! Polynomial utilities against their quadratic definitions: naive
//! convolution, direct evaluation, and written-out sums.

use std::vec::Vec;

use super::{divide_by_root, divide_by_root_iter, dot, evaluate, geometric_sum, multiply};
use crate::field::{Fp, PallasBase, PallasScalar, PastaField, PrimeModulus};

/// A deterministic spread of distinct nonzero field elements.
fn elements<M: PrimeModulus>(count: usize) -> Vec<PastaField<M>> {
    let mut current = PastaField::<M>::DELTA;
    (0..count)
        .map(|index| {
            current = current.mul(&PastaField::<M>::DELTA);
            current.add(&PastaField::<M>::from_u64(index as u64 + 1))
        })
        .collect()
}

fn naive_convolution<M: PrimeModulus>(
    a: &[PastaField<M>],
    b: &[PastaField<M>],
) -> Vec<PastaField<M>> {
    let mut product = std::vec![PastaField::ZERO; a.len() + b.len() - 1];
    for (i, lhs) in a.iter().enumerate() {
        for (j, rhs) in b.iter().enumerate() {
            product[i + j] = product[i + j].add(&lhs.mul(rhs));
        }
    }
    product
}

#[test]
fn evaluate_agrees_with_dot_against_powers() {
    fn check<M: PrimeModulus>() {
        for count in [1, 2, 7, 31] {
            let coefficients = elements::<M>(count);
            for x in elements::<M>(3) {
                let mut powers = Vec::with_capacity(count);
                let mut power = PastaField::ONE;
                for _ in 0..count {
                    powers.push(power);
                    power = power.mul(&x);
                }
                assert_eq!(
                    dot(powers.iter(), coefficients.iter()),
                    evaluate(&coefficients, x)
                );
            }
        }
        assert_eq!(
            evaluate::<PastaField<M>, _>(&[], PastaField::DELTA),
            PastaField::ZERO
        );
    }

    check::<PallasBase>();
    check::<PallasScalar>();
}

/// `p(y) = q(y) * (y - root) + p(root)` for the quotient `q`, whether or not
/// `root` divides `p`, and both quotient forms agree.
#[test]
fn divide_by_root_satisfies_the_quotient_identity() {
    fn check<M: PrimeModulus>() {
        for degree in [1, 2, 5, 15] {
            let p = elements::<M>(degree + 1);
            for root in elements::<M>(2) {
                let mut quotient = std::vec![PastaField::ZERO; degree];
                divide_by_root(&p, root, &mut quotient);
                let mut from_iterator: Vec<PastaField<M>> =
                    divide_by_root_iter(p.iter().copied(), root).collect();
                from_iterator.reverse();
                assert_eq!(from_iterator, quotient);
                for y in elements::<M>(3) {
                    let expected = evaluate(&quotient, y)
                        .mul(&y.sub(&root))
                        .add(&evaluate(&p, root));
                    assert_eq!(evaluate(&p, y), expected);
                }
            }
        }
    }

    check::<PallasBase>();
    check::<PallasScalar>();
}

/// A constant has no linear factor: the quotient is empty and the constant is
/// the remainder.
#[test]
fn dividing_a_constant_yields_an_empty_quotient() {
    divide_by_root(&[<Fp>::from_u64(7)], <Fp>::from_u64(3), &mut []);
    assert_eq!(
        divide_by_root_iter([<Fp>::from_u64(7)], <Fp>::from_u64(3)).count(),
        0
    );
}

#[test]
#[should_panic(expected = "without coefficients")]
fn dividing_nothing_panics() {
    divide_by_root::<Fp>(&[], <Fp>::from_u64(3), &mut []);
}

#[test]
#[should_panic(expected = "quotient length")]
fn dividing_into_the_wrong_length_panics() {
    let p = [<Fp>::ONE, <Fp>::ONE, <Fp>::ONE];
    divide_by_root(&p, <Fp>::from_u64(3), &mut [<Fp>::ZERO; 3]);
}

#[test]
fn geometric_sum_matches_the_written_out_sum() {
    fn check<M: PrimeModulus>() {
        for ratio in [
            PastaField::<M>::ZERO,
            PastaField::ONE,
            PastaField::from_u64(3),
            PastaField::DELTA,
        ] {
            for terms in 0..70 {
                let mut naive = PastaField::ZERO;
                let mut power = PastaField::ONE;
                for _ in 0..terms {
                    naive = naive.add(&power);
                    power = power.mul(&ratio);
                }
                assert_eq!(
                    geometric_sum(ratio, terms),
                    naive,
                    "ratio {ratio:?}, {terms} terms"
                );
            }
        }
    }

    check::<PallasBase>();
    check::<PallasScalar>();
}

/// Products agree with naive convolution around algorithm crossovers and
/// domain-size changes, with full, short, and empty transform scratch.
#[test]
fn multiply_agrees_with_naive_convolution_on_both_paths() {
    fn check<M: PrimeModulus>() {
        let mut scratch = std::vec![PastaField::<M>::ZERO; 512];
        for (left, right) in [
            (1, 1),
            (2, 3),
            (17, 17),
            (33, 12),
            (54, 54),
            (55, 55),
            (64, 64),
            (65, 65),
            (81, 81),
            (82, 82),
            (100, 3),
            (3, 100),
            (129, 128),
        ] {
            let a = elements::<M>(left);
            let b = elements::<M>(right);
            let expected = naive_convolution(&a, &b);
            let mut product = std::vec![PastaField::ZERO; left + right - 1];
            multiply(&a, &b, &mut product, &mut scratch);
            assert_eq!(product, expected, "lengths {left} and {right}");
            // Insufficient scratch must select the same exact convolution.
            let required = 2 * product.len().next_power_of_two();
            for capacity in [0, required - 1] {
                product.fill(PastaField::DELTA);
                multiply(&a, &b, &mut product, &mut scratch[..capacity]);
                assert_eq!(
                    product, expected,
                    "lengths {left} and {right}, scratch {capacity}"
                );
            }
        }

        let a = elements::<M>(5);
        let mut product = std::vec![PastaField::ZERO; 5];
        multiply(&a, &[PastaField::ONE], &mut product, &mut scratch);
        assert_eq!(product, a);
        multiply(&[PastaField::ONE], &a, &mut product, &mut scratch);
        assert_eq!(product, a);

        multiply::<PastaField<M>>(&[], &a, &mut [], &mut scratch);
        multiply::<PastaField<M>>(&a, &[], &mut [], &mut scratch);
        multiply::<PastaField<M>>(&[], &[], &mut [], &mut scratch);
    }

    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
fn large_thin_products_keep_the_schoolbook_path() {
    fn check<M: PrimeModulus>() {
        let mut scratch = std::vec![PastaField::<M>::DELTA; 2 * 65536];
        for (left, right) in [(32768, 33), (33, 32768)] {
            let a = std::vec![PastaField::ONE; left];
            let b = std::vec![PastaField::ONE; right];
            let mut product = std::vec![PastaField::DELTA; left + right - 1];
            multiply(&a, &b, &mut product, &mut scratch);

            // All-one coefficients give the integer count of contributing
            // pairs, independent of either field multiplication algorithm.
            for (index, actual) in product.iter().enumerate() {
                let count = (index + 1).min(left).min(right).min(product.len() - index);
                assert_eq!(*actual, PastaField::from_u64(count as u64));
            }
            // There is enough scratch for three transforms, but these thin
            // inputs are cheaper without them. The former estimate used it.
            assert!(scratch.iter().all(|value| *value == PastaField::DELTA));
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
#[should_panic(expected = "product length")]
fn multiply_rejects_the_wrong_product_length() {
    let a = [<Fp>::ONE, <Fp>::ONE];
    multiply(&a, &a, &mut [<Fp>::ZERO; 2], &mut []);
}

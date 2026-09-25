use crate::fft::execution::count_expansions;
use crate::field::pasta::test_support::field_samples;
use crate::field::{FftField, Fp, PallasBase, PallasScalar, PastaField, PrimeModulus};
use num_bigint::BigUint;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::vec::Vec;

fn elements<M: PrimeModulus>(count: usize) -> Vec<PastaField<M>> {
    field_samples::<M>().take(count).collect()
}

use super::{multiply, multiply_default};

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

/// Products agree with naive convolution around algorithm crossovers and
/// domain-size changes, with full, short, and empty transform scratch.
#[test]
fn multiply_agrees_with_naive_convolution_on_both_paths() {
    fn check<M: PrimeModulus>() {
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
            (128, 129),
            (130, 129),
            (65, 256),
            (256, 65),
        ] {
            let a = elements::<M>(left);
            let b = elements::<M>(right).into_iter().rev().collect::<Vec<_>>();
            let expected = naive_convolution(&a, &b);
            let mut product = std::vec![PastaField::ZERO; left + right - 1];
            let required = 2 * product.len().next_power_of_two();
            let mut scratch = std::vec![PastaField::<M>::DELTA; required + 3];
            for capacity in [0, required - 1, required, required + 3] {
                product.fill(PastaField::DELTA);
                scratch.fill(PastaField::DELTA);
                multiply(&a, &b, &mut product, &mut scratch[..capacity]);
                assert_eq!(
                    product, expected,
                    "lengths {left} and {right}, scratch {capacity}"
                );
                assert!(
                    scratch[required..]
                        .iter()
                        .all(|value| *value == PastaField::DELTA)
                );
            }

            // The default remains available to other field implementations.
            multiply_default(&a, &b, &mut product, &mut scratch);
            assert_eq!(product, expected);
        }

        let mut scratch = std::vec![PastaField::<M>::DELTA; 32];
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
fn pasta_products_dispatch_to_prefix_expansions_with_a_fused_product() {
    fn check<M: PrimeModulus>() {
        for (left, right) in [(64usize, 64usize), (129, 128), (65, 256), (256, 65)] {
            let a = elements::<M>(left);
            let b = elements::<M>(right);
            let mut product = std::vec![PastaField::ZERO; left + right - 1];
            let size = product.len().next_power_of_two();
            let mut scratch = std::vec![PastaField::DELTA; 2 * size];
            let (calls, residues, products) = count_expansions(|| {
                multiply(&a, &b, &mut product, &mut scratch);
            });
            assert_eq!(calls, 2);
            assert_eq!(
                residues,
                size / left.next_power_of_two() + size / right.next_power_of_two()
            );
            assert_eq!(products, 1);
            assert_eq!(product, naive_convolution(&a, &b));

            // Correct results alone cannot distinguish the previous strategy.
            assert_eq!(
                count_expansions(|| {
                    multiply_default(&a, &b, &mut product, &mut scratch);
                }),
                (0, 0, 0)
            );
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
fn expansion_products_match_integer_convolution() {
    fn check<M: PrimeModulus>() {
        let a = elements::<M>(129);
        let b = elements::<M>(128).into_iter().rev().collect::<Vec<_>>();
        let integers = |values: &[PastaField<M>]| {
            values
                .iter()
                .map(|value| BigUint::from_bytes_le(&value.to_bytes()))
                .collect::<Vec<_>>()
        };
        let mut expected = std::vec![BigUint::from(0u8); a.len() + b.len() - 1];
        let left = integers(&a);
        let right = integers(&b);
        for (i, lhs) in left.iter().enumerate() {
            for (j, rhs) in right.iter().enumerate() {
                expected[i + j] += lhs * rhs;
            }
        }
        let modulus = BigUint::from_bytes_le(&M::MODULUS.map(u64::to_le_bytes).concat());
        let mut product = std::vec![PastaField::DELTA; expected.len()];
        let mut scratch = std::vec![PastaField::DELTA; 2 * product.len().next_power_of_two()];
        multiply(&a, &b, &mut product, &mut scratch);
        for (actual, expected) in product.iter().zip(expected) {
            assert_eq!(
                BigUint::from_bytes_le(&actual.to_bytes()),
                expected % &modulus
            );
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
fn large_expansion_products_match_integer_pair_counts() {
    fn check<M: PrimeModulus>() {
        for (left, right) in [(4097usize, 4096usize), (2049, 8192), (32768, 32768)] {
            let a = std::vec![PastaField::<M>::ONE; left];
            let b = std::vec![PastaField::ONE; right];
            let mut product = std::vec![PastaField::DELTA; left + right - 1];
            let mut scratch = std::vec![PastaField::DELTA; 2 * product.len().next_power_of_two()];
            multiply(&a, &b, &mut product, &mut scratch);
            for (index, actual) in product.iter().enumerate() {
                let count = (index + 1).min(left).min(right).min(product.len() - index);
                assert_eq!(*actual, PastaField::from_u64(count as u64));
            }
        }
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
fn multiply_rejects_the_wrong_product_length_before_writes() {
    let a = [<Fp>::ONE, <Fp>::ONE];
    for multiply in [multiply::<Fp>, <Fp as FftField>::multiply_polynomials] {
        for left in [&a[..], &[]] {
            let mut product = [<Fp>::DELTA; 2];
            let mut scratch = [<Fp>::ONE; 8];
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    multiply(left, &a, &mut product, &mut scratch);
                }))
                .is_err()
            );
            assert_eq!(product, [Fp::DELTA; 2]);
            assert_eq!(scratch, [Fp::ONE; 8]);
        }
    }
}

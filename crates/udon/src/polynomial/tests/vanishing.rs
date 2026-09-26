use super::*;
use crate::field::pasta::test_support::{field_samples, integer, modulus};
use crate::field::{PallasBase, PallasScalar};
use crate::polynomial::{divide_linear_in_place, divide_monic_in_place};
use num_bigint::BigUint;
use std::{vec, vec::Vec};

fn from_raw<M: PrimeModulus>(raw: &BigUint) -> PastaField<M> {
    let digits = raw.to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    PastaField::from_montgomery_limbs(limbs)
}

fn canonical<M: PrimeModulus, S: ReductionState>(values: &[PastaField<M, S>]) -> Vec<BigUint> {
    let p = modulus::<M>();
    let inverse_r = (BigUint::from(1u8) << 256usize).modinv(&p).unwrap();
    values
        .iter()
        .map(|value| {
            let raw = integer(&value.montgomery_limbs());
            assert!(raw < &p * 2u8);
            raw * &inverse_r % &p
        })
        .collect()
}

fn multiply(a: &[BigUint], b: &[BigUint], p: &BigUint) -> Vec<BigUint> {
    let mut product = vec![BigUint::from(0u8); a.len() + b.len() - 1];
    for (i, a) in a.iter().enumerate() {
        for (j, b) in b.iter().enumerate() {
            product[i + j] = (&product[i + j] + a * b) % p;
        }
    }
    product
}

fn check_roots<M: PrimeModulus, S: ReductionState>(roots: &[PastaField<M, S>]) {
    let p = modulus::<M>();
    let roots_integer = canonical(roots);
    let sentinel = from_raw::<M>(&(&p * 2u8 - 1u8));
    let mut output = vec![sentinel; roots.len() + 4];
    let extent = vanishing_polynomial(roots, &mut output).unwrap();
    assert_eq!(extent, roots.len() + 1);
    assert!(
        output[extent..]
            .iter()
            .all(|v| v.montgomery_limbs() == sentinel.montgomery_limbs())
    );
    let actual = canonical(&output[..extent]);
    assert_eq!(actual.last().unwrap(), &BigUint::from(1u8));

    // Build a balanced product tree using out-of-place integer convolution.
    let mut factors: Vec<_> = roots_integer
        .iter()
        .map(|r| vec![(&p - r) % &p, BigUint::from(1u8)])
        .collect();
    if factors.is_empty() {
        factors.push(vec![BigUint::from(1u8)]);
    }
    while factors.len() > 1 {
        factors = factors
            .chunks(2)
            .map(|pair| {
                if pair.len() == 1 {
                    pair[0].clone()
                } else {
                    multiply(&pair[0], &pair[1], &p)
                }
            })
            .collect();
    }
    assert_eq!(actual, factors[0]);
    for root in &roots_integer {
        let mut power = BigUint::from(1u8);
        let mut sum = BigUint::from(0u8);
        for coefficient in &actual {
            sum = (sum + coefficient * &power) % &p;
            power = power * root % &p;
        }
        assert_eq!(sum, BigUint::from(0u8));
    }
    // Repeated linear factors must divide out with zero remainder, preserving
    // multiplicities; dividing by the complete prepared polynomial gives one.
    let mut divided = output[..extent].to_vec();
    let split = divide_monic_in_place(&mut divided, &output[..extent]).unwrap();
    assert!(divided[..split].iter().all(PastaField::is_zero));
    assert!(divided[split].is_one());
    let mut offset = 0;
    for root in roots {
        let split = divide_linear_in_place(&mut output[offset..extent], root);
        assert_eq!(split, 1);
        assert!(output[offset].is_zero());
        offset += split;
    }
    assert_eq!(extent - offset, 1);
    assert!(output[offset].is_one());
}

fn vanishing_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let mut roots: Vec<_> = field_samples::<M>().take(33).collect();
    roots[..6].copy_from_slice(&[
        PastaField::ZERO,
        from_raw(&p),
        PastaField::ONE,
        PastaField::<M>::ONE.neg(),
        from_raw(&(&p * 2u8 - 1u8)),
        from_raw(&(&p + 1u8)),
    ]);
    for length in [0, 1, 2, 3, 4, 8, 17, 33] {
        check_roots(&roots[..length]);
        check_roots(
            &roots[..length]
                .iter()
                .map(|r| r.reduce())
                .collect::<Vec<_>>(),
        );
        for root in &roots[..6] {
            check_roots(&vec![*root; length]);
        }
    }
}

#[test]
fn vanishing_coefficients_match_integer_product_tree() {
    vanishing_field::<PallasBase>();
    vanishing_field::<PallasScalar>();
}

fn capacity_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let original = [from_raw::<M>(&(&p * 2u8 - 1u8)); 8];
    let mut output = original;
    for root_count in [0, 1, 2, 3, 7] {
        let roots = vec![PastaField::<M>::ONE; root_count];
        for capacity in 0..=root_count {
            assert_eq!(
                vanishing_polynomial(&roots, &mut output[..capacity]),
                Err(VanishingError::OutputTooShort {
                    required: root_count + 1,
                    actual: capacity
                })
            );
            assert_eq!(
                output.map(|v| v.montgomery_limbs()),
                original.map(|v| v.montgomery_limbs())
            );
        }
        let mut exact = vec![original[0]; root_count + 1];
        assert_eq!(vanishing_polynomial(&roots, &mut exact), Ok(root_count + 1));
    }
}

#[test]
fn insufficient_capacity_preserves_storage() {
    assert_eq!(vanishing_extent(0), Ok(1));
    assert_eq!(vanishing_extent(usize::MAX - 1), Ok(usize::MAX));
    assert_eq!(
        vanishing_extent(usize::MAX),
        Err(VanishingError::SizeOverflow)
    );
    capacity_field::<PallasBase>();
    capacity_field::<PallasScalar>();
}

use super::*;
use crate::fft::Domain;
use crate::fft::tests::{direct, inputs, inverse_direct, ordered, reduced};
use crate::field::{PallasBase, PallasScalar};

fn interpreted_codelets<M: PrimeModulus>() {
    for steps in [&RADIX4[..], &RADIX8[..]] {
        let size = if steps.len() == 4 { 4 } else { 8 };
        let domain = Domain::<PastaField<M>>::for_size(size).unwrap().subgroup();
        let input = inputs(size);
        for inverse in [false, true] {
            for dif in [false, true] {
                let mut interpreted = if dif {
                    input.clone()
                } else {
                    ordered(&input, ElementOrder::BitReversed)
                };
                for index in 0..steps.len() {
                    let step = steps[if dif { steps.len() - 1 - index } else { index }];
                    assert_eq!(step.right - step.left, step.block / 2);
                    assert!(step.exponent < step.block / 2 && step.right < size);
                    let root = if inverse {
                        domain.domain().inverse_root()
                    } else {
                        domain.domain().root()
                    };
                    let power = root.pow_u64((size / step.block * step.exponent) as u64);
                    let left = interpreted[step.left];
                    let right = interpreted[step.right];
                    if dif {
                        interpreted[step.left] = left.add(&right);
                        interpreted[step.right] = left.sub(&right).mul(&power);
                    } else {
                        let product = right.mul(&power);
                        interpreted[step.left] = left.add(&product);
                        interpreted[step.right] = left.sub(&product);
                    }
                }
                let expected = if inverse {
                    inverse_direct(&input, domain, false)
                } else {
                    direct(&input, domain)
                };
                assert_eq!(
                    reduced(&interpreted),
                    reduced(
                        &(if dif {
                            ordered(&expected, ElementOrder::BitReversed)
                        } else {
                            expected
                        })
                    )
                );
            }
        }
    }
}

#[test]
fn generated_codelet_schedule_interpreter_matches_direct_sums() {
    interpreted_codelets::<PallasBase>();
    interpreted_codelets::<PallasScalar>();
}

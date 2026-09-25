use super::*;
use crate::exec::SerialExecutor;
use crate::field::{Fp, PallasBase, PallasScalar, Reduced};
use crate::test_support::field_samples;
use bento::bytes_of_slice;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicUsize, Ordering},
    vec,
    vec::Vec,
};

mod composition;
mod constant_prefix;
mod contracts;
mod expansion_engine;
mod group;
mod lagrange;
mod operations;
mod pipelines;
mod vanishing;

use expansion_engine::ExpansionStrategy;

fn reduced<M: PrimeModulus>(values: &[PastaField<M>]) -> Vec<PastaField<M, Reduced>> {
    values.iter().map(|value| value.reduce()).collect()
}

struct Prepared<M: PrimeModulus> {
    forward: Vec<PastaField<M>>,
    inverse: Vec<PastaField<M>>,
    finish: Vec<PastaField<M>>,
}

impl<M: PrimeModulus> Prepared<M> {
    fn new(domain: CosetDomain<M>) -> Self {
        let requirements = TableRequirements::for_size(domain.size()).unwrap();
        assert_eq!(requirements, TableRequirements::for_domain(domain));
        let mut result = Self {
            forward: vec![PastaField::ZERO; requirements.twiddles],
            inverse: vec![PastaField::ZERO; requirements.twiddles],
            finish: vec![PastaField::ZERO; requirements.twiddles],
        };
        TablesMut {
            forward: Some(&mut result.forward),
            inverse: Some(&mut result.inverse),
            inverse_finish: Some(&mut result.finish),
        }
        .prepare(domain);
        result
    }

    fn tables(&self) -> Tables<'_, M> {
        Tables {
            forward: Some(&self.forward),
            inverse: Some(&self.inverse),
            inverse_finish: Some(&self.finish),
        }
    }
}

fn inputs<M: PrimeModulus>(size: usize) -> Vec<PastaField<M>> {
    let mut samples = field_samples();
    (0..size)
        .map(|index| match index % 17 {
            0 => PastaField::ZERO,
            1 => PastaField::<_>::ONE.neg(),
            _ => samples.next().unwrap(),
        })
        .collect()
}

fn evaluate<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    point: PastaField<M>,
) -> PastaField<M> {
    coefficients
        .iter()
        .rev()
        .fold(PastaField::ZERO, |acc, coefficient| {
            acc.mul(&point).add(coefficient)
        })
}

fn direct<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    domain: CosetDomain<M>,
) -> Vec<PastaField<M>> {
    let mut point = domain.shift();
    (0..domain.size())
        .map(|_| {
            let value = evaluate(coefficients, point);
            point = point.mul(&domain.domain().root());
            value
        })
        .collect()
}

fn reference_coset<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    domain: CosetDomain<M>,
) -> Vec<PastaField<M>> {
    let mut values = vec![PastaField::ZERO; domain.size()];
    let mut scale = PastaField::ONE;
    for (out, coefficient) in values.iter_mut().zip(coefficients) {
        *out = coefficient.mul(&scale);
        scale = scale.mul(&domain.shift());
    }
    reference::transform(&mut values, &domain.domain().root());
    values
}

fn assert_loose_bound<M: PrimeModulus>(values: &[PastaField<M>]) {
    for value in values {
        assert!(
            value
                .montgomery_limbs()
                .iter()
                .rev()
                .cmp(M::TWICE_MODULUS.iter().rev())
                .is_lt()
        );
        assert_eq!(
            (PastaField::<M>::from_bytes(value.to_bytes())).map(|value| value.reduce()),
            (Some(*value)).map(|value| value.reduce())
        );
    }
}

fn small_transforms<M: PrimeModulus>() {
    // Debug must be available with only the public field-modulus bound.
    fn assert_debug<T: core::fmt::Debug>() {}
    assert_debug::<Domain<M>>();
    assert_debug::<CosetDomain<M>>();
    assert_debug::<Transform<'_, M>>();
    assert_debug::<Tables<'_, M>>();
    assert_debug::<TablesMut<'_, M>>();
    assert_debug::<Expansion<'_, M>>();
    assert_debug::<Class<'_, M>>();

    const OPTIONS: [Strategy; 3] = [
        Strategy::SERIAL,
        Strategy {
            tile_len: 1,
            columns_per_task: 3,
            max_tasks: 3,
        },
        Strategy {
            tile_len: 4,
            columns_per_task: 3,
            max_tasks: 5,
        },
    ];
    const SCRATCH: [[usize; OPTIONS.len()]; 7] = {
        let mut counts = [[0; OPTIONS.len()]; 7];
        let mut log = 0;
        while log < counts.len() {
            let mut index = 0;
            while index < OPTIONS.len() {
                counts[log][index] = match OPTIONS[index].requirements(1 << log) {
                    Ok(required) => required,
                    Err(_) => panic!("unsupported test configuration"),
                };
                index += 1;
            }
            log += 1;
        }
        counts
    };
    for log in 0..=6 {
        let subgroup = Domain::<M>::new(log).unwrap();
        let by_size = Domain::<M>::for_size(1 << log).unwrap();
        assert_eq!(by_size.size(), subgroup.size());
        assert_eq!(by_size.log_size(), log);
        assert_eq!((by_size.root()).reduce(), (subgroup.root()).reduce());
        assert_eq!(
            (by_size.inverse_root()).reduce(),
            (subgroup.inverse_root()).reduce()
        );
        assert_eq!(
            (by_size.size_inverse()).reduce(),
            (subgroup.size_inverse()).reduce()
        );
        let coefficients = inputs(subgroup.size());
        for coset in [false, true] {
            let domain = if coset {
                subgroup.coset()
            } else {
                subgroup.subgroup()
            };
            let shift: PastaField<M> = if coset {
                PastaField::ZETA
            } else {
                PastaField::ONE
            };
            assert_eq!(domain.shift().reduce(), shift.reduce());
            assert_eq!(
                domain.inverse_shift().reduce(),
                shift.invert().unwrap().reduce()
            );
            let prepared = Prepared::new(domain);
            let expected = direct(&coefficients, domain);
            // Every table is independently optional, including combinations
            // where final scales are prepared but ordinary twiddles are not.
            for mask in 0..8 {
                let tables = Tables {
                    forward: (mask & 1 != 0).then_some(prepared.forward.as_slice()),
                    inverse: (mask & 2 != 0).then_some(prepared.inverse.as_slice()),
                    inverse_finish: (mask & 4 != 0).then_some(prepared.finish.as_slice()),
                };
                let plan = tables.bind(domain);
                for (index, options) in OPTIONS.into_iter().enumerate() {
                    let count = SCRATCH[log as usize][index];
                    assert_eq!(plan.scratch_requirements_with(options).unwrap(), count);
                    let sentinel = PastaField::from_u64(99);
                    let mut scratch = vec![sentinel; count + 3];
                    let mut output = vec![PastaField::ZERO; coefficients.len()];
                    plan.forward_into_with(
                        &coefficients,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(reduced(&output), reduced(&expected), "forward log={log}");
                    assert_loose_bound(&output);
                    plan.inverse_with(&mut output, options, &SerialExecutor, &mut scratch)
                        .unwrap();
                    assert_eq!(
                        reduced(&output),
                        reduced(&coefficients),
                        "inverse log={log}"
                    );
                    let evaluations = expected.clone();
                    plan.inverse_into_with(
                        &evaluations,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(
                        reduced(&output),
                        reduced(&coefficients),
                        "inverse_into log={log}"
                    );
                    assert_eq!(reduced(&evaluations), reduced(&expected));
                    // Construct the input order independently of Transform::permute.
                    for (row, value) in expected.iter().enumerate() {
                        let bits = (0..log).fold(0, |bits, bit| (bits << 1) | ((row >> bit) & 1));
                        output[bits] = *value;
                    }
                    plan.inverse_bit_reversed_with(
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(
                        reduced(&output),
                        reduced(&coefficients),
                        "inverse_bit_reversed log={log}"
                    );
                    assert_eq!(
                        bytes_of_slice(&scratch[count..]),
                        bytes_of_slice(&[sentinel; 3])
                    );
                    assert_loose_bound(&scratch);
                }
            }
        }
    }
}

#[test]
fn small_dfts_both_fields_all_table_modes() {
    small_transforms::<PallasBase>();
    small_transforms::<PallasScalar>();
}

fn prefixes<M: PrimeModulus>() {
    for log in 0..=8 {
        let domain = Domain::<M>::new(log).unwrap().coset();
        let prepared = Prepared::new(domain);
        let input = inputs(domain.size());
        for table in [Tables::default(), prepared.tables()] {
            let plan = table.bind(domain);
            for len in [
                0,
                1,
                2,
                3,
                5,
                domain.size() / 2,
                domain.size() - 1,
                domain.size(),
            ]
            .into_iter()
            .filter(|&len| len <= domain.size())
            {
                let expected = reference_coset(&input[..len], domain);
                for tile_len in [1, 4, 16, 256] {
                    let options = Strategy {
                        tile_len,
                        columns_per_task: 3,
                        max_tasks: 3,
                    };
                    let mut scratch =
                        vec![PastaField::ZERO; plan.scratch_requirements_with(options).unwrap()];
                    let mut output = vec![PastaField::ONE; domain.size()];
                    plan.forward_prefix_with(
                        &input[..len],
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(
                        reduced(&output),
                        reduced(&expected),
                        "prefix log={log} len={len} tile={tile_len}"
                    );
                }
            }
        }
    }
}

#[test]
fn zero_suffix_skips_rounds_across_tile_boundaries() {
    prefixes::<PallasBase>();
    prefixes::<PallasScalar>();
}

struct Threads;
impl Executor for Threads {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        std::thread::scope(|scope| {
            let left = scope.spawn(left);
            let right = right();
            (
                left.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic)),
                right,
            )
        })
    }
}

fn source_sizes<M: PrimeModulus>() {
    for log in 11..=14 {
        let domain = Domain::<M>::new(log).unwrap().coset();
        let prepared = Prepared::new(domain);
        let plan = prepared.tables().bind(domain);
        let coefficients = inputs(domain.size());
        let expected = reference_coset(&coefficients, domain);
        let options = Strategy {
            tile_len: 2048,
            columns_per_task: 257,
            max_tasks: 3,
        };
        let mut scratch = vec![PastaField::ZERO; plan.scratch_requirements_with(options).unwrap()];
        let mut output = coefficients.clone();
        plan.forward_with(&mut output, options, &Threads, &mut scratch)
            .unwrap();
        assert_eq!(reduced(&output), reduced(&expected));
        plan.inverse_with(&mut output, options, &Threads, &mut scratch)
            .unwrap();
        assert_eq!(reduced(&output), reduced(&coefficients));
    }
}

#[test]
fn original_domain_sizes_with_scoped_parallel_execution() {
    source_sizes::<PallasBase>();
    source_sizes::<PallasScalar>();
}

fn larger_transform<M: PrimeModulus>() {
    // Verify the complete root ladder without allocating enormous transforms.
    for log in 1..=32 {
        let root = PastaField::<M>::root_of_unity(log).unwrap();
        let inverse = PastaField::<M>::root_of_unity_inverse(log).unwrap();
        assert_eq!(
            (root.square()).reduce(),
            (PastaField::<_>::root_of_unity(log - 1).unwrap()).reduce()
        );
        assert_eq!(
            (inverse.square()).reduce(),
            (PastaField::<_>::root_of_unity_inverse(log - 1).unwrap()).reduce()
        );
        assert_eq!(
            (root.mul(&inverse)).reduce(),
            (PastaField::<_>::ONE).reduce()
        );
        assert_eq!(
            (root.pow_u64(1u64 << (log - 1))).reduce(),
            (PastaField::<_>::ONE.neg()).reduce()
        );
    }
    let subgroup = Domain::<M>::for_size(1 << 16).unwrap();
    assert_eq!(subgroup.log_size(), 16);
    let domain = subgroup.coset();
    let prepared = Prepared::new(domain);
    let plan = prepared.tables().bind(domain);
    let coefficients = inputs(domain.size());
    let expected = reference_coset(&coefficients, domain);
    let options = Strategy {
        max_tasks: 8,
        ..Strategy::default()
    };
    let mut scratch = vec![PastaField::ZERO; plan.scratch_requirements_with(options).unwrap()];
    let executor = CountJoins::default();
    let mut output = coefficients.clone();
    plan.forward_with(&mut output, options, &executor, &mut scratch)
        .unwrap();
    assert_eq!(reduced(&output), reduced(&expected));
    assert!(executor.0.load(Ordering::Relaxed) > 0);
    plan.inverse_with(&mut output, options, &executor, &mut scratch)
        .unwrap();
    assert_eq!(reduced(&output), reduced(&coefficients));

    let mut reference = coefficients.clone();
    reference::transform(&mut reference, &subgroup.root());
    reference::inverse_transform(
        &mut reference,
        &subgroup.inverse_root(),
        &subgroup.size_inverse(),
    );
    assert_eq!(reduced(&reference), reduced(&coefficients));
}

#[test]
fn larger_transforms_and_all_root_orders_match_reference() {
    larger_transform::<PallasBase>();
    larger_transform::<PallasScalar>();
}

fn expansions<M: PrimeModulus>() {
    for log in [0, 1, 3, 6] {
        let base_domain = Domain::<M>::new(log).unwrap();
        let base_tables = Prepared::new(base_domain.subgroup());
        let base = base_tables.tables().bind(base_domain.subgroup());
        let coefficients = inputs(base_domain.size());
        let evaluations = direct(&coefficients, base_domain.subgroup());
        for extra in [0, 1, 2, 3] {
            for coset in [false, true] {
                let domain = {
                    let subgroup = Domain::new(log + extra).unwrap();
                    if coset {
                        subgroup.coset()
                    } else {
                        subgroup.subgroup()
                    }
                };
                let mut scales =
                    vec![
                        PastaField::ZERO;
                        ExpansionScales::<M>::requirements(base.domain().size(), domain.size())
                            .unwrap()
                    ];
                let scales = ExpansionScales::prepare(
                    base.domain().size(),
                    domain,
                    ExpansionScaleNormalization::Coefficients,
                    &mut scales,
                )
                .unwrap();
                for scales in [None, Some(scales)] {
                    let expansion = Expansion::new(base, domain, scales).unwrap();
                    let expected = direct(&coefficients, domain);
                    for options in [
                        ExpansionStrategy::SERIAL,
                        ExpansionStrategy {
                            max_residue_tasks: 3,
                            ..ExpansionStrategy::SERIAL
                        },
                        ExpansionStrategy {
                            max_residue_tasks: 1,
                            transform: Strategy {
                                tile_len: 4,
                                columns_per_task: 3,
                                max_tasks: 3,
                            },
                        },
                        ExpansionStrategy {
                            max_residue_tasks: 3,
                            transform: Strategy {
                                tile_len: 4,
                                columns_per_task: 3,
                                max_tasks: 3,
                            },
                        },
                    ] {
                        let mut output = vec![PastaField::ZERO; domain.size()];
                        let count = expansion.coefficient_scratch_with(options).unwrap();
                        let mut scratch = vec![PastaField::ONE; count + 1];
                        expansion
                            .coefficients_with(
                                &coefficients,
                                &mut output,
                                options,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        assert_eq!((scratch[count]).reduce(), (PastaField::<_>::ONE).reduce());
                        let view = expansion.view(&output);
                        for (row, expected) in expected.iter().enumerate() {
                            assert_eq!(
                                (view.get(row)).map(|value| value.reduce()),
                                (Some(expected)).map(|value| value.reduce())
                            );
                        }
                        let count = expansion.evaluation_scratch_with(options).unwrap();
                        scratch.fill(PastaField::ONE);
                        expansion
                            .evaluations_with(
                                &evaluations,
                                &mut output,
                                options,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        assert!(
                            scratch[count..]
                                .iter()
                                .all(|value| value.reduce() == PastaField::ONE)
                        );
                        let view = expansion.view(&output);
                        for (row, expected) in expected.iter().enumerate() {
                            assert_eq!(
                                (view.get(row)).map(|value| value.reduce()),
                                (Some(expected)).map(|value| value.reduce())
                            );
                        }
                        for len in [1, coefficients.len().min(3), coefficients.len().min(10)] {
                            let short = &coefficients[..len];
                            let mut product = vec![PastaField::ZERO; domain.size()];
                            expansion
                                .short_product_with(
                                    short,
                                    view,
                                    &mut product,
                                    options,
                                    &SerialExecutor,
                                    &mut scratch,
                                )
                                .unwrap();
                            let short_values = direct(short, domain);
                            let product = expansion.view(&product);
                            for row in 0..domain.size() {
                                assert_eq!(
                                    (product.get(row)).map(|value| value.reduce()),
                                    (Some(&expected[row].mul(&short_values[row])))
                                        .map(|value| value.reduce())
                                );
                            }
                        }
                        expansion
                            .coefficients_with(
                                &[],
                                &mut output,
                                options,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        assert!(
                            output
                                .iter()
                                .all(|value| value.reduce() == PastaField::ZERO)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn residue_expansion_and_fused_short_products_match_direct_evaluation() {
    expansions::<PallasBase>();
    expansions::<PallasScalar>();
}

fn large_expansion<M: PrimeModulus>() {
    let base_domain = Domain::<M>::new(11).unwrap();
    let prepared = Prepared::new(base_domain.subgroup());
    let base = prepared.tables().bind(base_domain.subgroup());
    let coefficients = inputs(base_domain.size());
    for extra in [1, 3] {
        let domain = Domain::new(11 + extra).unwrap().coset();
        let expansion = Expansion::new(base, domain, None).unwrap();
        let expected = reference_coset(&coefficients, domain);
        let mut output = vec![PastaField::ZERO; domain.size()];
        let options = ExpansionStrategy {
            max_residue_tasks: 3,
            transform: Strategy {
                tile_len: 256,
                columns_per_task: 127,
                max_tasks: 3,
            },
        };
        let mut scratch =
            vec![PastaField::ZERO; expansion.coefficient_scratch_with(options).unwrap()];
        expansion
            .coefficients_with(&coefficients, &mut output, options, &Threads, &mut scratch)
            .unwrap();
        let mut natural = vec![PastaField::ZERO; domain.size()];
        expansion.layout().copy_to_natural(&output, &mut natural);
        assert_eq!(reduced(&natural), reduced(&expected));
        let evaluations = reference_coset(&coefficients, base_domain.subgroup());
        expansion
            .evaluations_with(&evaluations, &mut output, options, &Threads, &mut scratch)
            .unwrap();
        expansion.layout().copy_to_natural(&output, &mut natural);
        assert_eq!(reduced(&natural), reduced(&expected));
        let factor = expansion.view(&output);
        let mut product = vec![PastaField::ZERO; domain.size()];
        expansion
            .short_product_with(
                &coefficients[..5],
                factor,
                &mut product,
                options,
                &Threads,
                &mut scratch,
            )
            .unwrap();
        let short_values = direct(&coefficients[..5], domain);
        let product = expansion.view(&product);
        for row in 0..domain.size() {
            assert_eq!(
                (product.get(row)).map(|value| value.reduce()),
                (Some(&expected[row].mul(&short_values[row]))).map(|value| value.reduce())
            );
        }
    }
}

#[test]
fn original_two_and_eight_residue_expansions_match_zero_padded_fft() {
    large_expansion::<PallasBase>();
    large_expansion::<PallasScalar>();
}

#[derive(Default)]
struct CountJoins(AtomicUsize);

impl Executor for CountJoins {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        self.0.fetch_add(1, Ordering::SeqCst);
        SerialExecutor.join(left, right)
    }
}

impl CountJoins {
    fn take(&self) -> usize {
        self.0.swap(0, Ordering::SeqCst)
    }
}

struct FailAt {
    calls: AtomicUsize,
    index: usize,
}

impl Executor for FailAt {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        let fail = self.calls.fetch_add(1, Ordering::SeqCst) == self.index;
        let results = SerialExecutor.join(left, right);
        if fail {
            panic!("interrupt transform");
        }
        results
    }
}

#[test]
fn every_expansion_transform_uses_the_callers_executor_and_options() {
    let base = Transform::new(Domain::<PallasBase>::new(6).unwrap().subgroup());
    let coefficients = inputs(base.domain().size());
    let options = ExpansionStrategy {
        // With no joins across residues, every observed join is intra-residue.
        max_residue_tasks: 1,
        transform: Strategy {
            tile_len: 8,
            columns_per_task: 3,
            max_tasks: 4,
        },
    };
    let mut scratch = vec![Fp::ZERO; base.scratch_requirements_with(options.transform).unwrap()];
    let executor = CountJoins::default();
    let mut evaluations = coefficients.clone();
    base.forward_with(&mut evaluations, options.transform, &executor, &mut scratch)
        .unwrap();
    let forward_joins = executor.take();
    let mut inverse = evaluations.clone();
    base.inverse_with(&mut inverse, options.transform, &executor, &mut scratch)
        .unwrap();
    let inverse_joins = executor.take();
    assert!(forward_joins > 0 && inverse_joins > 0);

    for extra in [0, 1, 3] {
        let domain = Domain::new(6 + extra).unwrap().coset();
        let expansion = Expansion::new(base, domain, None).unwrap();
        let residues = expansion.layout().residues();
        let mut output = vec![Fp::ZERO; domain.size()];
        expansion
            .coefficients_with(&coefficients, &mut output, options, &executor, &mut scratch)
            .unwrap();
        assert!(executor.take() >= residues);
        let expected = output.clone();
        expansion
            .evaluations_with(&evaluations, &mut output, options, &executor, &mut scratch)
            .unwrap();
        assert!(executor.take() > residues);
        assert_eq!(reduced(&output), reduced(&expected));
        let ones = vec![Fp::ONE; domain.size()];
        let factor = expansion.view(&ones);
        expansion
            .short_product_with(
                &coefficients[..5],
                factor,
                &mut output,
                options,
                &executor,
                &mut scratch,
            )
            .unwrap();
        let pruned_joins = executor.take();
        assert!(pruned_joins > 0);
        let expected_short = direct(&coefficients[..5], domain);
        let view = expansion.view(&output);
        for (row, expected) in expected_short.iter().enumerate() {
            assert_eq!(
                (view.get(row)).map(|value| value.reduce()),
                (Some(expected)).map(|value| value.reduce())
            );
        }

        // Whole transforms still allow callers to parallelize only residues
        // without reserving scratch for tiled transforms.
        let options = ExpansionStrategy {
            max_residue_tasks: 3,
            ..ExpansionStrategy::SERIAL
        };
        assert_eq!(expansion.coefficient_scratch_with(options).unwrap(), 0);
        expansion
            .coefficients_with(&coefficients, &mut output, options, &executor, &mut [])
            .unwrap();
        assert_eq!(executor.take(), residues.min(3) - 1);
        assert_eq!(reduced(&output), reduced(&expected));
    }
}

fn prepared_evaluation_expansions<M: PrimeModulus>() {
    let subgroup = Domain::<M>::new(6).unwrap().subgroup();
    let prepared = Prepared::new(subgroup);
    let coefficients = inputs::<M>(subgroup.size());
    let evaluations = direct(&coefficients, subgroup);
    let saved = evaluations.clone();
    for extra in [0, 1, 3] {
        for coset in [false, true] {
            let extended = {
                let subgroup = Domain::new(6 + extra).unwrap();
                if coset {
                    subgroup.coset()
                } else {
                    subgroup.subgroup()
                }
            };
            let expected = reference_coset(&coefficients, extended);
            let mut scales = vec![PastaField::ZERO; extended.size()];
            let scales = ExpansionScales::prepare(
                subgroup.size(),
                extended,
                ExpansionScaleNormalization::Coefficients,
                &mut scales,
            )
            .unwrap();
            for mask in 0..8 {
                let tables = Tables {
                    forward: (mask & 1 != 0).then_some(prepared.forward.as_slice()),
                    inverse: (mask & 2 != 0).then_some(prepared.inverse.as_slice()),
                    inverse_finish: (mask & 4 != 0).then_some(prepared.finish.as_slice()),
                };
                let expansion =
                    Expansion::new(tables.bind(subgroup), extended, Some(scales)).unwrap();
                for transform in [
                    Strategy::SERIAL,
                    Strategy {
                        tile_len: 8,
                        columns_per_task: 3,
                        max_tasks: 3,
                    },
                ] {
                    let options = ExpansionStrategy {
                        max_residue_tasks: 3,
                        transform,
                    };
                    let count = expansion.evaluation_scratch_with(options).unwrap();
                    let mut scratch = vec![PastaField::ONE; count + 1];
                    let mut output = vec![PastaField::ZERO; extended.size()];
                    expansion
                        .evaluations_with(
                            &evaluations,
                            &mut output,
                            options,
                            &SerialExecutor,
                            &mut scratch,
                        )
                        .unwrap();
                    let view = expansion.view(&output);
                    for (row, expected) in expected.iter().enumerate() {
                        assert_eq!(
                            (view.get(row)).map(|value| value.reduce()),
                            (Some(expected)).map(|value| value.reduce())
                        );
                    }
                    assert_eq!(bytes_of_slice(&evaluations), bytes_of_slice(&saved));
                    assert_eq!((scratch[count]).reduce(), (PastaField::<_>::ONE).reduce());
                    assert_loose_bound(&output);
                    assert_loose_bound(&scratch);
                }
            }
        }
    }
}

#[test]
fn prepared_evaluation_expansion_supports_every_base_table_subset() {
    prepared_evaluation_expansions::<PallasBase>();
    prepared_evaluation_expansions::<PallasScalar>();
}

fn constant_prefixes<M: PrimeModulus>() {
    let base = Transform::new(Domain::<M>::new(6).unwrap().subgroup());
    let transform = Strategy {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 4,
    };
    let options = ExpansionStrategy {
        max_residue_tasks: 3,
        transform,
    };
    let executor = CountJoins::default();
    let constant = [PastaField::from_u64(19)];
    for extra in [0, 1, 3] {
        for coset in [false, true] {
            let domain = {
                let subgroup = Domain::new(6 + extra).unwrap();
                if coset {
                    subgroup.coset()
                } else {
                    subgroup.subgroup()
                }
            };
            let plan = Transform::new(domain);
            let expansion = Expansion::new(base, domain, None).unwrap();
            let required = expansion
                .coefficient_scratch_with(options)
                .unwrap()
                .max(plan.scratch_requirements_with(transform).unwrap());
            let mut scratch = vec![PastaField::ONE; required + 1];
            let mut output = vec![PastaField::ZERO; domain.size()];
            for coefficients in [&constant[..0], &constant[..]] {
                let expected = coefficients.first().copied().unwrap_or(PastaField::ZERO);
                for expanded in [false, true] {
                    if expanded {
                        expansion
                            .coefficients_with(
                                coefficients,
                                &mut output,
                                options,
                                &executor,
                                &mut scratch,
                            )
                            .unwrap();
                    } else {
                        plan.forward_prefix_with(
                            coefficients,
                            &mut output,
                            transform,
                            &executor,
                            &mut scratch,
                        )
                        .unwrap();
                    }
                    assert_eq!(executor.take(), 0);
                    assert!(
                        output
                            .iter()
                            .all(|&value| value.reduce() == expected.reduce())
                    );
                    assert!(
                        scratch
                            .iter()
                            .all(|&value| value.reduce() == PastaField::ONE)
                    );
                }
            }
            let factor = inputs::<M>(domain.size());
            expansion
                .short_product_with(
                    &constant,
                    expansion.view(&factor),
                    &mut output,
                    options,
                    &executor,
                    &mut scratch,
                )
                .unwrap();
            assert_eq!(
                executor.take(),
                expansion.layout().residues().min(options.max_residue_tasks) - 1
            );
            for (actual, factor) in output.iter().zip(&factor) {
                assert_eq!((*actual).reduce(), (constant[0].mul(factor)).reduce());
            }
            assert!(
                scratch
                    .iter()
                    .all(|&value| value.reduce() == PastaField::ONE)
            );
            if extra == 0 && !coset {
                let input = inputs::<M>(base.domain().size());
                expansion
                    .evaluations_with(&input, &mut output, options, &executor, &mut scratch)
                    .unwrap();
                assert_eq!(reduced(&output), reduced(&input));
                assert_eq!(executor.take(), 0);
                assert!(
                    scratch
                        .iter()
                        .all(|&value| value.reduce() == PastaField::ONE)
                );
            }
        }
    }
}

#[test]
fn constant_prefixes_and_subgroup_copies_skip_transform_scheduling() {
    constant_prefixes::<PallasBase>();
    constant_prefixes::<PallasScalar>();
}

#[test]
fn expansion_validates_options_and_partitioned_scratch_before_mutation() {
    let base = Transform::new(Domain::<PallasBase>::new(6).unwrap().subgroup());
    let input = inputs(base.domain().size());
    let transform = Strategy {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 3,
    };
    let per_transform = base.scratch_requirements_with(transform).unwrap();
    for extra in [0, 1, 3] {
        let expansion =
            Expansion::new(base, Domain::new(6 + extra).unwrap().subgroup(), None).unwrap();
        let mut output = vec![Fp::ONE; expansion.layout().size()];
        let factors = output.clone();
        let factor = expansion.view(&factors);
        for tasks in [1, 2, 3, usize::MAX] {
            let options = ExpansionStrategy {
                max_residue_tasks: tasks,
                transform,
            };
            let count = expansion.coefficient_scratch_with(options).unwrap();
            let eval_count = expansion.evaluation_scratch_with(options).unwrap();
            assert_eq!(
                count,
                per_transform * tasks.min(expansion.layout().residues())
            );
            assert_eq!(
                eval_count,
                per_transform * tasks.min(expansion.layout().residues() - 1).max(1)
            );
            let mut scratch = vec![Fp::ONE; count - 1];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = expansion.coefficients_with(
                        &input,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    );
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = expansion.coefficients_with(
                        &[],
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    );
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = expansion.short_product_with(
                        &input[..5],
                        factor,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    );
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = expansion.evaluations_with(
                        &input,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch[..eval_count - 1],
                    );
                }))
                .is_err()
            );
            assert_eq!(reduced(&output), reduced(&factors));
            assert!(scratch.iter().all(|value| value.reduce() == Fp::ONE));
        }
        let valid = ExpansionStrategy {
            max_residue_tasks: 2,
            transform,
        };
        for options in [
            ExpansionStrategy {
                max_residue_tasks: 0,
                ..valid
            },
            ExpansionStrategy {
                transform: Strategy {
                    tile_len: 3,
                    ..transform
                },
                ..valid
            },
            ExpansionStrategy {
                transform: Strategy {
                    columns_per_task: 0,
                    ..transform
                },
                ..valid
            },
            ExpansionStrategy {
                transform: Strategy {
                    max_tasks: 0,
                    ..transform
                },
                ..valid
            },
        ] {
            let mut scratch = vec![Fp::ONE; 512];
            assert_eq!(
                expansion.coefficient_scratch_with(options),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.evaluation_scratch_with(options),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.coefficients_with(
                    &input,
                    &mut output,
                    options,
                    &SerialExecutor,
                    &mut scratch
                ),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.evaluations_with(
                    &input,
                    &mut output,
                    options,
                    &SerialExecutor,
                    &mut scratch
                ),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.short_product_with(
                    &input[..5],
                    factor,
                    &mut output,
                    options,
                    &SerialExecutor,
                    &mut scratch
                ),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(reduced(&output), reduced(&factors));
            assert!(scratch.iter().all(|value| value.reduce() == Fp::ONE));
        }
    }
}

fn classed<M: PrimeModulus>(log: u32) {
    let domain = Domain::<M>::new(log).unwrap().coset();
    let smaller = Domain::<M>::new(log - 1).unwrap().coset();
    let smallest = Domain::<M>::new(log - 2).unwrap().subgroup();
    let full_coefficients = inputs(domain.size());
    let small_coefficients = inputs(smaller.size());
    let smallest_coefficients = inputs(smallest.size());
    let expected: Vec<_> = full_coefficients
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .add(
                    small_coefficients
                        .get(index)
                        .unwrap_or(&PastaField::<_>::ZERO),
                )
                .add(
                    smallest_coefficients
                        .get(index)
                        .unwrap_or(&PastaField::<_>::ZERO),
                )
        })
        .collect();
    let full_values = reference_coset(&full_coefficients, domain);
    let small_values = reference_coset(&small_coefficients, smaller);
    let smallest_values = reference_coset(&smallest_coefficients, smallest);
    let prepared = Prepared::new(domain);
    let small_prepared = Prepared::new(smaller);
    let plan = prepared.tables().bind(domain);
    let small_plan = small_prepared.tables().bind(smaller);
    for order in [ElementOrder::Natural, ElementOrder::BitReversed] {
        let layout = if order == ElementOrder::Natural {
            EvaluationLayout::Natural
        } else {
            EvaluationLayout::BitReversed
        };
        let mut full = vec![PastaField::ZERO; domain.size()];
        let mut small = vec![PastaField::ZERO; smaller.size()];
        // Callers can scatter chunks and strided rows directly into the declared
        // order before giving the class buffers to an interpolation plan.
        for (chunk, values) in full_values.chunks(7).enumerate() {
            for (offset, value) in values.iter().enumerate() {
                full[layout.index(chunk * 7 + offset, domain.size()).unwrap()] = *value;
            }
        }
        for start in 0..2 {
            for row in (start..small_values.len()).step_by(2) {
                small[layout.index(row, smaller.size()).unwrap()] = small_values[row];
            }
        }
        for tasks in [1, 3] {
            let mut values = [full.clone(), small.clone(), smallest_values.clone()];
            let transforms = [plan, small_plan, Transform::new(smallest)].map(|plan| {
                run::FftPlan::with_strategy(
                    plan,
                    TransformRequest {
                        input_order: if plan.domain().size() == smallest.size() {
                            ElementOrder::Natural
                        } else {
                            order
                        },
                        ..TransformRequest::new(Direction::Inverse)
                    },
                    core::num::NonZeroUsize::new(if log > 8 { 2048 } else { 4 }).unwrap(),
                    Codelet::Radix2,
                )
                .unwrap()
            });
            let plan = run::InterpolationPlan::with_transforms(transforms, false);
            let mut scratch: [_; 3] = core::array::from_fn(|i| {
                vec![PastaField::ONE; plan.snapshot_fields(i).unwrap() + 2]
            });
            plan.execute_with(
                values.each_mut().map(Vec::as_mut_slice),
                scratch.each_mut().map(Vec::as_mut_slice),
                core::num::NonZeroUsize::new(tasks).unwrap(),
                &Threads,
            );
            assert_eq!(reduced(&values[0]), reduced(&expected));
            assert_eq!(reduced(&values[1]), reduced(&small_coefficients));
            assert_eq!(reduced(&values[2]), reduced(&smallest_coefficients));
            for buffer in &scratch {
                assert_eq!(
                    bytes_of_slice(&buffer[buffer.len() - 2..]),
                    bytes_of_slice(&[PastaField::<M>::ONE; 2])
                );
                assert_loose_bound(buffer);
            }
        }
    }
}

#[test]
fn class_interpolation_fuses_coefficients_and_supports_strided_scatter() {
    classed::<PallasBase>(2);
    classed::<PallasScalar>(8);
    classed::<PallasBase>(14);
    classed::<PallasScalar>(14);
}

#[test]
fn layouts_and_subdomain_rows_are_distinct_from_coefficient_tiles() {
    for (size, count) in [(0, 1), (3, 1), (8, 0), (8, 3), (8, 16)] {
        assert_eq!(
            ResidueLayout::new(size, count),
            Err(FftError::InvalidLayout)
        );
    }
    for residues in [1, 2, 4, 8] {
        let layout = ResidueLayout::new(32, residues).unwrap();
        let input = inputs::<PallasBase>(32);
        let mut stored = [Fp::ZERO; 32];
        let mut natural = [Fp::ZERO; 32];
        layout.copy_from_natural(&input, &mut stored);
        layout.copy_to_natural(&stored, &mut natural);
        assert_eq!(reduced(&input), reduced(&natural));
        let view = EvaluationView::bind(
            &stored,
            Domain::for_size(32).unwrap().subgroup(),
            EvaluationLayout::Residues(layout),
        );
        for (row, expected) in input.iter().enumerate() {
            assert_eq!(
                (view.get(row)).map(|value| value.reduce()),
                (Some(expected)).map(|value| value.reduce())
            );
            assert_eq!(layout.natural_row(layout.index(row).unwrap()), Some(row));
            assert_eq!(
                (view.get_extended_row(row * 4, Domain::for_size(128).unwrap().subgroup()))
                    .map(|value| value.reduce()),
                (Some(expected)).map(|value| value.reduce())
            );
            assert_eq!(
                (view.get_extended_row(row * 4 + 1, Domain::for_size(128).unwrap().subgroup()))
                    .map(|value| value.reduce()),
                None
            );
        }
        assert_eq!((view.get(32)).map(|value| value.reduce()), None);
        assert!(
            view.get_extended_row(0, Domain::for_size(128).unwrap().coset())
                .is_none()
        );
        assert_eq!(
            (view.get_extended_row(0, Domain::for_size(16).unwrap().subgroup()))
                .map(|value| value.reduce()),
            None
        );
        for (input_len, output_len) in [(31, 32), (33, 32), (32, 31), (32, 33)] {
            let source = vec![Fp::ONE; input_len];
            let mut destination = vec![Fp::ZERO; output_len];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    layout.copy_from_natural(&source, &mut destination);
                }))
                .is_err()
            );
            assert_eq!(destination, vec![Fp::ZERO; output_len]);
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    layout.copy_to_natural(&source, &mut destination);
                }))
                .is_err()
            );
            assert_eq!(destination, vec![Fp::ZERO; output_len]);
        }
    }
}

#[test]
fn size_queries_validate_without_constructing_domains() {
    const _: () = {
        let serial = Strategy::SERIAL;
        let expansion = ExpansionStrategy::SERIAL;
        let tables = match TableRequirements::for_size(1) {
            Ok(required) => required,
            Err(_) => panic!("singleton rejected"),
        };
        assert!(tables.twiddles == 0);
        assert!(matches!(serial.requirements(1), Ok(0)));
        assert!(matches!(expansion.coefficient_requirements(1, 1), Ok(0)));
        assert!(matches!(expansion.evaluation_requirements(1, 1), Ok(0)));
        let invalid_sizes = [0, 3, usize::MAX];
        let mut index = 0;
        while index < invalid_sizes.len() {
            let size = invalid_sizes[index];
            assert!(matches!(
                TableRequirements::for_size(size),
                Err(FftError::InvalidSize)
            ));
            assert!(matches!(
                serial.requirements(size),
                Err(FftError::InvalidSize)
            ));
            assert!(matches!(
                expansion.coefficient_requirements(size, 16),
                Err(FftError::InvalidSize)
            ));
            assert!(matches!(
                expansion.evaluation_requirements(1, size),
                Err(FftError::InvalidSize)
            ));
            index += 1;
        }
        let invalid_options = [
            Strategy {
                tile_len: 0,
                ..serial
            },
            Strategy {
                tile_len: 3,
                ..serial
            },
            Strategy {
                columns_per_task: 0,
                ..serial
            },
            Strategy {
                max_tasks: 0,
                ..serial
            },
        ];
        index = 0;
        while index < invalid_options.len() {
            let options = invalid_options[index];
            assert!(matches!(
                options.requirements(1),
                Err(FftError::InvalidExecution)
            ));
            let expansion = ExpansionStrategy {
                transform: options,
                ..expansion
            };
            assert!(matches!(
                expansion.coefficient_requirements(1, 1),
                Err(FftError::InvalidExecution)
            ));
            assert!(matches!(
                expansion.evaluation_requirements(1, 1),
                Err(FftError::InvalidExecution)
            ));
            index += 1;
        }
        let invalid = ExpansionStrategy {
            max_residue_tasks: 0,
            ..expansion
        };
        assert!(matches!(
            invalid.coefficient_requirements(1, 1),
            Err(FftError::InvalidExecution)
        ));
        assert!(matches!(
            invalid.evaluation_requirements(1, 1),
            Err(FftError::InvalidExecution)
        ));
        assert!(matches!(
            expansion.coefficient_requirements(8, 4),
            Err(FftError::InvalidLayout)
        ));
        assert!(matches!(
            expansion.evaluation_requirements(8, 4),
            Err(FftError::InvalidLayout)
        ));
    };

    for log in 0..usize::BITS {
        let size = 1usize << log;
        let expected = Domain::<PallasBase>::for_size(size).map(|_| ());
        assert_eq!(TableRequirements::for_size(size).map(|_| ()), expected);
        assert_eq!(Strategy::SERIAL.requirements(size).map(|_| ()), expected);
        assert_eq!(
            ExpansionStrategy::SERIAL
                .coefficient_requirements(size, size)
                .map(|_| ()),
            expected
        );
        assert_eq!(
            ExpansionStrategy::SERIAL
                .evaluation_requirements(1, size)
                .map(|_| ()),
            expected
        );
    }
    let saturated = Strategy {
        tile_len: 4,
        columns_per_task: usize::MAX,
        max_tasks: usize::MAX,
    };
    assert!(saturated.requirements(16).is_ok());
    let expansion = ExpansionStrategy {
        max_residue_tasks: usize::MAX,
        transform: saturated,
    };
    assert!(expansion.coefficient_requirements(16, 64).is_ok());
    assert!(expansion.evaluation_requirements(16, 64).is_ok());
}

#[test]
fn invalid_descriptions_and_short_scratch_do_not_mutate_buffers() {
    assert!(matches!(
        Domain::<PallasBase>::new(33),
        Err(FftError::InvalidSize)
    ));
    assert!(matches!(
        Domain::<PallasBase>::new(u32::MAX),
        Err(FftError::InvalidSize)
    ));
    for size in [0, 3, 7, usize::MAX] {
        assert!(Domain::<PallasBase>::for_size(size).is_err());
    }
    let domain = Domain::new(6).unwrap().subgroup();
    if usize::BITS == 32 {
        assert!(matches!(
            Domain::<PallasBase>::new(32),
            Err(FftError::SizeOverflow)
        ));
    } else {
        assert_eq!(
            Domain::<PallasBase>::new(32).unwrap().size() as u64,
            1u64 << 32
        );
    }
    let plan = Transform::new(domain);
    let options = Strategy {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 3,
    };
    let required = plan.scratch_requirements_with(options).unwrap();
    let original = inputs::<PallasBase>(64);
    let mut output = original.clone();
    let mut scratch = vec![Fp::ONE; required - 1];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.forward_with(&mut output, options, &SerialExecutor, &mut scratch);
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert_eq!(
        bytes_of_slice(&scratch),
        bytes_of_slice(&vec![<Fp>::ONE; required - 1])
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ =
                plan.inverse_bit_reversed_with(&mut output, options, &SerialExecutor, &mut scratch);
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.forward_into_with(
                &original,
                &mut output,
                options,
                &SerialExecutor,
                &mut scratch,
            );
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert_eq!(
        bytes_of_slice(&scratch),
        bytes_of_slice(&vec![<Fp>::ONE; required - 1])
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.inverse_into_with(
                &original,
                &mut output,
                options,
                &SerialExecutor,
                &mut scratch,
            );
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert_eq!(
        bytes_of_slice(&scratch),
        bytes_of_slice(&vec![<Fp>::ONE; required - 1])
    );
    for (input_len, output_len) in [(63, 64), (65, 64), (64, 63), (64, 65)] {
        let input = vec![Fp::ONE; input_len];
        let mut output = vec![Fp::ZERO; output_len];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = plan.forward_into_with(
                    &input,
                    &mut output,
                    Strategy::SERIAL,
                    &SerialExecutor,
                    &mut [],
                );
            }))
            .is_err()
        );
        assert_eq!(
            bytes_of_slice(&output),
            bytes_of_slice(&vec![<Fp>::ZERO; output_len])
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = plan.inverse_into_with(
                    &input,
                    &mut output,
                    Strategy::SERIAL,
                    &SerialExecutor,
                    &mut [],
                );
            }))
            .is_err()
        );
        assert_eq!(
            bytes_of_slice(&output),
            bytes_of_slice(&vec![<Fp>::ZERO; output_len])
        );
        if output_len != 64 {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = plan.inverse_bit_reversed_with(
                        &mut output,
                        Strategy::SERIAL,
                        &SerialExecutor,
                        &mut [],
                    );
                }))
                .is_err()
            );
            assert_eq!(
                bytes_of_slice(&output),
                bytes_of_slice(&vec![<Fp>::ZERO; output_len])
            );
        }
    }
    for options in [
        Strategy {
            tile_len: 3,
            ..options
        },
        Strategy {
            columns_per_task: 0,
            ..options
        },
        Strategy {
            max_tasks: 0,
            ..options
        },
    ] {
        assert_eq!(
            plan.forward_with(&mut output, options, &SerialExecutor, &mut []),
            Err(FftError::InvalidExecution)
        );
        assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    }
    assert!(matches!(
        run::InterpolationPlan::new(
            [
                (Transform::new(domain), ElementOrder::Natural),
                (
                    Transform::new(Domain::new(7).unwrap().subgroup()),
                    ElementOrder::Natural,
                ),
            ],
            false,
            StorageLayout::Contiguous,
            crate::exec::ExecutionOptions::default(),
        ),
        Err(FftError::InvalidLayout)
    ));
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    let expansion = Expansion::new(plan, Domain::new(7).unwrap().subgroup(), None).unwrap();
    let options = ExpansionStrategy {
        max_residue_tasks: 2,
        transform: options,
    };
    let mut expanded = [Fp::ONE; 128];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = expansion.evaluations_with(
                &original,
                &mut expanded,
                options,
                &SerialExecutor,
                &mut [],
            );
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&expanded), bytes_of_slice(&[<Fp>::ONE; 128]));
    let too_long = FftError::InvalidPrefix {
        min: 0,
        max: 64,
        actual: 65,
    };
    assert_eq!(
        expansion.coefficients_with(
            &[Fp::ONE; 65],
            &mut expanded,
            options,
            &SerialExecutor,
            &mut []
        ),
        Err(too_long)
    );
    assert_eq!(bytes_of_slice(&expanded), bytes_of_slice(&[<Fp>::ONE; 128]));
    assert_eq!(
        plan.forward_prefix_with(
            &[Fp::ONE; 65],
            &mut output,
            Strategy::SERIAL,
            &SerialExecutor,
            &mut []
        ),
        Err(too_long)
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert_eq!(
        std::format!("{too_long}"),
        "input prefix must contain 0..=64 elements, received 65"
    );
    let factor_values = [Fp::ONE; 128];
    let factor = expansion.view(&factor_values);
    for len in [0, 65] {
        let error = expansion
            .short_product_with(
                &vec![Fp::ONE; len],
                factor,
                &mut expanded,
                ExpansionStrategy::SERIAL,
                &SerialExecutor,
                &mut [],
            )
            .unwrap_err();
        assert_eq!(
            error,
            FftError::InvalidPrefix {
                min: 1,
                max: 64,
                actual: len
            }
        );
        assert_eq!(bytes_of_slice(&expanded), bytes_of_slice(&[<Fp>::ONE; 128]));
    }
}

#[test]
fn table_preparation_checks_all_lengths_before_writing() {
    let domain = Domain::<PallasBase>::new(3).unwrap().coset();
    let mut valid = [Fp::from_u64(17); 4];
    let mut wrong = [Fp::ONE; 3];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = TablesMut {
                inverse: Some(&mut valid),
                forward: Some(&mut wrong),
                ..TablesMut::default()
            }
            .prepare(domain);
        }))
        .is_err()
    );
    assert_eq!(
        bytes_of_slice(&valid),
        bytes_of_slice(&[<Fp>::from_u64(17); 4])
    );
    assert_eq!(bytes_of_slice(&wrong), bytes_of_slice(&[<Fp>::ONE; 3]));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = TablesMut {
                forward: Some(&mut valid),
                inverse_finish: Some(&mut wrong),
                ..TablesMut::default()
            }
            .prepare(domain);
        }))
        .is_err()
    );
    assert_eq!(
        bytes_of_slice(&valid),
        bytes_of_slice(&[<Fp>::from_u64(17); 4])
    );
    assert_eq!(bytes_of_slice(&wrong), bytes_of_slice(&[<Fp>::ONE; 3]));
}

#[test]
fn loose_values_remain_valid_on_unwind_and_serial_join_completes_both_jobs() {
    let a = <Fp>::ONE.neg();
    let b = <Fp>::from_u64(2).neg();
    let mut values = [a, b];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let (left, right) = values.split_at_mut(1);
            crate::field::fft::butterfly(&mut left[0], &mut right[0], None);
            panic!("interrupt loose region");
        }))
        .is_err()
    );
    assert_eq!(reduced(&values), reduced(&[a.add(&b), a.sub(&b)]));
    assert_loose_bound(&values);
    let count = AtomicUsize::new(0);
    assert!(
        catch_unwind(|| SerialExecutor.join(
            || panic!("first job"),
            || {
                count.fetch_add(1, Ordering::SeqCst);
            }
        ))
        .is_err()
    );
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn executor_panics_leave_public_buffers_within_loose_bounds() {
    struct Panics;
    impl Executor for Panics {
        fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
        where
            L: FnOnce() -> A + Send,
            R: FnOnce() -> B + Send,
            A: Send,
            B: Send,
        {
            SerialExecutor.join(left, right);
            panic!("executor failure");
        }
    }
    let domain = Domain::<PallasBase>::new(6).unwrap().subgroup();
    let plan = Transform::new(domain);
    let options = Strategy {
        tile_len: 8,
        columns_per_task: 4,
        max_tasks: 2,
    };
    let mut values = inputs(domain.size());
    let mut scratch = vec![Fp::ZERO; plan.scratch_requirements_with(options).unwrap()];
    assert!(
        catch_unwind(AssertUnwindSafe(|| plan.forward_with(
            &mut values,
            options,
            &Panics,
            &mut scratch
        )))
        .is_err()
    );
    assert_loose_bound(&values);
    assert_loose_bound(&scratch);
    // The private fused path also rejects a class interrupted by an executor.
    let mut class = Class::new(plan, &mut values, ElementOrder::Natural);
    assert!(
        catch_unwind(AssertUnwindSafe(|| interpolate_classes(
            &mut class,
            &mut [],
            options,
            &Panics,
            &mut scratch
        )))
        .is_err()
    );
    assert_loose_bound(class.values);
    assert_loose_bound(&scratch);
    let partial = class.values.to_vec();
    let scratch_before = scratch.clone();
    assert_eq!(
        interpolate_classes(&mut class, &mut [], options, &SerialExecutor, &mut scratch),
        Err(FftError::InvalidClassState)
    );
    assert_eq!(bytes_of_slice(class.values), bytes_of_slice(&partial));
    assert_eq!(bytes_of_slice(&scratch), bytes_of_slice(&scratch_before));
}

#[test]
fn inverse_panics_leave_outputs_and_scratch_within_loose_bounds() {
    fn check<M: PrimeModulus>() {
        let options = Strategy {
            tile_len: 16,
            columns_per_task: 3,
            max_tasks: 3,
        };
        let subgroup = Domain::<M>::new(7).unwrap();
        let input = inputs(subgroup.size());
        for coset in [false, true] {
            let domain = if coset {
                subgroup.coset()
            } else {
                subgroup.subgroup()
            };
            let prepared = Prepared::new(domain);
            for tables in [Tables::default(), prepared.tables()] {
                let plan = tables.bind(domain);
                let count = plan.scratch_requirements_with(options).unwrap();
                let mut scratch = vec![PastaField::ONE; count + 2];
                let joins = CountJoins::default();
                plan.inverse_with(&mut input.clone(), options, &joins, &mut scratch)
                    .unwrap();
                // Interrupt each scheduling boundary, including after terminal
                // kernels have stored their results and after only some
                // columns have been copied back to the caller's output.
                for index in 0..joins.take() {
                    let mut values = input.clone();
                    scratch.fill(PastaField::ONE);
                    let executor = FailAt {
                        calls: AtomicUsize::new(0),
                        index,
                    };
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| plan.inverse_with(
                            &mut values,
                            options,
                            &executor,
                            &mut scratch,
                        )))
                        .is_err()
                    );
                    assert_loose_bound(&values);
                    assert_loose_bound(&scratch);
                    assert_eq!(
                        bytes_of_slice(&scratch[count..]),
                        bytes_of_slice(&[PastaField::<M>::ONE; 2])
                    );
                }
            }
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
fn nested_expansion_panics_leave_all_scratch_partitions_canonical() {
    let base = Transform::new(Domain::<PallasBase>::new(6).unwrap().subgroup());
    let expansion = Expansion::new(base, Domain::new(9).unwrap().subgroup(), None).unwrap();
    let options = ExpansionStrategy {
        max_residue_tasks: 3,
        transform: Strategy {
            tile_len: 8,
            columns_per_task: 3,
            max_tasks: 3,
        },
    };
    let input = inputs(base.domain().size());
    let mut scratch = vec![Fp::ONE; expansion.coefficient_scratch_with(options).unwrap()];
    let count = CountJoins::default();
    base.inverse_with(&mut input.clone(), options.transform, &count, &mut scratch)
        .unwrap();
    let inverse_joins = count.take();
    let factors = vec![Fp::ONE; expansion.layout().size()];
    let factor = expansion.view(&factors);
    for operation in 0..3 {
        let mut output = factors.clone();
        let executor = FailAt {
            calls: AtomicUsize::new(0),
            // For evaluation input, reach the residue jobs after the inverse.
            index: if operation == 1 { inverse_joins + 1 } else { 1 },
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| match operation {
                0 => expansion.coefficients_with(
                    &input,
                    &mut output,
                    options,
                    &executor,
                    &mut scratch,
                ),
                1 => expansion.evaluations_with(
                    &input,
                    &mut output,
                    options,
                    &executor,
                    &mut scratch,
                ),
                _ => expansion.short_product_with(
                    &input[..5],
                    factor,
                    &mut output,
                    options,
                    &executor,
                    &mut scratch,
                ),
            }))
            .is_err()
        );
        assert_loose_bound(&output);
        assert_loose_bound(&scratch);
    }
}

#[test]
fn generic_reference_supports_a_foreign_field() {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct F17(u32);
    impl reference::Twiddle for F17 {
        const ONE: Self = Self(1);
        fn multiply(&self, rhs: &Self) -> Self {
            Self(self.0 * rhs.0 % 17)
        }
        fn square(&self) -> Self {
            self.multiply(self)
        }
    }
    impl reference::Butterfly<Self> for F17 {
        fn scaled(&self, rhs: &Self) -> Self {
            Self(self.0 * rhs.0 % 17)
        }
        fn add(&self, rhs: &Self) -> Self {
            Self((self.0 + rhs.0) % 17)
        }
        fn negated(&self) -> Self {
            Self((17 - self.0) % 17)
        }
    }
    for size in [1usize, 2, 4, 8, 16] {
        let root = F17(3u32.pow(16 / size as u32) % 17);
        let input: Vec<_> = (0..size).map(|i| F17((i as u32 + 7) % 17)).collect();
        let mut output = input.clone();
        reference::transform(&mut output, &root);
        for (row, output) in output.iter().enumerate() {
            let mut expected = 0;
            for (column, value) in input.iter().enumerate() {
                let power = (0..row * column).fold(1, |p, _| p * root.0 % 17);
                expected = (expected + value.0 * power) % 17;
            }
            assert_eq!(*output, F17(expected));
        }
        let inverse_root = F17((1..17).find(|x| x * root.0 % 17 == 1).unwrap());
        let inverse_size = F17((1..17).find(|x| x * size as u32 % 17 == 1).unwrap());
        reference::inverse_transform(&mut output, &inverse_root, &inverse_size);
        assert_eq!(output, input);
    }
    for size in [0, 3] {
        let mut values = vec![F17(1); size];
        assert!(
            catch_unwind(AssertUnwindSafe(|| reference::transform(
                &mut values,
                &F17(1)
            )))
            .is_err()
        );
    }
}

#[test]
fn generic_reference_forward_supports_noninvertible_lengths() {
    use reference::{Butterfly, Twiddle};

    // (Z/4Z)[i]/(i^2 + 1) has a valid fourth root even though four is zero.
    // Requiring an invertible length for the forward transform would exclude it.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Z4i(u32, u32);
    impl Twiddle for Z4i {
        const ONE: Self = Self(1, 0);
        fn multiply(&self, rhs: &Self) -> Self {
            Self(
                (self.0 * rhs.0 + 16 - self.1 * rhs.1) % 4,
                (self.0 * rhs.1 + self.1 * rhs.0) % 4,
            )
        }
        fn square(&self) -> Self {
            self.multiply(self)
        }
    }
    impl Butterfly<Self> for Z4i {
        fn scaled(&self, rhs: &Self) -> Self {
            self.multiply(rhs)
        }
        fn add(&self, rhs: &Self) -> Self {
            Self((self.0 + rhs.0) % 4, (self.1 + rhs.1) % 4)
        }
        fn negated(&self) -> Self {
            Self((4 - self.0) % 4, (4 - self.1) % 4)
        }
    }
    fn power(value: Z4i, exponent: usize) -> Z4i {
        (0..exponent).fold(Z4i::ONE, |acc, _| acc.multiply(&value))
    }

    for size in [1, 2, 4] {
        let root = power(Z4i(0, 1), 4 / size);
        assert_eq!(power(root, size), Z4i::ONE);
        for exponent in 1..size {
            assert_ne!(power(root, exponent), Z4i::ONE);
            let character_sum =
                (0..size).fold(Z4i(0, 0), |sum, i| sum.add(&power(root, i * exponent)));
            assert_eq!(character_sum, Z4i(0, 0));
        }
        if size > 1 {
            assert_eq!(power(root, size / 2), Z4i::ONE.negated());
            // A nonzero annihilator proves the length is not a unit.
            assert_eq!(Z4i(size as u32 % 4, 0).multiply(&Z4i(2, 0)), Z4i(0, 0));
        }
        for fixture in 0..=size {
            let input: Vec<_> = (0..size)
                .map(|i| {
                    if fixture == size {
                        Z4i((i as u32 + 1) % 4, (i as u32 + 3) % 4)
                    } else {
                        Z4i(u32::from(i == fixture), 0)
                    }
                })
                .collect();
            let expected: Vec<_> = (0..size)
                .map(|row| {
                    input
                        .iter()
                        .enumerate()
                        .fold(Z4i(0, 0), |sum, (column, value)| {
                            sum.add(&value.multiply(&power(root, row * column)))
                        })
                })
                .collect();
            let mut output = input;
            reference::transform(&mut output, &root);
            assert_eq!(output, expected, "size={size} fixture={fixture}");
        }
    }
}

use super::*;
use crate::field::{Fp, PallasBase, PallasScalar};
use crate::test_support::field_samples;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicUsize, Ordering},
    vec,
    vec::Vec,
};

mod composition;
mod contracts;
mod group;
mod operations;
mod pipelines;

struct Prepared<M: PrimeModulus> {
    forward: Vec<PastaField<M>>,
    inverse: Vec<PastaField<M>>,
    finish: Vec<PastaField<M>>,
    scales: Vec<PastaField<M>>,
}

impl<M: PrimeModulus> Prepared<M> {
    fn new(domain: CosetDomain<M>) -> Self {
        let requirements = TableRequirements::for_size(domain.size()).unwrap();
        assert_eq!(requirements, TableRequirements::for_domain(domain));
        let mut result = Self {
            forward: vec![PastaField::ZERO; requirements.twiddles],
            inverse: vec![PastaField::ZERO; requirements.twiddles],
            finish: vec![PastaField::ZERO; requirements.twiddles],
            scales: vec![PastaField::ZERO; requirements.inverse_scales],
        };
        TablesMut {
            forward: Some(&mut result.forward),
            inverse: Some(&mut result.inverse),
            inverse_finish: Some(&mut result.finish),
            inverse_scales: Some(&mut result.scales),
        }
        .prepare(domain)
        .unwrap();
        result
    }

    fn tables(&self) -> Tables<'_, M> {
        Tables {
            forward: Some(&self.forward),
            inverse: Some(&self.inverse),
            inverse_finish: Some(&self.finish),
            inverse_scales: Some(&self.scales),
        }
    }
}

fn inputs<M: PrimeModulus>(size: usize) -> Vec<PastaField<M>> {
    let mut samples = field_samples();
    (0..size)
        .map(|index| match index % 17 {
            0 => PastaField::ZERO,
            1 => PastaField::ONE.neg(),
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

fn assert_canonical<M: PrimeModulus>(values: &[PastaField<M>]) {
    for value in values {
        assert!(
            value
                .montgomery_limbs()
                .iter()
                .rev()
                .cmp(M::MODULUS.iter().rev())
                .is_lt()
        );
        assert_eq!(PastaField::<M>::from_bytes(value.to_bytes()), Some(*value));
    }
}

fn small_transforms<M: PrimeModulus>() {
    // Debug must be available with only the public field-modulus bound.
    fn assert_debug<T: core::fmt::Debug>() {}
    assert_debug::<Domain<M>>();
    assert_debug::<CosetDomain<M>>();
    assert_debug::<Plan<'_, M>>();
    assert_debug::<Tables<'_, M>>();
    assert_debug::<TablesMut<'_, M>>();
    assert_debug::<Expansion<'_, M>>();
    assert_debug::<Class<'_, M>>();

    const OPTIONS: [ExecutionOptions; 3] = [
        ExecutionOptions::serial(),
        ExecutionOptions {
            tile_len: 1,
            columns_per_task: 3,
            max_tasks: 3,
        },
        ExecutionOptions {
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
                    Ok(required) => required.field_elements,
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
        assert_eq!(by_size.root(), subgroup.root());
        assert_eq!(by_size.inverse_root(), subgroup.inverse_root());
        assert_eq!(by_size.size_inverse(), subgroup.size_inverse());
        let coefficients = inputs(subgroup.size());
        for shift in [
            PastaField::ONE,
            PastaField::zeta(),
            PastaField::zeta_inverse(),
            PastaField::from_u64(7),
        ] {
            let domain = subgroup.coset(shift).unwrap();
            let prepared = Prepared::new(domain);
            let expected = direct(&coefficients, domain);
            // Every table is independently optional, including combinations
            // where final scales are prepared but ordinary twiddles are not.
            for mask in 0..16 {
                let tables = Tables {
                    forward: (mask & 1 != 0).then_some(prepared.forward.as_slice()),
                    inverse: (mask & 2 != 0).then_some(prepared.inverse.as_slice()),
                    inverse_finish: (mask & 4 != 0).then_some(prepared.finish.as_slice()),
                    inverse_scales: (mask & 8 != 0).then_some(prepared.scales.as_slice()),
                };
                let plan = Plan::new(tables.bind(domain).unwrap());
                for (index, options) in OPTIONS.into_iter().enumerate() {
                    let count = SCRATCH[log as usize][index];
                    assert_eq!(
                        plan.scratch_requirements(options).unwrap().field_elements,
                        count
                    );
                    let sentinel = PastaField::from_u64(99);
                    let mut scratch = vec![sentinel; count + 3];
                    let mut output = vec![PastaField::ZERO; coefficients.len()];
                    plan.forward_into(
                        &coefficients,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(output, expected, "forward log={log}");
                    assert_canonical(&output);
                    plan.inverse(&mut output, options, &SerialExecutor, &mut scratch)
                        .unwrap();
                    assert_eq!(output, coefficients, "inverse log={log}");
                    let evaluations = expected.clone();
                    plan.inverse_into(
                        &evaluations,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(output, coefficients, "inverse_into log={log}");
                    assert_eq!(evaluations, expected);
                    // Construct the input order independently of Plan::permute.
                    for (row, value) in expected.iter().enumerate() {
                        let bits = (0..log).fold(0, |bits, bit| (bits << 1) | ((row >> bit) & 1));
                        output[bits] = *value;
                    }
                    plan.inverse_bit_reversed(&mut output, options, &SerialExecutor, &mut scratch)
                        .unwrap();
                    assert_eq!(output, coefficients, "inverse_bit_reversed log={log}");
                    assert_eq!(&scratch[count..], &[sentinel; 3]);
                    assert_canonical(&scratch);
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
        let domain = Domain::<M>::new(log)
            .unwrap()
            .coset(PastaField::from_u64(7))
            .unwrap();
        let prepared = Prepared::new(domain);
        let input = inputs(domain.size());
        for table in [Tables::default(), prepared.tables()] {
            let plan = Plan::new(table.bind(domain).unwrap());
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
                    let options = ExecutionOptions {
                        tile_len,
                        columns_per_task: 3,
                        max_tasks: 3,
                    };
                    let mut scratch =
                        vec![
                            PastaField::ZERO;
                            plan.scratch_requirements(options).unwrap().field_elements
                        ];
                    let mut output = vec![PastaField::ONE; domain.size()];
                    plan.forward_prefix(
                        &input[..len],
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(
                        output, expected,
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
        let domain = Domain::<M>::new(log)
            .unwrap()
            .coset(PastaField::zeta())
            .unwrap();
        let prepared = Prepared::new(domain);
        let plan = Plan::new(prepared.tables().bind(domain).unwrap());
        let coefficients = inputs(domain.size());
        let expected = reference_coset(&coefficients, domain);
        let options = ExecutionOptions {
            tile_len: 2048,
            columns_per_task: 257,
            max_tasks: 3,
        };
        let mut scratch =
            vec![PastaField::ZERO; plan.scratch_requirements(options).unwrap().field_elements];
        let mut output = coefficients.clone();
        plan.forward(&mut output, options, &Threads, &mut scratch)
            .unwrap();
        assert_eq!(output, expected);
        plan.inverse(&mut output, options, &Threads, &mut scratch)
            .unwrap();
        assert_eq!(output, coefficients);
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
        assert_eq!(root.square(), PastaField::root_of_unity(log - 1).unwrap());
        assert_eq!(
            inverse.square(),
            PastaField::root_of_unity_inverse(log - 1).unwrap()
        );
        assert_eq!(root.mul(&inverse), PastaField::ONE);
        assert_eq!(root.pow_u64(1u64 << (log - 1)), PastaField::ONE.neg());
    }
    let subgroup = Domain::<M>::for_size(1 << 16).unwrap();
    assert_eq!(subgroup.log_size(), 16);
    let domain = subgroup.coset(PastaField::from_u64(7)).unwrap();
    let prepared = Prepared::new(domain);
    let plan = Plan::new(prepared.tables().bind(domain).unwrap());
    let coefficients = inputs(domain.size());
    let expected = reference_coset(&coefficients, domain);
    let options = ExecutionOptions {
        max_tasks: 8,
        ..ExecutionOptions::default()
    };
    let mut scratch =
        vec![PastaField::ZERO; plan.scratch_requirements(options).unwrap().field_elements];
    let executor = CountJoins::default();
    let mut output = coefficients.clone();
    plan.forward(&mut output, options, &executor, &mut scratch)
        .unwrap();
    assert_eq!(output, expected);
    assert!(executor.0.load(Ordering::Relaxed) > 0);
    plan.inverse(&mut output, options, &executor, &mut scratch)
        .unwrap();
    assert_eq!(output, coefficients);

    let mut reference = coefficients.clone();
    reference::transform(&mut reference, &subgroup.root());
    reference::inverse_transform(
        &mut reference,
        &subgroup.inverse_root(),
        &subgroup.size_inverse(),
    );
    assert_eq!(reference, coefficients);
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
        let base = Plan::new(base_tables.tables().bind(base_domain.subgroup()).unwrap());
        let coefficients = inputs(base_domain.size());
        let evaluations = direct(&coefficients, base_domain.subgroup());
        for extra in [0, 1, 2, 3] {
            for shift in [PastaField::ONE, PastaField::zeta(), PastaField::from_u64(7)] {
                let domain = Domain::new(log + extra).unwrap().coset(shift).unwrap();
                let expansion = Expansion::new(base, domain, None).unwrap();
                let mut scales = vec![PastaField::ZERO; expansion.scale_count()];
                let scales = expansion.prepare_scales(&mut scales).unwrap();
                for scales in [None, Some(scales)] {
                    let expansion = Expansion::new(base, domain, scales).unwrap();
                    expansion.validate_scales().unwrap();
                    let expected = direct(&coefficients, domain);
                    for options in [
                        ExpansionOptions::serial(),
                        ExpansionOptions {
                            max_residue_tasks: 3,
                            ..ExpansionOptions::serial()
                        },
                        ExpansionOptions {
                            max_residue_tasks: 1,
                            transform: ExecutionOptions {
                                tile_len: 4,
                                columns_per_task: 3,
                                max_tasks: 3,
                            },
                        },
                        ExpansionOptions {
                            max_residue_tasks: 3,
                            transform: ExecutionOptions {
                                tile_len: 4,
                                columns_per_task: 3,
                                max_tasks: 3,
                            },
                        },
                    ] {
                        let mut output = vec![PastaField::ZERO; domain.size()];
                        let count = expansion
                            .coefficient_scratch(options)
                            .unwrap()
                            .field_elements;
                        let mut scratch = vec![PastaField::ONE; count + 1];
                        expansion
                            .coefficients(
                                &coefficients,
                                &mut output,
                                options,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        assert_eq!(scratch[count], PastaField::ONE);
                        let view = expansion.view(&output).unwrap();
                        for (row, expected) in expected.iter().enumerate() {
                            assert_eq!(view.get(row), Some(expected));
                        }
                        let count = expansion
                            .evaluation_scratch(options)
                            .unwrap()
                            .field_elements;
                        scratch.fill(PastaField::ONE);
                        expansion
                            .evaluations(
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
                                .all(|value| *value == PastaField::ONE)
                        );
                        let view = expansion.view(&output).unwrap();
                        for (row, expected) in expected.iter().enumerate() {
                            assert_eq!(view.get(row), Some(expected));
                        }
                        for len in [1, coefficients.len().min(3), coefficients.len().min(10)] {
                            let short = &coefficients[..len];
                            let mut product = vec![PastaField::ZERO; domain.size()];
                            expansion
                                .short_product(
                                    short,
                                    view,
                                    &mut product,
                                    options,
                                    &SerialExecutor,
                                    &mut scratch,
                                )
                                .unwrap();
                            let short_values = direct(short, domain);
                            let product = expansion.view(&product).unwrap();
                            for row in 0..domain.size() {
                                assert_eq!(
                                    product.get(row),
                                    Some(&expected[row].mul(&short_values[row]))
                                );
                            }
                        }
                        expansion
                            .coefficients(&[], &mut output, options, &SerialExecutor, &mut scratch)
                            .unwrap();
                        assert!(output.iter().all(|value| *value == PastaField::ZERO));
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
    let base = Plan::new(prepared.tables().bind(base_domain.subgroup()).unwrap());
    let coefficients = inputs(base_domain.size());
    for extra in [1, 3] {
        let domain = Domain::new(11 + extra)
            .unwrap()
            .coset(PastaField::zeta())
            .unwrap();
        let expansion = Expansion::new(base, domain, None).unwrap();
        let expected = reference_coset(&coefficients, domain);
        let mut output = vec![PastaField::ZERO; domain.size()];
        let options = ExpansionOptions {
            max_residue_tasks: 3,
            transform: ExecutionOptions {
                tile_len: 256,
                columns_per_task: 127,
                max_tasks: 3,
            },
        };
        let mut scratch = vec![
            PastaField::ZERO;
            expansion
                .coefficient_scratch(options)
                .unwrap()
                .field_elements
        ];
        expansion
            .coefficients(&coefficients, &mut output, options, &Threads, &mut scratch)
            .unwrap();
        let mut natural = vec![PastaField::ZERO; domain.size()];
        expansion
            .layout()
            .copy_to_natural(&output, &mut natural)
            .unwrap();
        assert_eq!(natural, expected);
        let evaluations = reference_coset(&coefficients, base_domain.subgroup());
        expansion
            .evaluations(&evaluations, &mut output, options, &Threads, &mut scratch)
            .unwrap();
        expansion
            .layout()
            .copy_to_natural(&output, &mut natural)
            .unwrap();
        assert_eq!(natural, expected);
        let factor = expansion.view(&output).unwrap();
        let mut product = vec![PastaField::ZERO; domain.size()];
        expansion
            .short_product(
                &coefficients[..5],
                factor,
                &mut product,
                options,
                &Threads,
                &mut scratch,
            )
            .unwrap();
        let short_values = direct(&coefficients[..5], domain);
        let product = expansion.view(&product).unwrap();
        for row in 0..domain.size() {
            assert_eq!(
                product.get(row),
                Some(&expected[row].mul(&short_values[row]))
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
    let base = Plan::without_tables(Domain::<PallasBase>::new(6).unwrap().subgroup());
    let coefficients = inputs(base.domain().size());
    let options = ExpansionOptions {
        // With no joins across residues, every observed join is intra-residue.
        max_residue_tasks: 1,
        transform: ExecutionOptions {
            tile_len: 8,
            columns_per_task: 3,
            max_tasks: 4,
        },
    };
    let mut scratch = vec![
        Fp::ZERO;
        base.scratch_requirements(options.transform)
            .unwrap()
            .field_elements
    ];
    let executor = CountJoins::default();
    let mut evaluations = coefficients.clone();
    base.forward(&mut evaluations, options.transform, &executor, &mut scratch)
        .unwrap();
    let forward_joins = executor.take();
    let mut inverse = evaluations.clone();
    base.inverse(&mut inverse, options.transform, &executor, &mut scratch)
        .unwrap();
    let inverse_joins = executor.take();
    assert!(forward_joins > 0 && inverse_joins > 0);

    for extra in [0, 1, 3] {
        let domain = Domain::new(6 + extra)
            .unwrap()
            .coset(Fp::from_u64(7))
            .unwrap();
        let expansion = Expansion::new(base, domain, None).unwrap();
        let residues = expansion.layout().residues();
        let mut output = vec![Fp::ZERO; domain.size()];
        expansion
            .coefficients(&coefficients, &mut output, options, &executor, &mut scratch)
            .unwrap();
        assert!(executor.take() >= residues);
        let expected = output.clone();
        expansion
            .evaluations(&evaluations, &mut output, options, &executor, &mut scratch)
            .unwrap();
        assert!(executor.take() > residues);
        assert_eq!(output, expected);
        let ones = vec![Fp::ONE; domain.size()];
        let factor = expansion.view(&ones).unwrap();
        expansion
            .short_product(
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
        let view = expansion.view(&output).unwrap();
        for (row, expected) in expected_short.iter().enumerate() {
            assert_eq!(view.get(row), Some(expected));
        }

        // Whole transforms still allow callers to parallelize only residues
        // without reserving scratch for tiled transforms.
        let options = ExpansionOptions {
            max_residue_tasks: 3,
            ..ExpansionOptions::serial()
        };
        assert_eq!(
            expansion
                .coefficient_scratch(options)
                .unwrap()
                .field_elements,
            0
        );
        expansion
            .coefficients(&coefficients, &mut output, options, &executor, &mut [])
            .unwrap();
        assert_eq!(executor.take(), residues.min(3) - 1);
        assert_eq!(output, expected);
    }
}

fn prepared_evaluation_expansions<M: PrimeModulus>() {
    let subgroup = Domain::<M>::new(6).unwrap().subgroup();
    let prepared = Prepared::new(subgroup);
    let coefficients = inputs::<M>(subgroup.size());
    let evaluations = direct(&coefficients, subgroup);
    let saved = evaluations.clone();
    for extra in [0, 1, 3] {
        for shift in [PastaField::ONE, PastaField::zeta(), PastaField::from_u64(7)] {
            let extended = Domain::new(6 + extra).unwrap().coset(shift).unwrap();
            let expected = reference_coset(&coefficients, extended);
            let mut scales = vec![PastaField::ZERO; extended.size()];
            let scales = Expansion::new(Plan::without_tables(subgroup), extended, None)
                .unwrap()
                .prepare_scales(&mut scales)
                .unwrap();
            for mask in 0..16 {
                let tables = Tables {
                    forward: (mask & 1 != 0).then_some(prepared.forward.as_slice()),
                    inverse: (mask & 2 != 0).then_some(prepared.inverse.as_slice()),
                    inverse_finish: (mask & 4 != 0).then_some(prepared.finish.as_slice()),
                    inverse_scales: (mask & 8 != 0).then_some(prepared.scales.as_slice()),
                };
                let expansion = Expansion::new(
                    Plan::new(tables.bind(subgroup).unwrap()),
                    extended,
                    Some(scales),
                )
                .unwrap();
                for transform in [
                    ExecutionOptions::serial(),
                    ExecutionOptions {
                        tile_len: 8,
                        columns_per_task: 3,
                        max_tasks: 3,
                    },
                ] {
                    let options = ExpansionOptions {
                        max_residue_tasks: 3,
                        transform,
                    };
                    let count = expansion
                        .evaluation_scratch(options)
                        .unwrap()
                        .field_elements;
                    let mut scratch = vec![PastaField::ONE; count + 1];
                    let mut output = vec![PastaField::ZERO; extended.size()];
                    expansion
                        .evaluations(
                            &evaluations,
                            &mut output,
                            options,
                            &SerialExecutor,
                            &mut scratch,
                        )
                        .unwrap();
                    let view = expansion.view(&output).unwrap();
                    for (row, expected) in expected.iter().enumerate() {
                        assert_eq!(view.get(row), Some(expected));
                    }
                    assert_eq!(evaluations, saved);
                    assert_eq!(scratch[count], PastaField::ONE);
                    assert_canonical(&output);
                    assert_canonical(&scratch);
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
    let base = Plan::without_tables(Domain::<M>::new(6).unwrap().subgroup());
    let transform = ExecutionOptions {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 4,
    };
    let options = ExpansionOptions {
        max_residue_tasks: 3,
        transform,
    };
    let executor = CountJoins::default();
    let constant = [PastaField::from_u64(19)];
    for extra in [0, 1, 3] {
        for shift in [PastaField::ONE, PastaField::zeta(), PastaField::from_u64(7)] {
            let domain = Domain::new(6 + extra).unwrap().coset(shift).unwrap();
            let plan = Plan::without_tables(domain);
            let expansion = Expansion::new(base, domain, None).unwrap();
            let required = expansion
                .coefficient_scratch(options)
                .unwrap()
                .field_elements
                .max(plan.scratch_requirements(transform).unwrap().field_elements);
            let mut scratch = vec![PastaField::ONE; required + 1];
            let mut output = vec![PastaField::ZERO; domain.size()];
            for coefficients in [&constant[..0], &constant[..]] {
                let expected = coefficients.first().copied().unwrap_or(PastaField::ZERO);
                for expanded in [false, true] {
                    if expanded {
                        expansion
                            .coefficients(
                                coefficients,
                                &mut output,
                                options,
                                &executor,
                                &mut scratch,
                            )
                            .unwrap();
                    } else {
                        plan.forward_prefix(
                            coefficients,
                            &mut output,
                            transform,
                            &executor,
                            &mut scratch,
                        )
                        .unwrap();
                    }
                    assert_eq!(executor.take(), 0);
                    assert!(output.iter().all(|&value| value == expected));
                    assert!(scratch.iter().all(|&value| value == PastaField::ONE));
                }
            }
            let factor = inputs::<M>(domain.size());
            expansion
                .short_product(
                    &constant,
                    expansion.view(&factor).unwrap(),
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
                assert_eq!(*actual, constant[0].mul(factor));
            }
            assert!(scratch.iter().all(|&value| value == PastaField::ONE));
            if extra == 0 && shift == PastaField::ONE {
                let input = inputs::<M>(base.domain().size());
                expansion
                    .evaluations(&input, &mut output, options, &executor, &mut scratch)
                    .unwrap();
                assert_eq!(output, input);
                assert_eq!(executor.take(), 0);
                assert!(scratch.iter().all(|&value| value == PastaField::ONE));
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
    let base = Plan::without_tables(Domain::<PallasBase>::new(6).unwrap().subgroup());
    let input = inputs(base.domain().size());
    let transform = ExecutionOptions {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 3,
    };
    let per_transform = base.scratch_requirements(transform).unwrap().field_elements;
    for extra in [0, 1, 3] {
        let expansion =
            Expansion::new(base, Domain::new(6 + extra).unwrap().subgroup(), None).unwrap();
        let mut output = vec![Fp::ONE; expansion.layout().size()];
        let factors = output.clone();
        let factor = expansion.view(&factors).unwrap();
        for tasks in [1, 2, 3, usize::MAX] {
            let options = ExpansionOptions {
                max_residue_tasks: tasks,
                transform,
            };
            let count = expansion
                .coefficient_scratch(options)
                .unwrap()
                .field_elements;
            let eval_count = expansion
                .evaluation_scratch(options)
                .unwrap()
                .field_elements;
            assert_eq!(
                count,
                per_transform * tasks.min(expansion.layout().residues())
            );
            assert_eq!(
                eval_count,
                per_transform * tasks.min(expansion.layout().residues() - 1).max(1)
            );
            let mut scratch = vec![Fp::ONE; count - 1];
            let error = Err(FftError::ScratchTooSmall {
                required: count,
                provided: count - 1,
            });
            assert_eq!(
                expansion.coefficients(&input, &mut output, options, &SerialExecutor, &mut scratch),
                error
            );
            assert_eq!(
                expansion.coefficients(&[], &mut output, options, &SerialExecutor, &mut scratch),
                error
            );
            assert_eq!(
                expansion.short_product(
                    &input[..5],
                    factor,
                    &mut output,
                    options,
                    &SerialExecutor,
                    &mut scratch
                ),
                error
            );
            assert_eq!(
                expansion.evaluations(
                    &input,
                    &mut output,
                    options,
                    &SerialExecutor,
                    &mut scratch[..eval_count - 1]
                ),
                Err(FftError::ScratchTooSmall {
                    required: eval_count,
                    provided: eval_count - 1
                })
            );
            assert_eq!(output, factors);
            assert!(scratch.iter().all(|value| *value == Fp::ONE));
        }
        let valid = ExpansionOptions {
            max_residue_tasks: 2,
            transform,
        };
        for options in [
            ExpansionOptions {
                max_residue_tasks: 0,
                ..valid
            },
            ExpansionOptions {
                transform: ExecutionOptions {
                    tile_len: 3,
                    ..transform
                },
                ..valid
            },
            ExpansionOptions {
                transform: ExecutionOptions {
                    columns_per_task: 0,
                    ..transform
                },
                ..valid
            },
            ExpansionOptions {
                transform: ExecutionOptions {
                    max_tasks: 0,
                    ..transform
                },
                ..valid
            },
        ] {
            let mut scratch = vec![Fp::ONE; 512];
            assert_eq!(
                expansion.coefficient_scratch(options),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.evaluation_scratch(options),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.coefficients(&input, &mut output, options, &SerialExecutor, &mut scratch),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.evaluations(&input, &mut output, options, &SerialExecutor, &mut scratch),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.short_product(
                    &input[..5],
                    factor,
                    &mut output,
                    options,
                    &SerialExecutor,
                    &mut scratch
                ),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(output, factors);
            assert!(scratch.iter().all(|value| *value == Fp::ONE));
        }
    }
}

fn classed<M: PrimeModulus>(log: u32) {
    let domain = Domain::<M>::new(log)
        .unwrap()
        .coset(PastaField::zeta())
        .unwrap();
    let smaller = Domain::<M>::new(log - 1)
        .unwrap()
        .coset(PastaField::from_u64(7))
        .unwrap();
    let smallest = Domain::<M>::new(log - 2).unwrap().subgroup();
    let full_coefficients = inputs(domain.size());
    let small_coefficients = inputs(smaller.size());
    let smallest_coefficients = inputs(smallest.size());
    let expected: Vec<_> = full_coefficients
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .add(small_coefficients.get(index).unwrap_or(&PastaField::ZERO))
                .add(
                    smallest_coefficients
                        .get(index)
                        .unwrap_or(&PastaField::ZERO),
                )
        })
        .collect();
    let full_values = reference_coset(&full_coefficients, domain);
    let small_values = reference_coset(&small_coefficients, smaller);
    let smallest_values = reference_coset(&smallest_coefficients, smallest);
    let prepared = Prepared::new(domain);
    let small_prepared = Prepared::new(smaller);
    let plan = Plan::new(prepared.tables().bind(domain).unwrap());
    let small_plan = Plan::new(small_prepared.tables().bind(smaller).unwrap());
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
            let transforms = [plan, small_plan, Plan::without_tables(smallest)].map(|plan| {
                run::FftPlan::new(
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
            let plan = run::InterpolationPlan::new(transforms, false).unwrap();
            let mut scratch: [_; 3] = core::array::from_fn(|i| {
                vec![PastaField::ONE; plan.snapshot_fields(i).unwrap() + 2]
            });
            plan.execute(
                values.each_mut().map(Vec::as_mut_slice),
                scratch.each_mut().map(Vec::as_mut_slice),
                core::num::NonZeroUsize::new(tasks).unwrap(),
                &Threads,
            )
            .unwrap();
            assert_eq!(values[0], expected);
            assert_eq!(values[1], small_coefficients);
            assert_eq!(values[2], smallest_coefficients);
            for buffer in &scratch {
                assert_eq!(&buffer[buffer.len() - 2..], &[PastaField::ONE; 2]);
                assert_canonical(buffer);
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
        layout.copy_from_natural(&input, &mut stored).unwrap();
        layout.copy_to_natural(&stored, &mut natural).unwrap();
        assert_eq!(input, natural);
        let view = EvaluationView::bind(
            &stored,
            Domain::for_size(32).unwrap().subgroup(),
            EvaluationLayout::Residues(layout),
        )
        .unwrap();
        for (row, expected) in input.iter().enumerate() {
            assert_eq!(view.get(row), Some(expected));
            assert_eq!(layout.natural_row(layout.index(row).unwrap()), Some(row));
            assert_eq!(
                view.get_extended_row(row * 4, Domain::for_size(128).unwrap().subgroup()),
                Some(expected)
            );
            assert_eq!(
                view.get_extended_row(row * 4 + 1, Domain::for_size(128).unwrap().subgroup()),
                None
            );
        }
        assert_eq!(view.get(32), None);
        assert_eq!(
            view.get_extended_row(0, Domain::for_size(16).unwrap().subgroup()),
            None
        );
        for (input_len, output_len) in [(31, 32), (33, 32), (32, 31), (32, 33)] {
            let source = vec![Fp::ONE; input_len];
            let mut destination = vec![Fp::ZERO; output_len];
            let expected = Err(FftError::LengthMismatch {
                buffer: if input_len != 32 { "input" } else { "output" },
                expected: 32,
                actual: if input_len != 32 {
                    input_len
                } else {
                    output_len
                },
            });
            assert_eq!(
                layout.copy_from_natural(&source, &mut destination),
                expected
            );
            assert_eq!(destination, vec![Fp::ZERO; output_len]);
            assert_eq!(layout.copy_to_natural(&source, &mut destination), expected);
            assert_eq!(destination, vec![Fp::ZERO; output_len]);
        }
    }
}

#[test]
fn size_queries_validate_without_constructing_domains() {
    const _: () = {
        let serial = ExecutionOptions::serial();
        let expansion = ExpansionOptions::serial();
        let tables = match TableRequirements::for_size(1) {
            Ok(required) => required,
            Err(_) => panic!("singleton rejected"),
        };
        assert!(tables.twiddles == 0 && tables.inverse_scales == 0);
        assert!(matches!(
            serial.requirements(1),
            Ok(ScratchRequirements { field_elements: 0 })
        ));
        assert!(matches!(
            expansion.coefficient_requirements(1, 1),
            Ok(ScratchRequirements { field_elements: 0 })
        ));
        assert!(matches!(
            expansion.evaluation_requirements(1, 1),
            Ok(ScratchRequirements { field_elements: 0 })
        ));
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
            ExecutionOptions {
                tile_len: 0,
                ..serial
            },
            ExecutionOptions {
                tile_len: 3,
                ..serial
            },
            ExecutionOptions {
                columns_per_task: 0,
                ..serial
            },
            ExecutionOptions {
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
            let expansion = ExpansionOptions {
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
        let invalid = ExpansionOptions {
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
        assert_eq!(
            ExecutionOptions::serial().requirements(size).map(|_| ()),
            expected
        );
        assert_eq!(
            ExpansionOptions::serial()
                .coefficient_requirements(size, size)
                .map(|_| ()),
            expected
        );
        assert_eq!(
            ExpansionOptions::serial()
                .evaluation_requirements(1, size)
                .map(|_| ()),
            expected
        );
    }
    let saturated = ExecutionOptions {
        tile_len: 4,
        columns_per_task: usize::MAX,
        max_tasks: usize::MAX,
    };
    assert!(saturated.requirements(16).is_ok());
    let expansion = ExpansionOptions {
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
    assert!(matches!(
        domain.domain().coset(Fp::ZERO),
        Err(FftError::ZeroShift)
    ));
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
    let plan = Plan::without_tables(domain);
    let options = ExecutionOptions {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 3,
    };
    let required = plan.scratch_requirements(options).unwrap().field_elements;
    let original = inputs::<PallasBase>(64);
    let mut output = original.clone();
    let mut scratch = vec![Fp::ONE; required - 1];
    assert!(matches!(
        plan.forward(&mut output, options, &SerialExecutor, &mut scratch),
        Err(FftError::ScratchTooSmall { .. })
    ));
    assert_eq!(output, original);
    assert_eq!(scratch, vec![Fp::ONE; required - 1]);
    assert!(matches!(
        plan.inverse_bit_reversed(&mut output, options, &SerialExecutor, &mut scratch),
        Err(FftError::ScratchTooSmall { .. })
    ));
    assert_eq!(output, original);
    assert!(matches!(
        plan.forward_into(
            &original,
            &mut output,
            options,
            &SerialExecutor,
            &mut scratch
        ),
        Err(FftError::ScratchTooSmall { .. })
    ));
    assert_eq!(output, original);
    assert_eq!(scratch, vec![Fp::ONE; required - 1]);
    assert!(matches!(
        plan.inverse_into(
            &original,
            &mut output,
            options,
            &SerialExecutor,
            &mut scratch
        ),
        Err(FftError::ScratchTooSmall { .. })
    ));
    assert_eq!(output, original);
    assert_eq!(scratch, vec![Fp::ONE; required - 1]);
    for (input_len, output_len) in [(63, 64), (65, 64), (64, 63), (64, 65)] {
        let input = vec![Fp::ONE; input_len];
        let mut output = vec![Fp::ZERO; output_len];
        let forward_error = plan
            .forward_into(
                &input,
                &mut output,
                ExecutionOptions::serial(),
                &SerialExecutor,
                &mut [],
            )
            .unwrap_err();
        assert_eq!(output, vec![Fp::ZERO; output_len]);
        let error = plan
            .inverse_into(
                &input,
                &mut output,
                ExecutionOptions::serial(),
                &SerialExecutor,
                &mut [],
            )
            .unwrap_err();
        assert_eq!(forward_error, error);
        let (buffer, actual) = if input_len != 64 {
            ("input", input_len)
        } else {
            ("output", output_len)
        };
        assert_eq!(
            error,
            FftError::LengthMismatch {
                buffer,
                expected: 64,
                actual
            }
        );
        assert_eq!(
            std::format!("{error}"),
            std::format!("{buffer}: expected 64 elements, received {actual}")
        );
        assert_eq!(output, vec![Fp::ZERO; output_len]);
        if output_len != 64 {
            assert!(matches!(
                plan.inverse_bit_reversed(
                    &mut output,
                    ExecutionOptions::serial(),
                    &SerialExecutor,
                    &mut []
                ),
                Err(FftError::LengthMismatch {
                    buffer: "values",
                    ..
                })
            ));
            assert_eq!(output, vec![Fp::ZERO; output_len]);
        }
    }
    for options in [
        ExecutionOptions {
            tile_len: 3,
            ..options
        },
        ExecutionOptions {
            columns_per_task: 0,
            ..options
        },
        ExecutionOptions {
            max_tasks: 0,
            ..options
        },
    ] {
        assert_eq!(
            plan.forward(&mut output, options, &SerialExecutor, &mut []),
            Err(FftError::InvalidExecution)
        );
        assert_eq!(output, original);
    }
    let inverse = |domain| {
        run::FftPlan::new(
            Plan::without_tables(domain),
            TransformRequest::new(Direction::Inverse),
            core::num::NonZeroUsize::new(8).unwrap(),
            Codelet::Radix2,
        )
        .unwrap()
    };
    assert!(matches!(
        run::InterpolationPlan::new(
            [inverse(domain), inverse(Domain::new(7).unwrap().subgroup()),],
            false
        ),
        Err(FftError::InvalidLayout)
    ));
    assert_eq!(output, original);
    let expansion = Expansion::new(plan, Domain::new(7).unwrap().subgroup(), None).unwrap();
    let options = ExpansionOptions {
        max_residue_tasks: 2,
        transform: options,
    };
    let mut expanded = [Fp::ONE; 128];
    assert!(
        expansion
            .evaluations(&original, &mut expanded, options, &SerialExecutor, &mut [])
            .is_err()
    );
    assert_eq!(expanded, [Fp::ONE; 128]);
    let too_long = FftError::InvalidPrefix {
        min: 0,
        max: 64,
        actual: 65,
    };
    assert_eq!(
        expansion.coefficients(
            &[Fp::ONE; 65],
            &mut expanded,
            options,
            &SerialExecutor,
            &mut []
        ),
        Err(too_long)
    );
    assert_eq!(expanded, [Fp::ONE; 128]);
    assert_eq!(
        plan.forward_prefix(
            &[Fp::ONE; 65],
            &mut output,
            ExecutionOptions::serial(),
            &SerialExecutor,
            &mut []
        ),
        Err(too_long)
    );
    assert_eq!(output, original);
    assert_eq!(
        std::format!("{too_long}"),
        "input prefix must contain 0..=64 elements, received 65"
    );
    let factor_values = [Fp::ONE; 128];
    let factor = expansion.view(&factor_values).unwrap();
    for len in [0, 65] {
        let error = expansion
            .short_product(
                &vec![Fp::ONE; len],
                factor,
                &mut expanded,
                ExpansionOptions::serial(),
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
        assert_eq!(expanded, [Fp::ONE; 128]);
    }
}

#[test]
fn table_preparation_checks_all_lengths_before_writing_and_validates_contents() {
    let domain = Domain::<PallasBase>::new(3)
        .unwrap()
        .coset(Fp::from_u64(7))
        .unwrap();
    let mut valid = [Fp::from_u64(17); 4];
    let mut wrong = [Fp::ONE; 3];
    assert!(
        TablesMut {
            inverse: Some(&mut valid),
            forward: Some(&mut wrong),
            ..TablesMut::default()
        }
        .prepare(domain)
        .is_err()
    );
    assert_eq!(valid, [Fp::from_u64(17); 4]);
    assert_eq!(wrong, [Fp::ONE; 3]);
    assert!(matches!(
        TablesMut {
            forward: Some(&mut valid),
            inverse_scales: Some(&mut wrong),
            ..TablesMut::default()
        }
        .prepare(domain),
        Err(FftError::LengthMismatch {
            buffer: "inverse_scales",
            expected: 4,
            actual: 3
        })
    ));
    assert_eq!(valid, [Fp::from_u64(17); 4]);
    assert_eq!(wrong, [Fp::ONE; 3]);
    let mut prepared = Prepared::new(domain);
    prepared.inverse[1] = Fp::ZERO;
    assert!(matches!(
        prepared.tables().bind(domain),
        Err(FftError::InvalidTables)
    ));
    assert_eq!(
        prepared.tables().validate(domain),
        Err(FftError::InvalidTables)
    );
    let mut prepared = Prepared::new(domain);
    prepared.finish[1] = Fp::ZERO;
    assert_eq!(
        prepared.tables().validate(domain),
        Err(FftError::InvalidTables)
    );
    let invalid: &Fp = bento::AlignedBytes([0xff; 32]).as_value();
    prepared.forward[0] = *invalid;
    assert_eq!(
        prepared.tables().validate(domain),
        Err(FftError::InvalidTables)
    );
    let expansion = Expansion::new(
        Plan::without_tables(domain.domain().subgroup()),
        domain,
        None,
    )
    .unwrap();
    let mut scales = [Fp::ZERO; 8];
    expansion.prepare_scales(&mut scales).unwrap();
    scales[2] = Fp::ZERO;
    assert!(matches!(
        ExpansionScales::bind(
            domain.size(),
            domain,
            ExpansionScaleNormalization::Coefficients,
            &scales
        ),
        Err(FftError::InvalidTables)
    ));
}

#[test]
fn loose_regions_normalize_on_unwind_and_serial_join_completes_both_jobs() {
    let a = Fp::ONE.neg();
    let b = Fp::from_u64(2).neg();
    let mut values = [a, b];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let guard = crate::field::fft::Guard::new(&mut values);
            let (left, right) = guard.values.split_at_mut(1);
            crate::field::fft::butterfly(&mut left[0], &mut right[0], None);
            panic!("interrupt loose region");
        }))
        .is_err()
    );
    assert_eq!(values, [a.add(&b), a.sub(&b)]);
    assert_canonical(&values);
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
fn executor_panics_leave_public_buffers_canonical() {
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
    let plan = Plan::without_tables(domain);
    let options = ExecutionOptions {
        tile_len: 8,
        columns_per_task: 4,
        max_tasks: 2,
    };
    let mut values = inputs(domain.size());
    let mut scratch = vec![Fp::ZERO; plan.scratch_requirements(options).unwrap().field_elements];
    assert!(
        catch_unwind(AssertUnwindSafe(|| plan.forward(
            &mut values,
            options,
            &Panics,
            &mut scratch
        )))
        .is_err()
    );
    assert_canonical(&values);
    assert_canonical(&scratch);
    // The private fused path also rejects a class interrupted by an executor.
    let mut class = Class::new(plan, &mut values, ElementOrder::Natural).unwrap();
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
    assert_canonical(class.values);
    assert_canonical(&scratch);
    let partial = class.values.to_vec();
    let scratch_before = scratch.clone();
    assert_eq!(
        interpolate_classes(&mut class, &mut [], options, &SerialExecutor, &mut scratch),
        Err(FftError::InvalidClassState)
    );
    assert_eq!(class.values, partial);
    assert_eq!(scratch, scratch_before);
}

#[test]
fn inverse_panics_leave_normalized_outputs_and_scratch_canonical() {
    fn check<M: PrimeModulus>() {
        let options = ExecutionOptions {
            tile_len: 16,
            columns_per_task: 3,
            max_tasks: 3,
        };
        let subgroup = Domain::<M>::new(7).unwrap();
        let input = inputs(subgroup.size());
        for shift in [
            PastaField::ONE,
            PastaField::zeta(),
            PastaField::zeta_inverse(),
            PastaField::from_u64(7),
        ] {
            let domain = subgroup.coset(shift).unwrap();
            let prepared = Prepared::new(domain);
            for tables in [Tables::default(), prepared.tables()] {
                let plan = Plan::new(tables.bind(domain).unwrap());
                let count = plan.scratch_requirements(options).unwrap().field_elements;
                let mut scratch = vec![PastaField::ONE; count + 2];
                let joins = CountJoins::default();
                plan.inverse(&mut input.clone(), options, &joins, &mut scratch)
                    .unwrap();
                // Interrupt each scheduling boundary, including after terminal
                // kernels have disarmed scratch guards and after only some
                // columns have been copied back to the caller's output.
                for index in 0..joins.take() {
                    let mut values = input.clone();
                    scratch.fill(PastaField::ONE);
                    let executor = FailAt {
                        calls: AtomicUsize::new(0),
                        index,
                    };
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| plan.inverse(
                            &mut values,
                            options,
                            &executor,
                            &mut scratch,
                        )))
                        .is_err()
                    );
                    assert_canonical(&values);
                    assert_canonical(&scratch);
                    assert_eq!(&scratch[count..], &[PastaField::ONE; 2]);
                }
            }
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
fn nested_expansion_panics_leave_all_scratch_partitions_canonical() {
    let base = Plan::without_tables(Domain::<PallasBase>::new(6).unwrap().subgroup());
    let expansion = Expansion::new(base, Domain::new(9).unwrap().subgroup(), None).unwrap();
    let options = ExpansionOptions {
        max_residue_tasks: 3,
        transform: ExecutionOptions {
            tile_len: 8,
            columns_per_task: 3,
            max_tasks: 3,
        },
    };
    let input = inputs(base.domain().size());
    let mut scratch = vec![
        Fp::ONE;
        expansion
            .coefficient_scratch(options)
            .unwrap()
            .field_elements
    ];
    let count = CountJoins::default();
    base.inverse(&mut input.clone(), options.transform, &count, &mut scratch)
        .unwrap();
    let inverse_joins = count.take();
    let factors = vec![Fp::ONE; expansion.layout().size()];
    let factor = expansion.view(&factors).unwrap();
    for operation in 0..3 {
        let mut output = factors.clone();
        let executor = FailAt {
            calls: AtomicUsize::new(0),
            // For evaluation input, reach the residue jobs after the inverse.
            index: if operation == 1 { inverse_joins + 1 } else { 1 },
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| match operation {
                0 => expansion.coefficients(&input, &mut output, options, &executor, &mut scratch),
                1 => expansion.evaluations(&input, &mut output, options, &executor, &mut scratch),
                _ => expansion.short_product(
                    &input[..5],
                    factor,
                    &mut output,
                    options,
                    &executor,
                    &mut scratch
                ),
            }))
            .is_err()
        );
        assert_canonical(&output);
        assert_canonical(&scratch);
    }
}

#[test]
fn generic_reference_supports_a_foreign_field() {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct F17(u32);
    impl reference::Twiddle for F17 {
        fn one() -> Self {
            Self(1)
        }
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

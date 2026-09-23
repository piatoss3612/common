use super::shared_scalars::measure;
use super::{Buffers, Pool};
use criterion::{BenchmarkId, Criterion};
use std::hint::black_box;
use zakura_udon::{
    curve::{AffinePoint, PastaCurve, Point, ProjectivePoint, msm::SuffixBasis},
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    field::{CanonicalUint, PastaField, PrimeModulus},
};

struct Samples(u64);
impl Samples {
    fn word(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn field<M: PrimeModulus>(&mut self) -> PastaField<M> {
        let mut limbs = std::array::from_fn(|_| self.word());
        limbs[3] &= (1 << 62) - 1;
        PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap()
    }
}

fn binary<C: PastaCurve>(point: Point<C>, limbs: [u64; 4]) -> ProjectivePoint<C> {
    let mut sum = ProjectivePoint::IDENTITY;
    let base = point.to_projective();
    for bit in (0..256).rev() {
        sum = sum.double();
        if (limbs[bit / 64] >> (bit % 64)) & 1 != 0 {
            sum = sum.add(&base);
        }
    }
    sum
}

fn reference<C: PastaCurve>(
    bases: &[Point<C>],
    scalars: &[PastaField<C::Scalar>],
) -> ProjectivePoint<C> {
    bases
        .iter()
        .zip(scalars)
        .fold(ProjectivePoint::IDENTITY, |sum, (base, scalar)| {
            sum.add(&binary(*base, scalar.to_canonical_uint().limbs()))
        })
}

fn from_u128<M: PrimeModulus>(n: u128) -> PastaField<M> {
    PastaField::from_canonical_uint(CanonicalUint::from_limbs([
        n as u64,
        (n >> 64) as u64,
        0,
        0,
    ]))
    .unwrap()
}

fn bases<C: PastaCurve>(n: usize) -> Vec<Point<C>> {
    let mut samples = Samples(0x243f_6a88_85a3_08d3);
    (0..n)
        .map(|_| {
            AffinePoint::GENERATOR
                .mul_projective(&samples.field())
                .to_point()
        })
        .collect()
}

enum Row<C: PastaCurve> {
    Unsigned(Vec<u128>),
    Field(Vec<PastaField<C::Scalar>>),
}
impl<C: PastaCurve> Row<C> {
    fn new(shape: &str, n: usize) -> Self {
        let mut samples = Samples(0xa409_3822_299f_31d0);
        match shape {
            "sorted_small" | "sorted_offset" => {
                let mut value = if shape == "sorted_offset" {
                    1_u128 << 96
                } else {
                    0
                };
                Self::Unsigned(
                    (0..n)
                        .map(|_| {
                            value += (samples.word() % 8) as u128;
                            value
                        })
                        .collect(),
                )
            }
            "integer_runs_small" | "integer_runs_offset" => Self::Unsigned(
                (0..n)
                    .map(|i| {
                        (if shape == "integer_runs_offset" {
                            1_u128 << 96
                        } else {
                            0
                        }) + 13 * (i / 16) as u128
                    })
                    .collect(),
            ),
            "field_runs" => {
                let mut value = PastaField::ZERO;
                Self::Field(
                    (0..n)
                        .map(|i| {
                            if i % 16 == 0 {
                                value = samples.field();
                            }
                            value
                        })
                        .collect(),
                )
            }
            "dense" => Self::Field((0..n).map(|_| samples.field()).collect()),
            _ => unreachable!(),
        }
    }
    fn fields(&self) -> Vec<PastaField<C::Scalar>> {
        match self {
            Self::Unsigned(s) => s.iter().map(|x| from_u128(*x)).collect(),
            Self::Field(s) => s.clone(),
        }
    }
    fn direct<'a>(&'a self, bases: &'a [Point<C>]) -> zakura_udon::msm::Input<'a, C> {
        let selection = zakura_udon::msm::Selection::new(zakura_udon::msm::Bases::Points(bases));
        match self {
            Self::Unsigned(s) => selection.with_unsigned(s),
            Self::Field(s) => selection.with_scalars(s),
        }
    }
    fn difference<'a>(
        &self,
        basis: SuffixBasis<'a, C>,
        unsigned: &'a mut [u128],
        field: &'a mut [PastaField<C::Scalar>],
    ) -> zakura_udon::msm::Input<'a, C> {
        match self {
            Self::Unsigned(s) => basis.with_monotone_unsigned(s, unsigned).unwrap(),
            Self::Field(s) => basis.with_scalars(s, field),
        }
    }
}

pub(super) fn bench<C: PastaCurve>(c: &mut Criterion, name: &str) {
    let points = bases::<C>(2048);
    let options = ExecutionOptions::default();
    for n in [32, 256, 2048] {
        let points = &points[..n];
        let mut output = vec![Point::IDENTITY; n];
        let mut projective = vec![ProjectivePoint::IDENTITY; n];
        let mut field = vec![PastaField::ZERO; n];
        let mut group = c.benchmark_group(format!("{name}/suffix_prepare"));
        for (index, capacity) in [32, n].into_iter().enumerate() {
            if index == 1 && n == 32 {
                continue;
            }
            group.bench_function(BenchmarkId::new(format!("scratch_{capacity}"), n), |b| {
                b.iter(|| {
                    black_box(SuffixBasis::prepare(
                        black_box(points),
                        &mut output,
                        &mut projective[..capacity],
                        &mut field[..capacity],
                    ));
                })
            });
        }
        group.finish();
        let mut retained_output = vec![Point::IDENTITY; n];
        let retained =
            SuffixBasis::prepare(points, &mut retained_output, &mut projective, &mut field);
        for shape in [
            "sorted_small",
            "sorted_offset",
            "integer_runs_small",
            "integer_runs_offset",
            "field_runs",
            "dense",
        ] {
            let row = Row::<C>::new(shape, n);
            let expected = reference(points, &row.fields());
            let (mut unsigned, mut fields) = match &row {
                Row::Unsigned(_) => (vec![0; n], vec![]),
                Row::Field(_) => (vec![], vec![PastaField::ZERO; n]),
            };
            let input = row.direct(points);
            let mut direct_scratch = Buffers::new(input.requirements(options).unwrap());
            assert_eq!(
                input
                    .execute(options, &SerialExecutor, direct_scratch.borrow())
                    .unwrap(),
                expected
            );
            let input = row.difference(retained, &mut unsigned, &mut fields);
            let mut difference_scratch = Buffers::new(input.requirements(options).unwrap());
            assert_eq!(
                input
                    .execute(options, &SerialExecutor, difference_scratch.borrow())
                    .unwrap(),
                expected
            );
            let mut group = c.benchmark_group(format!("{name}/differences/{shape}"));
            group.bench_function(BenchmarkId::new("direct", n), |b| {
                b.iter(|| {
                    let input = black_box(&row).direct(black_box(points));
                    black_box(
                        input
                            .execute(options, &SerialExecutor, direct_scratch.borrow())
                            .unwrap(),
                    );
                })
            });
            group.bench_function(BenchmarkId::new("retained_suffix", n), |b| {
                b.iter(|| {
                    let input =
                        black_box(&row).difference(black_box(retained), &mut unsigned, &mut fields);
                    black_box(
                        input
                            .execute(options, &SerialExecutor, difference_scratch.borrow())
                            .unwrap(),
                    );
                })
            });
            group.bench_function(BenchmarkId::new("prepare_and_execute", n), |b| {
                b.iter(|| {
                    let basis = SuffixBasis::prepare(
                        black_box(points),
                        &mut output,
                        &mut projective,
                        &mut field,
                    );
                    let input = black_box(&row).difference(basis, &mut unsigned, &mut fields);
                    black_box(
                        input
                            .execute(options, &SerialExecutor, difference_scratch.borrow())
                            .unwrap(),
                    );
                })
            });
            group.finish();
            if n == 2048 {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(4)
                    .build()
                    .unwrap();
                for (tasks, limit, label) in [
                    (1, 65536, "serial_cap_64k"),
                    (4, 65536, "workers_4_cap_64k"),
                    (4, usize::MAX, "workers_4_cap_all"),
                ] {
                    let options = ExecutionOptions::default()
                        .with_task_budget(TaskBudget::new(tasks).unwrap())
                        .with_memory_limit(limit);
                    let input = row.direct(points);
                    let mut direct_scratch = Buffers::new(input.requirements(options).unwrap());
                    assert_eq!(
                        pool.install(|| input
                            .execute(options, &Pool, direct_scratch.borrow())
                            .unwrap()),
                        expected
                    );
                    let input = row.difference(retained, &mut unsigned, &mut fields);
                    let mut difference_scratch = Buffers::new(input.requirements(options).unwrap());
                    assert_eq!(
                        pool.install(|| input
                            .execute(options, &Pool, difference_scratch.borrow())
                            .unwrap()),
                        expected
                    );
                    let mut group =
                        c.benchmark_group(format!("{name}/differences/{shape}/{label}"));
                    for method in ["direct", "retained_suffix", "prepare_and_execute"] {
                        group.bench_function(BenchmarkId::new(method, n), |b| {
                            measure(b, Some(&pool), || {
                                let (input, scratch) = match method {
                                    "direct" => (
                                        black_box(&row).direct(black_box(points)),
                                        direct_scratch.borrow(),
                                    ),
                                    "retained_suffix" => (
                                        black_box(&row).difference(
                                            black_box(retained),
                                            &mut unsigned,
                                            &mut fields,
                                        ),
                                        difference_scratch.borrow(),
                                    ),
                                    _ => {
                                        let basis = SuffixBasis::prepare(
                                            black_box(points),
                                            &mut output,
                                            &mut projective,
                                            &mut field,
                                        );
                                        (
                                            black_box(&row).difference(
                                                basis,
                                                &mut unsigned,
                                                &mut fields,
                                            ),
                                            difference_scratch.borrow(),
                                        )
                                    }
                                };
                                black_box(input.execute(options, &Pool, scratch).unwrap());
                            });
                        });
                    }
                    group.finish();
                }
            }
        }
    }
}

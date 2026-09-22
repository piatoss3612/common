//! A fixed-pool application driver for internal scheduler tests.
//!
//! Application kernels use Udon's private frontier here to exercise the same
//! failure and publication paths as arithmetic tasks. Downstream schedulers own
//! their application-task state alongside the public arithmetic runs.
//!
//! Seven rotating lanes arbitrate two MSMs, two FFTs, application work, immediate
//! MSM consumers, and the round challenge fence. Every dispatch first obtains
//! all actual guards and an admission permit. There is no per-operation worker
//! allowance, queue growth, or blocking acquisition in the coordinator.

use super::{admission::*, fft_run, msm_run, run_pool};
use crate::exec::run::frontier::ReadyRange;
use spin::{RwLock, RwLockWriteGuard};
use std::num::NonZeroUsize;
use std::{vec, vec::Vec};
use zakura_udon::{
    curve::{
        AffinePoint, Pallas, ProjectivePoint,
        msm::{
            Bases, Input,
            run::{
                Buffers as MsmBuffers, MsmKernel, MsmOutput, MsmPlan, MsmRun,
                Resources as MsmResources,
            },
        },
    },
    exec::{
        ExecutionOptions, TaskBudget,
        run::{Completion, Frontier, Identity, Kernel, Outcome, Task, TaskStorage},
    },
    fft::{
        Direction, Domain, StorageLayout, Transform, TransformRequest,
        run::{Buffers as FftBuffers, FftKernel, FftPlan, FftRun, Resources as FftResources},
    },
    field::{CanonicalUint, Fp, Fq, PallasBase},
};

const CLASSES: usize = 7;
const RETAINED: Resources<CLASSES> = Resources([1, 1, 1, 1, 0, 1, 0]);
const TEMPORARY: Resources<CLASSES> = Resources([0, 0, 0, 0, 1, 0, 1]);
const ZERO: Resources<CLASSES> = Resources::ZERO;
const FRONTIER: usize = 32;
const APP_CHUNKS: usize = 8;
pub const CEILING: usize = 16 * 1024 * 1024;

pub struct Fixture {
    bases: Vec<AffinePoint<Pallas>>,
    scalars: Vec<Fq>,
    coefficients: Vec<Fp>,
    rounds: Vec<[MsmPlan<Pallas>; 2]>,
    fft_plans: Vec<[FftPlan<'static, PallasBase>; 2]>,
    terms: usize,
    msm: [msm_run::Arena<Pallas>; 2],
    fft: [fft_run::Arena<PallasBase>; 2],
    scratch: Vec<RwLock<msm_run::Work<Pallas>>>,
    app: Vec<RwLock<Vec<u64>>>,
}

impl Fixture {
    pub fn new(terms: usize, scratch_slots: usize) -> Self {
        Self::build(terms, scratch_slots)
    }

    fn build(terms: usize, scratch_slots: usize) -> Self {
        assert!(terms >= 8 && terms.is_power_of_two() && scratch_slots > 0);
        let bases = (0..terms)
            .scan(ProjectivePoint::<Pallas>::GENERATOR, |point, _| {
                let result = *point.to_point().as_affine().unwrap();
                *point = point.add(&ProjectivePoint::GENERATOR);
                Some(result)
            })
            .collect();
        let mut seed = 0x243f_6a88_85a3_08d3_u64;
        let scalars = (0..terms)
            .map(|_| {
                let mut limbs = core::array::from_fn(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed
                });
                limbs[3] &= (1 << 62) - 1;
                Fq::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap()
            })
            .collect();
        let coefficients = (0..16384).map(|i| Fp::from_u64(i * i + 1)).collect();
        let rounds: Vec<_> = (0..11.min(terms.ilog2() as usize + 1))
            .map(|round| {
                let n = terms >> round;
                [n, (n / 8).max(1)].map(|n| {
                    MsmPlan::new(
                        n,
                        ExecutionOptions::default()
                            .with_task_budget(TaskBudget::new(scratch_slots).unwrap()),
                    )
                    .unwrap()
                })
            })
            .collect();
        let fft_plans: Vec<_> = (0..rounds.len())
            .map(|round| {
                [14 - round.min(3) as u32, 11].map(|log| {
                    FftPlan::new(
                        Transform::new(Domain::new(log).unwrap().subgroup()),
                        TransformRequest::new(Direction::Forward),
                        StorageLayout::Fragments {
                            length: NonZeroUsize::new(1024).unwrap(),
                            whole_bank: false,
                        },
                        ExecutionOptions::default()
                            .with_task_budget(TaskBudget::new(scratch_slots).unwrap()),
                    )
                    .unwrap()
                })
            })
            .collect();
        Self {
            bases,
            scalars,
            coefficients,
            terms,
            msm: core::array::from_fn(|op| msm_run::Arena::for_plans(rounds.iter().map(|p| p[op]))),
            fft: fft_plans[0].map(fft_run::Arena::new),
            scratch: (0..scratch_slots)
                .map(|_| {
                    RwLock::new(msm_run::Work::new(
                        rounds.iter().flatten().map(MsmPlan::temporary),
                    ))
                })
                .collect(),
            app: (0..APP_CHUNKS)
                .map(|_| RwLock::new(vec![0; 1024]))
                .collect(),
            rounds,
            fft_plans,
        }
    }

    fn input(&self, round: usize, op: usize) -> Input<'_, Pallas> {
        let n = self.terms >> round;
        let n = if op == 0 { n } else { (n / 8).max(1) };
        Input::new(Bases::Affine(&self.bases[..n]), &self.scalars[..n]).unwrap()
    }
}

struct Bundle<'a, R> {
    lease: R,
    permit: TaskPermit<'a, CLASSES>,
}
impl<R: MsmResources<Pallas>> MsmResources<Pallas> for Bundle<'_, R> {
    fn buffers(&mut self) -> MsmBuffers<'_, Pallas> {
        self.lease.buffers()
    }
}
impl<R: FftResources<PallasBase>> FftResources<PallasBase> for Bundle<'_, R> {
    fn buffers(&mut self) -> FftBuffers<'_, PallasBase> {
        self.lease.buffers()
    }
}

enum AppKernel {
    Update { round: usize, start: usize },
    Consume(ProjectivePoint<Pallas>),
    Fence(usize),
}
impl Kernel<Bundle<'_, RwLockWriteGuard<'_, Vec<u64>>>> for AppKernel {
    type Output = ();
    fn execute(&mut self, resources: &mut Bundle<'_, RwLockWriteGuard<'_, Vec<u64>>>) {
        let values = &mut resources.lease;
        match *self {
            Self::Update { round, start } => {
                for (i, value) in values.iter_mut().enumerate() {
                    *value = value
                        .wrapping_add((start + i + round) as u64)
                        .rotate_left(17);
                }
            }
            Self::Consume(point) => {
                std::hint::black_box(point);
            }
            Self::Fence(round) => {
                std::hint::black_box((round, &values));
            }
        }
    }
}

type MsmTask<'a, 'i> = Task<'a, MsmKernel<'i, Pallas>, Bundle<'a, msm_run::Lease<'i, Pallas>>>;
type FftTask<'a, 'i> =
    Task<'a, FftKernel<'static, PallasBase>, Bundle<'a, fft_run::Lease<'i, PallasBase>>>;
type AppTask<'a, 'i> = Task<'a, AppKernel, Bundle<'a, RwLockWriteGuard<'i, Vec<u64>>>>;
#[expect(
    clippy::large_enum_variant,
    reason = "fixed envelopes avoid task allocation"
)]
enum Work<'a, 'i> {
    Msm(usize, MsmTask<'a, 'i>),
    Fft(usize, FftTask<'a, 'i>),
    App(usize, AppTask<'a, 'i>),
}
#[expect(
    clippy::large_enum_variant,
    reason = "fixed envelopes avoid task allocation"
)]
enum Receipt<'a, 'i> {
    Msm(
        usize,
        Completion<'a, Bundle<'a, msm_run::Lease<'i, Pallas>>, MsmOutput<Pallas>>,
    ),
    Fft(
        usize,
        Completion<
            'a,
            Bundle<'a, fft_run::Lease<'i, PallasBase>>,
            Result<(), zakura_udon::fft::FftError>,
        >,
    ),
    App(
        usize,
        Completion<'a, Bundle<'a, RwLockWriteGuard<'i, Vec<u64>>>, ()>,
    ),
}
impl<'a, 'i> run_pool::Work for Work<'a, 'i> {
    type Completion = Receipt<'a, 'i>;
    fn execute(&mut self) {
        match self {
            Self::Msm(_, t) => t.execute().unwrap(),
            Self::Fft(_, t) => t.execute().unwrap(),
            Self::App(_, t) => t.execute().unwrap(),
        }
    }
    fn complete(self) -> Self::Completion {
        match self {
            Self::Msm(i, t) => Receipt::Msm(i, t.finish()),
            Self::Fft(i, t) => Receipt::Fft(i, t.finish()),
            Self::App(i, t) => Receipt::App(i, t.finish()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub bytes: usize,
    pub peak: Resources<CLASSES>,
    pub tasks: usize,
    pub early_consumers: usize,
    pub rounds: usize,
    pub results: [ProjectivePoint<Pallas>; 2],
}

/// Calls `use_driver` after all allocations and worker creation. The supplied
/// closure repeats the same shrinking workload using that pool and arena.
#[allow(dead_code)] // The benchmark omits the independent arithmetic oracle.
pub fn scoped<O>(
    fixture: &Fixture,
    workers: usize,
    queue: usize,
    use_driver: impl FnOnce(&mut dyn FnMut() -> Stats) -> O,
) -> O {
    scoped_impl::<false, O>(fixture, workers, queue, use_driver)
}

/// Checks every round against a scalar ladder and sampled polynomial evaluation.
/// Kept separate from the measured driver so oracles never enter timings.
pub fn scoped_checked<O>(
    fixture: &Fixture,
    workers: usize,
    queue: usize,
    use_driver: impl FnOnce(&mut dyn FnMut() -> Stats) -> O,
) -> O {
    scoped_impl::<true, O>(fixture, workers, queue, use_driver)
}

fn scoped_impl<const CHECK: bool, O>(
    fixture: &Fixture,
    workers: usize,
    queue: usize,
    use_driver: impl FnOnce(&mut dyn FnMut() -> Stats) -> O,
) -> O {
    let mut msm_identity = [Identity::new(), Identity::new()];
    let mut fft_identity = [Identity::new(), Identity::new()];
    let mut app_identity = [Identity::new(), Identity::new(), Identity::new()];
    let mut msm_slots = [const { [const { TaskStorage::EMPTY }; FRONTIER] }; 2];
    let mut fft_slots = [const { [const { TaskStorage::EMPTY }; FRONTIER] }; 2];
    let mut app_slots = [const { [const { TaskStorage::EMPTY }; APP_CHUNKS] }; 3];
    let [mi0, mi1] = &mut msm_identity;
    let [ms0, ms1] = &mut msm_slots;
    let mut msm = [
        MsmRun::new(fixture.rounds[0][0], fixture.input(0, 0), mi0, ms0).unwrap(),
        MsmRun::new(fixture.rounds[0][1], fixture.input(0, 1), mi1, ms1).unwrap(),
    ];
    let [fi0, fi1] = &mut fft_identity;
    let [fs0, fs1] = &mut fft_slots;
    let mut fft = [
        FftRun::new(fixture.fft_plans[0][0], false, fi0, fs0).unwrap(),
        FftRun::new(fixture.fft_plans[0][1], false, fi1, fs1).unwrap(),
    ];
    let [ai0, ai1, ai2] = &mut app_identity;
    let [as0, as1, as2] = &mut app_slots;
    let mut app = [
        Frontier::new(ai0, as0, APP_CHUNKS).unwrap(),
        Frontier::new(ai1, as1, 2).unwrap(),
        Frontier::new(ai2, as2, 1).unwrap(),
    ];
    let mut admission_identity = Identity::new();
    let mut segments = [SegmentStorage::EMPTY];
    let mut started = false;
    run_pool::scoped(workers, queue, |pool| {
        let metadata = pool.queue_bytes()
            + size_of_val(&msm)
            + size_of_val(&fft)
            + size_of_val(&app)
            + (4 * FRONTIER + 3 * APP_CHUNKS) * size_of::<TaskStorage>()
            + 8 * size_of::<Identity>()
            + size_of_val(&segments)
            + size_of::<Admission<'_, CLASSES>>()
            + size_of::<Stats>()
            + size_of::<[Option<zakura_udon::curve::msm::run::Request<'_>>; FRONTIER]>()
            + size_of::<[Option<zakura_udon::fft::run::Request<'_>>; FRONTIER]>()
            + size_of::<ReadyRange<'_>>()
            + size_of::<Option<Segment<'_>>>()
            + size_of_val(fixture)
            + fixture.rounds.capacity() * size_of::<[MsmPlan<Pallas>; 2]>()
            + fixture.fft_plans.capacity() * size_of::<[FftPlan<'_, PallasBase>; 2]>()
            + (fixture.scratch.capacity() - fixture.scratch.len())
                * size_of::<RwLock<msm_run::Work<Pallas>>>()
            + 1024; // Conservative fixed coordinator locals, counters, and alignment.
        let layout = ArenaLayout {
            classes: [
                BlockClass {
                    blocks: 1,
                    block_bytes: fixture.msm[0].bytes(),
                },
                BlockClass {
                    blocks: 1,
                    block_bytes: fixture.msm[1].bytes(),
                },
                BlockClass {
                    blocks: 1,
                    block_bytes: fixture.fft[0].bytes(),
                },
                BlockClass {
                    blocks: 1,
                    block_bytes: fixture.fft[1].bytes(),
                },
                BlockClass {
                    blocks: fixture.scratch.len(),
                    block_bytes: fixture.scratch[0].read().bytes()
                        + size_of::<RwLock<msm_run::Work<Pallas>>>()
                        - size_of::<msm_run::Work<Pallas>>(),
                },
                BlockClass {
                    blocks: 1,
                    block_bytes: fixture.app.capacity() * size_of::<RwLock<Vec<u64>>>()
                        + fixture
                            .app
                            .iter()
                            .map(|s| s.read().capacity() * size_of::<u64>())
                            .sum::<usize>(),
                },
                // Queue envelope storage is already charged by queue_bytes above.
                BlockClass {
                    blocks: queue,
                    block_bytes: 0,
                },
            ],
            metadata_bytes: metadata,
        };
        let bytes = layout.check(CEILING).unwrap();
        assert!(layout.check(bytes - 1).is_err());
        let mut admission =
            Admission::new(&mut admission_identity, &mut segments, layout.capacity());
        let mut execute = || {
            let mut expected_app = [[0; 3]; APP_CHUNKS];
            if CHECK {
                for (chunk, expected) in fixture.app.iter().zip(&mut expected_app) {
                    let chunk = chunk.read();
                    *expected = [chunk[0], chunk[17], chunk[1023]];
                }
            }
            let mut stats = Stats {
                bytes,
                peak: ZERO,
                tasks: 0,
                early_consumers: 0,
                rounds: 0,
                results: [ProjectivePoint::IDENTITY; 2],
            };
            let mut cursor = 0;
            for _ in 0..2 {
                for round in 0..fixture.rounds.len() {
                    if started {
                        for op in 0..2 {
                            msm[op]
                                .rebind(fixture.rounds[round][op], fixture.input(round, op))
                                .unwrap();
                            fft[op].rebind(fixture.fft_plans[round][op], false).unwrap();
                        }
                        app[0].restart(APP_CHUNKS).unwrap();
                        app[1].restart(2).unwrap();
                        app[2].restart(1).unwrap();
                    }
                    started = true;
                    for op in 0..2 {
                        fixture.fft[op]
                            .write(&fixture.coefficients[..fixture.fft_plans[round][op].size()]);
                    }
                    let segment = admission
                        .admit(Profile {
                            retained: RETAINED,
                            temporary: TEMPORARY,
                        })
                        .unwrap();
                    let initial = admission.try_task(&segment, ZERO, RETAINED).unwrap();
                    admission.finish_task(initial, true).unwrap();
                    let mut mr = [None; FRONTIER];
                    let mut fr = core::array::from_fn::<_, FRONTIER, _>(|_| None);
                    let mut ar = [ReadyRange::EMPTY];
                    while !app[2].is_complete() {
                        // One successful claim advances the cursor. A scan skips
                        // unavailable bundles and dependent work without waiting.
                        while pool.available() {
                            let mut submitted = false;
                            for distance in 0..7 {
                                let lane = (cursor + distance) % 7;
                                let task = if lane < 2 {
                                    let op = lane;
                                    let count = msm[op].ready(&mut mr);
                                    let mut task = None;
                                    for request in mr[..count].iter().flatten() {
                                        task = msm[op]
                                            .try_claim(*request, || {
                                                let lease = fixture.msm[op]
                                                    .acquire(*request, &fixture.scratch)?;
                                                let r = request.scratch;
                                                let scratch = usize::from(
                                                    r.affine()
                                                        | r.projective()
                                                        | r.field()
                                                        | r.indices()
                                                        != 0,
                                                );
                                                let permit = admission
                                                    .try_task(
                                                        &segment,
                                                        Resources([0, 0, 0, 0, scratch, 0, 1]),
                                                        ZERO,
                                                    )
                                                    .ok()?;
                                                Some(Bundle { lease, permit })
                                            })
                                            .unwrap();
                                        if task.is_some() {
                                            break;
                                        }
                                    }
                                    task.map(|t| Work::Msm(op, t))
                                } else if lane < 4 {
                                    let op = lane - 2;
                                    let count = fft[op].ready(&mut fr);
                                    let mut task = None;
                                    for request in fr[..count].iter().flatten() {
                                        task = fft[op]
                                            .try_claim(request.clone(), || {
                                                let lease =
                                                    fixture.fft[op].acquire(request, &[], &[])?;
                                                let permit = admission
                                                    .try_task(
                                                        &segment,
                                                        Resources([0, 0, 0, 0, 0, 0, 1]),
                                                        ZERO,
                                                    )
                                                    .ok()?;
                                                Some(Bundle { lease, permit })
                                            })
                                            .unwrap();
                                        if task.is_some() {
                                            break;
                                        }
                                    }
                                    task.map(|t| Work::Fft(op, t))
                                } else {
                                    let op = lane - 4;
                                    let enabled = op == 0
                                        || op == 1 && app[0].is_complete()
                                        || op == 2
                                            && app[0].is_complete()
                                            && app[1].is_complete()
                                            && fft.iter().all(FftRun::is_complete);
                                    let mut task = None;
                                    if enabled && app[op].ready(&mut ar) > 0 {
                                        for key in ar[0].tasks() {
                                            let kernel = match op {
                                                0 => AppKernel::Update {
                                                    round,
                                                    start: key.index() * 1024,
                                                },
                                                1 => {
                                                    let Some(result) = msm[key.index()].result()
                                                    else {
                                                        continue;
                                                    };
                                                    AppKernel::Consume(result)
                                                }
                                                _ => AppKernel::Fence(round),
                                            };
                                            task = app[op]
                                                .try_claim(key, kernel, || {
                                                    let lease =
                                                        fixture.app[key.index()].try_write()?;
                                                    let permit = admission
                                                        .try_task(
                                                            &segment,
                                                            Resources([0, 0, 0, 0, 0, 0, 1]),
                                                            ZERO,
                                                        )
                                                        .ok()?;
                                                    Some(Bundle { lease, permit })
                                                })
                                                .unwrap();
                                            if task.is_some() {
                                                if op == 1
                                                    && msm[1 - key.index()].result().is_none()
                                                {
                                                    stats.early_consumers += 1;
                                                }
                                                break;
                                            }
                                        }
                                    }
                                    task.map(|t| Work::App(op, t))
                                };
                                if let Some(task) = task {
                                    assert!(pool.submit(task).is_ok());
                                    stats.tasks += 1;
                                    cursor = (lane + 1) % 7;
                                    submitted = true;
                                    break;
                                }
                            }
                            if !submitted {
                                break;
                            }
                        }
                        // A dispatched task has a complete bundle and is finite.
                        // No receipt here would indicate a dependency/admission bug.
                        let receipt = pool.receive().expect("admitted segment has no escape task");
                        match receipt {
                            Receipt::Msm(op, receipt) => {
                                let published = msm[op].complete(receipt).unwrap();
                                assert_eq!(published.outcome, Outcome::Success);
                                assert_eq!(published.error, None);
                                let Bundle { lease, permit } = published.resources;
                                drop(lease);
                                admission.finish_task(permit, true).unwrap();
                                if let Some(result) = published.result {
                                    stats.results[op] = result;
                                }
                            }
                            Receipt::Fft(op, receipt) => {
                                let published = fft[op].complete(receipt).unwrap();
                                assert_eq!(published.outcome, Outcome::Success);
                                assert_eq!(published.error, None);
                                let Bundle { lease, permit } = published.resources;
                                drop(lease);
                                admission.finish_task(permit, true).unwrap();
                            }
                            Receipt::App(op, receipt) => {
                                let index = receipt.key().index();
                                let completed = app[op].complete(receipt).unwrap();
                                assert_eq!(completed.outcome, Outcome::Success);
                                let Bundle { lease, permit } = completed.resources;
                                drop(lease);
                                admission.finish_task(permit, true).unwrap();
                                if op == 1 {
                                    let mut release = ZERO;
                                    release.0[index] = 1;
                                    admission.release_retained(&segment, release).unwrap();
                                }
                            }
                        }
                    }
                    admission
                        .release_retained(&segment, Resources([0, 0, 1, 1, 0, 1, 0]))
                        .unwrap();
                    admission.retire(segment).unwrap();
                    assert_eq!(admission.used(), ZERO);
                    if CHECK {
                        for (op, result) in stats.results.iter().enumerate() {
                            let n = fixture.input(round, op).len();
                            let scalar = fixture.scalars[..n].iter().enumerate().fold(
                                Fq::ZERO,
                                |sum, (i, scalar)| {
                                    sum.add(&scalar.mul(&Fq::from_u64(i as u64 + 1)))
                                },
                            );
                            let mut expected = ProjectivePoint::IDENTITY;
                            for byte in scalar.to_bytes().iter().rev() {
                                for bit in (0..8).rev() {
                                    expected = expected.double();
                                    if byte & (1 << bit) != 0 {
                                        expected = expected.add(&ProjectivePoint::GENERATOR);
                                    }
                                }
                            }
                            assert_eq!(*result, expected, "MSM round {round}, operation {op}");
                            let n = fixture.fft_plans[round][op].size();
                            let root = Domain::<PallasBase>::for_size(n).unwrap().root();
                            for index in [0, 1, n / 3, n / 2, n - 1] {
                                let point = root.pow_u64(index as u64);
                                let expected = fixture.coefficients[..n]
                                    .iter()
                                    .rev()
                                    .fold(Fp::ZERO, |sum, coefficient| {
                                        sum.mul(&point).add(coefficient)
                                    });
                                assert_eq!(
                                    fixture.fft[op].values[index / 1024].read()[index % 1024],
                                    expected,
                                    "FFT round {round}, operation {op}, row {index}"
                                );
                            }
                        }
                        for (chunk, expected) in expected_app.iter_mut().enumerate() {
                            let actual = fixture.app[chunk].read();
                            for (index, expected) in [0, 17, 1023].into_iter().zip(expected) {
                                *expected = expected
                                    .wrapping_add((chunk * 1024 + index + round) as u64)
                                    .rotate_left(17);
                                assert_eq!(actual[index], *expected);
                            }
                        }
                    }
                    stats.rounds += 1;
                }
            }
            stats.peak = admission.peak();
            stats
        };
        use_driver(&mut execute)
    })
}

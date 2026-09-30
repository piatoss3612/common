//! Prepared storage, deterministic inputs, and executor probes.

use super::*;

pub(in crate::fft) fn reduced<M: PrimeModulus>(
    values: &[PastaField<M>],
) -> Vec<PastaField<M, Reduced>> {
    values.iter().map(|value| value.reduce()).collect()
}

pub(super) struct Prepared<M: PrimeModulus> {
    pub(super) forward: Vec<PastaField<M>>,
    pub(super) inverse: Vec<PastaField<M>>,
    pub(super) finish: Vec<PastaField<M>>,
}

impl<M: PrimeModulus> Prepared<M> {
    pub(super) fn new(domain: CosetDomain<M>) -> Self {
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

    pub(super) fn tables(&self) -> Tables<'_, M> {
        Tables {
            forward: Some(&self.forward),
            inverse: Some(&self.inverse),
            inverse_finish: Some(&self.finish),
        }
    }
}

pub(in crate::fft) fn inputs<M: PrimeModulus>(size: usize) -> Vec<PastaField<M>> {
    let mut samples = field_samples();
    (0..size)
        .map(|index| match index % 17 {
            0 => PastaField::ZERO,
            1 => PastaField::<_>::ONE.neg(),
            _ => samples.next().unwrap(),
        })
        .collect()
}

pub(super) fn assert_loose_bound<M: PrimeModulus>(values: &[PastaField<M>]) {
    let bound = twice_modulus::<M>();
    for value in values {
        assert!(integer(&value.montgomery_limbs()) < bound);
        assert_eq!(
            (PastaField::<M>::from_bytes(value.to_bytes())).map(|value| value.reduce()),
            (Some(*value)).map(|value| value.reduce())
        );
    }
}

pub(super) struct Threads;
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

#[derive(Default)]
pub(super) struct CountJoins(pub(super) AtomicUsize);

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
    pub(super) fn take(&self) -> usize {
        self.0.swap(0, Ordering::SeqCst)
    }
}

pub(super) struct FailAt {
    pub(super) calls: AtomicUsize,
    pub(super) index: usize,
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

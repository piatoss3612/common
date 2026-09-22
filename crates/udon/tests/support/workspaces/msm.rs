use zakura_udon::curve::msm::run::{BatchPlan, JobStorage, WorkerStorage};
use zakura_udon::{
    curve::{
        AffinePoint, CurveError, PastaCurve, ProjectivePoint,
        msm::{Input, Requirements, ScalarStorage, Scratch},
    },
    exec::{ExecutionOptions, Executor},
    field::PastaField,
};

pub struct MsmWorkspace<C: PastaCurve> {
    scalars: Vec<ScalarStorage<C>>,
    digits: Vec<u8>,
    affine: Vec<AffinePoint<C>>,
    projective: Vec<ProjectivePoint<C>>,
    field: Vec<PastaField<C::Base>>,
    indices: Vec<usize>,
    jobs: Vec<JobStorage>,
    workers: Vec<WorkerStorage>,
}

impl<C: PastaCurve> MsmWorkspace<C> {
    pub fn new() -> Self {
        Self {
            scalars: Vec::new(),
            digits: Vec::new(),
            affine: Vec::new(),
            projective: Vec::new(),
            field: Vec::new(),
            indices: Vec::new(),
            jobs: Vec::new(),
            workers: Vec::new(),
        }
    }

    /// Prepares a borrowed run, growing initialized storage as needed.
    ///
    /// The run borrows scalar rows and separate metadata and scratch fields.
    /// Dropping it releases the workspace for the next batch.
    pub fn prepare<'a, 'i>(
        &'a mut self,
        inputs: &'a [Input<'i, C>],
        options: ExecutionOptions,
    ) -> Result<MsmRun<'a, 'i, C>, CurveError> {
        let (jobs, workers) = BatchPlan::<C>::storage_len(inputs.len(), options)?;
        self.jobs.resize(jobs, JobStorage::EMPTY);
        self.workers.resize(workers, WorkerStorage::EMPTY);
        let plan = BatchPlan::new(inputs, options, &mut self.jobs, &mut self.workers)?;
        // Size arithmetic scratch from the resolved plan. Metadata is separate.
        let required = plan.requirements();
        self.scalars.resize(required.scalars(), ScalarStorage::ZERO);
        self.digits.resize(required.digits(), 0);
        self.affine
            .resize(required.affine(), AffinePoint::GENERATOR);
        self.projective
            .resize(required.projective(), ProjectivePoint::IDENTITY);
        self.field.resize(required.field(), PastaField::ZERO);
        self.indices.resize(required.indices(), 0);
        let scratch = Scratch::new(
            &mut self.scalars,
            &mut self.digits,
            &mut self.affine,
            &mut self.projective,
            &mut self.field,
            &mut self.indices,
        );
        Ok(MsmRun { plan, scratch })
    }

    pub fn capacities(&self) -> [usize; 8] {
        [
            self.scalars.capacity(),
            self.digits.capacity(),
            self.affine.capacity(),
            self.projective.capacity(),
            self.field.capacity(),
            self.indices.capacity(),
            self.jobs.capacity(),
            self.workers.capacity(),
        ]
    }

    pub fn capacity_bytes(&self) -> usize {
        let sizes = [
            size_of::<ScalarStorage<C>>(),
            1,
            size_of::<AffinePoint<C>>(),
            size_of::<ProjectivePoint<C>>(),
            size_of::<PastaField<C::Base>>(),
            size_of::<usize>(),
            size_of::<JobStorage>(),
            size_of::<WorkerStorage>(),
        ];
        self.capacities()
            .into_iter()
            .zip(sizes)
            .map(|(n, size)| n * size)
            .sum()
    }
}

pub struct MsmRun<'a, 'i, C: PastaCurve> {
    plan: BatchPlan<'a, 'i, C>,
    scratch: Scratch<'a, C>,
}

impl<C: PastaCurve> MsmRun<'_, '_, C> {
    pub fn requirements(&self) -> Requirements {
        self.plan.requirements()
    }

    pub fn temporary_bytes(&self) -> usize {
        self.plan.temporary_bytes()
    }

    /// Execution has access only to slices and cannot grow the workspace.
    pub fn execute<E: Executor>(
        &mut self,
        output: &mut [ProjectivePoint<C>],
        executor: &E,
    ) -> Result<(), CurveError> {
        self.plan.execute(output, executor, self.scratch.reborrow())
    }
}

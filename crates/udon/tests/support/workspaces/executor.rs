use rayon::ThreadPool;
use zakura_udon::exec::{Executor, SerialExecutor, TaskBudget};

/// Runs scoped joins on a caller-selected pool.
///
/// Every join enters that pool, including calls from outside a worker or from
/// another pool's worker.
pub struct RayonExecutor<'pool>(pub &'pool ThreadPool);

impl Executor for RayonExecutor<'_> {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        self.0.install(|| rayon::join(left, right))
    }
}

/// Reserve one allowance for serial side work, giving the remainder to MSM.
///
/// A single allowance executes both branches serially, including on unwind.
pub fn with_side_work<E, L, R, A, B>(executor: &E, budget: TaskBudget, left: L, right: R) -> (A, B)
where
    E: Executor,
    L: FnOnce(TaskBudget) -> A + Send,
    R: FnOnce(TaskBudget) -> B + Send,
    A: Send,
    B: Send,
{
    match budget.split_at(budget.get() - 1) {
        Some((main, side)) => executor.join(|| left(main), || right(side)),
        None => SerialExecutor.join(|| left(TaskBudget::SERIAL), || right(TaskBudget::SERIAL)),
    }
}

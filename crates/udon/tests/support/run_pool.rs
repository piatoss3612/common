//! Application-owned scoped workers and movable typed lease guards.
//!
//! All queue and block allocation happens before dispatch. The coordinator
//! acquires actual block guards and accounting permits before submitting a task.
//! Pending work holds neither a worker nor a partial resource bundle.

use std::{
    collections::VecDeque,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Condvar, Mutex},
};
use zakura_udon::exec::run::{Completion, Kernel, Task};

pub(crate) trait Work {
    type Completion;
    fn execute(&mut self);
    // Must not panic. Kernel panics have already been caught with self retained.
    fn complete(self) -> Self::Completion;
}

impl<'a, K: Kernel<R>, R> Work for Task<'a, K, R> {
    type Completion = Completion<'a, R, K::Output>;
    fn execute(&mut self) {
        Task::execute(self).unwrap();
    }
    fn complete(self) -> Self::Completion {
        self.finish()
    }
}

impl<T: Work> Work for (usize, T) {
    type Completion = (usize, T::Completion);
    fn execute(&mut self) {
        self.1.execute();
    }
    fn complete(self) -> Self::Completion {
        (self.0, self.1.complete())
    }
}

struct Queues<T: Work> {
    ready: VecDeque<T>,
    complete: VecDeque<T::Completion>,
    outstanding: usize,
    stop: bool,
}

struct Shared<T: Work> {
    queues: Mutex<Queues<T>>,
    work: Condvar,
    completion: Condvar,
    capacity: usize,
    workers: usize,
}

pub(crate) struct Pool<'a, T: Work> {
    shared: &'a Shared<T>,
}

impl<T: Work> Pool<'_, T> {
    pub(crate) fn available(&self) -> bool {
        self.shared.queues.lock().unwrap().outstanding < self.shared.capacity
    }

    // Capacity represents task envelopes in every state, including the receipt.
    // Check available before acquiring a bundle; the single coordinator is the
    // only dispatcher, so that credit cannot be taken concurrently.
    pub(crate) fn submit(&mut self, task: T) -> Result<(), T> {
        let mut queues = self.shared.queues.lock().unwrap();
        if queues.outstanding == self.shared.capacity {
            return Err(task);
        }
        queues.outstanding += 1;
        queues.ready.push_back(task);
        self.shared.work.notify_one();
        Ok(())
    }

    pub(crate) fn receive(&mut self) -> Option<T::Completion> {
        let mut queues = self.shared.queues.lock().unwrap();
        loop {
            if let Some(completion) = queues.complete.pop_front() {
                queues.outstanding -= 1;
                return Some(completion);
            }
            if queues.outstanding == 0 {
                return None;
            }
            queues = self.shared.completion.wait(queues).unwrap();
        }
    }

    pub(crate) fn queue_bytes(&self) -> usize {
        let queues = self.shared.queues.lock().unwrap();
        size_of::<Shared<T>>() + size_of::<Self>()
            + queues.ready.capacity() * size_of::<T>()
            + queues.complete.capacity() * size_of::<T::Completion>()
            // Bound owned worker and coordinator envelopes in addition to the
            // full backing arrays, including their temporarily vacant slots.
            + (self.shared.workers.min(self.shared.capacity) + 2)
                * (size_of::<T>() + size_of::<T::Completion>())
    }
}

impl<T: Work> Drop for Pool<'_, T> {
    fn drop(&mut self) {
        self.shared.queues.lock().unwrap().stop = true;
        self.shared.work.notify_all();
    }
}

pub(crate) fn scoped<T: Work + Send, O>(
    workers: usize,
    capacity: usize,
    coordinator: impl FnOnce(&mut Pool<'_, T>) -> O,
) -> O
where
    T::Completion: Send,
{
    assert!(workers > 0 && capacity > 0);
    let shared: Shared<T> = Shared {
        queues: Mutex::new(Queues {
            ready: VecDeque::with_capacity(capacity),
            complete: VecDeque::with_capacity(capacity),
            outstanding: 0,
            stop: false,
        }),
        work: Condvar::new(),
        completion: Condvar::new(),
        capacity,
        workers,
    };
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let shared = &shared;
            scope.spawn(move || {
                loop {
                    let task = {
                        let mut queues = shared.queues.lock().unwrap();
                        loop {
                            if let Some(task) = queues.ready.pop_front() {
                                break Some(task);
                            }
                            if queues.stop {
                                break None;
                            }
                            queues = shared.work.wait(queues).unwrap();
                        }
                    };
                    let Some(mut task) = task else {
                        break;
                    };
                    let _ = catch_unwind(AssertUnwindSafe(|| task.execute()));
                    let completion = task.complete();
                    shared.queues.lock().unwrap().complete.push_back(completion);
                    shared.completion.notify_one();
                }
            });
        }
        coordinator(&mut Pool { shared: &shared })
    })
}

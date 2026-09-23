//! Structured dispatch shared by the typed FFT and MSM drivers.

// Keep the run and resource types inferred at each call site without adding a
// coordinator trait to the public incremental execution protocol.
macro_rules! dispatch {
    ($run:expr, $requests:expr, $leases:expr, $executor:expr, $message:literal) => {{
        let run = $run;
        let requests = $requests;
        let mut leases = $leases;
        let executor = $executor;
        if requests.len() == 1 {
            let mut task = run
                .try_claim(requests[0].take().unwrap(), || leases.next())
                .expect("structured claim")
                .expect("complete resource iterator");
            task.execute().expect("fresh task");
            let published = run.complete(task.finish()).expect("structured receipt");
            assert!(published.error.is_none(), $message);
        } else {
            let mut tasks: [Option<_>; 32] = ::core::array::from_fn(|_| None);
            let count = requests.len();
            for ((slot, request), lease) in tasks.iter_mut().zip(requests).zip(leases) {
                *slot = run
                    .try_claim(request.take().expect("ready request"), || Some(lease))
                    .expect("structured task claim");
            }
            $crate::exec::for_each_task_mut(&mut tasks[..count], executor, |_, task| {
                task.as_mut()
                    .expect("complete resource iterator")
                    .execute()
                    .expect("fresh task")
            });
            let mut error = None;
            for task in &mut tasks[..count] {
                let published = run
                    .complete(task.take().unwrap().finish())
                    .expect("structured receipt");
                error = error.or(published.error);
            }
            assert!(error.is_none(), $message);
        }
    }};
}

pub(crate) use dispatch;

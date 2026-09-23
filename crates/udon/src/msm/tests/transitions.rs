//! Automatic window changes, with an oracle that stays cheap for large rows.

use super::super::{recode::Geometry, test_support::count_kernels};
use super::*;

fn transitions<C: PastaCurve>() {
    let g = Point::<C>::GENERATOR;
    let bases = [
        g,
        g.neg(),
        Point::IDENTITY,
        g.to_projective().double().to_point(),
    ];
    // Public plans cap preparation at 8192 terms, so the selector's 32768-term
    // width is unreachable here. Cover the reachable window changes and the
    // chunk boundary, checking actual execution as well as the result.
    for (boundary, before, after) in [
        (BOOTH_MIN, Geometry::Joint, Geometry::Booth(6)),
        (192, Geometry::Booth(6), Geometry::Booth(7)),
        (512, Geometry::Booth(7), Geometry::Booth(8)),
        (4096, Geometry::Booth(8), Geometry::Booth(10)),
        (8192, Geometry::Booth(10), Geometry::Booth(10)),
    ] {
        for n in [boundary - 1, boundary, boundary + 1] {
            let mut scalars: Vec<_> = field_samples::<C::Scalar>().take(n).collect();
            for pair in scalars.chunks_mut(8) {
                if pair.len() > 1 {
                    pair[1] = pair[0];
                }
            }
            let indices: Vec<_> = (0..n).map(|i| (i % bases.len()) as u32).collect();
            // All bases are known multiples of G. Accumulate those coefficients
            // and use one binary ladder, bypassing MSM decomposition and buckets.
            let coefficient = scalars
                .iter()
                .zip(&indices)
                .fold(PastaField::ZERO, |sum, (k, i)| match i {
                    0 => sum.add(k),
                    1 => sum.sub(k),
                    2 => sum,
                    3 => sum.add(&k.double()),
                    _ => unreachable!(),
                });
            let expected =
                scalar::multiply(&coefficient, |sum| sum.add_mixed(g.as_affine().unwrap()));
            let input = Input::indexed(Bases::Points(&bases), &indices, &scalars).unwrap();
            for tasks in [1, 3] {
                let options =
                    ExecutionOptions::default().with_task_budget(TaskBudget::new(tasks).unwrap());
                let required = input.requirements(options).unwrap();
                let mut buffers = Buffers::new(required);
                let geometry = if tasks > 1 && n >= 4096 {
                    Geometry::Booth(11)
                } else if n < boundary {
                    before
                } else {
                    after
                };
                for _ in 0..2 {
                    let (actual, calls) = count_kernels(|| {
                        input
                            .execute(options, &SerialExecutor, buffers.borrow())
                            .unwrap()
                    });
                    assert_eq!(actual, expected, "n={n}, tasks={tasks}");
                    assert!(
                        calls.for_geometry(geometry) > 0,
                        "n={n}, tasks={tasks}, geometry={geometry:?}, calls={calls:?}"
                    );
                    assert_eq!(calls.total(), geometry.windows() * n.div_ceil(8192));
                    buffers.tails(required);
                }
            }
        }
    }
}

#[test]
fn automatic_window_transitions_match_a_single_ladder() {
    transitions::<Pallas>();
    transitions::<Vesta>();
}

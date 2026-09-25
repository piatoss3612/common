use super::*;

fn classed<M: PrimeModulus>(log: u32) {
    let domain = Domain::<PastaField<M>>::new(log).unwrap().coset();
    let smaller = Domain::<PastaField<M>>::new(log - 1).unwrap().coset();
    let smallest = Domain::<PastaField<M>>::new(log - 2).unwrap().subgroup();
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
                execution::FftPlan::with_strategy(
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
            let plan = execution::InterpolationPlan::with_transforms(transforms, false);
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

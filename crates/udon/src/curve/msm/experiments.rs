//! Opt-in native controls. Counters and fixture checks are outside timed loops.
use super::*;
use crate::{
    curve::{CurveTableEntry, Pallas, Vesta, reduce},
    exec::SerialExecutor,
    test_support::field_samples,
};
use std::{
    hint::black_box,
    time::{Duration, Instant},
    vec,
    vec::Vec,
};

fn timing(name: &str, n: usize, bytes: usize, mut f: impl FnMut()) {
    f();
    let mut samples = [0.0_f64; 7];
    for sample in &mut samples {
        let start = Instant::now();
        let mut runs = 0;
        while start.elapsed() < Duration::from_millis(30) {
            f();
            runs += 1;
        }
        *sample = start.elapsed().as_secs_f64() * 1e6 / f64::from(runs);
    }
    samples.sort_by(f64::total_cmp);
    std::println!(
        "{name},{n},{bytes},{:.3},{:.3},{:.3}",
        samples[3],
        samples[0],
        samples[6]
    );
}

#[test]
#[ignore = "isolated phase timings and counters; run alone with --nocapture"]
fn phases() {
    fn check<C: PastaCurve>() {
        std::println!("curve={}", core::any::type_name::<C>());
        for budget in [1, 12] {
            let options =
                BatchOptions::default().with_task_budget(TaskBudget::new(budget).unwrap());
            for (name, options) in [
                ("uncapped", options),
                (
                    "pass_512",
                    BatchOptions::new(
                        options
                            .arithmetic()
                            .with_max_terms_per_pass(core::num::NonZeroUsize::new(512)),
                    )
                    .with_task_budget(options.task_budget()),
                ),
                ("limit_32768", options.with_memory_limit(32768)),
            ] {
                let r = Input::<C>::requirements_for_len(1 << 20, options).unwrap();
                std::println!(
                    "sizing_only,1048576,{budget},{name},{},counts={r:?}",
                    r.bytes::<C>().unwrap()
                );
            }
        }
        for n in [128, 1024, 8192] {
            let scalars: Vec<_> = field_samples::<C::Scalar>().take(n).collect();
            let bases: Vec<_> = scalars
                .iter()
                .map(|s| {
                    *AffinePoint::<C>::GENERATOR
                        .mul_projective(s)
                        .to_point()
                        .as_affine()
                        .unwrap()
                })
                .collect();
            let mut canonical = vec![scalars[0].to_canonical_uint(); n];
            timing("canonicalize", n, n * 32, || {
                for (dst, s) in canonical.iter_mut().zip(&scalars) {
                    *dst = s.to_canonical_uint();
                }
                black_box(&canonical);
            });
            let mut halves = vec![(0, 0); n];
            timing("glv_from_canonical", n, n * 32, || {
                for (dst, &s) in halves.iter_mut().zip(&canonical) {
                    *dst = crate::curve::glv::decompose_canonical::<C>(s);
                }
                black_box(&halves);
            });
            let options = BatchOptions::new(
                ArithmeticOptions::DEFAULT
                    .with_kernel(Kernel::Booth {
                        width: Some(8),
                        accumulation: Accumulation::Auto,
                    })
                    .unwrap(),
            );
            let input = Input::new(Bases::Affine(&bases), &scalars);
            let mut records = vec![ScalarStorage::ZERO; n];
            PreparedScalars::<C>::prepare(
                &scalars,
                &mut records,
                TaskBudget::SERIAL,
                &SerialExecutor,
            );
            let geometry = recode::Geometry::Booth(8);
            let mut digits = vec![0; geometry.storage_len(n).unwrap()];
            recode::write(&records, geometry, &mut digits);
            let count = geometry.buckets();
            let mut points = vec![AffinePoint::GENERATOR; n * 2];
            let mut starts = vec![0; count];
            let mut lens = vec![0; count];
            let mut cursors = vec![0; count];
            // Isolate the ordinary dense first-pass count/scatter loops. The
            // full kernel below separately includes later passes and survivors.
            let mut count_terms = |starts: &mut [usize], lens: &mut [usize]| {
                lens.fill(0);
                recode::rows(&digits, n, 0..n, geometry, 0, |_, a, b| {
                    for d in [a, b] {
                        if d != 0 {
                            lens[usize::from(d.unsigned_abs()) - 1] += 1;
                        }
                    }
                });
                let mut total = 0;
                for i in 0..count {
                    starts[i] = total;
                    cursors[i] = total;
                    total += lens[i];
                }
            };
            timing("count_window_0", n, count * 24, || {
                count_terms(&mut starts, &mut lens);
                black_box(&lens);
            });
            let mut scatter = |points: &mut [AffinePoint<C>]| {
                cursors.copy_from_slice(&starts);
                recode::rows(&digits, n, 0..n, geometry, 0, |term, a, b| {
                    for (half, d) in [a, b].into_iter().enumerate() {
                        if d != 0 {
                            let p = bases[term].rotated(half);
                            let i = usize::from(d.unsigned_abs()) - 1;
                            points[cursors[i]] = if d < 0 { p.neg() } else { p };
                            cursors[i] += 1;
                        }
                    }
                });
            };
            timing("scatter_window_0", n, n * 128 + count * 24, || {
                scatter(&mut points);
                black_box(&points);
            });
            let total: usize = lens.iter().sum();
            let mut fields = vec![PastaField::ZERO; total];
            let mut level_points = points[..total].to_vec();
            let mut level_lens = lens.clone();
            let mut level_index = 0;
            while level_lens.iter().any(|&len| len > 1) {
                let saved_points = level_points.clone();
                let saved_lens = level_lens.clone();
                timing(
                    &std::format!("reduce_level_{level_index}_with_reset"),
                    n,
                    total * 32,
                    || {
                        level_points.copy_from_slice(&saved_points);
                        level_lens.copy_from_slice(&saved_lens);
                        black_box(reduce::reduce_level::<C, false>(
                            &mut level_points,
                            &starts,
                            &mut level_lens,
                            &mut fields,
                        ));
                        black_box(&level_points);
                    },
                );
                level_index += 1;
            }
            let mut levels = Vec::new();
            reduce::reduce_with::<C, false>(
                &mut points[..total],
                &starts,
                &mut lens,
                &mut fields,
                |pairs, terms| levels.push((pairs, terms)),
            );
            std::println!(
                "counts,{n},scanned_terms={},deposits={total},inversions={},levels={levels:?}",
                2 * n,
                levels.iter().filter(|&&(pairs, _)| pairs != 0).count()
            );
            let survivors: Vec<_> = starts
                .iter()
                .zip(&lens)
                .map(|(&i, &len)| {
                    if len == 0 {
                        AffinePoint::GENERATOR
                    } else {
                        points[i]
                    }
                })
                .collect();
            timing("weighted_collapse_window_0", n, count * 64, || {
                black_box(reduce::collapse(&survivors, &lens));
            });
            let r = input.requirements_with(options).unwrap();
            let mut affine = vec![AffinePoint::GENERATOR; r.affine()];
            let mut projective = vec![ProjectivePoint::IDENTITY; r.projective()];
            let mut fields = vec![PastaField::ZERO; r.field()];
            let mut indices = vec![0; r.indices()];
            let mut results = vec![ProjectivePoint::IDENTITY; geometry.windows()];
            let mut work = kernels::Work {
                affine: &mut affine,
                projective: &mut projective,
                field: &mut fields,
                indices: &mut indices,
            };
            timing("all_window_kernels", n, r.bytes::<C>().unwrap(), || {
                for (window, result) in results.iter_mut().enumerate() {
                    *result = kernels::run(
                        &input,
                        &records,
                        &digits,
                        kernels::Task {
                            offset: 0,
                            window,
                            pass: n,
                            geometry,
                            accumulation: Accumulation::Affine,
                        },
                        &mut work,
                    );
                }
                black_box(&results);
            });
            let fold = || {
                let mut sum = ProjectivePoint::IDENTITY;
                for p in results.iter().rev() {
                    for _ in 0..geometry.width() {
                        sum = sum.double();
                    }
                    sum = sum.add(p);
                }
                sum
            };
            let mut buffers = tests::Buffers::new(r);
            assert_eq!(
                fold(),
                input
                    .execute_with(options, &SerialExecutor, buffers.borrow())
                    .unwrap()
            );
            timing("final_recombination", n, results.len() * 96, || {
                black_box(fold());
            });
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

// Keep all windows' projective buckets while preparing/recoding short chunks.
// This bounds digit lifetime but pays windows*buckets projective storage.
fn retained_windows<C: PastaCurve>(
    bases: &[AffinePoint<C>],
    scalars: &[PastaField<C::Scalar>],
    records: &mut [ScalarStorage<C>],
    digits: &mut [u8],
    buckets: &mut [ProjectivePoint<C>],
    geometry: recode::Geometry,
) -> ProjectivePoint<C> {
    buckets.fill(ProjectivePoint::IDENTITY);
    let chunk_size = records.len();
    for (chunk, source) in scalars.chunks(chunk_size).enumerate() {
        let records = &mut records[..source.len()];
        PreparedScalars::<C>::prepare(source, records, TaskBudget::SERIAL, &SerialExecutor);
        let digits = &mut digits[..geometry.storage_len(source.len()).unwrap()];
        recode::write(records, geometry, digits);
        for (window, buckets) in buckets.chunks_exact_mut(geometry.buckets()).enumerate() {
            recode::rows(
                digits,
                source.len(),
                0..source.len(),
                geometry,
                window,
                |term, a, b| {
                    let p = bases[chunk * chunk_size + term];
                    for (digit, p) in [(a, p), (b, p.endomorphism())] {
                        if digit != 0 {
                            let p = if digit < 0 { p.neg() } else { p };
                            let j = usize::from(digit.unsigned_abs()) - 1;
                            buckets[j] = buckets[j].add_mixed(&p);
                        }
                    }
                },
            );
        }
    }
    let mut sum = ProjectivePoint::IDENTITY;
    for buckets in buckets.chunks_exact(geometry.buckets()).rev() {
        for _ in 0..geometry.width() {
            sum = sum.double();
        }
        let mut running = ProjectivePoint::IDENTITY;
        for bucket in buckets.iter().rev() {
            running = running.add(bucket);
            sum = sum.add(&running);
        }
    }
    sum
}

#[test]
#[ignore = "native timing experiment; run alone with --nocapture"]
fn native_controls() {
    fn check<C: PastaCurve>() {
        std::println!("curve={}", core::any::type_name::<C>());
        let scalars: Vec<_> = field_samples::<C::Scalar>().take(8192).collect();
        let g = AffinePoint::<C>::GENERATOR;
        let bases: Vec<_> = scalars
            .iter()
            .map(|s| *g.mul_projective(s).to_point().as_affine().unwrap())
            .collect();
        for n in [16, 128, 1024, 8192] {
            for occupancy in [1, 2, 17, 128] {
                let starts: Vec<_> = (0..n).step_by(occupancy).collect();
                let lens: Vec<_> = starts.iter().map(|&i| occupancy.min(n - i)).collect();
                let mut points = bases[..n].to_vec();
                let mut lengths = lens.clone();
                let mut fields = vec![PastaField::ZERO; n * 3];
                let mut writes = vec![0; n / 2];
                let mut histogram = Vec::new();
                reduce::reduce_with::<C, false>(
                    &mut points,
                    &starts,
                    &mut lengths,
                    &mut fields[..n],
                    |pairs, terms| histogram.push((pairs, terms)),
                );
                let expected = points.clone();
                let expected_lens = lengths.clone();
                std::println!("levels,{n},{occupancy},{histogram:?}");
                for mode in [0, 1, 2, 0] {
                    let mut compute = || {
                        points.copy_from_slice(&bases[..n]);
                        lengths.copy_from_slice(&lens);
                        match mode {
                            0 => reduce::reduce_original(
                                &mut points,
                                &starts,
                                &mut lengths,
                                &mut fields,
                                &mut writes,
                            ),
                            1 => {
                                reduce::reduce(&mut points, &starts, &mut lengths, &mut fields[..n])
                            }
                            _ => reduce::reduce_with::<C, true>(
                                &mut points,
                                &starts,
                                &mut lengths,
                                &mut fields[..n],
                                |_, _| {},
                            ),
                        }
                        black_box(&points);
                    };
                    compute();
                    assert_eq!(lengths, expected_lens);
                    for (&i, &len) in starts.iter().zip(&lengths) {
                        if len != 0 {
                            assert_eq!(points[i], expected[i]);
                        }
                    }
                    timing(
                        &std::format!("reducer_{mode}_occupancy_{occupancy}"),
                        n,
                        if mode == 0 { n / 2 * 200 } else { n / 2 * 64 },
                        || {
                            points.copy_from_slice(&bases[..n]);
                            lengths.copy_from_slice(&lens);
                            match mode {
                                0 => reduce::reduce_original(
                                    &mut points,
                                    &starts,
                                    &mut lengths,
                                    &mut fields,
                                    &mut writes,
                                ),
                                1 => reduce::reduce(
                                    &mut points,
                                    &starts,
                                    &mut lengths,
                                    &mut fields[..n],
                                ),
                                _ => reduce::reduce_with::<C, true>(
                                    &mut points,
                                    &starts,
                                    &mut lengths,
                                    &mut fields[..n],
                                    |_, _| {},
                                ),
                            }
                            black_box(&points);
                        },
                    );
                }
            }
        }
        for n in [128, 1024, 8192] {
            let input = Input::new(Bases::Affine(&bases[..n]), &scalars[..n]);
            let options = BatchOptions::default();
            let mut buffers = tests::Buffers::new(input.requirements_with(options).unwrap());
            let expected = input
                .execute_with(options, &SerialExecutor, buffers.borrow())
                .unwrap();
            for width in [4, 6, 8] {
                let geometry = recode::Geometry::Booth(width);
                let mut all = vec![ScalarStorage::ZERO; n];
                PreparedScalars::<C>::prepare(
                    &scalars[..n],
                    &mut all,
                    TaskBudget::SERIAL,
                    &SerialExecutor,
                );
                let mut packed = vec![0; geometry.storage_len(n).unwrap()];
                timing(
                    &std::format!("recode_width_{width}"),
                    n,
                    packed.len(),
                    || recode::write(&all, geometry, &mut packed),
                );
                let mut occupancy = vec![0usize; geometry.buckets()];
                for window in 0..geometry.windows() {
                    occupancy.fill(0);
                    recode::rows(&packed, n, 0..n, geometry, window, |_, a, b| {
                        for d in [a, b] {
                            if d != 0 {
                                occupancy[usize::from(d.unsigned_abs()) - 1] += 1;
                            }
                        }
                    });
                    std::println!(
                        "occupancy,{n},{width},{window},active={},max={},deposits={}",
                        occupancy.iter().filter(|&&x| x != 0).count(),
                        occupancy.iter().max().unwrap(),
                        occupancy.iter().sum::<usize>()
                    );
                }
                for chunk in [32, 256] {
                    let mut records = vec![ScalarStorage::ZERO; chunk.min(n)];
                    let mut digits = vec![0; geometry.storage_len(records.len()).unwrap()];
                    let mut buckets =
                        vec![ProjectivePoint::IDENTITY; geometry.windows() * geometry.buckets()];
                    let bytes = records.len() * core::mem::size_of::<ScalarStorage<C>>()
                        + digits.len()
                        + buckets.len() * core::mem::size_of::<ProjectivePoint<C>>();
                    assert_eq!(
                        retained_windows(
                            &bases[..n],
                            &scalars[..n],
                            &mut records,
                            &mut digits,
                            &mut buckets,
                            geometry
                        ),
                        expected
                    );
                    timing(
                        &std::format!("retained_windows_w{width}_chunk{chunk}"),
                        n,
                        bytes,
                        || {
                            black_box(retained_windows(
                                &bases[..n],
                                &scalars[..n],
                                &mut records,
                                &mut digits,
                                &mut buckets,
                                geometry,
                            ));
                        },
                    );
                    let o = BatchOptions::new(
                        options
                            .arithmetic()
                            .with_kernel(Kernel::Booth {
                                width: Some(u32::from(width)),
                                accumulation: Accumulation::Projective,
                            })
                            .unwrap()
                            .with_chunk_size(NonZeroUsize::new(chunk).unwrap()),
                    );
                    let r = input.requirements_with(o).unwrap();
                    let mut buffers = tests::Buffers::new(r);
                    assert_eq!(
                        input
                            .execute_with(o, &SerialExecutor, buffers.borrow())
                            .unwrap(),
                        expected
                    );
                    timing(
                        &std::format!("complete_chunks_w{width}_chunk{chunk}"),
                        n,
                        r.bytes::<C>().unwrap(),
                        || {
                            black_box(
                                input
                                    .execute_with(o, &SerialExecutor, buffers.borrow())
                                    .unwrap(),
                            );
                        },
                    );
                }
            }
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

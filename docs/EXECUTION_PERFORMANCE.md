# Incremental execution measurements

On the measured Apple M4 Max, the application run driver completes the fixed
shrinking workload in 26.0–26.3 ms with 15,616,760 bytes of working provision.
The synchronous static-partition comparison takes 32.8–33.6 ms. Seven successive
measurements show a 20.6–22.3% elapsed-time reduction under the same 16 MiB
ceiling. This measures one application policy and machine; it does not establish
a universal scheduler or arithmetic grain.

The [execution guide](EXECUTION.md) describes ownership and admission. The
[benchmark](../crates/udon/benches/execution.rs) and
[measurement data](measurements/execution.csv) make the workload and estimates
reviewable. Timings use Criterion 0.8, release optimization, the pinned Rust
1.91.0 toolchain, and default features on a 16-CPU Apple M4 Max with 128 GiB RAM.
Measurements were taken on September 14, 2026, in the working revision based on
`d454936`. No compilation or test workload ran concurrently with the final
timings. Thread placement, frequency, and system activity were not pinned.

## Workload and accounting

Each iteration runs eleven shrinking rounds twice. Each round contains two
MSMs starting at 8,192 and 1,024 terms, FFTs starting at 16,384 and 2,048 fields,
8,192 integer updates, result consumers, and a challenge fence. MSM sizes halve;
the larger FFT halves down to 2,048 fields. Inputs and plans are prepared before
timing. Pools and working allocations are retained across timed iterations.

The application driver uses seven compatible MSM scratch bundles, fixed FFT
fragments, a 16-entry outstanding-work limit, and fair rotation across seven
work lanes. It exposes 1,376 bounded tasks per iteration. Each of 22 rounds
dispatches the smaller MSM's consumer before unrelated arithmetic finishes.
The checked integration fixture compares MSMs with a scalar ladder, FFT samples
with direct polynomial evaluation, and application values with a scalar
reference. The benchmark omits those expensive oracles.

The run fixture charges full allocated capacities, lease metadata, retained
banks, run state, frontier and admission descriptors, queues, worker envelopes,
and fixed coordinator metadata. Its componentwise peak is
`[1, 1, 1, 1, 7, 1, 16]`: two MSM banks, two FFT banks, seven MSM task bundles,
one application bank, and sixteen outstanding queue permits. This is the peak
of each class, not necessarily a simultaneously observed vector. Idle scratch
still counts against the ceiling.

The synchronous partition comparison provisions 9,875,872 bytes of arithmetic
banks and their owning buffer headers. Its log omits structured-driver stack
metadata; it has over 6 MiB of headroom under the ceiling. Runtime stacks and
pool internals are outside both working-storage accounts. This comparison uses
less arithmetic storage than the incremental configuration; the constraint is
an equal ceiling, not equal used bytes. The application driver spends the
remaining capacity on shared scratch to improve throughput. Full-budget
independent synchronous jobs require 27,431,328 bytes, and the synchronous MSM
batch configuration requires 24,324,800 bytes at sixteen workers, so the harness
rejects both before timing. The `usize::MAX` batch configuration has the same
storage as the uncapped batch after migration.

Eight width-11 scratch bundles exceeded the run ceiling and were rejected by
the fixture's arena accounting; seven fit. Earlier eight-bundle timings with
width-10 geometry are not results for the final configuration.

| Successive trial | Synchronous partition | Incremental runs | Time reduction |
| --- | --- | --- | --- |
| 1 | 32.828 ms | 26.029 ms | 20.7% |
| 2 | 33.192 ms | 26.279 ms | 20.8% |
| 3 | 33.124 ms | 26.097 ms | 21.2% |
| 4 | 33.305 ms | 26.098 ms | 21.6% |
| 5 | 33.126 ms | 26.311 ms | 20.6% |
| 6 | 33.355 ms | 26.179 ms | 21.5% |
| 7, final driver | 33.641 ms | 26.144 ms | 22.3% |

Before the synchronous migration, the recorded static-partition comparison was
31.726 ms. The final run measurements also improve on that result by 17–18%.
The current comparison remains in the benchmark so future changes can be
tested against a reproducible structured policy using the same kernels.

## Isolated operation cost

The final before/after control contains 45 synchronous cases: three MSM lengths
and three FFT lengths, at one, four, and sixteen workers; FFTs include both
directions and subgroup and generic coset domains. The identical
[control source](measurements/execution-control.rs) was compiled against a fresh
archive of `d454936` and the final candidate. Both binaries were built before
timing, then run sequentially. Their results appear as `control-before1` and
`control-after1` in the CSV.

The largest slowdown in that complete matrix was 2.9%. A second consecutive
control pair repeated the two small subgroup FFTs at one worker, the 1,024-term
MSM at four workers, and the 8,192-term MSM at sixteen workers. Those slowdowns
were 3.1%, 3.3%, 1.9%, and −0.4%, respectively. No repeated fresh-control case
exceeded 5%. These are local migration checks, not guaranteed performance bounds.

| Representative control case | Before | After | Change |
| --- | --- | --- | --- |
| MSM 32 terms, 1 worker | 180.01 µs | 180.10 µs | +0.05% |
| MSM 1,024 terms, 4 workers | 906.42 µs | 927.68 µs | +2.3% |
| MSM 8,192 terms, 4 workers | 5.5506 ms | 5.5250 ms | −0.5% |
| MSM 8,192 terms, 16 workers | 2.2171 ms | 2.2470 ms | +1.3% |
| Subgroup forward FFT 64, 1 worker | 3.8034 µs | 3.9141 µs | +2.9% |
| Subgroup inverse FFT 16,384, 16 workers | 708.36 µs | 722.96 µs | +2.1% |
| Coset inverse FFT 16,384, 16 workers | 755.79 µs | 761.74 µs | +0.8% |

The CSV also preserves the earlier matrix (`before` / `legacy`) and successive
candidate trials (`after1` through `after7`). The first matrix comparison stayed
below 5%, but later measurements against that older baseline showed 5.2–5.4%
slowdowns for small subgroup FFTs and up to 9.0% for the sixteen-worker MSM.
Rebuilding the control exposed baseline drift: its 32-term serial MSM moved
from 173.74 to 180.01 µs, and its small subgroup forward FFT moved from 3.7080 to
3.8034 µs. The sixteen-worker MSM varied between control repeats as well.
Benchmark binary context and system activity remain uncontrolled. Retaining
these measurements avoids treating a selected historical baseline as stable.
The current main harness's `synchronous` label always means the migrated API;
it does not select a second legacy implementation.

Initial four-worker MSM runs used a join barrier after every scratch-sized wave
and were about 7% slower at 1,024 terms. The default driver now gives each joined
branch an explicitly owned scratch bundle for successive bounded window
kernels, reducing that overhead. The application interface still publishes
individual tasks and permits global interleaving.

The final driver also separates the preparation allowance from window scratch
count. A width-11 MSM has twelve windows, but a sixteen-task allowance can run
sixteen bounded preparation tasks using disjoint retained slices. This avoids
limiting preparation to twelve tasks or provisioning unused window scratch.

Geometry matters as much as dispatch. At 8,192 terms and four workers, the
initial worker-independent width-10 plan took 6.5866 ms, while width 11 took
5.5772 ms against the 5.5165 ms before result. The automatic plan now selects
width 11 at grains of at least 4,096. For FFTs, column panels preserve the
existing blocked arithmetic path; forcing paired tile stages for every shape
can cost more. The CSV includes explicit `runs`, `stage`, and `blocked` modes
for examining these choices separately from the public API.

## Reproduction and remaining measurements

Run the full current matrix:

```console
cargo bench --locked -p zakura-udon --bench execution
```

Repeat the final mixed comparison and the sensitive MSM cases:

```console
cargo bench --locked -p zakura-udon --bench execution -- 'execution/synchronous/shrinking/Partition/16|execution/runs/shrinking/grain8192-scratch7/16|execution/isolated/msm/(1024/.*/4|8192/.*/16)'
```

The harness uses 300 ms warmup, ten mixed-workload samples, twenty isolated
samples, and a one-second measurement target. Criterion may extend that target
for longer iterations. The CSV preserves reported estimates and confidence
interval endpoints in microseconds. CI uses `-- --test` to exercise arithmetic
and admissibility without asserting machine-specific timings.

These commands and the control below reproduce the historical revisions used
for this report. The incremental mixed driver now lives in the test fixtures;
the current public MSM batch entry point is `msm::run::BatchPlan`.

To repeat the independent control, copy `measurements/execution-control.rs`
from this guide's directory to `crates/udon/benches/execution_control.rs` in
isolated before and after checkouts. Add a `[[bench]]` entry named
`execution_control` with `harness = false` to each Udon manifest. Build both with
`cargo bench --locked -p zakura-udon --bench execution_control --no-run`, then
run that command without `--no-run` sequentially. The control uses only public
synchronous APIs available in both revisions and the same sampling settings.

The benchmark also includes one, three, and four workers, smaller MSM grains,
four scratch bundles, independent full budgets, and explicit memory caps. These
allow an application to measure the storage/throughput tradeoff before choosing
its own policy. Unmeasured premises remain: other CPUs, NUMA placement, realistic
challenge computation, larger retained consumer graphs, bandwidth/cache-token
policies, and automatically chosen grains across algorithm families. No
bandwidth or cache speedup is claimed by the current measurements.

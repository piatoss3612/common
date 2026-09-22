# Execution measurements

The [execution benchmark](../crates/udon/benches/execution.rs) compares caller
concurrency policies and memory limits. Use it with the
[execution guide](EXECUTION.md) to measure an application's scratch capacity
and worker allowance. A faster isolated kernel does not by
itself establish a better mixed-workload policy.

## Workload and accounting

The current mixed suite, `execution/synchronous/shrinking`, runs eleven shrinking
rounds twice. Each round contains two MSMs starting at 8,192 and 1,024 terms,
FFTs starting at 16,384 and 2,048 fields, 8,192 integer updates, and a result
consumer/challenge fence. MSM sizes halve; the larger FFT halves down to 2,048.
Input binding and FFT domain/table plans are prepared before timing. Pools,
arithmetic scratch, and output arrays are retained. Timing includes pool entry,
FFT input copying, execution through the convenience APIs, and their per-call
planning work.

The suite compares `Partition`, `Independent`, `Batched`, `UnlimitedCap`, and
`Capped` policies at one, three, four, and sixteen workers. Partition divides
the allowance across branches; independent jobs each receive the full allowance.
Batched modes share MSM planning, with either no explicit cap, an unlimited cap,
or a 1 MiB MSM cap. Each case must also fit the fixture's 16 MiB ceiling before
it is admitted for timing.

That ceiling charges the two MSM scratch owners and their allocated capacities,
FFT scratch and value capacities, and the integer work array. It is a fixture
account, not process RSS or a bound on runtime stacks, pool internals, borrowed
inputs, or all planning metadata. Idle provisioned arithmetic storage still
counts. Obtain the printed capacity and admission decision from the current
run instead of reusing byte totals for another plan or driver.

Application-controlled mixed incremental execution is covered by the
[execution tests](../crates/udon/tests/execution/mixed.rs), including
independent arithmetic checks and admission. The current mixed benchmark does
not time that driver.

## Isolated operation cost

The isolated suite covers 32-, 1,024-, and 8,192-term Pallas MSMs and Fp FFTs
of 64, 2,048, and 16,384 elements. FFTs include both directions on subgroups and
cosets. Every shape runs at one, four, and sixteen workers. MSM modes compare
`synchronous` convenience execution and bounded `runs`; FFT modes compare
`synchronous` execution and reusable `planned` execution.
The recorded measurements below compare stage and blocked implementations.

Input binding, reusable plans, scratch allocation, and pool entry are outside
isolated timing. FFT input copying is included in every mode. Per-call
convenience planning remains included where the API performs it. Small shapes
expose dispatch overhead; larger ones expose arithmetic and memory geometry.
An allowance of sixteen tasks does not require sixteen simultaneous kernels.

The [measurement data](measurements/execution.csv) retains the complete
126-case isolated matrix from session `after4`, a September 14, 2026 working
candidate based on `d454936`. The session label identifies the recorded data;
it is not a comparison against another revision. Measurements used Criterion
0.8, Rust 1.91.0, default features, and a 16-CPU Apple M4 Max with 128 GiB RAM.
The measured drivers differ from the current suite, so these estimates
illustrate geometry tradeoffs and do not measure current latency. The CSV
retains estimate and confidence interval endpoints in microseconds.

Representative 8,192-term MSM estimates from that session were:

| Workers | Synchronous, µs | Bounded runs, µs |
| ---: | ---: | ---: |
| 1 | 21,084 | 21,224 |
| 4 | 5,536.8 | 5,540.1 |
| 16 | 2,377.7 | 2,335.6 |

The sixteen-worker intervals overlap: [2,339.4, 2,431.4] and
[2,296.6, 2,392.3] µs. These small differences do not establish a general
advantage for either driver.

Size-16,384 subgroup inverse FFTs show a larger geometry effect:

| Workers | Synchronous, µs | Stage, µs | Blocked, µs |
| ---: | ---: | ---: | ---: |
| 1 | 2,712.9 | 2,965.6 | 2,713.1 |
| 4 | 803.59 | 860.40 | 805.16 |
| 16 | 738.99 | 921.20 | 742.03 |

Blocked execution retains column panels to improve locality. For size-64
subgroup forward transforms at one worker, the same modes took 3.9076, 3.8745,
and 3.8772 µs, respectively. The larger transform's ranking does not imply
that every shape benefits from blocked geometry. The
[FFT strategy report](FFT_PERFORMANCE.md) covers ordering and retained tables.

## Reproduction and interpretation

Run the full current matrix, or select policies and isolated cases:

```console
cargo bench --locked -p zakura-udon --bench execution
cargo bench --locked -p zakura-udon --bench execution -- 'execution/synchronous/shrinking/(Partition|Capped)/(4|16)|execution/isolated/msm/(1024/.*/4|8192/.*/16)'
cargo bench --locked -p zakura-udon --bench execution -- 'execution/isolated/fft/16384/false/Inverse'
```

The harness uses 300 ms warmup, ten mixed-workload samples, twenty isolated
samples, and a one-second measurement target. Criterion can extend that target
for longer iterations. CI's `-- --test` mode exercises cases and admission
without collecting timings; the separate correctness suite owns independent
arithmetic oracles.

For a revision comparison, use identical applicable benchmark cases in isolated
checkouts. Build both binaries before timing, then run them sequentially and
repeat unchanged controls. Do not run builds or tests alongside measurements.
Historical control runs showed several-percent baseline drift; a confidence
interval within one run does not capture that drift. Thread placement,
frequency, and system activity were not pinned for the recorded session.

The measurements do not establish performance on other CPUs or NUMA systems,
with realistic challenge computation, or with larger retained consumer graphs.
They do not determine a universal grain or prove bandwidth/cache improvements.
Measure the complete workload under its actual ownership and memory policy.

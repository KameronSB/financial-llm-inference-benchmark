# Financial LLM Inference Benchmark

A small Python project measuring the quality and performance of local
language models on financial table extraction.

The project uses the official Ollama Python client, structured JSON
responses, mechanical evidence checks and a fixed subset of TAT-QA.

## Completed experiment: v0.1

The experiment compares two quantisations of Qwen2.5 7B Instruct:

- `qwen2.5:7b-instruct-q4_K_M`
- `qwen2.5:7b-instruct-q8_0`

Both configurations were evaluated on all **211 prepared test questions
across 134 table contexts**, with three repetitions per configuration.

All six runs completed without API errors or invalid structured outputs.
Predictions were identical across the three repetitions within each
configuration.

### Results

Quality results below use the 211 distinct questions. Timing results
pool the three repetitions, giving 633 measured requests per configuration.

| Metric | Q4_K_M | Q8_0 |
|---|---:|---:|
| Correct numeric value and scale | **156/211 (73.93%)** | 151/211 (71.56%) |
| Correct numeric value, ignoring scale | 197/211 (93.36%) | **198/211 (93.84%)** |
| Passed mechanical cell checks | 196/211 | 199/211 |
| Correct value and scale, and passed checks | **152/211** | 144/211 |
| Reference-incorrect answers that passed checks | 44 | 55 |
| Mean request latency | **1.041 s** | 1.115 s |
| Pooled median request latency | **0.974 s** | 1.068 s |
| Pooled p95 request latency | 1.628 s | **1.568 s** |
| Aggregate generation throughput | **83.83 tokens/s** | 55.52 tokens/s |
| Loaded size reported by Ollama | **5.1 GB** | 8.3 GB |

For this workload, Q4 achieved five more strictly correct answers,
lower average latency and a smaller reported loaded size. Q8 achieved
slightly better numeric-only accuracy and lower p95 latency.

This is an observed result for this dataset, prompt and machine. It does
not establish that Q4 is generally more accurate than Q8.

### Results by repetition

| Run | Strict correct | Mean request (s) | Median request (s) | p95 request (s) | Generation (tokens/s) |
|---|---:|---:|---:|---:|---:|
| Q4 r1 | 156/211 | 1.037 | 0.974 | 1.610 | 83.42 |
| Q4 r2 | 156/211 | 1.039 | 0.959 | 1.618 | 84.27 |
| Q4 r3 | 156/211 | 1.046 | 0.975 | 1.634 | 83.79 |
| Q8 r1 | 151/211 | 1.124 | 1.078 | 1.563 | 55.15 |
| Q8 r2 | 151/211 | 1.111 | 1.065 | 1.570 | 55.66 |
| Q8 r3 | 151/211 | 1.111 | 1.062 | 1.568 | 55.77 |

See [comparison.csv](results/v0.1/comparison.csv) for the per-run
measurements and [results/v0.1](results/v0.1/) for summaries and raw evidence.

## Task and dataset

Each request supplies a question, an indexed table and associated
paragraphs. The model returns a numeric value, its reported scale,
and zero-based row and column coordinates, or a structured `not_found`
response.

Reference answers are retained for evaluation and are excluded from the
model request.

The data comes from [TAT-QA](https://github.com/NExTplusplus/TAT-QA),
pinned to source commit
[870accc41953dcde885aabeb963d94aabdc0fbc3](https://github.com/NExTplusplus/TAT-QA/tree/870accc41953dcde885aabeb963d94aabdc0fbc3).

The preparation filter selects numeric span answers from table or
table-text questions, excludes comparison questions, and requires the
signed reference value to occur in a whole table cell.

| Prepared split | Questions |
|---|---:|
| Train | 1,818 |
| Development | 233 |
| Test | 211 |

The experiment covers the complete prepared test subset. It does not
cover the full original TAT-QA benchmark, arbitrary financial reasoning,
PDF parsing or document retrieval.

Dataset provenance and attribution are recorded in
[data/manifest.json](data/manifest.json). See also the
[TAT-QA paper](https://aclanthology.org/2021.acl-long.254/).

## What the quality metrics mean

- **Strict accuracy:** the parsed numeric value and reported scale both
  match the reference. A cell-check pass is not required for this metric.
- **Numeric-only accuracy:** the parsed value matches the reference,
  ignoring scale.
- **Cell-check pass:** the returned value matches the cited table cell,
  coordinates are valid, and supported scale checks pass.
- **Correct and check-passing:** both strict correctness and the mechanical
  checks pass.

Abstentions, invalid outputs and request failures count as incorrect.

The verifier establishes consistency with a cited cell. It does not
establish that the selected metric or year answers the question, or fully
validate units against surrounding headings and prose.

This distinction matters: 44 Q4 answers and 55 Q8 answers passed the
mechanical checks while failing strict reference scoring.

### Main error pattern

Scale interpretation was the largest source of strict mismatches:

| Error category | Q4_K_M | Q8_0 |
|---|---:|---:|
| Correct number, different reference scale | 41 | 47 |
| Incorrect found numeric value | 13 | 12 |
| Abstention | 1 | 1 |

For example, question `82a48abedc1e5cfed2abad3c9a10ae56` has a row labelled
“Group operating profit (£m)”. Q4 returned 283.8 million; Q8 returned
283.8 thousand. Both passed the value-to-cell check.

Review also identified possible reference-scale inconsistencies in
`2fed773590cac936ba03262eedeadff6` and
`0009bd87a55c3494b46c041853bd6f10`: their reference scales are empty,
although the supplied paragraphs describe values in thousands.
The original labels and published scores remain unchanged.

## Environment and measurement procedure

The recorded experiment used:

- AMD Radeon RX 6900 XT with 16 GB VRAM.
- Ubuntu 24.04.4 LTS and Python 3.12.3.
- Ollama version output `0.40.17`; Ollama Python package `0.6.3`.
- The `llamacpp` runner, with post-run snapshots showing the selected
  model at `100% GPU`.
- Context length 8192, maximum output tokens 256, temperature 0 and seed 42.
- Sequential requests with concurrency 1 and non-streaming responses.

The GPU backend was not independently confirmed as ROCm or Vulkan.

Both benchmark models were unloaded before each run. A synthetic warm-up
was recorded separately, then all 211 test questions were processed in
dataset order. Model order was Q4/Q8, Q8/Q4, then Q4/Q8 across the three rounds.

The original [environment record](results/v0.1/environment.json) contains
model digests, dataset hash and the clean experiment source commit:
[58ba7fd](https://github.com/KameronSB/financial-llm-inference-benchmark/tree/58ba7fdaa9b1e0d24981f13c7cb54fbb7efb8c2d).

### Timing definitions

Request latency is measured around `extract()`. It includes the request
and output parsing, and excludes subsequent evidence verification and
result-file writing. Warm-ups are excluded.

Generation throughput is total native `eval_count` divided by total
native `eval_duration` in seconds. It excludes model loading and prompt
processing.

p95 uses linear interpolation, implemented with
`statistics.quantiles(..., n=100, method="inclusive")[94]`.
Pooled percentiles are calculated from all 633 recorded request times,
rather than averaging the three per-run percentiles.

All measured responses contribute to timing, regardless of correctness.

## Project files

| File | Purpose |
|---|---|
| `prepare_data.py` | Build the pinned dataset subset |
| `extract.py` | Request structured extraction through Ollama |
| `verify.py` | Check the prediction against its cited cell |
| `run.py` | Run examples and save responses, metadata and measurements |
| `evaluate.py` | Score an existing run without inference |
| `publish_results.py` | Generate the frozen v0.1 publication files |

The publication directory contains six regenerated summaries,
`comparison.csv`, the original `environment.json`, and `raw-runs.tar.gz`.

The archive preserves all 19 original experiment files: six raw runs,
six summaries, six GPU-residency snapshots and the environment record.

## Reproduce the published scores

On a fresh clone, restore the archived inputs and regenerate the
publication files:

```bash
uv sync --locked

mkdir -p results/local/v0.1
tar -xzf results/v0.1/raw-runs.tar.gz -C results/local/v0.1

uv run python publish_results.py
```

If the original files already exist in `results/local/v0.1`, skip the
archive extraction.

The publication script checks matching inputs and settings, reuses the
existing evaluator, and requires regenerated summaries to match the
saved originals. It makes no model requests and leaves local inputs
unchanged.

To re-score one saved run:

```bash
uv run python evaluate.py \
  results/local/v0.1/qwen7b-q4-test211-warm-r1.json
```

## Run new inference

Install Ollama separately and ensure its service is running. For a small
development run:

```bash
uv sync --locked

ollama pull qwen2.5:7b-instruct-q4_K_M

uv run python run.py \
  --model qwen2.5:7b-instruct-q4_K_M \
  --split dev \
  --limit 10 \
  --out results/local/q4-dev-new.json

uv run python evaluate.py results/local/q4-dev-new.json
```

The runner performs a synthetic warm-up automatically and refuses to
overwrite an existing run file.

For a complete prepared test run, use `--split test --limit 211`.
Use `--model qwen2.5:7b-instruct-q8_0` for the Q8 configuration after
pulling that model.

Exact model digests are recorded in the environment file; a changed
digest represents a different model artifact.

## Limitations and next work

The three repetitions provide 633 timing observations per configuration,
but only 211 distinct quality cases. Questions also share table contexts.

The measurements describe a warm, sequential workload using normal
Ollama caching. They do not measure concurrent serving capacity.
Reported loaded sizes are snapshots, rather than peak VRAM measurements.
Energy consumption and monetary cost per request were not measured.

The preparation filter can retain numeric headers such as years, and
some reference labels need further review. These limitations are retained
in the frozen v0.1 results.

Planned v0.2 work is a separate vLLM serving experiment covering concurrent
requests, time to first token, latency percentiles, throughput and cost
on documented hardware.

## Background

This project builds on
[financial-extraction-verification](https://github.com/KameronSB/financial-extraction-verification),
with local Ollama inference and reproducible quality/performance evaluation.
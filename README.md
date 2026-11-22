# Financial LLM Inference Benchmark

A small Python project for evaluating local LLMs on financial table
extraction using the official Ollama Python client.

The project will measure answer accuracy and inference performance
using a fixed subset of TAT-QA.

## Files

- prepare_data.py prepares the dataset.
- extract.py sends structured extraction requests to Ollama.
- verify.py checks answers against their cited table cells.
- run.py runs examples and saves responses and measurements.
- evaluate.py scores saved runs.

## Background

Based on:
https://github.com/KameronSB/financial-extraction-verification

Dataset:
https://github.com/NExTplusplus/TAT-QA

## Status

Initial setup. Extraction and benchmarking are being implemented.
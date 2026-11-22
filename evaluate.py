"""Evaluate one saved run without making any API calls."""

import argparse
import json
import math
from statistics import median
from pathlib import Path

from verify import parse_number

def performance_metrics(records: list[dict]) -> dict:
    elapsed = [record["elapsed_seconds"] for record in records]
    generation = []
    prompt_seconds = []
    load_seconds = []

    for record in records:
        raw = record.get("raw_response") or {}
        if not isinstance(raw, dict):
            continue

        tokens = raw.get("eval_count")
        duration = raw.get("eval_duration")
        if (type(tokens) is int and tokens > 0
                and type(duration) is int and duration > 0):
            generation.append((tokens, duration / 1_000_000_000))

        duration = raw.get("prompt_eval_duration")
        if type(duration) is int and duration >= 0:
            prompt_seconds.append(duration / 1_000_000_000)

        duration = raw.get("load_duration")
        if type(duration) is int and duration >= 0:
            load_seconds.append(duration / 1_000_000_000)

    generated_tokens = sum(tokens for tokens, _ in generation)
    generation_seconds = sum(seconds for _, seconds in generation)

    return {
        "request_attempts": len(records),
        "median_request_seconds": median(elapsed) if elapsed else None,
        "generation_measurements": len(generation),
        "attempts_without_usable_generation_metrics": (
            len(records) - len(generation)
        ),
        "generation_tokens_per_second": (
            generated_tokens / generation_seconds
            if generation_seconds > 0 else None
        ),
        "prompt_measurements": len(prompt_seconds),
        "median_prompt_seconds": (
            median(prompt_seconds) if prompt_seconds else None
        ),
        "load_measurements": len(load_seconds),
        "total_model_load_seconds": sum(load_seconds) if load_seconds else None,
    }

def evaluate(path: Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(run, dict) or type(run.get("format_version")) is not int or run["format_version"] != 1:
        raise ValueError("Expected a format_version=1 run file created by run.py.")
    planned = run.get("selected_count")
    records = run.get("records")
    if type(planned) is not int or planned < 1:
        raise ValueError("selected_count must be a positive integer.")
    if not isinstance(records, list) or not records:
        raise ValueError("The run has no saved records to evaluate.")
    if len(records) > planned:
        raise ValueError("There are more records than selected examples.")
    if type(run.get("completed")) is not bool:
        raise ValueError("The run must include a boolean completed field.")
    if run["completed"] and len(records) != planned:
        raise ValueError("A completed run must contain every selected example.")
    if not all(isinstance(run.get(key), str) and run[key]
               for key in ("model", "split", "dataset_sha256")):
        raise ValueError("The run is missing model, split or dataset_sha256 metadata.")

    identifiers, correct, passed = set(), [], []
    tokens = {"input_tokens": 0, "output_tokens": 0}
    total_seconds = 0.0
    for index, record in enumerate(records, 1):
        try:
            identifier = record["id"]
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                raise ValueError("missing or duplicate example ID")
            identifiers.add(identifier)
            outcome = record["outcome"]
            if outcome not in {"ok", "api_error", "invalid_output"}:
                raise ValueError("unknown outcome")
            prediction = record["prediction"]
            if prediction is not None and not isinstance(prediction, dict):
                raise ValueError("prediction must be an object or null")
            if outcome == "ok" and (prediction is None or prediction.get("status") not in {"found", "not_found"}):
                raise ValueError("successful responses must contain a found/not_found prediction")
            did_pass = record["verification"]["passed"]
            if type(did_pass) is not bool:
                raise ValueError("verification.passed must be boolean")
            found = outcome == "ok" and prediction is not None and prediction.get("status") == "found"
            if did_pass and not found:
                raise ValueError("only a successful found prediction can pass verification")
            gold_value = parse_number(record["gold"]["value"])
            gold_scale = record["gold"]["scale"]
            answer_correct = False
            if found:
                try:
                    answer_correct = (parse_number(prediction.get("value")) == gold_value
                                      and prediction.get("scale") == gold_scale)
                except (ValueError, TypeError):
                    pass
            correct.append(answer_correct)
            passed.append(did_pass)
            seconds = record["elapsed_seconds"]
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
                raise ValueError("elapsed_seconds must be a finite nonnegative number")
            total_seconds += seconds
            for key in tokens:
                count = record["usage"][key]
                if type(count) is not int or count < 0:
                    raise ValueError("token counts must be nonnegative integers")
                tokens[key] += count
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid record {index}: {error}") from error

    accepted = sum(passed)
    accepted_correct = sum(is_correct and did_pass for is_correct, did_pass in zip(correct, passed))
    summary = {
        "model": run["model"], "split": run["split"], "dataset_sha256": run["dataset_sha256"],
        "planned_count": planned, "processed_count": len(records), "unprocessed": planned - len(records),
        "successful_responses": sum(row["outcome"] == "ok" for row in records),
        "api_errors": sum(row["outcome"] == "api_error" for row in records),
        "invalid_outputs": sum(row["outcome"] == "invalid_output" for row in records),
        "flagged_for_review": sum(row["outcome"] == "ok" and not flag for row, flag in zip(records, passed)),
        "correct_answers": sum(correct), "accepted_answers": accepted,
        "correct_accepted_answers": accepted_correct,
        "answer_accuracy_over_planned": sum(correct) / planned,
        "pass_rate_over_planned": accepted / planned,
        "accepted_accuracy": accepted_correct / accepted if accepted else None,
        "accepted_error_rate": (accepted - accepted_correct) / accepted if accepted else None,
        "incorrect_passed_ids": [row["id"] for row, good, flag in zip(records, correct, passed) if flag and not good],
        "recorded_input_tokens": tokens["input_tokens"],
        "recorded_output_tokens": tokens["output_tokens"],
        "records_with_usage": sum(row.get("usage_known", False) for row in records),
        "records_without_usage": sum(not row.get("usage_known", False) for row in records),
        "average_elapsed_seconds": total_seconds / len(records),
    }
    summary["performance"] = performance_metrics(records)
    
    output = path.with_name(path.stem + ".summary.json")
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="A run JSON file saved by run.py.")
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.file), indent=2))
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
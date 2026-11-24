"""Run a small sequential financial extraction study with Ollama."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv
from ollama import Client

from extract import Extraction, KEEP_ALIVE, OPTIONS, PROMPT, extract
from verify import verify

ROOT = Path(__file__).resolve().parent
REQUEST_TIMEOUT = 180.0


def save_run(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--out", default="results/local/sample.json")
    parser.add_argument(
        "--exclude-contexts-from",
        help="Exclude every table context present in this saved run.",
    )
    parser.add_argument(
        "--model",
        help="Ollama model tag; overrides OLLAMA_MODEL in .env.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")

    load_dotenv(ROOT / ".env", override=True, encoding="utf-8-sig")
    model = (
        args.model
        if args.model is not None
        else os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    ).strip()

    if not model:
        raise ValueError("The model name must not be empty.")

    data_path = ROOT / "data" / f"{args.split}.json"
    if not data_path.exists():
        raise ValueError("Prepare the data first: uv run python prepare_data.py")
    data_bytes = data_path.read_bytes()
    dataset_sha256 = hashlib.sha256(data_bytes).hexdigest()
    all_examples = json.loads(data_bytes)

    excluded_contexts = set()
    exclusion = None

    if args.exclude_contexts_from:
        exclusion_path = ROOT / args.exclude_contexts_from
        exclusion_bytes = exclusion_path.read_bytes()
        previous = json.loads(exclusion_bytes)

        if (
            not isinstance(previous, dict)
            or previous.get("format_version") != 1
            or previous.get("split") != args.split
            or previous.get("dataset_sha256") != dataset_sha256
        ):
            raise ValueError(
                "The exclusion run must use this dataset version and split."
            )

        previous_records = previous.get("records")
        if not isinstance(previous_records, list) or not previous_records:
            raise ValueError("The exclusion run has no records.")

        known_contexts = {
            example["id"]: example["context_id"]
            for example in all_examples
        }
        for record in previous_records:
            if not isinstance(record, dict):
                raise ValueError("Invalid record in the exclusion run.")
            identifier = record.get("id")
            context = record.get("context_id")
            if (
                not isinstance(identifier, str)
                or identifier not in known_contexts
                or not isinstance(context, str)
                or not context
                or context != known_contexts[identifier]
            ):
                raise ValueError(
                    "An exclusion record does not match the current dataset."
                )
            excluded_contexts.add(context)

        exclusion = {
            "file": args.exclude_contexts_from,
            "sha256": hashlib.sha256(exclusion_bytes).hexdigest(),
            "context_ids": sorted(excluded_contexts),
        }

    eligible = [
        example for example in all_examples
        if example["context_id"] not in excluded_contexts
    ]
    if not eligible:
        raise ValueError("No eligible examples remain after applying exclusions.")
    if len(eligible) < args.limit:
        raise ValueError(
            f"Only {len(eligible)} eligible examples remain; "
            f"choose --limit at most {len(eligible)}."
        )
    examples = eligible[:args.limit]

    output = ROOT / args.out
    if output.suffix != ".json":
        raise ValueError("--out must end in .json.")
    if output.exists():
        raise ValueError(f"{output} already exists. Choose a new output filename.")
    output.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "format_version": 1,
        "provider": "ollama",
        "model": model,
        "ollama_python_version": version("ollama"),
        "split": args.split,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "selected_count": len(examples),
        "completed": False,
        "dataset_sha256": dataset_sha256,
        "selection": {
            "method": "dataset_order",
            "requested_limit": args.limit,
            "eligible_count": len(eligible),
            "selected_context_count": len({
                example["context_id"] for example in examples
            }),
            "selected_ids": [example["id"] for example in examples],
            "exclusion": exclusion,
        },        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "settings": {
            "options": OPTIONS,
            "schema": Extraction.model_json_schema(),
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "request_timeout_seconds": REQUEST_TIMEOUT,
        },
        "warmup": None,
        "records": [],
    }
    save_run(output, result)
    client = Client(timeout=REQUEST_TIMEOUT)

    warmup_example = {
        "question": "What was revenue in 2024?",
        "table": [
            ["Metric (in millions)", "2024"],
            ["Revenue", "10.0"],
        ],
        "paragraphs": [],
    }

    warmup_start = time.perf_counter()
    warmup_answer = extract(
        client,
        model,
        warmup_example["question"],
        warmup_example["table"],
        warmup_example["paragraphs"],
    )

    result["warmup"] = {
        "input": warmup_example,
        "elapsed_seconds": round(
            time.perf_counter() - warmup_start, 6
        ),
        "response": warmup_answer,
    }
    save_run(output, result)

    if warmup_answer["outcome"] == "api_error":
        raise SystemExit(
            f"Warm-up failed: {warmup_answer['error']}\n"
            f"Run saved to {output}; no benchmark requests were made."
        )

    print(
        f"Warm-up: {warmup_answer['outcome']} "
        "(excluded from benchmark scores and timings)",
        flush=True,
    )


    for index, example in enumerate(examples, 1):
        record = {
            **example,
            "verification": {
                "passed": False, "reasons": [], "cell": None,
                "normalised_value": None,
            },
        }
        start = time.perf_counter()
        answer = extract(
            client, model, example["question"],
            example["table"], example["paragraphs"],
        )
        record["elapsed_seconds"] = round(time.perf_counter() - start, 6)
        record.update(answer)
        if record["outcome"] == "ok":
            record["verification"] = verify(example["table"], record["prediction"])

        result["records"].append(record)
        failed_request = record["outcome"] == "api_error"
        result["completed"] = index == len(examples) and not failed_request
        save_run(output, result)
        print(
            f"{index}/{len(examples)} | {record['outcome']} | "
            f"checks_passed={record['verification']['passed']}",
            flush=True,
        )
        if failed_request:
            raise SystemExit(f"{record['error']}\nPartial run saved to {output}")

    print(f"Saved: {output}")
    if len(examples) == 1:
        fields = ["question", "prediction", "verification", "gold",
                  "error", "usage", "elapsed_seconds"]
        print(json.dumps({key: record[key] for key in fields}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Interrupted. Any previously completed records were saved.") from None
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
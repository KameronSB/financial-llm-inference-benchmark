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
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1.")

    load_dotenv(ROOT / ".env", override=True, encoding="utf-8-sig")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M").strip()
    if not model:
        raise ValueError("OLLAMA_MODEL is empty in .env.")

    data_path = ROOT / "data" / f"{args.split}.json"
    if not data_path.exists():
        raise ValueError("Prepare the data first: uv run python prepare_data.py")
    data_bytes = data_path.read_bytes()
    examples = json.loads(data_bytes)[:args.limit]
    if not examples:
        raise ValueError("The selected dataset is empty.")

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
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "settings": {
            "options": OPTIONS,
            "schema": Extraction.model_json_schema(),
            "stream": False,
            "keep_alive": KEEP_ALIVE,
            "request_timeout_seconds": REQUEST_TIMEOUT,
        },
        "records": [],
    }
    save_run(output, result)
    client = Client(timeout=REQUEST_TIMEOUT)

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
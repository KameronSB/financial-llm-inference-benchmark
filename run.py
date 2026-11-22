"""Check the API connection or extract a selected TAT-QA split."""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv

from extract import PROMPT, extract
from verify import verify

ROOT = Path(__file__).resolve().parent


def api_error_message(error: openai.APIError) -> str:
    status = getattr(error, "status_code", None)
    help_text = {
        401: "Check OPENAI_API_KEY in this project's .env; authentication failed.",
        403: "The API project does not permit this request or model.",
        404: "Check OPENAI_MODEL and the models available to your API project.",
        429: "Check API quota/billing and rate limits before retrying.",
        400: "The API rejected the request; check the configured model and SDK version.",
    }.get(status, "Check your network or the API service, then retry.")
    # Do not save the server's full error message, which may contain key fragments.
    return f"{type(error).__name__} (HTTP {status}): {help_text}"


def save_run(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="One small API request; no dataset needed.")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--limit", type=int, default=10, help="Number of examples; 0 means the entire split.")
    parser.add_argument("--out", default="results/sample.json", help="A new results filename.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be 0 or greater.")

    # Explicit project-file precedence avoids a stale PowerShell key overriding .env.
    load_dotenv(ROOT / ".env", override=True, encoding="utf-8-sig")
    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip()
    if not key or key == "replace_with_your_api_key":
        raise ValueError("Copy .env.example to .env and put your API key in .env.")
    if not model:
        raise ValueError("OPENAI_MODEL is empty in .env.")
    client = openai.OpenAI(api_key=key, timeout=60.0, max_retries=2)

    if args.check:
        result = extract(
            client, model, "What was revenue in 2025?",
            [["Metric", "2025"], ["Revenue", "12.5"]],
            [{"order": 1, "text": "Amounts are in millions."}],
        )
        print("API connection and structured output succeeded.")
        print(json.dumps(result, indent=2))
        return

    data_path = ROOT / "data" / f"{args.split}.json"
    if not data_path.exists():
        raise ValueError("Dataset missing. Run: uv run python prepare_data.py")
    data_bytes = data_path.read_bytes()
    examples = json.loads(data_bytes)
    if args.limit:
        examples = examples[:args.limit]
    if not examples:
        raise ValueError("The selected split is empty.")
    output = ROOT / args.out
    if output.suffix.lower() != ".json":
        raise ValueError("--out must end in .json.")
    if output.exists():
        raise ValueError(f"{output.name} already exists. Choose another --out filename.")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "format_version": 1, "model": model, "split": args.split,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "selected_count": len(examples), "completed": False,
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "settings": {"reasoning_effort": "none", "max_output_tokens": 2000},
        "records": [],
    }
    save_run(output, result)

    for index, example in enumerate(examples, 1):
        record = {
            **example, "outcome": "ok", "prediction": None, "error": None,
            "verification": {"passed": False, "reasons": [], "cell": None,
                             "normalised_value": None},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "usage_known": False,
        }
        start = time.perf_counter()
        try:
            # Only these three source fields cross the extraction boundary.
            answer = extract(client, model, example["question"],
                             example["table"], example["paragraphs"])
            record.update(answer)
            record["verification"] = verify(example["table"], answer["prediction"])
        except openai.APIError as error:
            record["outcome"] = "api_error"
            record["error"] = api_error_message(error)
        except ValueError:
            record["outcome"] = "invalid_output"
            record["error"] = "No usable structured output (invalid, incomplete or refused)."
        record["elapsed_seconds"] = round(time.perf_counter() - start, 4)
        result["records"].append(record)
        result["completed"] = index == len(examples) and record["outcome"] != "api_error"
        save_run(output, result)
        print(f"{index}/{len(examples)} | {record['outcome']} | "
              f"checks_passed={record['verification']['passed']}", flush=True)
        if record["outcome"] == "api_error":
            raise SystemExit(f"Stopped. {record['error']}\nPartial results: {output}")

    print(f"Saved: {output}")
    if len(examples) == 1:
        record = result["records"][0]
        print(json.dumps({key: record[key] for key in
                         ["question", "prediction", "verification", "gold"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except openai.APIError as error:
        raise SystemExit(api_error_message(error)) from None
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from None
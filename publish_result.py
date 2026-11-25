"""Generate the v0.1 release from saved local runs, without model requests."""

import csv
import gzip
import io
import json
import shutil
import tarfile
from pathlib import Path
from statistics import mean, median, quantiles
from tempfile import TemporaryDirectory

from evaluate import evaluate, performance_metrics
from verify import parse_number

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "local" / "v0.1"
DESTINATION = ROOT / "results" / "v0.1"
MODELS = {
    "q4": "qwen2.5:7b-instruct-q4_K_M",
    "q8": "qwen2.5:7b-instruct-q8_0",
}
INPUT_FIELDS = ("id", "context_id", "question", "table", "paragraphs", "gold")


def numeric_correct(record: dict) -> bool:
    prediction = record.get("prediction") or {}
    if record["outcome"] != "ok" or prediction.get("status") != "found":
        return False
    try:
        return parse_number(prediction["value"]) == parse_number(record["gold"]["value"])
    except (ValueError, TypeError):
        return False


def timings(records: list[dict]) -> dict:
    seconds = [record["elapsed_seconds"] for record in records]
    return {
        "mean_request_seconds": mean(seconds),
        "median_request_seconds": median(seconds),
        "p95_request_seconds": quantiles(seconds, n=100, method="inclusive")[94],
        "generation_tokens_per_second": performance_metrics(records)[
            "generation_tokens_per_second"
        ],
    }


def main() -> None:
    runs = [
        (quant, repetition, f"qwen7b-{quant}-test211-warm-r{repetition}")
        for quant in MODELS for repetition in (1, 2, 3)
    ]
    names = ["environment.json"] + [
        stem + suffix for _, _, stem in runs
        for suffix in (".json", ".summary.json", ".ps.txt")
    ]
    originals = {name: (SOURCE / name).read_bytes() for name in names}
    environment = json.loads(originals["environment.json"])
    rows, reference = [], None
    pooled = {quant: [] for quant in MODELS}

    # Evaluate temporary copies: evaluate() writes a sibling summary file.
    with TemporaryDirectory() as directory:
        stage = Path(directory)
        for quant, repetition, stem in runs:
            run = json.loads(originals[stem + ".json"])
            records = run["records"]
            identifiers = [record["id"] for record in records]
            if (
                run["completed"] is not True
                or run["split"] != "test"
                or run["model"] != MODELS[quant]
                or run["selected_count"] != 211
                or len(records) != 211
                or len(set(identifiers)) != 211
                or run["selection"]["selected_ids"] != identifiers
                or run["selection"]["exclusion"] is not None
                or run["dataset_sha256"] != environment["test_sha256"]
                or run["warmup"]["response"]["outcome"] != "ok"
            ):
                raise ValueError(f"Unexpected or incomplete frozen run: {stem}")

            inputs = [{key: record[key] for key in INPUT_FIELDS} for record in records]
            signature = (
                inputs, run["dataset_sha256"], run["prompt_sha256"],
                run["settings"], run["warmup"]["input"],
            )
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(f"Inputs or inference settings differ: {stem}")

            temporary_raw = stage / (stem + ".json")
            temporary_raw.write_bytes(originals[stem + ".json"])
            summary = evaluate(temporary_raw)
            if summary != json.loads(originals[stem + ".summary.json"]):
                raise ValueError(f"Regenerated summary differs from original: {stem}")
            if summary["performance"]["generation_measurements"] != len(records):
                raise ValueError(f"Missing generation measurements: {stem}")

            numeric = sum(numeric_correct(record) for record in records)
            rows.append({
                "configuration": quant.upper(), "repetition": repetition,
                "questions": len(records),
                "contexts": len({record["context_id"] for record in records}),
                "strict_correct": summary["correct_answers"],
                "numeric_correct": numeric,
                "cell_checks_passed": summary["accepted_answers"],
                "strict_correct_and_checks_passed": summary["correct_accepted_answers"],
                "reference_incorrect_but_checks_passed": (
                    summary["accepted_answers"] - summary["correct_accepted_answers"]
                ),
                "scale_only_mismatches": numeric - summary["correct_answers"],
                "api_errors": summary["api_errors"],
                "invalid_outputs": summary["invalid_outputs"],
                **timings(records),
                "warmup_seconds_excluded": run["warmup"]["elapsed_seconds"],
            })
            pooled[quant].extend(records)

        with (stage / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (stage / "environment.json").write_bytes(originals["environment.json"])

        # Fixed archive metadata makes repeat executions produce identical bytes.
        with (stage / "raw-runs.tar.gz").open("wb") as handle:
            with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for name, content in sorted(originals.items()):
                        entry = tarfile.TarInfo(name)
                        entry.size, entry.mode, entry.mtime = len(content), 0o644, 0
                        archive.addfile(entry, io.BytesIO(content))

        outputs = [stem + ".summary.json" for _, _, stem in runs] + [
            "comparison.csv", "environment.json", "raw-runs.tar.gz",
        ]
        DESTINATION.mkdir(parents=True, exist_ok=True)
        for name in outputs:
            shutil.copyfile(stage / name, DESTINATION / name)

    for row in rows:
        print(
            f"{row['configuration']} r{row['repetition']}: "
            f"strict={row['strict_correct']}/211, numeric={row['numeric_correct']}/211"
        )
    for quant, records in pooled.items():
        metrics = timings(records)
        print(
            f"{quant.upper()} pooled ({len(records)} requests): "
            f"mean={metrics['mean_request_seconds']:.3f}s, "
            f"median={metrics['median_request_seconds']:.3f}s, "
            f"p95={metrics['p95_request_seconds']:.3f}s, "
            f"generation={metrics['generation_tokens_per_second']:.2f} tokens/s"
        )
    print(f"Generated {len(outputs)} publication files in {DESTINATION}")
    print(f"Archive contains {len(originals)} original files; local inputs unchanged.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"Publication stopped: {error}") from None

"""Download a pinned TAT-QA release and select direct numeric cell questions."""

import hashlib
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from verify import SCALES, parse_number

ROOT = Path(__file__).resolve().parent
COMMIT = "870accc41953dcde885aabeb963d94aabdc0fbc3"
BASE = f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{COMMIT}/dataset_raw"
SOURCES = {
    "train": ("tatqa_dataset_train.json",
              "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69"),
    "dev": ("tatqa_dataset_dev.json",
            "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5"),
    "test": ("tatqa_dataset_test_gold.json",
             "c4d08418359c1d76468dec420ee748a37f48c06b63cb8ec2766f19d5d314b597"),
}


def select_examples(contexts: list[dict]) -> list[dict]:
    examples = []
    for context in contexts:
        table = context["table"]["table"]
        numeric_cells = set()
        for row in table:
            for cell in row:
                try:
                    numeric_cells.add(parse_number(cell))
                except ValueError:
                    pass
        for question in context["questions"]:
            answers = question.get("answer")
            if (question.get("answer_type") != "span"
                    or question.get("req_comparison", False)
                    or question.get("answer_from") not in {"table", "table-text"}
                    or not isinstance(answers, list) or len(answers) != 1):
                continue
            try:
                value = parse_number(answers[0])
            except ValueError:
                continue
            scale = question.get("scale", "") or ""
            if "%" in str(answers[0]):
                if scale not in {"", "percent"}:
                    continue
                scale = "percent"
            if scale not in SCALES or value not in numeric_cells:
                continue
            examples.append({
                "id": question["uid"],
                "context_id": context["table"]["uid"],
                "question": question["question"],
                "table": table,
                "paragraphs": [
                    {"order": p["order"], "text": p["text"]}
                    for p in context["paragraphs"]
                ],
                "gold": {"value": format(value, "f"), "scale": scale},
            })
    return examples


def main() -> None:
    data_dir = ROOT / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "TAT-QA direct numeric table-cell subset",
        "source_commit": COMMIT,
        "source": "https://github.com/NExTplusplus/TAT-QA",
        "paper": "https://aclanthology.org/2021.acl-long.254/",
        "license": "CC BY 4.0",
        "selection": "Single numeric span; no comparison; table/table-text; "
                     "at least one whole cell equals the signed reference value.",
        "normalization": "Percent signs in reference answers imply percent scale. "
                         "Reference values stay in reported units.",
        "splits": {},
    }
    for split, (filename, expected_hash) in SOURCES.items():
        url = f"{BASE}/{filename}"
        raw_path = raw_dir / filename
        if raw_path.exists():
            raw = raw_path.read_bytes()
        else:
            print(f"Downloading {split}...")
            request = Request(url, headers={"User-Agent": "financial-extraction-verification"})
            with urlopen(request, timeout=60) as response:
                raw = response.read()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Source checksum mismatch: {raw_path}")
        raw_path.write_bytes(raw)
        examples = select_examples(json.loads(raw))
        if not examples:
            raise ValueError(f"No examples selected for {split}.")
        output = data_dir / f"{split}.json"
        output.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["splits"][split] = {
            "url": url, "source_sha256": expected_hash, "count": len(examples),
            "prepared_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
        print(f"{split}: {len(examples)} examples -> {output.name}")
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, URLError, ValueError) as error:
        raise SystemExit(f"Data preparation failed: {error}") from None
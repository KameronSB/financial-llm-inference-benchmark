"""Mechanical evidence checks. This module never reads reference answers."""

import re
from decimal import Decimal

SCALES = {"": "1", "thousand": "1000", "million": "1000000",
          "billion": "1000000000", "percent": "0.01"}


def parse_number(value: str) -> Decimal:
    """Read a whole financial number; do not extract digits from prose."""
    text = str(value).strip().replace("−", "-").replace("–", "-")
    text = re.sub(r"[\s,$£€%]", "", text)
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        raise ValueError(f"Not a supported whole number: {value!r}")
    return Decimal(text)


def verify(table: list[list[str]], prediction: dict) -> dict:
    result = {"passed": False, "reasons": [], "cell": None,
              "normalised_value": None}
    if prediction["status"] != "found":
        result["reasons"].append("model_did_not_find_an_answer")
        return result

    row, column = prediction["row"], prediction["column"]
    if (type(row) is not int or type(column) is not int
            or not 0 <= row < len(table)
            or not 0 <= column < len(table[row])):
        result["reasons"].append("invalid_cell_coordinates")
        return result

    cell = str(table[row][column])
    result["cell"] = cell
    try:
        source_value = parse_number(cell)
        predicted_value = parse_number(prediction["value"])
    except ValueError:
        result["reasons"].append("value_or_cell_is_not_numeric")
        return result

    if source_value != predicted_value:
        result["reasons"].append("value_does_not_match_cited_cell")
    scale = prediction["scale"]
    if scale not in SCALES:
        result["reasons"].append("unsupported_scale")
        return result
    if "%" in cell and scale != "percent":
        result["reasons"].append("percent_cell_requires_percent_scale")

    result["normalised_value"] = format(
        predicted_value * Decimal(SCALES[scale]), "f"
    )
    result["passed"] = not result["reasons"]
    return result
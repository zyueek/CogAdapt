"""Selected experimental functions; see provenance/code_excerpts.json.

This review module omits model loading, datasets, and experiment runners.
"""

from __future__ import annotations
import keyword
import math
import re
import numpy as np


def token_category(text: str) -> str:
    """Map a visible token piece to a cross-language category used by the V8 gaze prior."""

    clean = text.replace("\u0120", " ").replace("\u2581", " ").replace("\u010a", "\n").strip()
    if not clean:
        return "whitespace"
    if clean in keyword.kwlist or clean in {
        "public", "private", "protected", "static", "final", "int", "double", "boolean",
        "String", "void", "new", "null", "true", "false",
    }:
        return f"keyword:{clean}"
    if re.fullmatch(r"[-+*/%<>=!&|^~?:]+", clean):
        return "operator"
    if re.fullmatch(r"[(){}\[\],.;]+", clean):
        return "punctuation"
    if re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", clean):
        return "number"
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", clean):
        return "identifier"
    if clean.startswith(('"', "'")):
        return "string"
    return "other"


def _hotspot_lines(solution: str) -> set[int]:
    """Mark V8-positive conditions, returns, updates, recursion, and loop boundaries."""
    lines = solution.splitlines()
    function_names = re.findall(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)", solution, re.MULTILINE)
    result: set[int] = set()
    patterns = (
        r"\bif\b|\belif\b|\belse\b",
        r"\bfor\b|\bwhile\b|\bbreak\b|\bcontinue\b",
        r"\breturn\b",
        r"(?<![=!<>])=(?!=)|\+=|-=|\*=|/=|//=|%=",
    )
    for index, line in enumerate(lines):
        if any(re.search(pattern, line) for pattern in patterns):
            result.add(index)
        if any(re.search(rf"\b{re.escape(name)}\s*\(", line) for name in function_names):
            # The definition line is not a recursive call.
            if not re.match(r"^\s*(?:async\s+)?def\b", line):
                result.add(index)
    return result


def _hotspot_token_flags(offsets: list[tuple[int, int]], solution: str) -> list[bool]:
    lines = _hotspot_lines(solution)
    starts = [0]
    for match in re.finditer("\n", solution):
        starts.append(match.end())
    flags = []
    for start, _ in offsets:
        line = max(0, int(np.searchsorted(starts, start, side="right") - 1))
        flags.append(line in lines)
    return flags


def task_weights(eeg_scores, similarity_weights, human_weighting=True):
    """Standalone translation of V10/V11 dataset task weighting.

    Inputs cover unique training records, before target-matched repetition.
    The final normalized weights need not remain inside the EEG clipping bounds.
    """
    eeg = np.asarray(eeg_scores, dtype=float)
    similarity = np.asarray(similarity_weights, dtype=float)
    if eeg.ndim != 1 or eeg.shape != similarity.shape or not eeg.size:
        raise ValueError("Expected equally sized nonempty vectors")
    if not np.isfinite(eeg).all() or not np.isfinite(similarity).all():
        raise ValueError("Weights require finite inputs")
    if (similarity <= 0).any():
        raise ValueError("Similarity weights must be positive")
    raw = np.asarray(
        [np.clip(math.exp(0.8 * float(z)), 0.4, 2.0) * q
         for z, q in zip(eeg, similarity, strict=True)]
    ) if human_weighting else similarity.copy()
    return raw / raw.mean()


def completion_weights(token_pieces, hotspot_flags, category_prior, human_weighting=True):
    """Standalone translation of the dataset's completion-token weighting.

    Token pieces must be the actual tokenizer pieces; offsets/hotspot flags refer
    to the completion including its formatting suffix. No EEG sample is an input.
    """
    if len(token_pieces) != len(hotspot_flags):
        raise ValueError("Each token needs one hotspot flag")
    values = []
    for piece, hotspot in zip(token_pieces, hotspot_flags, strict=True):
        if human_weighting:
            score = category_prior.get(token_category(str(piece)), 0.0)
            value = float(np.clip(math.exp(0.45 * score), 0.65, 1.5))
            if hotspot:
                value *= 1.35
        else:
            value = 1.0
        values.append(value)
    result = np.asarray(values, dtype=float)
    return result / result.mean() if result.size else result

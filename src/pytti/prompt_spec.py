"""
The single parser for pytti's prompt DSL.

Grammar (one grammar for semantic prompts, direct image prompts, and
weight fields):

    prompt      := text [":" weight [":" stop]]
    weight      := expr ["_" mask ["_" cutoff]]
    mask        := geometric-key | "[" path "]" | "-[" path "]" | semantic-text
    text        := free text | "[" path "]" | bare path (direct prompts)

Colons are only separators at bracket depth zero, and never when they belong
to a URL scheme ("https://...") or a Windows drive letter ("C:\\..."), so
paths — bracketed or bare — survive parsing on every platform. Weights and
stops stay expression strings (evaluated per step by parametric_eval).

Everything downstream consumes the typed `PromptSpec` / `MaskSpec` — no
caller re-parses strings.
"""

from __future__ import annotations

from pathlib import Path

from attrs import define

GEOMETRIC_MASK_KEYS = ("a", "r", "l", "d", "u", "n", "f")

# historical sentinel: "no cutoff given" (see mask_semantic)
DEFAULT_CUTOFF = "0.5000873264"


@define
class MaskAll:
    pass


@define
class MaskGeometric:
    key: str  # one of GEOMETRIC_MASK_KEYS minus "a"


@define
class MaskImage:
    path: str
    inverted: bool


@define
class MaskVideo:
    path: str
    inverted: bool


@define
class MaskSemantic:
    text: str


MaskSpec = MaskAll | MaskGeometric | MaskImage | MaskVideo | MaskSemantic


@define
class PromptSpec:
    text: str  # prompt text, or "[path]"-style image reference
    weight: str  # expression, e.g. "1", "10*t", "bass"
    stop: str  # expression, e.g. "-inf", "0.8"
    mask: MaskSpec
    cutoff: str  # expression; mask threshold
    prompt_string: str  # the original, for display/round-tripping

    def image_path(self) -> str | None:
        """If the text is a bracketed [path], return the path."""
        text = self.text.strip()
        if text.startswith("[") and text.endswith("]"):
            return text[1:-1].strip()
        return None


def _is_scheme_colon(string: str, i: int) -> bool:
    """True if string[i] == ':' belongs to a URL scheme like https://"""
    return string.startswith("//", i + 1) and string[:i].isalpha()


def _is_drive_colon(string: str, i: int, segment_start: int) -> bool:
    """True if string[i] == ':' looks like a Windows drive letter (C:\\ or C:/)."""
    segment = string[segment_start:i]
    return len(segment) == 1 and segment.isalpha() and string[i + 1 : i + 2] in ("\\", "/")


def split_toplevel(string: str, sep: str, maxsplit: int) -> list[str]:
    """
    Split on `sep` at bracket depth zero, skipping URL-scheme and
    drive-letter colons when sep is ':'.
    """
    parts = []
    depth = 0
    start = 0
    i = 0
    while i < len(string) and len(parts) < maxsplit:
        ch = string[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            if sep == ":" and (
                _is_scheme_colon(string, i) or _is_drive_colon(string, i, start)
            ):
                i += 1
                continue
            parts.append(string[start:i])
            start = i + 1
        i += 1
    parts.append(string[start:])
    return parts


def parse_mask_token(token: str) -> MaskSpec:
    token = token.strip()
    inverted = token.startswith("-")
    if inverted:
        token = token[1:]
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if inner.startswith("-"):
            inner = inner[1:]
            inverted = True
        if Path(inner).suffix.lower() == ".mp4":
            return MaskVideo(path=inner, inverted=inverted)
        return MaskImage(path=inner, inverted=inverted)
    if token == "" or token == "a":
        return MaskAll()
    if token in GEOMETRIC_MASK_KEYS:
        return MaskGeometric(key=token)
    # bare .mp4/.png masks (direct-prompt style, no brackets)
    suffix = Path(token).suffix.lower()
    if suffix == ".mp4":
        return MaskVideo(path=token, inverted=inverted)
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        return MaskImage(path=token, inverted=inverted)
    return MaskSemantic(text=("-" + token) if inverted else token)


def parse_weight_spec(weight_field: str) -> tuple[str, MaskSpec, str]:
    """
    Parse a weight field like '1', '2_[mask.png]', '1_r_0.3' into
    (weight_expr, mask, cutoff_expr).
    """
    parts = split_toplevel(weight_field, "_", maxsplit=2)
    weight = parts[0].strip() or "1"
    mask_token = parts[1] if len(parts) > 1 else ""
    cutoff = parts[2].strip() if len(parts) > 2 else DEFAULT_CUTOFF
    return weight, parse_mask_token(mask_token), cutoff or DEFAULT_CUTOFF


def parse_prompt_spec(prompt_string: str) -> PromptSpec:
    """
    Parse a full prompt string into a PromptSpec. Raises ValueError for
    prompts with no text.
    """
    parts = split_toplevel(prompt_string, ":", maxsplit=2)
    text = parts[0].strip()
    weight_field = parts[1].strip() if len(parts) > 1 else "1"
    stop = parts[2].strip() if len(parts) > 2 else "-inf"
    if not text:
        raise ValueError(
            f"Prompt {prompt_string!r} has no text — check for stray '|' or ':' "
            "separators in your scenes."
        )
    weight, mask, cutoff = parse_weight_spec(weight_field or "1")
    return PromptSpec(
        text=text,
        weight=weight,
        stop=stop or "-inf",
        mask=mask,
        cutoff=cutoff,
        prompt_string=prompt_string,
    )

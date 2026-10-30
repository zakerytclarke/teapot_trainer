"""Local IFEval checks.

The instruction ids and the pass rules follow the public
``instruction_following_eval`` checkers. Language id uses ``langdetect`` when
it is installed. Otherwise English is ASCII text, and other languages are
checked by script or a few function words. A missing detector does not invent
a pass for a Latin language other than English.
"""

from __future__ import annotations

import json
import re

_CONSTRAINED_RESPONSES = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)

_SCRIPTS = {
    "ko": (0xAC00, 0xD7A3),
    "ru": (0x0400, 0x04FF),
    "bg": (0x0400, 0x04FF),
    "ar": (0x0600, 0x06FF),
    "fa": (0x0600, 0x06FF),
    "ur": (0x0600, 0x06FF),
    "hi": (0x0900, 0x097F),
    "mr": (0x0900, 0x097F),
    "ne": (0x0900, 0x097F),
    "bn": (0x0980, 0x09FF),
    "pa": (0x0A00, 0x0A7F),
    "gu": (0x0A80, 0x0AFF),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "th": (0x0E00, 0x0E7F),
}

_LATIN_STOPS = {
    "de": {"und", "der", "die", "nicht", "ein"},
    "pt": {"não", "uma", "para", "com"},
    "it": {"che", "non", "una", "per"},
    "fi": {"ja", "on", "ei"},
    "vi": {"không", "của", "một"},
    "sw": {"na", "ya", "kwa"},
}


def _compare(relation: str | None, actual: int, expected: int | None) -> bool:
    if expected is None:
        return False
    if relation == "less than":
        return actual < expected
    if relation == "at least":
        return actual >= expected
    return False


def _count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def _count_sentences(text: str) -> int:
    return len([part for part in re.split(r"[.!?]+", text) if part.strip()])


def _matches_language(text: str, code: str) -> bool:
    try:
        import langdetect
    except ImportError:
        langdetect = None
    if langdetect is not None:
        try:
            return langdetect.detect(text) == code
        except langdetect.LangDetectException:
            return True
    if code == "en":
        letters = [ch for ch in text if ch.isalpha()]
        return bool(letters) and all(ord(ch) < 128 for ch in letters)
    span = _SCRIPTS.get(code)
    if span is not None:
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return False
        hits = sum(1 for ch in letters if span[0] <= ord(ch) <= span[1])
        return hits / len(letters) >= 0.3
    stops = _LATIN_STOPS.get(code, set())
    words = set(re.findall(r"\w+", text.lower()))
    return bool(words & stops)


def _check(instruction_id: str, raw: dict, value: str) -> bool:
    args = raw if isinstance(raw, dict) else {}
    if instruction_id == "keywords:existence":
        keywords = [str(word) for word in (args.get("keywords") or []) if str(word)]
        return bool(keywords) and all(re.search(re.escape(word), value, flags=re.IGNORECASE) for word in keywords)
    if instruction_id == "keywords:frequency":
        keyword = str(args.get("keyword") or "")
        if not keyword:
            return False
        found = len(re.findall(re.escape(keyword), value, flags=re.IGNORECASE))
        return _compare(args.get("relation"), found, args.get("frequency"))
    if instruction_id == "keywords:forbidden_words":
        words = [str(word) for word in (args.get("forbidden_words") or []) if str(word)]
        return bool(words) and all(
            re.search(r"\b" + re.escape(word) + r"\b", value, flags=re.IGNORECASE) is None for word in words
        )
    if instruction_id == "keywords:letter_frequency":
        letter = str(args.get("letter") or "").lower()
        if len(letter) != 1:
            return False
        found = sum(1 for ch in value.lower() if ch == letter)
        return _compare(args.get("let_relation"), found, args.get("let_frequency"))
    if instruction_id == "language:response_language":
        code = str(args.get("language") or "")
        return bool(code) and _matches_language(value, code)
    if instruction_id == "length_constraints:number_words":
        return _compare(args.get("relation"), _count_words(value), args.get("num_words"))
    if instruction_id == "length_constraints:number_sentences":
        return _compare(args.get("relation"), _count_sentences(value), args.get("num_sentences"))
    if instruction_id == "length_constraints:number_paragraphs":
        paragraphs = re.split(r"\s?\*\*\*\s?", value)
        count = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                if index == 0 or index == len(paragraphs) - 1:
                    count -= 1
                else:
                    return False
        expected = args.get("num_paragraphs")
        return expected is not None and count == expected
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        paragraphs = re.split(r"\n\n", value)
        count = len(paragraphs)
        for paragraph in paragraphs:
            if not paragraph.strip():
                count -= 1
        nth = args.get("nth_paragraph")
        expected_paragraphs = args.get("num_paragraphs")
        first = str(args.get("first_word") or "").lower()
        if not isinstance(nth, int) or nth <= 0 or nth > count or not first:
            return False
        paragraph = paragraphs[nth - 1].strip()
        if not paragraph:
            return False
        word = paragraph.split()[0].strip().lstrip("'\"").lower()
        cleaned = ""
        for letter in word:
            if letter in {".", ",", "?", "!", "'", '"'}:
                break
            cleaned += letter
        return count == expected_paragraphs and cleaned == first
    if instruction_id == "detectable_content:number_placeholders":
        expected = args.get("num_placeholders")
        return isinstance(expected, int) and len(re.findall(r"\[.*?\]", value)) >= expected
    if instruction_id == "detectable_content:postscript":
        marker = str(args.get("postscript_marker") or "")
        lowered = value.lower()
        if marker == "P.P.S":
            pattern = r"\s*p\.\s?p\.\s?s.*$"
        elif marker == "P.S.":
            pattern = r"\s*p\.\s?s\..*$"
        elif marker:
            pattern = r"\s*" + re.escape(marker.lower()) + r".*$"
        else:
            return False
        return bool(re.findall(pattern, lowered, flags=re.MULTILINE))
    if instruction_id == "detectable_format:number_bullet_lists":
        stars = re.findall(r"^\s*\*[^*].*$", value, flags=re.MULTILINE)
        dashes = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
        expected = args.get("num_bullets")
        return isinstance(expected, int) and len(stars) + len(dashes) == expected
    if instruction_id == "detectable_format:constrained_response":
        stripped = value.strip()
        return any(option in stripped for option in _CONSTRAINED_RESPONSES)
    if instruction_id == "detectable_format:number_highlighted_sections":
        singles = re.findall(r"\*[^\n*]+\*", value)
        doubles = re.findall(r"\*\*[^\n*]+\*\*", value)
        count = sum(1 for item in singles if item.strip("*").strip())
        count += sum(1 for item in doubles if item.removeprefix("**").removesuffix("**").strip())
        expected = args.get("num_highlights")
        return isinstance(expected, int) and count >= expected
    if instruction_id == "detectable_format:multiple_sections":
        spliter = str(args.get("section_spliter") or "")
        expected = args.get("num_sections")
        if not spliter or not isinstance(expected, int):
            return False
        pattern = r"\s?" + re.escape(spliter) + r"\s?\d+\s?"
        return len(re.split(pattern, value)) - 1 >= expected
    if instruction_id == "detectable_format:json_format":
        cleaned = (
            value.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(cleaned)
        except ValueError:
            return False
        return True
    if instruction_id == "detectable_format:title":
        titles = re.findall(r"<<[^\n]+>>", value)
        return any(title.lstrip("<").rstrip(">").strip() for title in titles)
    if instruction_id == "combination:two_responses":
        parts = []
        chunks = value.split("******")
        for index, chunk in enumerate(chunks):
            if not chunk.strip():
                if index != 0 and index != len(chunks) - 1:
                    return False
            else:
                parts.append(chunk)
        return len(parts) == 2 and parts[0].strip() != parts[1].strip()
    if instruction_id == "combination:repeat_prompt":
        prompt = str(args.get("prompt_to_repeat") or "").strip().lower()
        return bool(prompt) and value.strip().lower().startswith(prompt)
    if instruction_id == "startend:end_checker":
        phrase = str(args.get("end_phrase") or "").strip().lower()
        return bool(phrase) and value.strip().strip('"').lower().endswith(phrase)
    if instruction_id == "startend:quotation":
        stripped = value.strip()
        return len(stripped) > 1 and stripped[0] == '"' and stripped[-1] == '"'
    if instruction_id == "change_case:capital_word_frequency":
        words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", value)
        capitals = sum(1 for word in words if word.isupper())
        return _compare(args.get("capital_relation"), capitals, args.get("capital_frequency"))
    if instruction_id == "change_case:english_capital":
        return value.isupper() and _matches_language(value, "en")
    if instruction_id == "change_case:english_lowercase":
        return value.islower() and _matches_language(value, "en")
    if instruction_id == "punctuation:no_comma":
        return re.search(r",", value) is None
    return False


def _variants(response: str) -> list[str]:
    lines = response.split("\n")
    stripped = [
        response,
        "\n".join(lines[1:]),
        "\n".join(lines[:-1]),
        "\n".join(lines[1:-1]),
    ]
    starred = [item.replace("*", "") for item in stripped]
    variants = []
    seen = set()
    for item in stripped + starred:
        if item not in seen:
            seen.add(item)
            variants.append(item)
    return variants


def score_ifeval(row: dict, prediction: str) -> dict[str, float]:
    """Prompt-level and instruction-level strict and loose accuracy for one example."""
    ids = list(row.get("instruction_id_list") or [])
    kwargs = list(row.get("kwargs") or [])
    empty = {
        "prompt_strict": 0.0,
        "prompt_loose": 0.0,
        "instruction_strict": 0.0,
        "instruction_loose": 0.0,
    }
    if not ids:
        return empty
    text = prediction or ""
    variants = _variants(text)
    strict_hits = []
    loose_hits = []
    for index, instruction_id in enumerate(ids):
        raw = kwargs[index] if index < len(kwargs) else {}
        strict_hits.append(1.0 if _check(instruction_id, raw, text) else 0.0)
        loose_hits.append(1.0 if any(_check(instruction_id, raw, variant) for variant in variants) else 0.0)
    return {
        "prompt_strict": 1.0 if all(strict_hits) else 0.0,
        "prompt_loose": 1.0 if all(loose_hits) else 0.0,
        "instruction_strict": sum(strict_hits) / len(strict_hits),
        "instruction_loose": sum(loose_hits) / len(loose_hits),
    }

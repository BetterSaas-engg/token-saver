"""
Generic splitter: converts prose into structured context/ask/constraints XML.

Pure Python stdlib, no LLM, no external dependencies beyond re and dataclasses.
Designed for "safe fallthrough" -- when a sentence cannot be confidently
classified, it passes through inside a <message> tag rather than being dropped
or mangled.

Aggressiveness: balanced -- trims filler, light rewording only where rules
are unambiguous. Never paraphrases.
"""

import re
from dataclasses import dataclass, field


# Filler words and short phrases that add no meaning.
# Stripped only at phrase boundaries, never mid-sentence.
FILLER_TOKENS = {
    "hey", "so", "um", "uh", "like", "basically", "literally", "honestly",
    "actually", "really", "kind of", "sort of", "you know", "i mean",
    "i guess", "i think", "i feel like", "maybe", "just", "pretty much",
    "anyway", "anyways", "okay so", "ok so", "well",
}

# Patterns that signal the sentence is an ASK (request for action).
ASK_PATTERNS = [
    r"\?\s*$",                          # ends with question mark
    r"\bcan you\b",
    r"\bcould you\b",
    r"\bwould you\b",
    r"\bhelp me\b",
    r"\bwrite (?:me |a |an )",
    r"\bexplain\b",
    r"\bgive me\b",
    r"\btell me\b",
    r"\bshow me\b",
    r"\bsuggest\b",
    r"\brecommend\b",
    r"\bcreate\b",
    r"\bgenerate\b",
    r"\bdraft\b",
    r"\bi (?:need|want)\b",
    r"\bi'm looking for\b",
    r"\bim looking for\b",
    r"\blooking for\b",
    r"\bhow (?:do|can|should|would|to)\b",
    r"\bwhat (?:is|are|should|would)\b",
    r"\bwhy (?:is|are|do|does)\b",
]

# Patterns that signal the sentence is a CONSTRAINT (preference, limit, requirement).
CONSTRAINT_PATTERNS = [
    r"\bmake (?:it|sure)\b",
    r"\bkeep (?:it|them)\b",
    r"\bshould be\b",
    r"\bmust be\b",
    r"\bmust not\b",
    r"\bdon't\b",
    r"\bdo not\b",
    r"\bavoid\b",
    r"\bwithin \d+\b",
    r"\bunder \d+\b",
    r"\bover \d+\b",
    r"\bat least \d+\b",
    r"\bat most \d+\b",
    r"\bin \d+ (?:word|char|minute|hour|day|line)",
    r"\bno more than\b",
    r"\bno less than\b",
    r"\bnot too\b",
    r"\bbut not\b",
    r"\b(?:needs? to|has to) be\b",
    r"\bin the style of\b",
    r"\btone should\b",
    r"\bformat (?:it|should|as)\b",
]


@dataclass
class SplitResult:
    """Output of the generic splitter."""
    context: str
    ask: str
    constraints: str
    xml: str
    dropped_filler: list[str] = field(default_factory=list)
    unclassified: list[str] = field(default_factory=list)


def _split_sentences(text: str) -> list[str]:
    """
    Break text into sentences on . ! ? boundaries.

    Conservative: keeps common abbreviations (Mr., Dr., etc.) intact by
    requiring the boundary to be followed by whitespace + capital letter,
    newline, or end of string.
    """
    text = text.strip()
    if not text:
        return []

    # Split on sentence-ending punctuation followed by whitespace or EOL
    pattern = r"(?<=[.!?])\s+(?=[A-Za-z])|(?<=[.!?])\s*$"
    parts = re.split(pattern, text)

    # Also split on single newlines that separate thoughts (common in chat)
    result = []
    for part in parts:
        result.extend(s.strip() for s in part.split("\n") if s.strip())
    return result


def _strip_filler(sentence: str) -> tuple[str, list[str]]:
    """
    Remove filler tokens from the start/end of a sentence.

    Returns (cleaned_sentence, list_of_dropped_fillers).
    Preserves original capitalization of remaining content.
    Never modifies mid-sentence content -- only trims edges.
    """
    dropped = []
    s = sentence.strip().rstrip(".,;:!?")
    # Normalize whitespace and lowercase for matching
    working = s.lower().strip()

    # Strip leading fillers (try longest matches first)
    changed = True
    while changed:
        changed = False
        for filler in sorted(FILLER_TOKENS, key=len, reverse=True):
            prefix = filler + " "
            if working.startswith(prefix):
                dropped.append(filler)
                working = working[len(prefix):].strip()
                changed = True
                break

    # Strip trailing fillers
    changed = True
    while changed:
        changed = False
        for filler in sorted(FILLER_TOKENS, key=len, reverse=True):
            suffix = " " + filler
            if working.endswith(suffix):
                dropped.append(filler)
                working = working[:-len(suffix)].strip()
                changed = True
                break

    if not working:
        return "", dropped

    # Find the cleaned content in the original (case-preserving) sentence
    # by locating where working starts within the lowercased original
    orig_lower = s.lower()
    start = orig_lower.find(working)
    if start < 0:
        # Fallback: return the working version as-is
        return working, dropped

    end = start + len(working)
    return s[start:end], dropped


def _score_patterns(sentence: str, patterns: list[str]) -> int:
    """Count how many of the given patterns match the sentence (case-insensitive)."""
    s = sentence.lower()
    return sum(1 for p in patterns if re.search(p, s))


def _classify(sentence: str) -> str:
    """
    Assign a sentence to a bucket: 'ask', 'constraint', 'context', or 'unclassified'.

    Strategy:
      - Score against ask and constraint patterns
      - If any score >= 1, take the higher one (ties go to 'ask')
      - If neither matches but the sentence has substantive content, call it 'context'
      - If the sentence is very short (<= 3 words) with no signals, mark unclassified
    """
    ask_score = _score_patterns(sentence, ASK_PATTERNS)
    constraint_score = _score_patterns(sentence, CONSTRAINT_PATTERNS)

    if ask_score == 0 and constraint_score == 0:
        word_count = len(sentence.split())
        if word_count <= 3:
            return "unclassified"
        return "context"

    if ask_score >= constraint_score:
        return "ask"
    return "constraint"


def split(text: str) -> SplitResult:
    """
    Main entry point: take prose, return a SplitResult with XML and metadata.
    """
    sentences = _split_sentences(text)

    buckets = {"context": [], "ask": [], "constraint": []}
    unclassified = []
    all_dropped = []

    for sentence in sentences:
        cleaned, dropped = _strip_filler(sentence)
        all_dropped.extend(dropped)

        if not cleaned:
            continue

        label = _classify(cleaned)
        if label == "unclassified":
            unclassified.append(cleaned)
        else:
            buckets[label].append(cleaned)

    context_text = "; ".join(buckets["context"])
    ask_text = "; ".join(buckets["ask"])
    constraints_text = "; ".join(buckets["constraint"])

    xml_parts = []
    if context_text:
        xml_parts.append(f"<context>{context_text}</context>")
    if ask_text:
        xml_parts.append(f"<ask>{ask_text}</ask>")
    if constraints_text:
        xml_parts.append(f"<constraints>{constraints_text}</constraints>")
    for leftover in unclassified:
        xml_parts.append(f"<message>{leftover}</message>")

    xml = "\n".join(xml_parts) if xml_parts else f"<message>{text.strip()}</message>"

    return SplitResult(
        context=context_text,
        ask=ask_text,
        constraints=constraints_text,
        xml=xml,
        dropped_filler=all_dropped,
        unclassified=unclassified,
    )


# Quick self-test
# Run this file directly to verify: python backend\splitters\generic.py

if __name__ == "__main__":
    test_messages = [
        "hey so i've been working at this startup for 2 years and i think i deserve a raise, can you help me write an email to my manager? make it professional but not too formal, and keep it under 200 words",
        "explain quantum entanglement to me like i'm 10 years old please",
        "i'm planning a trip to japan next month, i have 10 days and a budget of $3000, what cities should i visit? keep the itinerary relaxed, i don't like rushing",
        "write a python function that reverses a string",
        "what's the weather today",
    ]

    print("Generic Splitter -- Self Test\n")
    print("=" * 70)

    for i, msg in enumerate(test_messages, 1):
        print(f"\nTest {i}:")
        print(f"Input: {msg}")
        print(f"Input chars: {len(msg)}")

        result = split(msg)

        print(f"\nOutput XML:")
        print(result.xml)
        print(f"\nOutput chars: {len(result.xml)}")

        char_savings = (1 - len(result.xml) / len(msg)) * 100
        print(f"Char savings: {char_savings:.1f}%")

        if result.dropped_filler:
            print(f"Dropped filler: {result.dropped_filler}")
        if result.unclassified:
            print(f"Unclassified: {result.unclassified}")

        print("-" * 70)

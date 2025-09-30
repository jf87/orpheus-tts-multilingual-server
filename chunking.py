#!/usr/bin/env python3
"""
Advanced text chunking module with German-specific abbreviation handling.

This module provides intelligent text chunking for TTS systems, with support
for soft length limits to preserve natural sentence boundaries.
"""

import re
from typing import List, Optional

# German abbreviations that should not trigger sentence breaks
GER_ABBREVIATIONS = {
    "z. b.", "z.b.", "u. a.", "u.a.", "u. ä.", "u.ä.", "bzw.", "usw.", "etc.",
    "ca.", "vgl.", "s.", "sog.", "bzgl.", "inkl.", "exkl.", "ggf.", "bspw.",
    "d. h.", "d.h.", "dr.", "prof.", "nr.", "abs.", "kap.", "str.", "hrsg.",
    "geb.", "od.", "evtl.", "i. d. r.", "i.d.r."
}

SENT_END = re.compile(r"""(?x)
    (.+?
     [\.!?…]+
     (?:["»«„“”‚‘’\)\]]+)?    # optional closing quotes/parens
    )
    (?=\s+|$)
""")

SOFT_BREAK = re.compile(r"([;:—–\-]\s+|,\s+)")
SPACE_SPLIT = re.compile(r"\s+")

def _norm(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    sentinel = " <§PAR§> "
    text = text.replace("\n\n", sentinel)
    text = re.sub(r"\s+", " ", text)
    return text.replace(sentinel, "\n\n").strip()

def _is_abbrev(fragment: str) -> bool:
    t = re.sub(r"\s+", " ", fragment.strip().lower())
    return t in GER_ABBREVIATIONS

def _sentences(paragraph: str) -> List[str]:
    out, last_end = [], 0
    for m in SENT_END.finditer(paragraph):
        cand = m.group(1).strip()
        tokens = cand.split()
        last4 = " ".join(tokens[-4:]) if tokens else ""
        # Avoid splitting after abbreviations and ordinals like "3."
        if _is_abbrev(last4) or (tokens and _is_abbrev(tokens[-1])) or re.search(r"\b\d+\.$", cand):
            continue
        out.append(cand)
        last_end = m.end(1)
    rest = paragraph[last_end:].strip()
    if rest:
        out.append(rest)
    return out

def _hard_wrap(text: str, max_len: int) -> List[str]:
    words = SPACE_SPLIT.split(text.strip())
    out, cur = [], ""
    for w in words:
        add = w if not cur else " " + w
        if len(cur) + len(add) > max_len:
            if cur:
                out.append(cur)
            if len(w) > max_len:
                for i in range(0, len(w), max_len):
                    out.append(w[i:i+max_len])
                cur = ""
            else:
                cur = w
        else:
            cur += add
    if cur:
        out.append(cur)
    return out

def _split_long_piece(piece: str, max_len: int) -> List[str]:
    if len(piece) <= max_len:
        return [piece]
    parts, cur = [], ""
    for seg in SOFT_BREAK.split(piece):
        if not seg:
            continue
        cand = (cur + seg).strip() if cur else seg.strip()
        if len(cand) <= max_len:
            cur = cand
        else:
            if cur:
                parts.append(cur)
                cur = ""
            if len(seg) > max_len:
                parts.extend(_hard_wrap(seg, max_len))
            else:
                cur = seg.strip()
    if cur:
        parts.append(cur)
    final = []
    for p in parts:
        final.extend(_hard_wrap(p, max_len) if len(p) > max_len else [p])
    return final

_ENDS_WITH_PUNCT = re.compile(r'[\.!?…]["»«„“”‚‘’\)\]]*\s*$')

def split_text_into_chunks_chars(
    text: str,
    max_chars: int = 200,
    prefer_end_punct: bool = True,
    soft_max_ratio: float = 0.85,
    max_sentences_per_chunk: Optional[int] = 2,
    soft_allowance: int = 40,
    soft_allow_ratio: float = 0.2,
) -> List[str]:
    """
    German-aware text chunking with soft length limits.

    This algorithm balances two competing goals:
    1. Keep chunks under max_chars for model efficiency
    2. Preserve natural sentence boundaries for better TTS output

    Soft overshoot strategy:
    - Start preferring to end chunks at soft_max_ratio * max_chars (e.g., 170 of 200)
    - Allow up to max_chars + min(soft_allowance, max_chars * soft_allow_ratio)
      to finish a complete sentence cleanly
    - Example: max_chars=200 → can extend to ~240 chars to avoid mid-sentence cuts

    Args:
        text: Input text to chunk
        max_chars: Target maximum characters per chunk
        prefer_end_punct: Prefer ending chunks at sentence boundaries
        soft_max_ratio: Start preferring to end around this fraction of max_chars
        max_sentences_per_chunk: Max sentences per chunk (None = unlimited)
        soft_allowance: Absolute max chars allowed beyond max_chars
        soft_allow_ratio: Percentage-based max chars allowed beyond max_chars

    Returns:
        List of text chunks
    """
    assert max_chars > 20, "max_chars too small"
    text = _norm(text)
    if not text:
        return []

    # Calculate soft and hard limits
    soft_cap_pref = int(max_chars * soft_max_ratio)  # Start preferring to end here
    soft_overshoot = min(soft_allowance, int(max_chars * soft_allow_ratio))  # Max overshoot allowed
    soft_max = max_chars + max(0, soft_overshoot)  # Absolute maximum with overshoot

    chunks, cur, sent_in_cur = [], "", 0

    def push():
        nonlocal cur, sent_in_cur
        if cur:
            chunks.append(cur.strip())
            cur, sent_in_cur = "", 0

    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        pieces = _sentences(para) or [para]

        for s in pieces:
            # If a single sentence is already over soft_max, we must split it.
            if len(s) > soft_max:
                push()
                chunks.extend(_split_long_piece(s, max_chars))
                continue

            if not cur:
                cur, sent_in_cur = s, 1
                continue

            # Try to add s
            candidate_len = len(cur) + 1 + len(s)

            # 1) Hard fit
            if candidate_len <= max_chars:
                # But if we're already beyond the "preferred" size, consider ending here for punctuation
                nearing_pref = prefer_end_punct and (len(cur) >= soft_cap_pref)
                hit_sentence_limit = (max_sentences_per_chunk and sent_in_cur >= max_sentences_per_chunk)
                if nearing_pref or hit_sentence_limit:
                    push()
                    cur, sent_in_cur = s, 1
                else:
                    cur = cur + " " + s
                    sent_in_cur += 1
                continue

            # 2) Soft overshoot: allow only if we'll end on a sentence and stay ≤ soft_max
            if prefer_end_punct and candidate_len <= soft_max and _ENDS_WITH_PUNCT.match(s):
                cur = cur + " " + s
                push()  # flush now since we used the soft overshoot to end nicely
                continue

            # 3) Otherwise, flush and start new chunk with s
            push()
            cur, sent_in_cur = s, 1

        # Paragraph boundary: flush
        push()

    return [c for c in chunks if c]


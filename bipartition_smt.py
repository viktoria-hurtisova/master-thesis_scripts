#!/usr/bin/env python3
"""bipartition_smt.py

Batch-rewrite SMT-LIB v2 files that contain exactly one gigantic assertion of the
form

    (assert (let ... (and φ₁ φ₂ … φₙ)))

into two separate assertions A and B, each wrapped with a `:named` attribute so
that they can be used for binary interpolation.

For every <file>.smt2 the script creates <file>.bi.smt2 with this structure::

    ; original meta-info and declarations come first …

    (assert (! (let ... (and φ₁ … φ_k)) :named A))
    (assert (! (let ... (and φ_{k+1} … φₙ)) :named B))
    (check-sat)
    (exit)

The split position k = ⌊n/2⌋ (first half vs. second half).  Feel free to adjust
`split_conjuncts()` if you need a different policy (e.g. keyword-based).

Usage::

    $ python scripts/bipartition_smt.py inputs/RandomCoupledUnsat

Requires only Python 3 (no external deps).
"""
from __future__ import annotations

import pathlib
import re
import sys
from typing import List, Tuple

def split_conjuncts(body: str) -> Tuple[str, str, str]:
    """Return (prefix, and_left, and_right).

    prefix  = text from beginning up to *and* (typically the whole `(let …`).
    and_left/right = `(and …)` terms for the two halves.
    """
    # Find the outermost (and ...) by looking for the deepest nested one
    # Since the structure is (let (...) (and ...)), we want the first (and
    and_start = -1
    i = 0
    while i < len(body):
        if body[i:i+4] == "(and":
            and_start = i
            break
        i += 1
    
    if and_start == -1:
        raise ValueError("Cannot locate (and …) in assert body")

    prefix = body[:and_start]

    # Find the matching closing paren for this (and
    depth = 0
    and_end = -1
    for j in range(and_start, len(body)):
        if body[j] == '(':
            depth += 1
        elif body[j] == ')':
            depth -= 1
            if depth == 0:
                and_end = j
                break
    
    if and_end == -1:
        raise ValueError("Unmatched parenthesis while extracting (and …)")

    and_block = body[and_start:and_end + 1]
    suffix = body[and_end + 1:]

    # Extract content inside (and ...)
    content = and_block[4:-1].strip()  # Remove "(and" and ")"

    # Split conjuncts at top level (depth 0)
    conjuncts = []
    current = []
    depth = 0
    i = 0
    
    while i < len(content):
        ch = content[i]
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch.isspace() and depth == 0:
            # End of a conjunct
            if current:
                conjunct = ''.join(current).strip()
                if conjunct:
                    conjuncts.append(conjunct)
                current = []
        else:
            current.append(ch)
        i += 1
    
    # Add the last conjunct
    if current:
        conjunct = ''.join(current).strip()
        if conjunct:
            conjuncts.append(conjunct)

    if len(conjuncts) < 2:
        raise ValueError(f"Need at least two conjuncts to split, found {len(conjuncts)}")

    # Take the first conjunct as the left side and the rest as the right side
    left = conjuncts[0]
    right = conjuncts[1]

    if len(conjuncts) > 2:
        right = f"(and {' '.join(conjuncts[1:])})"
    else:
        right = conjuncts[1]

    #left = f"(and {' '.join(conjuncts[:1])})"
    #right = f"(and {' '.join(conjuncts[1:])})"

    return prefix, left + suffix, right + suffix


# ---------------------------------------------------------------------------
# file processing
# ---------------------------------------------------------------------------

def transform(text: str) -> str:
    """Rewrite a single SMT-LIB text, return new text."""
    lines = text.splitlines()
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("(assert "):
            # Found the assert line - process it
            # Extract everything between (assert and the final )
            assert_content = stripped[8:-1]  # Remove "(assert " and ")"
            prefix, left, right = split_conjuncts(assert_content)
            
            # Add two new assertions
            result_lines.append(f"(assert (! {prefix}{left} :named A))")
            result_lines.append(f"(assert (! {prefix}{right} :named B))")
        else:
            # Keep the line as-is
            result_lines.append(line)
    
    return '\n'.join(result_lines)


# ---------------------------------------------------------------------------
# main script entry-point
# ---------------------------------------------------------------------------

def main(argv: List[str]):
    if len(argv) != 2 or argv[1] in {"-h", "--help"}:
        print("Usage: python bipartition_smt.py <directory>")
        sys.exit(1)

    target_dir = pathlib.Path(argv[1])
    if not target_dir.is_dir():
        print(f"error: {target_dir} is not a directory", file=sys.stderr)
        sys.exit(2)

    for smt_file in sorted(target_dir.glob("*.smt2")):
        # Skip files that are already bipartitioned
        if smt_file.name.endswith(".bi.smt2"):
            continue
            
        try:
            text = smt_file.read_text(encoding="utf-8")
            new_text = transform(text)
            out_file = smt_file.with_suffix(".bi.smt2")
            out_file.write_text(new_text, encoding="utf-8")
            print(f"✓ {smt_file.name} → {out_file.name}")
        except Exception as e:
            print(f"✗ {smt_file.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv)

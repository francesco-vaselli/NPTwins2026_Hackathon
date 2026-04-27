"""Validate a leaderboard-submission PR.

Reads the diff between two git refs and checks:

  1. The PR touches ONLY submissions/scorecards/<id>.json
     and submissions/notebooks/<id>.ipynb (exactly one of each).
  2. The two <id>s match.
  3. The JSON parses, has the required schema, and `id` matches the filename.
  4. The notebook is a valid .ipynb and has at least one executed code cell
     (proves the student actually ran it).

Files in the PR are read with `git show <ref>:<path>` — never executed.
The script itself lives on `main` and is trusted.

Exits 0 on success, non-zero with a printed reason on failure.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys


SCORECARD_RE = re.compile(r"^submissions/scorecards/([A-Za-z0-9_\-.]+)\.json$")
NOTEBOOK_RE  = re.compile(r"^submissions/notebooks/([A-Za-z0-9_\-.]+)\.ipynb$")

REQUIRED_SCORECARD_KEYS = {"ws_per_feature", "ws_sum", "c2st", "auc_delta_btag"}
REQUIRED_TOP_KEYS       = {"id", "name", "scorecard", "config"}


def fail(msg: str) -> "None":
    print(f"❌ {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return res.stdout


def changed_files(base: str, head: str) -> list[tuple[str, str]]:
    out = run(["git", "diff", "--name-status", f"{base}..{head}"])
    files = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        status, path = parts[0], parts[-1]
        files.append((status, path))
    return files


def read_at_ref(ref: str, path: str) -> str:
    return run(["git", "show", f"{ref}:{path}"])


def main() -> "None":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="git ref of the PR base (e.g. origin/main)")
    ap.add_argument("--head", required=True, help="git ref of the PR head (e.g. pr-head)")
    args = ap.parse_args()

    files = changed_files(args.base, args.head)
    if not files:
        fail("No file changes detected between base and head.")

    scorecards, notebooks, others = [], [], []
    for status, path in files:
        if status == "D":
            others.append(("delete", path))
            continue
        if SCORECARD_RE.match(path):
            scorecards.append(path)
        elif NOTEBOOK_RE.match(path):
            notebooks.append(path)
        else:
            others.append((status, path))

    if others:
        rendered = ", ".join(f"{s} {p}" for s, p in others)
        fail(
            "PR touches files outside submissions/scorecards or submissions/notebooks "
            f"(or deletes files). Disallowed changes: {rendered}"
        )
    if len(scorecards) != 1:
        fail(f"Expected exactly one scorecard JSON; found {len(scorecards)}: {scorecards}")
    if len(notebooks) != 1:
        fail(f"Expected exactly one notebook .ipynb; found {len(notebooks)}: {notebooks}")

    sc_id = SCORECARD_RE.match(scorecards[0]).group(1)
    nb_id = NOTEBOOK_RE.match(notebooks[0]).group(1)
    if sc_id != nb_id:
        fail(
            f"Scorecard id ({sc_id!r}) and notebook id ({nb_id!r}) must match — "
            "use the SAME <id> for both files."
        )

    raw_json = read_at_ref(args.head, scorecards[0])
    try:
        sub = json.loads(raw_json)
    except json.JSONDecodeError as e:
        fail(f"Scorecard JSON does not parse: {e}")

    missing = REQUIRED_TOP_KEYS - set(sub)
    if missing:
        fail(f"Scorecard JSON missing top-level keys: {sorted(missing)}")
    if sub["id"] != sc_id:
        fail(f"JSON 'id' field ({sub['id']!r}) does not match filename ({sc_id!r}).")

    sc = sub["scorecard"]
    if not isinstance(sc, dict):
        fail("'scorecard' must be a dict.")
    missing = REQUIRED_SCORECARD_KEYS - set(sc)
    if missing:
        fail(f"scorecard.* missing keys: {sorted(missing)}")
    for k in ("ws_sum", "c2st", "auc_delta_btag"):
        if not isinstance(sc[k], (int, float)):
            fail(f"scorecard.{k} must be a number, got {type(sc[k]).__name__}.")
        if sc[k] < 0:
            fail(f"scorecard.{k} must be non-negative, got {sc[k]}.")
    if not isinstance(sc["ws_per_feature"], list) or not sc["ws_per_feature"]:
        fail("scorecard.ws_per_feature must be a non-empty list.")

    raw_nb = read_at_ref(args.head, notebooks[0])
    try:
        nb = json.loads(raw_nb)
    except json.JSONDecodeError as e:
        fail(f"Notebook JSON does not parse: {e}")
    if "cells" not in nb or not isinstance(nb["cells"], list):
        fail("Notebook has no 'cells' array — is this a real .ipynb?")
    n_executed = sum(
        1 for c in nb["cells"]
        if c.get("cell_type") == "code" and c.get("outputs")
    )
    if n_executed == 0:
        fail(
            "Notebook has no code cells with outputs — please run the notebook "
            "end-to-end before submitting (Save with cells executed)."
        )

    print(f"✅ Submission '{sc_id}' validated.")
    print(f"   ws_sum         = {sc['ws_sum']:.4f}")
    print(f"   c2st           = {sc['c2st']:.4f}")
    print(f"   auc_delta_btag = {sc['auc_delta_btag']:.4f}")
    print(f"   notebook code cells with outputs: {n_executed}")


if __name__ == "__main__":
    main()

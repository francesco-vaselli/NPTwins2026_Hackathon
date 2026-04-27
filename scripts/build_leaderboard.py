"""Rebuild LEADERBOARD.md from submissions/scorecards/*.json.

Sorted by ws_sum ascending (lower = better).
Each row links to the executed notebook in submissions/notebooks/<id>.ipynb.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCORECARD_DIR = ROOT / "submissions" / "scorecards"
NOTEBOOK_DIR  = ROOT / "submissions" / "notebooks"
OUT           = ROOT / "LEADERBOARD.md"


def load_all() -> list[dict]:
    rows = []
    for path in sorted(SCORECARD_DIR.glob("*.json")):
        try:
            with path.open() as f:
                data = json.load(f)
            sc = data["scorecard"]
            rows.append({
                "id":             data.get("id", path.stem),
                "name":           data.get("name", "—"),
                "run_tag":        data.get("run_tag", "—"),
                "ws_sum":         float(sc["ws_sum"]),
                "c2st":           float(sc["c2st"]),
                "auc_delta_btag": float(sc["auc_delta_btag"]),
                "notebook":       (NOTEBOOK_DIR / f"{data.get('id', path.stem)}.ipynb"),
            })
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"⚠️  skipping {path.name}: {e}")
    rows.sort(key=lambda r: r["ws_sum"])
    return rows


def render(rows: list[dict]) -> str:
    lines = []
    lines.append("# 🏆 NP_Twins Leaderboard")
    lines.append("")
    lines.append(
        "Auto-generated from `submissions/scorecards/*.json`. "
        "Lower `ws_sum` is better."
    )
    lines.append("")
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    lines.append(f"_Last updated: {now} — {len(rows)} entries._")
    lines.append("")

    if not rows:
        lines.append("_No submissions yet. Be the first!_")
        lines.append("")
        return "\n".join(lines)

    lines.append("| # | Name | Run | `ws_sum` ↓ | `c2st` | `auc_delta_btag` | Notebook |")
    lines.append("|---|------|-----|-----------:|-------:|-----------------:|----------|")
    for i, r in enumerate(rows, 1):
        nb_rel = r["notebook"].relative_to(ROOT).as_posix()
        nb_link = f"[{r['id']}]({nb_rel})" if r["notebook"].exists() else r["id"]
        lines.append(
            f"| {i} | {r['name']} | `{r['run_tag']}` "
            f"| {r['ws_sum']:.4f} | {r['c2st']:.4f} | {r['auc_delta_btag']:.4f} "
            f"| {nb_link} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    rows = load_all()
    OUT.write_text(render(rows))
    print(f"Wrote {OUT} ({len(rows)} entries).")

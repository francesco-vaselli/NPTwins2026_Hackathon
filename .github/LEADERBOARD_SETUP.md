# Leaderboard CI — one-time repo setup

The two workflows under `.github/workflows/` plus the scripts under `scripts/`
implement a fully automated, no-manual-review leaderboard:

- `validate-and-merge.yml` — validates each PR that touches `submissions/**`
  and (if the validation passes) calls `gh pr merge --auto --squash`.
- `build-leaderboard.yml` — on push to `main`, rebuilds `LEADERBOARD.md`
  from `submissions/scorecards/*.json`.

For the auto-merge to actually fire, you must do these one-time steps in
GitHub repo **Settings** (the workflows alone are not enough):

## 1. Enable auto-merge
**Settings → General → Pull Requests** → tick **Allow auto-merge**.

## 2. Add a branch protection rule on `main`
**Settings → Branches → Add classic branch protection rule** (or **Rulesets**):

- Branch name pattern: `main`
- Require a pull request before merging: ✅
  - Required approvals: **0** (we're trusting the CI gate, not humans)
- Require status checks to pass: ✅
  - Add the check named **`validate`** (job name from `validate-and-merge.yml`).
  - Tick "Require branches to be up to date before merging" if you want
    students to rebase on top of the latest leaderboard.
- Restrict pushes that create matching branches: ✅ (so submissions can only
  land via PR, never via direct push from a maintainer's clone).

## 3. Allow GitHub Actions to write to PRs
**Settings → Actions → General → Workflow permissions**:

- "Read and write permissions": ✅
- "Allow GitHub Actions to create and approve pull requests": ✅

## 4. (Optional but recommended) Restrict workflow approvals
**Settings → Actions → General → Fork pull request workflows from outside
collaborators**:

- "Require approval for first-time contributors who are new to GitHub": ✅

This is a soft safety net — the workflow itself is already hardened (it
never executes PR code), but this stops random new accounts from spamming
the runner queue.

## What the validator enforces

A PR is auto-mergeable iff **all** of these hold:

- The diff touches **only** `submissions/scorecards/<id>.json` and
  `submissions/notebooks/<id>.ipynb`.
- Exactly one of each, and the two `<id>`s match.
- The JSON parses, has the required schema, and its `id` field matches the
  filename.
- The notebook is valid `.ipynb` JSON and has at least one code cell with
  recorded outputs (i.e. the student actually ran it before submitting).

Anything else (touching the baseline notebook, uploading random files,
empty-output notebooks, etc.) → validation fails → no auto-merge.

## What the validator does NOT do

- It does **not** re-run the notebook (cost/time), and it does **not**
  recompute the scorecard. The notebook in `submissions/notebooks/` is the
  audit trail — anyone can clone the repo and verify a top entry by
  re-running the notebook locally.
- It does not check `ws_sum` against the baseline. A regression entry is
  legal — it just won't rank well.

## Local sanity check

Before pushing the workflows, you can dry-run the validator against your
own branch:

```bash
python scripts/validate_submission.py --base origin/main --head HEAD
python scripts/build_leaderboard.py
```

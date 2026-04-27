# Leaderboard CI — one-time repo setup

The workflow under `.github/workflows/validate-and-merge.yml` plus the
scripts under `scripts/` implement a fully automated, no-manual-review
leaderboard:

- On every PR that touches `submissions/**`, the workflow validates the
  submission, merges the PR if validation passes, then rebuilds
  `LEADERBOARD.md` and pushes it back to `main` — all in a single job.
- `build-leaderboard.yml` is kept as a manual / fallback workflow
  (e.g. for direct edits to a scorecard on `main` outside the PR flow).

The workflow uses a **fine-grained Personal Access Token** stored as
the repo secret `LEADERBOARD_TOKEN`. This is required because:

1. The default `GITHUB_TOKEN` cannot call `enablePullRequestAutoMerge`
   (so we use direct `gh pr merge --squash` instead).
2. On personal repos, `github-actions[bot]` cannot be added to a
   ruleset bypass list, so a `GITHUB_TOKEN`-driven push to `main` is
   blocked by "require PR" rules.

A PAT minted by the repo admin acts as the admin user, which IS in the
ruleset bypass list, so its pushes go through.

For everything to work, do these one-time steps in GitHub repo
**Settings**:

## 1. Allow GitHub Actions to write
**Settings → Actions → General → Workflow permissions**:

- ✅ "Read and write permissions"
- ✅ "Allow GitHub Actions to create and approve pull requests"

## 2. Mint a fine-grained PAT and add as a repo secret

**Personal Settings (top-right avatar) → Developer settings → Personal
access tokens → Fine-grained tokens → Generate new token**:

- Token name: `NPTwins2026 leaderboard bot`
- Expiration: as you like (e.g. 90 days; renew before expiry)
- Resource owner: your account
- Repository access: **Only select repositories** → `NPTwins2026_Hackathon`
- Repository permissions:
  - **Contents**: Read and write
  - **Pull requests**: Read and write
  - **Metadata**: Read (auto-selected)
- Generate, copy the `github_pat_...` value once.

Then **Repo Settings → Secrets and variables → Actions → New repository
secret**:

- Name: `LEADERBOARD_TOKEN`
- Value: paste the PAT.

## 3. Branch protection on `main`
**Settings → Branches → Branch protection rule** (or **Rulesets**):

- Branch name pattern: `main`
- ✅ Require a pull request before merging
  - Required approvals: **0**
- ❌ **Do NOT** "Require status checks to pass" — leave this **off**.
  (Reason: the workflow that performs the merge is itself the validator,
  so its check is "in_progress" at merge time. Requiring it would
  deadlock the merge step.)
- ✅ "Restrict pushes that create matching branches" — so `main` can
  only be written via PR (or by the workflow's own leaderboard commit).

The workflow itself is the gate: it merges only when validation
succeeds, and forks have no write access to `upstream/main` regardless,
so students cannot bypass the validator.

## 4. Auto-merge — NOT needed

The "Allow auto-merge" repo setting and `gh pr merge --auto` are not
used in this design. You can leave the repo setting on or off; it has
no effect here.

## What the validator enforces

A PR is auto-mergeable iff **all** of these hold:

- The diff touches **only** `submissions/scorecards/<id>.json` and
  `submissions/notebooks/<id>.ipynb`.
- Exactly one of each, and the two `<id>`s match.
- The JSON parses, has the required schema, and its `id` field matches
  the filename.
- The notebook is valid `.ipynb` JSON and has at least one code cell
  with recorded outputs (i.e. the student actually ran it before
  submitting).

Anything else (touching the baseline notebook, uploading random files,
empty-output notebooks, etc.) → validation fails → no merge.

## What the validator does NOT do

- It does **not** re-run the notebook (cost/time), and it does **not**
  recompute the scorecard. The notebook in `submissions/notebooks/` is
  the audit trail — anyone can clone the repo and verify a top entry by
  re-running the notebook locally.
- It does not check `ws_sum` against the baseline. A regression entry
  is legal — it just won't rank well.

## Local sanity check

Before pushing the workflows, you can dry-run the validator against
your own branch:

```bash
python scripts/validate_submission.py --base origin/main --head HEAD
python scripts/build_leaderboard.py
```

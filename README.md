# Flow Matching — Tutorial on CMS Jet Data

A three-notebook tutorial that builds up **Flow Matching** from scratch and applies it to simulating reconstruction-level jet features from gen-level features at CMS (a fast ML surrogate for detector simulation).

## Notebooks

| File | What it covers |
|---|---|
| `01_flow_matching_fundamentals.ipynb` | Build Flow Matching on 2D toy data: ODEs, Euler + Heun integrators, conditional paths, the CFM loss, a tiny MLP vector field, backward integration for invertibility. ✅ self-validating via `test_flow_matching.py` |
| `02_conditional_flow_model.ipynb` | Apply conditional flow matching to CMS jet data — learn `p(reco \| gen)`. |
| `03_sampling_and_evaluation.ipynb` | Generate reco-level samples, evaluate with ROC curves, Wasserstein distance, corner plots. |

Each student notebook has a matching `*_solution.ipynb` with the reference implementation.

## Getting started

You have two options. **Colab is strongly recommended** for the tutorial — no local install, GPU available for free.

### Option A — Google Colab (zero setup ☁️)

Click the "Open in Colab" badge at the top of any notebook. The first cell installs the remaining dependencies and downloads the test suite automatically.

### Option B — Local setup with `venv` + `pip` (no extra tooling 🛠️)

`venv` and `pip` ship with Python itself — no extra installers. You just need Python ≥ 3.10.

```bash
git clone https://github.com/fvaselli/NP_Twins.git
cd NP_Twins

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Register the kernel with Jupyter so notebooks can find it
python -m ipykernel install --user --name flow-matching-tutorial --display-name "Flow Matching"

jupyter notebook
```

When the notebook opens, pick the **Flow Matching** kernel from `Kernel → Change kernel`.

### Option C — Local setup with `uv` (faster ⚡, needs one-time install)

[`uv`](https://docs.astral.sh/uv/) is a Rust-based drop-in replacement for `venv + pip`; on a cold cache it resolves and installs these deps in a handful of seconds instead of a minute. Trade-off: you have to install `uv` itself once.

```bash
# Install uv (one-time, one line)
curl -LsSf https://astral.sh/uv/install.sh | sh              # macOS / Linux
# or:  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# Then, same flow as Option B but with `uv`:
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
python -m ipykernel install --user --name flow-matching-tutorial --display-name "Flow Matching"
jupyter notebook
```

## Using the test suite

`test_flow_matching.py` ships with Notebook 1. The student notebook calls into it automatically after each task — run the cells as you go and watch the ✅s light up. If you prefer to run the tests by hand:

```python
from test_flow_matching import test_euler_integrate
test_euler_integrate(euler_integrate)
```

## Hardware

- **CPU-only is fine** for Notebook 1 (each 2D model trains in ~1 min).
- Notebooks 2 and 3 benefit from a GPU. Colab's free T4 is more than enough.

## Reference material

- `lecture-notes.pdf` — full lecture notes on Flow Matching (morning seminar)
- `An introduction to Flow Matching · Cambridge MLG Blog.pdf` — excellent external reference
- `Vaselli_2024_Mach._Learn.__Sci._Technol._5_035007.pdf` — the paper behind Notebooks 2 and 3
- `figs/` — pre-generated figures used in the notebooks

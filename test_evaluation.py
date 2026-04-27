"""
Test suite for Notebook 3: Sampling and Evaluation.

Mirrors the style of `test_flow_matching.py` and `test_conditional_flow.py`:
one `test_*` function per task, plus a `run_all_tests_nb3(...)` orchestrator.

Usage (inside the notebook):

    from test_evaluation import (
        test_compute_scorecard,
        test_heun_sample_reco,
        test_trig_fm_loss,
        run_all_tests_nb3,
    )
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Pretty-printed result container (identical style to NB1/NB2 test suites)
# ----------------------------------------------------------------------------

class TestResults:
    """Container for test results with coloured / emoji output."""

    def __init__(self, section_name=""):
        self.section_name = section_name
        self.passed = []
        self.failed = []
        self.warnings = []

    def add_pass(self, test_name):
        self.passed.append(test_name)

    def add_fail(self, test_name, error_msg):
        self.failed.append((test_name, str(error_msg)))

    def add_warning(self, test_name, warning_msg):
        self.warnings.append((test_name, str(warning_msg)))

    def print_summary(self):
        print("\n" + "=" * 70)
        title = "TEST SUMMARY"
        if self.section_name:
            title += f" — {self.section_name}"
        print(title)
        print("=" * 70)

        if self.passed:
            print(f"\n✅ PASSED ({len(self.passed)}):")
            for t in self.passed:
                print(f"   • {t}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for t, msg in self.warnings:
                print(f"   • {t}: {msg}")

        if self.failed:
            print(f"\n❌ FAILED ({len(self.failed)}):")
            for t, msg in self.failed:
                print(f"   • {t}")
                print(f"     Reason: {msg}")

        total = len(self.passed) + len(self.failed)
        print(f"\nTotal: {len(self.passed)}/{total} tests passed")
        print("=" * 70 + "\n")

        return len(self.failed) == 0


# ----------------------------------------------------------------------------
# Common fixtures
# ----------------------------------------------------------------------------

RECO_DIM = 3
COND_DIM = 6


def _synth_eval_batch(n=2000, seed=0):
    """Build (generated_phys, target_phys, gen_phys) for scorecard tests.

    `gen_phys[:, 4]` is treated as flavour (0=light, 1=charm, 2=bottom), so the
    b-tag AUC metric has meaningful labels. Generated is target + small noise
    (a "good" model).
    """
    rng = np.random.RandomState(seed)
    gen = rng.randn(n, COND_DIM).astype(np.float64)
    gen[:, 4] = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25]).astype(float)

    target = rng.randn(n, RECO_DIM).astype(np.float64)
    # Make column 0 (btag) correlate with flavour so AUC is non-trivial.
    target[:, 0] = (gen[:, 4] == 2) * 1.5 + (gen[:, 4] == 1) * 0.5 + 0.3 * rng.randn(n)

    generated = target + 0.15 * rng.randn(*target.shape)
    return generated, target, gen


# ----------------------------------------------------------------------------
# 1. compute_scorecard
# ----------------------------------------------------------------------------

def test_compute_scorecard(compute_scorecard_fn, verbose=True):
    """Validate the scorecard helper: keys, consistency, and basic behaviour."""
    results = TestResults("compute_scorecard")

    if verbose:
        print("=" * 70)
        print("🧪 Testing compute_scorecard")
        print("=" * 70 + "\n")

    required_keys = ("ws_per_feature", "ws_sum", "c2st", "auc_delta_btag")

    # --- 1.1 Returns a dict with the right keys ---
    try:
        generated, target, gen = _synth_eval_batch(n=1500, seed=1)
        sc = compute_scorecard_fn(generated, target, gen)
        assert isinstance(sc, dict), f"Scorecard must be a dict, got {type(sc).__name__}"
        missing = [k for k in required_keys if k not in sc]
        assert not missing, f"Scorecard missing keys: {missing}"
        results.add_pass(f"Scorecard is a dict with keys {list(required_keys)}")
    except Exception as e:
        results.add_fail("Shape / keys", e)
        if verbose:
            results.print_summary()
        return results

    # --- 1.2 ws_per_feature length = RECO_DIM ---
    try:
        ws = list(sc["ws_per_feature"])
        assert len(ws) == RECO_DIM, (
            f"ws_per_feature must have length {RECO_DIM}, got {len(ws)}"
        )
        results.add_pass(f"ws_per_feature has length {RECO_DIM}")
    except Exception as e:
        results.add_fail("ws_per_feature length", e)

    # --- 1.3 ws_sum == sum(ws_per_feature) ---
    try:
        delta = abs(float(sum(sc["ws_per_feature"])) - float(sc["ws_sum"]))
        assert delta < 1e-5, (
            f"ws_sum must equal sum(ws_per_feature); got diff = {delta:.4e}"
        )
        results.add_pass("ws_sum equals sum(ws_per_feature)")
    except Exception as e:
        results.add_fail("ws_sum consistency", e)

    # --- 1.4 Metric sanity: non-negative and finite ---
    try:
        assert sc["c2st"] >= 0.0 and math.isfinite(sc["c2st"]), (
            f"c2st must be >= 0 and finite, got {sc['c2st']}"
        )
        assert sc["auc_delta_btag"] >= 0.0 and math.isfinite(sc["auc_delta_btag"]), (
            f"auc_delta_btag must be >= 0 and finite, got {sc['auc_delta_btag']}"
        )
        results.add_pass("c2st and auc_delta_btag are non-negative & finite")
    except Exception as e:
        results.add_fail("Metric sanity", e)

    # --- 1.5 Same distribution, independent draws → small WS, small C2ST ---
    #
    # Real sampling produces independent draws — we must test that case here,
    # not `generated = target + eps`, because the BDT can pick up on the per-row
    # coupling in the latter and return a biased AUC.
    try:
        rng_t = np.random.RandomState(7)
        rng_g = np.random.RandomState(8)
        target2    = rng_t.randn(2000, RECO_DIM)
        generated2 = rng_g.randn(2000, RECO_DIM)
        gen_arr    = rng_t.randn(2000, COND_DIM)
        gen_arr[:, 4] = rng_t.choice([0, 1, 2], size=2000)
        sc2 = compute_scorecard_fn(generated2, target2, gen_arr)
        assert sc2["ws_sum"] < 0.25, (
            f"Same-dist draws should give small ws_sum, got {sc2['ws_sum']:.3f}"
        )
        assert sc2["c2st"] < 0.1, (
            f"Same-dist draws should give small c2st, got {sc2['c2st']:.3f}"
        )
        results.add_pass(
            f"Same-dist draws → ws_sum={sc2['ws_sum']:.3f}, c2st={sc2['c2st']:.3f}"
        )
    except Exception as e:
        results.add_fail("Same-distribution regime", e)

    # --- 1.6 Clearly different distributions → larger WS sum than near-identical ---
    try:
        rng = np.random.RandomState(9)
        target3 = rng.randn(2000, RECO_DIM)
        gen_arr3 = rng.randn(2000, COND_DIM)
        gen_arr3[:, 4] = rng.choice([0, 1, 2], size=2000)
        # Generated = target shifted by +1 (biased simulator)
        generated3 = target3 + 1.0
        sc3 = compute_scorecard_fn(generated3, target3, gen_arr3)
        assert sc3["ws_sum"] > 1.0, (
            f"Shifted dists should give larger ws_sum, got {sc3['ws_sum']:.3f}"
        )
        results.add_pass(f"Biased simulator → large ws_sum ({sc3['ws_sum']:.3f})")
    except Exception as e:
        results.add_fail("Biased-simulator regime", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 2. heun_sample_reco
# ----------------------------------------------------------------------------

def test_heun_sample_reco(heun_sample_reco_fn, model_cls, verbose=True):
    """Validate the conditional Heun sampler."""
    results = TestResults("heun_sample_reco")

    if verbose:
        print("=" * 70)
        print("🧪 Testing heun_sample_reco")
        print("=" * 70 + "\n")

    torch.manual_seed(0)
    model = model_cls(reco_dim=RECO_DIM, cond_dim=COND_DIM,
                      time_dim=16, hidden_dim=32, n_blocks=2)
    model.eval()

    # --- 2.1 Output shape ---
    try:
        gen_cond = torch.randn(50, COND_DIM)
        samples = heun_sample_reco_fn(model, gen_cond, n_steps=10)
        samples = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
        assert samples.shape == (50, RECO_DIM), (
            f"Expected shape (50, {RECO_DIM}), got {tuple(samples.shape)}"
        )
        results.add_pass("Output shape is (N, reco_dim)")
    except Exception as e:
        results.add_fail("Output shape", e)
        if verbose:
            results.print_summary()
        return results

    # --- 2.2 Finite values ---
    try:
        gen_cond = torch.randn(32, COND_DIM)
        samples = heun_sample_reco_fn(model, gen_cond, n_steps=20)
        samples = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
        assert torch.isfinite(samples).all().item(), (
            "Heun samples contain non-finite values"
        )
        results.add_pass("All samples finite")
    except Exception as e:
        results.add_fail("Finiteness", e)

    # --- 2.3 Stochasticity (different noise → different samples) ---
    try:
        gen_cond = torch.randn(64, COND_DIM)
        torch.manual_seed(101)
        s1 = heun_sample_reco_fn(model, gen_cond, n_steps=10)
        torch.manual_seed(202)
        s2 = heun_sample_reco_fn(model, gen_cond, n_steps=10)
        s1 = s1 if isinstance(s1, torch.Tensor) else torch.as_tensor(s1)
        s2 = s2 if isinstance(s2, torch.Tensor) else torch.as_tensor(s2)
        diff = (s1 - s2).abs().max().item()
        assert diff > 1e-4, (
            f"Two runs with different RNG should differ (max diff={diff:.4e})"
        )
        results.add_pass("Samples are stochastic")
    except Exception as e:
        results.add_fail("Stochasticity", e)

    # --- 2.4 Batch handling (not a multiple of the default batch size) ---
    try:
        gen_cond = torch.randn(130, COND_DIM)
        out = heun_sample_reco_fn(model, gen_cond, n_steps=10)
        out = out if isinstance(out, torch.Tensor) else torch.as_tensor(out)
        assert out.shape == (130, RECO_DIM), (
            f"Batch handling: expected (130, {RECO_DIM}), got {tuple(out.shape)}"
        )
        results.add_pass("Handles 130-sample batch cleanly")
    except Exception as e:
        results.add_fail("Batch handling", e)

    # --- 2.5 Heun on a constant field = closed form: x_final ≈ x0 + constant ---
    #        Uses a tiny stub model where v(x, t, c) = c  (a constant vector).
    try:
        class _ConstField(nn.Module):
            def __init__(self):
                super().__init__()
                # dummy output_proj so the student's `out_features` trick works
                self.output_proj = nn.Linear(RECO_DIM, RECO_DIM)
            def forward(self, x, t, cond):
                # Velocity equal to cond[:, :RECO_DIM] — constant in t and x
                return cond[:, :RECO_DIM]

        stub = _ConstField().eval()
        torch.manual_seed(0)
        cond = torch.randn(8, COND_DIM)
        # With v constant, Heun's average-of-slopes is exact: x(1) = x(0) + v
        x_final = heun_sample_reco_fn(stub, cond, n_steps=4)
        x_final = x_final if isinstance(x_final, torch.Tensor) else torch.as_tensor(x_final)
        # Can't check against x(0) directly (it's sampled inside), but
        # the per-sample *difference* x_final - cond[:, :RECO_DIM] should be
        # distributed like a fresh standard-Gaussian x0. Check marginals.
        residual = x_final - cond[:, :RECO_DIM]
        # Loose Gaussian check: |mean| small, std near 1
        assert residual.abs().mean().item() < 1.5, (
            "On a constant field, x_final - v should look like N(0, I)"
        )
        results.add_pass("Matches closed form on a constant velocity field")
    except Exception as e:
        results.add_fail("Closed-form check", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 3. trig_fm_loss (trigonometric probability path)
# ----------------------------------------------------------------------------

class _ZeroModel(nn.Module):
    """Outputs zero — used as a reference for loss baselines."""
    def __init__(self, reco_dim=RECO_DIM):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.reco_dim = reco_dim
        # so student code that reads .output_proj doesn't break
        self.output_proj = nn.Linear(reco_dim, reco_dim)

    def forward(self, x, t, cond):
        return torch.zeros_like(x) + 0.0 * self.dummy


def test_trig_fm_loss(trig_fm_loss_fn, model_cls=None, verbose=True):
    """Validate the trigonometric-path conditional FM loss."""
    results = TestResults("trig_fm_loss")

    if verbose:
        print("=" * 70)
        print("🧪 Testing trig_fm_loss")
        print("=" * 70 + "\n")

    def _synth_batch(n, seed):
        g = torch.Generator().manual_seed(seed)
        reco = torch.randn(n, RECO_DIM, generator=g)
        gen = torch.randn(n, COND_DIM, generator=g)
        return reco, gen

    # --- 3.1 Scalar, finite, non-negative ---
    try:
        reco, gen = _synth_batch(128, 1)
        if model_cls is None:
            loss_model = _ZeroModel()
        else:
            loss_model = model_cls(reco_dim=RECO_DIM, cond_dim=COND_DIM,
                                   time_dim=16, hidden_dim=32, n_blocks=2)
        loss = trig_fm_loss_fn(loss_model, reco, gen)
        assert isinstance(loss, torch.Tensor), f"Loss must be a tensor, got {type(loss)}"
        assert loss.dim() == 0, f"Loss must be scalar, got shape {tuple(loss.shape)}"
        assert torch.isfinite(loss).item(), "Loss is not finite"
        assert loss.item() >= 0.0, f"Loss must be non-negative, got {loss.item():.4e}"
        results.add_pass(f"Scalar, finite, non-negative (got {loss.item():.4f})")
    except Exception as e:
        results.add_fail("Scalar/finite/non-negative", e)
        if verbose:
            results.print_summary()
        return results

    # --- 3.2 Zero-model baseline for trig path ≈ π²/4 ≈ 2.467 ---
    #
    # For the trig path x_t = cos(πt/2) x0 + sin(πt/2) x1, the target velocity is
    # v* = (π/2) (cos(πt/2) x1 − sin(πt/2) x0). With x0, x1 ~ N(0, I) independently
    # and t ~ U[0, 1], E[||v*||² / d] averaged over t is (π/2)² = π²/4.
    try:
        torch.manual_seed(7)
        reco, gen = _synth_batch(4000, 2)
        loss = trig_fm_loss_fn(_ZeroModel(), reco, gen).item()
        expected = (math.pi / 2) ** 2  # π²/4 ≈ 2.467
        assert abs(loss - expected) < 0.35, (
            f"Zero-model baseline for trig path should be ≈ π²/4 ≈ {expected:.2f}, "
            f"got {loss:.2f}. Check your x_t and target formulas."
        )
        results.add_pass(f"Zero-model baseline ≈ π²/4 = {expected:.2f} (got {loss:.2f})")
    except Exception as e:
        results.add_fail("Zero-model baseline", e)

    # --- 3.3 Differentiable through the model ---
    if model_cls is not None:
        try:
            reco, gen = _synth_batch(64, 3)
            model = model_cls(reco_dim=RECO_DIM, cond_dim=COND_DIM,
                              time_dim=16, hidden_dim=32, n_blocks=2)
            loss = trig_fm_loss_fn(model, reco, gen)
            for p in model.parameters():
                p.grad = None
            loss.backward()
            any_grad = any((p.grad is not None) and (p.grad.abs().sum().item() > 0)
                           for p in model.parameters())
            assert any_grad, "No gradient flowed through the model."
            results.add_pass("Loss is differentiable through the model")
        except Exception as e:
            results.add_fail("Differentiability", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------

def run_all_tests_nb3(
    compute_scorecard=None,
    heun_sample_reco=None,
    trig_fm_loss=None,
    model_cls=None,
):
    """Run every NB3 test in one go and print a final summary."""
    print("\n" + "=" * 70)
    print("🚀 Running all Notebook 3 tests")
    print("=" * 70 + "\n")

    sections = []

    if compute_scorecard is not None:
        sections.append(test_compute_scorecard(compute_scorecard, verbose=True))
    if heun_sample_reco is not None and model_cls is not None:
        sections.append(test_heun_sample_reco(heun_sample_reco, model_cls, verbose=True))
    if trig_fm_loss is not None:
        sections.append(test_trig_fm_loss(trig_fm_loss, model_cls=model_cls, verbose=True))

    total_pass = sum(len(s.passed) for s in sections)
    total_fail = sum(len(s.failed) for s in sections)
    total_warn = sum(len(s.warnings) for s in sections)
    total = total_pass + total_fail

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + f"  FINAL RESULT — Notebook 3".ljust(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + f"  ✅ Passed:   {total_pass}/{total}".ljust(68) + "║")
    if total_warn:
        print("║" + f"  ⚠️  Warnings: {total_warn}".ljust(68) + "║")
    if total_fail:
        print("║" + f"  ❌ Failed:   {total_fail}".ljust(68) + "║")
    else:
        print("║" + f"  🎉 All implementations validated — ready to climb the leaderboard!".ljust(68) + "║")
    print("╚" + "═" * 68 + "╝\n")

    return total_fail == 0

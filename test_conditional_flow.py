"""
Test suite for Notebook 2: Conditional Flow Matching on CMS jet data.

Mirrors the style of `test_flow_matching.py`: one `test_*` function per task,
plus a `run_all_tests_nb2(...)` orchestrator that runs everything at the end.

Usage (inside the notebook):

    from test_conditional_flow import (
        test_sinusoidal_embedding,
        test_conditional_vector_field_model,
        test_conditional_fm_loss,
        test_train_conditional_model,
        test_generate_reco,
        run_all_tests_nb2,
    )

The tests only depend on PyTorch + NumPy — no CMS data is required, we
fabricate small synthetic batches so students can validate each building
block in a second.
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Pretty-printed result container (same style as NB1 test suite)
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


def _synth_batch(n=64, reco_dim=RECO_DIM, cond_dim=COND_DIM, seed=0):
    """Create a small synthetic (reco, gen) batch for testing."""
    g = torch.Generator().manual_seed(seed)
    reco = torch.randn(n, reco_dim, generator=g)
    gen = torch.randn(n, cond_dim, generator=g)
    return reco, gen


# ----------------------------------------------------------------------------
# 1. sinusoidal_embedding
# ----------------------------------------------------------------------------

def test_sinusoidal_embedding(sinusoidal_embedding, verbose=True):
    """Validate the sinusoidal time embedding."""
    results = TestResults("sinusoidal_embedding")

    if verbose:
        print("=" * 70)
        print("🧪 Testing sinusoidal_embedding")
        print("=" * 70 + "\n")

    # --- 1.1 Output shape ---
    try:
        t = torch.rand(32, 1)
        emb = sinusoidal_embedding(t, dim=16)
        assert emb.shape == (32, 16), (
            f"Expected shape (32, 16), got {tuple(emb.shape)}"
        )
        results.add_pass("Output shape is (batch, dim)")
    except Exception as e:
        results.add_fail("Output shape", e)
        if verbose:
            results.print_summary()
        return results

    # --- 1.2 Values are bounded in [-1, 1] ---
    try:
        t = torch.linspace(0, 1, 50).unsqueeze(1)
        emb = sinusoidal_embedding(t, dim=32)
        assert emb.abs().max().item() <= 1.0 + 1e-5, (
            f"Sinusoidal embedding values should lie in [-1, 1], "
            f"got max |value| = {emb.abs().max().item():.4f}"
        )
        results.add_pass("Values bounded in [-1, 1]")
    except Exception as e:
        results.add_fail("Bounded values", e)

    # --- 1.3 Different times give different embeddings ---
    try:
        t1 = torch.zeros(1, 1)
        t2 = torch.full((1, 1), 0.5)
        e1 = sinusoidal_embedding(t1, dim=16)
        e2 = sinusoidal_embedding(t2, dim=16)
        diff = (e1 - e2).abs().max().item()
        assert diff > 1e-3, (
            f"Embeddings at t=0 and t=0.5 should differ, got max |diff| = {diff:.4e}"
        )
        results.add_pass("Embedding depends on t")
    except Exception as e:
        results.add_fail("Time dependence", e)

    # --- 1.4 Supports different even dims ---
    try:
        t = torch.rand(4, 1)
        for dim in (8, 16, 32, 64):
            emb = sinusoidal_embedding(t, dim=dim)
            assert emb.shape == (4, dim), f"dim={dim}: shape {emb.shape}"
        results.add_pass("Works for a range of even dims")
    except Exception as e:
        results.add_fail("Multiple dims", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 2. ConditionalVectorField (model class)
# ----------------------------------------------------------------------------

def test_conditional_vector_field_model(model_cls, verbose=True):
    """Validate the ConditionalVectorField class: construction, forward, gradients."""
    results = TestResults("ConditionalVectorField")

    if verbose:
        print("=" * 70)
        print("🧪 Testing ConditionalVectorField")
        print("=" * 70 + "\n")

    # --- 2.1 Can be instantiated with defaults ---
    try:
        model = model_cls(reco_dim=RECO_DIM, cond_dim=COND_DIM,
                          time_dim=16, hidden_dim=64, n_blocks=2)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model has zero parameters"
        results.add_pass(f"Instantiated successfully ({n_params:,} params)")
    except Exception as e:
        results.add_fail("Instantiation", e)
        if verbose:
            results.print_summary()
        return results

    model.eval()

    # --- 2.2 Forward output shape ---
    try:
        x = torch.randn(8, RECO_DIM)
        t = torch.rand(8, 1)
        c = torch.randn(8, COND_DIM)
        out = model(x, t, c)
        assert out.shape == (8, RECO_DIM), (
            f"Expected output shape (8, {RECO_DIM}), got {tuple(out.shape)}"
        )
        results.add_pass("Forward output shape matches (batch, reco_dim)")
    except Exception as e:
        results.add_fail("Forward output shape", e)
        if verbose:
            results.print_summary()
        return results

    # --- 2.3 Output depends on condition ---
    try:
        torch.manual_seed(0)
        x = torch.randn(8, RECO_DIM)
        t = torch.full((8, 1), 0.5)
        c1 = torch.zeros(8, COND_DIM)
        c2 = torch.ones(8, COND_DIM) * 3.0
        o1 = model(x, t, c1)
        o2 = model(x, t, c2)
        diff = (o1 - o2).abs().max().item()
        assert diff > 1e-4, (
            f"Output should respond to cond; max |diff| = {diff:.4e}. "
            "Did you forget to concatenate `cond` into the input?"
        )
        results.add_pass("Output depends on `cond`")
    except Exception as e:
        results.add_fail("Condition dependence", e)

    # --- 2.4 Output depends on t ---
    try:
        torch.manual_seed(0)
        x = torch.randn(8, RECO_DIM)
        c = torch.randn(8, COND_DIM)
        t1 = torch.zeros(8, 1)
        t2 = torch.ones(8, 1)
        o1 = model(x, t1, c)
        o2 = model(x, t2, c)
        diff = (o1 - o2).abs().max().item()
        assert diff > 1e-4, (
            f"Output should respond to t; max |diff| = {diff:.4e}. "
            "Did you plug the time embedding into the input?"
        )
        results.add_pass("Output depends on `t`")
    except Exception as e:
        results.add_fail("Time dependence", e)

    # --- 2.5 Output depends on x ---
    try:
        t = torch.full((8, 1), 0.3)
        c = torch.randn(8, COND_DIM)
        x1 = torch.randn(8, RECO_DIM)
        x2 = x1 + 1.0
        o1 = model(x1, t, c)
        o2 = model(x2, t, c)
        diff = (o1 - o2).abs().max().item()
        assert diff > 1e-4, (
            f"Output should respond to x; max |diff| = {diff:.4e}"
        )
        results.add_pass("Output depends on `x`")
    except Exception as e:
        results.add_fail("Input dependence", e)

    # --- 2.6 Gradients flow to every parameter ---
    try:
        model.train()
        x = torch.randn(4, RECO_DIM, requires_grad=False)
        t = torch.rand(4, 1)
        c = torch.randn(4, COND_DIM)
        out = model(x, t, c)
        loss = out.pow(2).sum()
        # zero any stale grads, then backward
        for p in model.parameters():
            p.grad = None
        loss.backward()
        no_grad = [n for n, p in model.named_parameters() if p.grad is None]
        assert not no_grad, f"No grad for parameters: {no_grad[:3]}{'...' if len(no_grad)>3 else ''}"
        zero_grad = [n for n, p in model.named_parameters()
                     if p.grad is not None and p.grad.abs().max().item() == 0.0]
        # Some parameters can legitimately have zero grad for a specific input,
        # but if *everything* is zero the residual/output wiring is broken.
        if len(zero_grad) == len(list(model.parameters())):
            raise AssertionError("All parameter grads are zero — residual or output wiring is broken.")
        results.add_pass("Gradients flow to all parameters")
    except Exception as e:
        results.add_fail("Gradient flow", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 3. conditional_fm_loss
# ----------------------------------------------------------------------------

class _ZeroModel(nn.Module):
    """Outputs zero — used as a reference model for loss sanity checks."""
    def __init__(self, reco_dim=RECO_DIM):
        super().__init__()
        # One dummy parameter so the model is "trainable" but always outputs 0.
        self.dummy = nn.Parameter(torch.zeros(1))
        self.reco_dim = reco_dim

    def forward(self, x, t, cond):
        return torch.zeros_like(x) + 0.0 * self.dummy


def test_conditional_fm_loss(conditional_fm_loss, model_cls=None, verbose=True):
    """Validate the conditional flow matching loss."""
    results = TestResults("conditional_fm_loss")

    if verbose:
        print("=" * 70)
        print("🧪 Testing conditional_fm_loss")
        print("=" * 70 + "\n")

    # --- 3.1 Returns a scalar, finite, non-negative ---
    try:
        reco, gen = _synth_batch(n=128, seed=1)
        if model_cls is None:
            loss_model = _ZeroModel()
        else:
            loss_model = model_cls(reco_dim=RECO_DIM, cond_dim=COND_DIM,
                                   time_dim=16, hidden_dim=32, n_blocks=2)
        loss = conditional_fm_loss(loss_model, reco, gen)
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

    # --- 3.2 Zero model → elementwise mean loss ≈ 2  (reco ~ N(0,I), x0 ~ N(0,I)) ---
    try:
        torch.manual_seed(7)
        reco, gen = _synth_batch(n=4000, seed=2)
        zero_model = _ZeroModel()
        loss = conditional_fm_loss(zero_model, reco, gen).item()
        # target = reco - x0; each entry ~ N(0, 2) → E[(reco - x0)^2] = 2
        # loss = mean over batch AND reco_dim → ≈ 2.0, independent of reco_dim
        expected = 2.0
        assert abs(loss - expected) < 0.3, (
            f"For a zero model with Gaussian data, expected loss ≈ {expected:.1f}, "
            f"got {loss:.2f}. Did you use `reco - x0` as the target?"
        )
        results.add_pass(f"Zero-model baseline ≈ 2.0 (got {loss:.2f})")
    except Exception as e:
        results.add_fail("Zero-model baseline", e)

    # --- 3.3 Loss is differentiable through model parameters ---
    if model_cls is not None:
        try:
            reco, gen = _synth_batch(n=64, seed=3)
            model = model_cls(reco_dim=RECO_DIM, cond_dim=COND_DIM,
                              time_dim=16, hidden_dim=32, n_blocks=2)
            loss = conditional_fm_loss(model, reco, gen)
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
# 4. train_conditional_model
# ----------------------------------------------------------------------------

def test_train_conditional_model(train_conditional_model, model_cls, verbose=True):
    """Confirm that training reduces the loss."""
    results = TestResults("train_conditional_model")

    if verbose:
        print("=" * 70)
        print("🧪 Testing train_conditional_model")
        print("=" * 70 + "\n")

    try:
        torch.manual_seed(0)
        # Small synthetic dataset — with a learnable correlation between gen and reco
        # so training actually has something to latch onto.
        n = 2000
        gen = torch.randn(n, COND_DIM)
        # reco depends linearly on gen[:, :3] + noise
        reco = gen[:, :RECO_DIM] + 0.3 * torch.randn(n, RECO_DIM)

        val_gen = torch.randn(400, COND_DIM)
        val_reco = val_gen[:, :RECO_DIM] + 0.3 * torch.randn(400, RECO_DIM)

        model = model_cls(reco_dim=RECO_DIM, cond_dim=COND_DIM,
                          time_dim=16, hidden_dim=64, n_blocks=2)

        train_losses, val_losses = train_conditional_model(
            model, reco, gen, val_reco, val_gen,
            n_epochs=25, batch_size=256, lr=1e-3,
        )
        assert len(train_losses) == 25, (
            f"Expected 25 training-loss entries, got {len(train_losses)}"
        )
        assert len(val_losses) == 25, (
            f"Expected 25 val-loss entries, got {len(val_losses)}"
        )
        early = float(np.mean(train_losses[:3]))
        late = float(np.mean(train_losses[-3:]))
        assert late < early * 0.85, (
            f"Training loss should decrease meaningfully: "
            f"early={early:.4f}, late={late:.4f} (ratio {late/early:.2f})"
        )
        results.add_pass(f"Loss decreases: {early:.3f} → {late:.3f}")
    except Exception as e:
        results.add_fail("Training reduces loss", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 5. generate_reco
# ----------------------------------------------------------------------------

def test_generate_reco(generate_reco, model_cls, verbose=True):
    """Validate the conditional sampling function."""
    results = TestResults("generate_reco")

    if verbose:
        print("=" * 70)
        print("🧪 Testing generate_reco")
        print("=" * 70 + "\n")

    torch.manual_seed(0)
    model = model_cls(reco_dim=RECO_DIM, cond_dim=COND_DIM,
                      time_dim=16, hidden_dim=32, n_blocks=2)
    model.eval()

    # --- 5.1 Output shape ---
    try:
        gen_cond = torch.randn(50, COND_DIM)
        samples = generate_reco(model, gen_cond, n_steps=10)
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

    # --- 5.2 Finite values ---
    try:
        gen_cond = torch.randn(32, COND_DIM)
        samples = generate_reco(model, gen_cond, n_steps=20)
        samples = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
        assert torch.isfinite(samples).all().item(), "Generated samples contain non-finite values"
        results.add_pass("All generated values are finite")
    except Exception as e:
        results.add_fail("Finiteness", e)

    # --- 5.3 Samples are stochastic (different seeds → different samples) ---
    try:
        gen_cond = torch.randn(64, COND_DIM)
        torch.manual_seed(11)
        s1 = generate_reco(model, gen_cond, n_steps=10)
        torch.manual_seed(12)
        s2 = generate_reco(model, gen_cond, n_steps=10)
        s1 = s1 if isinstance(s1, torch.Tensor) else torch.as_tensor(s1)
        s2 = s2 if isinstance(s2, torch.Tensor) else torch.as_tensor(s2)
        diff = (s1 - s2).abs().max().item()
        assert diff > 1e-4, (
            f"Two calls with different RNG seeds should give different samples "
            f"(max |diff| = {diff:.4e}). Did you seed torch.randn inside the function?"
        )
        results.add_pass("Samples are stochastic (start from fresh noise)")
    except Exception as e:
        results.add_fail("Stochasticity", e)

    # --- 5.4 Batch handling: running in chunks gives the same shape as one shot ---
    try:
        gen_cond = torch.randn(130, COND_DIM)
        out = generate_reco(model, gen_cond, n_steps=10)
        out = out if isinstance(out, torch.Tensor) else torch.as_tensor(out)
        assert out.shape == (130, RECO_DIM), (
            f"Batch handling: expected (130, {RECO_DIM}), got {tuple(out.shape)}"
        )
        results.add_pass("Handles arbitrary batch sizes (130 samples)")
    except Exception as e:
        results.add_fail("Batch handling", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------

def run_all_tests_nb2(
    sinusoidal_embedding=None,
    model_cls=None,
    conditional_fm_loss=None,
    train_conditional_model=None,
    generate_reco=None,
):
    """Run every NB2 test in one go and print a final summary."""
    print("\n" + "=" * 70)
    print("🚀 Running all Notebook 2 tests")
    print("=" * 70 + "\n")

    sections = []

    if sinusoidal_embedding is not None:
        sections.append(test_sinusoidal_embedding(sinusoidal_embedding, verbose=True))
    if model_cls is not None:
        sections.append(test_conditional_vector_field_model(model_cls, verbose=True))
    if conditional_fm_loss is not None:
        sections.append(test_conditional_fm_loss(conditional_fm_loss, model_cls=model_cls, verbose=True))
    if train_conditional_model is not None and model_cls is not None:
        sections.append(test_train_conditional_model(train_conditional_model, model_cls, verbose=True))
    if generate_reco is not None and model_cls is not None:
        sections.append(test_generate_reco(generate_reco, model_cls, verbose=True))

    total_pass = sum(len(s.passed) for s in sections)
    total_fail = sum(len(s.failed) for s in sections)
    total_warn = sum(len(s.warnings) for s in sections)
    total = total_pass + total_fail

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + f"  FINAL RESULT — Notebook 2".ljust(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + f"  ✅ Passed:   {total_pass}/{total}".ljust(68) + "║")
    if total_warn:
        print("║" + f"  ⚠️  Warnings: {total_warn}".ljust(68) + "║")
    if total_fail:
        print("║" + f"  ❌ Failed:   {total_fail}".ljust(68) + "║")
    else:
        print("║" + f"  🎉 All implementations validated — ready to train!".ljust(68) + "║")
    print("╚" + "═" * 68 + "╝\n")

    return total_fail == 0

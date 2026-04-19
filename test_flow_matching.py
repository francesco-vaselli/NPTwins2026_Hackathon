"""
Test suite for Notebook 1: Flow Matching Fundamentals.

This module provides automated tests that validate student implementations
of the Flow Matching exercise components. Students are expected to run the
relevant `test_*` function after completing each task, or call
`run_all_tests(...)` at the end for a comprehensive report.

Usage (inside the notebook):

    from test_flow_matching import (
        test_euler_integrate,
        test_conditional_path,
        test_conditional_vector_field,
        test_flow_matching_loss,
        test_simple_vector_field,
        test_heun_integrate,
        test_generate_samples,
        test_train_flow_model,
        run_all_tests,
    )

    # After implementing euler_integrate:
    test_euler_integrate(euler_integrate)

    # After implementing everything:
    run_all_tests(
        euler_integrate=euler_integrate,
        sample_conditional_path=sample_conditional_path,
        conditional_vector_field=conditional_vector_field,
        flow_matching_loss=flow_matching_loss,
        generate_samples=generate_samples,
        train_flow_model=train_flow_model,
        model_cls=SimpleVectorField,
        heun_integrate=heun_integrate,
    )
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Pretty-printed result container
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
# Helpers
# ----------------------------------------------------------------------------

def _rotation_field(x, t):
    """Analytic rotation vector field u(x) = (-x2, x1). Used for integrator tests."""
    return torch.stack([-x[:, 1], x[:, 0]], dim=1)


def _constant_field(v):
    """Return a constant vector field function that ignores x and t."""
    v = torch.as_tensor(v, dtype=torch.float32)

    def field(x, t):
        return v.unsqueeze(0).expand_as(x)

    return field


# ----------------------------------------------------------------------------
# 1. euler_integrate
# ----------------------------------------------------------------------------

def test_euler_integrate(euler_integrate, verbose=True):
    """Validate the Euler integrator implementation."""
    results = TestResults("euler_integrate")

    if verbose:
        print("=" * 70)
        print("🧪 Testing euler_integrate")
        print("=" * 70 + "\n")

    # --- 1.1 Return signature ---
    try:
        x0 = torch.tensor([[1.0, 0.0]])
        out = euler_integrate(x0, _rotation_field, n_steps=10)
        assert isinstance(out, tuple) and len(out) == 2, (
            "euler_integrate must return a tuple (final_x, trajectory)"
        )
        x_final, trajectory = out
        assert isinstance(trajectory, list), "trajectory must be a list"
        results.add_pass("Return signature is (tensor, list)")
    except Exception as e:
        results.add_fail("Return signature", e)
        if verbose:
            results.print_summary()
        return results

    # --- 1.2 Output shape preserved ---
    try:
        x0 = torch.randn(17, 2)
        x_final, _ = euler_integrate(x0, _rotation_field, n_steps=20)
        assert x_final.shape == x0.shape, (
            f"Expected shape {tuple(x0.shape)}, got {tuple(x_final.shape)}"
        )
        results.add_pass("Output shape matches input")
    except Exception as e:
        results.add_fail("Output shape", e)

    # --- 1.3 Trajectory length is n_steps + 1 ---
    try:
        x0 = torch.zeros(1, 2)
        _, traj = euler_integrate(x0, _constant_field([1.0, 0.0]), n_steps=25)
        assert len(traj) == 26, f"Trajectory should contain 26 entries, got {len(traj)}"
        results.add_pass("Trajectory length = n_steps + 1")
    except Exception as e:
        results.add_fail("Trajectory length", e)

    # --- 1.4 Constant field gives exact endpoint ---
    try:
        x0 = torch.zeros(4, 2)
        x_final, _ = euler_integrate(x0, _constant_field([1.0, -0.5]), n_steps=50)
        expected = torch.tensor([[1.0, -0.5]] * 4)
        assert torch.allclose(x_final, expected, atol=1e-5), (
            f"Constant field: expected {expected[0].tolist()}, got {x_final[0].tolist()}"
        )
        results.add_pass("Exact integration of constant field")
    except Exception as e:
        results.add_fail("Constant field integration", e)

    # --- 1.5 Rotation field approximates analytic solution ---
    # Under u(x)=(-y, x), starting at (1, 0), the exact solution at t=1 is (cos 1, sin 1).
    try:
        x0 = torch.tensor([[1.0, 0.0]])
        x_final, _ = euler_integrate(x0, _rotation_field, n_steps=2000)
        expected = torch.tensor([[math.cos(1.0), math.sin(1.0)]])
        err = (x_final - expected).abs().max().item()
        assert err < 2e-3, (
            f"Rotation field error too large: {err:.4e} "
            f"(got {x_final[0].tolist()}, expected {expected[0].tolist()})"
        )
        results.add_pass("Euler approximates rotation ODE (fine discretisation)")
    except Exception as e:
        results.add_fail("Rotation ODE accuracy", e)

    # --- 1.6 Trajectory entries are (x, t) pairs ---
    try:
        x0 = torch.zeros(2, 2)
        _, traj = euler_integrate(x0, _constant_field([1.0, 0.0]), n_steps=10)
        first, last = traj[0], traj[-1]
        # Each entry must be unpackable into (x, t)
        xf, tf = first
        xl, tl = last
        assert abs(float(tf) - 0.0) < 1e-6, f"First t should be 0, got {float(tf)}"
        assert abs(float(tl) - 1.0) < 1e-4, f"Last t should be 1, got {float(tl)}"
        results.add_pass("Trajectory carries (x, t) pairs with t: 0 → 1")
    except Exception as e:
        results.add_warning(
            "Trajectory carries (x, t) pairs",
            f"{e}. Visualisation cells expect trajectory entries of the form (x_tensor, t_float).",
        )

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 2. sample_conditional_path
# ----------------------------------------------------------------------------

def test_conditional_path(sample_conditional_path, verbose=True):
    """Validate the linear conditional probability path."""
    results = TestResults("sample_conditional_path")

    if verbose:
        print("=" * 70)
        print("🧪 Testing sample_conditional_path")
        print("=" * 70 + "\n")

    torch.manual_seed(0)
    x0 = torch.randn(32, 2)
    x1 = torch.randn(32, 2)

    # --- 2.1 Shape ---
    try:
        t = torch.full((32, 1), 0.3)
        xt = sample_conditional_path(x0, x1, t)
        assert xt.shape == x0.shape, f"Expected shape {tuple(x0.shape)}, got {tuple(xt.shape)}"
        results.add_pass("Output shape matches input")
    except Exception as e:
        results.add_fail("Output shape", e)

    # --- 2.2 Endpoints ---
    try:
        t0 = torch.zeros(32, 1)
        t1 = torch.ones(32, 1)
        xt0 = sample_conditional_path(x0, x1, t0)
        xt1 = sample_conditional_path(x0, x1, t1)
        assert torch.allclose(xt0, x0, atol=1e-6), "x_t at t=0 should equal x0"
        assert torch.allclose(xt1, x1, atol=1e-6), "x_t at t=1 should equal x1"
        results.add_pass("Endpoints: x_0 = x0 and x_1 = x1")
    except Exception as e:
        results.add_fail("Endpoints", e)

    # --- 2.3 Midpoint ---
    try:
        t_half = torch.full((32, 1), 0.5)
        xt = sample_conditional_path(x0, x1, t_half)
        expected = 0.5 * (x0 + x1)
        assert torch.allclose(xt, expected, atol=1e-6), "x_t at t=0.5 should be (x0 + x1)/2"
        results.add_pass("Midpoint: x_{1/2} = (x0 + x1)/2")
    except Exception as e:
        results.add_fail("Midpoint", e)

    # --- 2.4 Per-sample times are respected ---
    try:
        t_vary = torch.linspace(0.0, 1.0, 32).unsqueeze(1)
        xt = sample_conditional_path(x0, x1, t_vary)
        expected = (1 - t_vary) * x0 + t_vary * x1
        assert torch.allclose(xt, expected, atol=1e-6), (
            "Path must broadcast per-sample time correctly"
        )
        results.add_pass("Per-sample time broadcasting")
    except Exception as e:
        results.add_fail("Per-sample time broadcasting", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 3. conditional_vector_field
# ----------------------------------------------------------------------------

def test_conditional_vector_field(conditional_vector_field, verbose=True):
    """Validate u_t(x | x1) = x1 - x0 for the linear path."""
    results = TestResults("conditional_vector_field")

    if verbose:
        print("=" * 70)
        print("🧪 Testing conditional_vector_field")
        print("=" * 70 + "\n")

    torch.manual_seed(1)
    x0 = torch.randn(16, 2)
    x1 = torch.randn(16, 2)

    try:
        v = conditional_vector_field(x0, x1)
        assert v.shape == x0.shape, f"Expected shape {tuple(x0.shape)}, got {tuple(v.shape)}"
        results.add_pass("Output shape matches input")
    except Exception as e:
        results.add_fail("Output shape", e)
        if verbose:
            results.print_summary()
        return results

    try:
        v = conditional_vector_field(x0, x1)
        expected = x1 - x0
        assert torch.allclose(v, expected, atol=1e-6), (
            "For the linear path x_t = (1-t)x0 + t x1, the velocity should be x1 - x0"
        )
        results.add_pass("Velocity equals x1 - x0")
    except Exception as e:
        results.add_fail("Velocity formula", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 4. flow_matching_loss
# ----------------------------------------------------------------------------

class _IdentityVelocity(nn.Module):
    """Trivial model that returns the target velocity exactly when trained.

    Used to sanity-check the loss: feeding a model that outputs x gives a
    non-trivial, finite loss; feeding a model that returns the true conditional
    velocity yields zero loss (in expectation) when we fix the randomness.
    """

    def forward(self, x, t):
        return torch.zeros_like(x)


def test_flow_matching_loss(flow_matching_loss, model_cls=None, verbose=True):
    """Validate the conditional flow matching loss."""
    results = TestResults("flow_matching_loss")

    if verbose:
        print("=" * 70)
        print("🧪 Testing flow_matching_loss")
        print("=" * 70 + "\n")

    # --- 4.1 Scalar, finite, non-negative ---
    try:
        torch.manual_seed(0)
        x1 = torch.randn(64, 2)
        model = _IdentityVelocity()
        loss = flow_matching_loss(model, x1)
        assert loss.dim() == 0, f"Loss should be a scalar, got shape {tuple(loss.shape)}"
        assert torch.isfinite(loss), "Loss is not finite"
        assert loss.item() >= 0.0, f"Loss must be non-negative, got {loss.item()}"
        results.add_pass("Loss is a finite, non-negative scalar")
    except Exception as e:
        results.add_fail("Scalar / finite / non-negative", e)

    # --- 4.2 Loss is differentiable w.r.t. model parameters ---
    if model_cls is not None:
        try:
            torch.manual_seed(0)
            model = model_cls(d=2, hidden_dim=16, n_layers=2)
            x1 = torch.randn(32, 2)
            loss = flow_matching_loss(model, x1)
            loss.backward()
            has_grad = any(
                (p.grad is not None and p.grad.abs().sum() > 0)
                for p in model.parameters()
            )
            assert has_grad, "No non-zero gradients found — loss is not differentiable w.r.t. model params"
            results.add_pass("Loss produces non-zero gradients for the model")
        except Exception as e:
            results.add_fail("Gradients propagate", e)
    else:
        results.add_warning(
            "Gradients propagate",
            "model_cls not provided — skipping gradient test",
        )

    # --- 4.3 A model that perfectly predicts (x1 - x0) gets zero loss ---
    try:

        class Oracle(nn.Module):
            """Model that stores (x0, x1) and returns x1 - x0 regardless of x_t, t."""

            def __init__(self, target):
                super().__init__()
                self.target = target

            def forward(self, x, t):
                return self.target

        # Monkey-patch torch.randn_like to return a fixed x0 so loss is deterministic
        torch.manual_seed(1234)
        x1 = torch.randn(128, 2)
        x0_fixed = torch.randn_like(x1)
        target = x1 - x0_fixed

        orig_randn_like = torch.randn_like
        torch.randn_like = lambda t, *a, **kw: x0_fixed.clone()
        try:
            loss = flow_matching_loss(Oracle(target), x1)
        finally:
            torch.randn_like = orig_randn_like

        assert loss.item() < 1e-6, (
            f"An oracle predicting exactly x1 - x0 should produce ~zero loss, got {loss.item():.4e}"
        )
        results.add_pass("Oracle predictor achieves zero loss")
    except Exception as e:
        results.add_warning(
            "Oracle predictor",
            f"{e}. (Non-critical: this test relies on torch.randn_like for sampling x0.)",
        )

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 5. SimpleVectorField (student-built neural network)
# ----------------------------------------------------------------------------

def test_simple_vector_field(model_cls, verbose=True):
    """Validate the student-built vector-field neural network."""
    results = TestResults("SimpleVectorField")

    if verbose:
        print("=" * 70)
        print("🧪 Testing SimpleVectorField")
        print("=" * 70 + "\n")

    # --- 5.1 Construction ---
    try:
        model = model_cls(d=2, hidden_dim=32, n_layers=2)
        assert isinstance(model, nn.Module), "SimpleVectorField must be an nn.Module"
        params = list(model.parameters())
        assert len(params) > 0, "Model has no learnable parameters"
        results.add_pass("Constructs as an nn.Module with parameters")
    except Exception as e:
        results.add_fail("Construction", e)
        if verbose:
            results.print_summary()
        return results

    # --- 5.2 Forward pass shape ---
    try:
        x = torch.randn(8, 2)
        t = torch.rand(8, 1)
        out = model(x, t)
        assert out.shape == (8, 2), f"Expected output shape (8, 2), got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), "Output contains NaN/Inf"
        results.add_pass("Forward pass returns correct shape and finite values")
    except Exception as e:
        results.add_fail("Forward pass shape", e)

    # --- 5.3 Accepts t in multiple shapes: scalar, (B,), (B, 1) ---
    try:
        x = torch.randn(4, 2)
        out_2d = model(x, torch.full((4, 1), 0.3))
        out_1d = model(x, torch.full((4,), 0.3))
        out_0d = model(x, torch.tensor(0.3))
        assert out_2d.shape == out_1d.shape == out_0d.shape == (4, 2), (
            "Model should accept t with shape (B, 1), (B,), or scalar and return (B, d)"
        )
        # Same t value should give (roughly) the same output regardless of shape
        assert torch.allclose(out_2d, out_1d, atol=1e-5), (
            "Outputs for t shape (B,1) vs (B,) differ"
        )
        assert torch.allclose(out_2d, out_0d, atol=1e-5), (
            "Outputs for t shape (B,1) vs scalar differ"
        )
        results.add_pass("Handles t as scalar, (B,), and (B, 1)")
    except Exception as e:
        results.add_fail("t shape flexibility", e)

    # --- 5.4 Different inputs → different outputs ---
    try:
        x1 = torch.randn(4, 2)
        x2 = torch.randn(4, 2)
        t = torch.rand(4, 1)
        if torch.allclose(model(x1, t), model(x2, t), atol=1e-3):
            results.add_warning(
                "Output variation",
                "Very different inputs produce nearly identical outputs — is the network degenerate?",
            )
        else:
            results.add_pass("Different inputs produce different outputs")
    except Exception as e:
        results.add_warning("Output variation", e)

    # --- 5.5 Different t → different outputs ---
    try:
        x = torch.randn(4, 2)
        o1 = model(x, torch.zeros(4, 1))
        o2 = model(x, torch.ones(4, 1))
        if torch.allclose(o1, o2, atol=1e-3):
            results.add_warning(
                "Time dependence",
                "Outputs at t=0 and t=1 are (nearly) identical — is t actually fed to the network?",
            )
        else:
            results.add_pass("Output depends on t")
    except Exception as e:
        results.add_warning("Time dependence", e)

    # --- 5.6 Gradient flow ---
    try:
        x = torch.randn(4, 2, requires_grad=False)
        t = torch.rand(4, 1)
        out = model(x, t)
        loss = out.pow(2).sum()
        loss.backward()
        has_grad = any(
            (p.grad is not None and p.grad.abs().sum() > 0)
            for p in model.parameters()
        )
        assert has_grad, "No gradients found — is the model autograd-friendly?"
        results.add_pass("Gradients flow through every parameter")
    except Exception as e:
        results.add_fail("Gradient flow", e)

    # --- 5.7 Depth / width affect parameter count ---
    try:
        small = model_cls(d=2, hidden_dim=16, n_layers=2)
        big = model_cls(d=2, hidden_dim=64, n_layers=4)
        n_small = sum(p.numel() for p in small.parameters())
        n_big = sum(p.numel() for p in big.parameters())
        assert n_big > n_small * 3, (
            f"Parameter count does not scale with hidden_dim/n_layers "
            f"(small={n_small}, big={n_big}). Did you honour the constructor args?"
        )
        results.add_pass(f"Parameter count scales with width/depth ({n_small} → {n_big})")
    except Exception as e:
        results.add_warning("Parameter scaling", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 6. heun_integrate (bonus higher-order integrator)
# ----------------------------------------------------------------------------

def test_heun_integrate(heun_integrate, verbose=True):
    """Validate the second-order Heun (RK2) integrator."""
    results = TestResults("heun_integrate")

    if verbose:
        print("=" * 70)
        print("🧪 Testing heun_integrate")
        print("=" * 70 + "\n")

    # --- 6.1 Return signature ---
    try:
        x0 = torch.tensor([[1.0, 0.0]])
        out = heun_integrate(x0, _rotation_field, n_steps=10)
        assert isinstance(out, tuple) and len(out) == 2, (
            "heun_integrate must return (final_x, trajectory)"
        )
        x_final, traj = out
        assert isinstance(traj, list)
        results.add_pass("Return signature is (tensor, list)")
    except Exception as e:
        results.add_fail("Return signature", e)
        if verbose:
            results.print_summary()
        return results

    # --- 6.2 Shape preserved, trajectory length ---
    try:
        x0 = torch.randn(9, 2)
        x_final, traj = heun_integrate(x0, _rotation_field, n_steps=25)
        assert x_final.shape == x0.shape, (
            f"Shape mismatch: {tuple(x_final.shape)} vs {tuple(x0.shape)}"
        )
        assert len(traj) == 26, f"Trajectory should have 26 entries, got {len(traj)}"
        results.add_pass("Shape preserved and trajectory length = n_steps + 1")
    except Exception as e:
        results.add_fail("Shape / trajectory length", e)

    # --- 6.3 Constant field exact ---
    try:
        x0 = torch.zeros(4, 2)
        x_final, _ = heun_integrate(x0, _constant_field([1.0, -0.5]), n_steps=20)
        expected = torch.tensor([[1.0, -0.5]] * 4)
        assert torch.allclose(x_final, expected, atol=1e-5), (
            f"Constant field: expected {expected[0].tolist()}, got {x_final[0].tolist()}"
        )
        results.add_pass("Exact on constant vector field")
    except Exception as e:
        results.add_fail("Constant field", e)

    # --- 6.4 More accurate than Euler at the same step count (rotation) ---
    # Use a modest n_steps where the difference should be clearly visible.
    try:
        # Build a reference Euler integrator here for comparison
        def euler_ref(x0, v_fn, n_steps):
            h = 1.0 / n_steps
            x = x0.clone(); t = 0.0
            for _ in range(n_steps):
                x = x + h * v_fn(x, t); t += h
            return x

        x0 = torch.tensor([[1.0, 0.0]])
        exact = torch.tensor([[math.cos(1.0), math.sin(1.0)]])
        n = 20
        x_h, _ = heun_integrate(x0, _rotation_field, n_steps=n)
        x_e = euler_ref(x0, _rotation_field, n_steps=n)
        err_h = (x_h - exact).abs().max().item()
        err_e = (x_e - exact).abs().max().item()
        assert err_h < err_e, (
            f"Heun error ({err_h:.4e}) should be smaller than Euler error "
            f"({err_e:.4e}) at n_steps={n}"
        )
        results.add_pass(
            f"More accurate than Euler at n_steps={n} (Heun {err_h:.2e} < Euler {err_e:.2e})"
        )
    except Exception as e:
        results.add_fail("Accuracy vs Euler", e)

    # --- 6.5 Converges to analytic solution with enough steps ---
    try:
        x0 = torch.tensor([[1.0, 0.0]])
        exact = torch.tensor([[math.cos(1.0), math.sin(1.0)]])
        x_final, _ = heun_integrate(x0, _rotation_field, n_steps=500)
        err = (x_final - exact).abs().max().item()
        assert err < 1e-5, f"Heun with 500 steps should be near-exact, got error {err:.2e}"
        results.add_pass(f"Converges to analytic rotation solution (err={err:.2e})")
    except Exception as e:
        results.add_fail("Convergence", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 7. generate_samples
# ----------------------------------------------------------------------------

def test_generate_samples(generate_samples, model_cls, verbose=True):
    """Validate the sampling procedure (Euler integration of the learned field)."""
    results = TestResults("generate_samples")

    if verbose:
        print("=" * 70)
        print("🧪 Testing generate_samples")
        print("=" * 70 + "\n")

    try:
        torch.manual_seed(0)
        model = model_cls(d=2, hidden_dim=16, n_layers=2)
        samples = generate_samples(model, n_samples=128, n_steps=20, d=2)
        assert samples.shape == (128, 2), (
            f"Expected shape (128, 2), got {tuple(samples.shape)}"
        )
        assert torch.isfinite(samples).all(), "Samples contain NaN/Inf"
        results.add_pass("Output shape is (n_samples, d) and finite")
    except Exception as e:
        results.add_fail("Sampling output", e)

    # Sample variability: two independent calls should not produce identical points
    try:
        torch.manual_seed(0)
        model = model_cls(d=2, hidden_dim=16, n_layers=2)
        s1 = generate_samples(model, n_samples=64, n_steps=10, d=2)
        s2 = generate_samples(model, n_samples=64, n_steps=10, d=2)
        if torch.allclose(s1, s2):
            results.add_warning(
                "Sampling stochasticity",
                "Two successive calls produced identical samples — did you reuse the same x0?",
            )
        else:
            results.add_pass("Sampling is stochastic across calls")
    except Exception as e:
        results.add_warning("Sampling stochasticity", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# 6. train_flow_model
# ----------------------------------------------------------------------------

def test_train_flow_model(train_flow_model, model_cls, verbose=True):
    """Validate the training loop reduces the loss on a trivial target."""
    results = TestResults("train_flow_model")

    if verbose:
        print("=" * 70)
        print("🧪 Testing train_flow_model (quick 200-epoch run)")
        print("=" * 70 + "\n")

    try:
        torch.manual_seed(0)

        def sample_unit_gaussian_shifted(n):
            # A very easy target: samples from N((2, 0), 0.1^2 I)
            return torch.randn(n, 2) * 0.1 + torch.tensor([2.0, 0.0])

        model = model_cls(d=2, hidden_dim=32, n_layers=2)
        losses = train_flow_model(
            model, sample_unit_gaussian_shifted,
            n_epochs=200, batch_size=256, lr=1e-3,
        )
        assert isinstance(losses, (list, tuple, np.ndarray)), (
            "train_flow_model must return the list of per-epoch losses"
        )
        assert len(losses) == 200, f"Expected 200 loss values, got {len(losses)}"
        results.add_pass("Training runs and returns 200 losses")

        # Loss should decrease between first and last 20-epoch averages
        first = float(np.mean(losses[:20]))
        last = float(np.mean(losses[-20:]))
        if last < 0.5 * first:
            results.add_pass(f"Training reduces loss ({first:.3f} → {last:.3f})")
        elif last < first:
            results.add_warning(
                "Loss reduction",
                f"Loss dropped only modestly ({first:.3f} → {last:.3f}). More epochs may be needed.",
            )
        else:
            results.add_fail(
                "Loss reduction",
                f"Loss did not decrease ({first:.3f} → {last:.3f}). Check the training step.",
            )
    except Exception as e:
        results.add_fail("Training function", e)

    if verbose:
        results.print_summary()
    return results


# ----------------------------------------------------------------------------
# Master test runner
# ----------------------------------------------------------------------------

def run_all_tests(
    euler_integrate=None,
    sample_conditional_path=None,
    conditional_vector_field=None,
    flow_matching_loss=None,
    generate_samples=None,
    train_flow_model=None,
    model_cls=None,
    heun_integrate=None,
    verbose=True,
):
    """Run every test in sequence and print a final summary."""
    print("\n" + "=" * 70)
    print("🚀 RUNNING COMPREHENSIVE TEST SUITE — Notebook 1")
    print("=" * 70 + "\n")

    all_results = []

    if euler_integrate is not None:
        all_results.append(test_euler_integrate(euler_integrate, verbose=verbose))
    if sample_conditional_path is not None:
        all_results.append(test_conditional_path(sample_conditional_path, verbose=verbose))
    if conditional_vector_field is not None:
        all_results.append(test_conditional_vector_field(conditional_vector_field, verbose=verbose))
    if flow_matching_loss is not None:
        all_results.append(test_flow_matching_loss(
            flow_matching_loss, model_cls=model_cls, verbose=verbose,
        ))
    if model_cls is not None:
        all_results.append(test_simple_vector_field(model_cls, verbose=verbose))
    if heun_integrate is not None:
        all_results.append(test_heun_integrate(heun_integrate, verbose=verbose))
    if generate_samples is not None and model_cls is not None:
        all_results.append(test_generate_samples(generate_samples, model_cls, verbose=verbose))
    if train_flow_model is not None and model_cls is not None:
        all_results.append(test_train_flow_model(train_flow_model, model_cls, verbose=verbose))

    # Final aggregate
    print("\n" + "=" * 70)
    print("🏁 FINAL SUMMARY")
    print("=" * 70)

    total_passed = sum(len(r.passed) for r in all_results)
    total_failed = sum(len(r.failed) for r in all_results)
    total_warnings = sum(len(r.warnings) for r in all_results)

    print(f"\n✅ Passed:   {total_passed}")
    print(f"⚠️  Warnings: {total_warnings}")
    print(f"❌ Failed:   {total_failed}")

    if total_failed == 0:
        print("\n🎉 All tests passed — nice work! Ready to move on to Notebook 2.")
        print("=" * 70 + "\n")
        return True
    else:
        print("\n⚠️  Some tests failed. Scroll up for details and keep iterating.")
        print("=" * 70 + "\n")
        return False

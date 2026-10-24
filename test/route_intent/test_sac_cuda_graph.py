"""SAC ``cuda_graph_update``: the captured update must be the eager update.

The graphed core is verified against the eager core on identical inputs
with a deterministic actor (mean action, so no sampling noise separates
the two), then reuse / invalidation / fallback rules are checked.
"""
import copy

import numpy as np
import pytest
import torch

from opencda_marl.builtin.algorithms.sac import SACAlgorithm

_MIN_FREE_BYTES = 1 << 31          # 2 GiB of headroom for these tiny networks


def _gpu_headroom() -> bool:
    """Enough free VRAM to run a small capture.

    On a shared GPU another process can leave only a few hundred MB free,
    and then every test here fails with an allocation error that says
    nothing about CUDA graphs.
    """
    if not torch.cuda.is_available():
        return False
    try:
        free, _total = torch.cuda.mem_get_info()
    except Exception:                                   # noqa: BLE001
        return True                                     # can't tell -> try
    return free >= _MIN_FREE_BYTES


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
needs_headroom = pytest.mark.skipif(
    torch.cuda.is_available() and not _gpu_headroom(),
    reason="less than 2 GiB free on the GPU (shared GPU); not a CUDA-graph failure")


def _cfg(**overrides):
    cfg = {"warmup_steps": 0, "memory_size": 512, "batch_size": 32,
           "actor_hidden_dims": [32, 32], "critic_hidden_dims": [32, 32],
           "lstm_hidden_size": 8, "tensorboard": {"enabled": False},
           "auto_entropy_tuning": True}
    cfg.update(overrides)
    return cfg


def _obs(rng, n):
    return {f"v{i}": rng.normal(size=5).astype(np.float32) for i in range(n)}


def _fill(algo, rng, n=96):
    for i in range(n):
        k = 2 + (i % 4)
        algo.store_transition(_obs(rng, k), "v0", 30.0, float(rng.normal()), _obs(rng, k), i % 9 == 0)


def _deterministic_actor(algo):
    """Replace sampling by the mean action so eager and graphed updates see
    identical actions; log-probs come from the same distribution."""
    actor = algo.actor
    real_forward = actor.forward

    def sample_with_aux(state):
        mean, log_std, aux = real_forward(state, return_aux=True)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std, validate_args=False)
        x_t = mean
        y_t = torch.tanh(x_t)
        action = y_t * actor.action_scale + actor.action_bias
        log_prob = dist.log_prob(x_t) - torch.log(actor.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, action, aux

    actor.sample_with_aux = sample_with_aux


def _params(algo):
    return {"actor": [p.detach().clone() for p in algo.actor.parameters()],
            "critic": [p.detach().clone() for p in algo.critic.parameters()],
            "target": [p.detach().clone() for p in algo.critic_target.parameters()],
            "log_alpha": algo.log_alpha.detach().clone()}


@cuda_only
@needs_headroom
def test_graphed_update_matches_eager_update():
    """After capture, every replayed update equals the eager update an
    identically-initialised learner performs on the same batch."""
    torch.manual_seed(0)
    graphed = SACAlgorithm(_cfg(cuda_graph_update=True), 5, 1)
    assert graphed.cuda_graph_update is True
    _deterministic_actor(graphed)
    _fill(graphed, np.random.default_rng(1))
    np.random.seed(10)
    graphed.update()                    # 3 warm-up updates + capture + first replay
    assert graphed._update_graph is not None

    # Eager twin cloned from the graphed learner's post-capture state.
    twin = SACAlgorithm(_cfg(cuda_graph_update=False), 5, 1)
    twin.actor.load_state_dict(graphed.actor.state_dict())
    twin.critic.load_state_dict(graphed.critic.state_dict())
    twin.critic_target.load_state_dict(graphed.critic_target.state_dict())
    twin.lstm_encoder.load_state_dict(graphed.lstm_encoder.state_dict())
    with torch.no_grad():
        twin.log_alpha.copy_(graphed.log_alpha)
    twin.alpha = graphed.alpha
    # deepcopy: load_state_dict keeps the SAME moment tensors when device and
    # dtype already match, which would make the twin mutate the graphed
    # learner's Adam state.
    twin.actor_optimizer.load_state_dict(copy.deepcopy(graphed.actor_optimizer.state_dict()))
    twin.critic_optimizer.load_state_dict(copy.deepcopy(graphed.critic_optimizer.state_dict()))
    twin.alpha_optimizer.load_state_dict(copy.deepcopy(graphed.alpha_optimizer.state_dict()))
    _deterministic_actor(twin)
    twin.memory = graphed.memory        # identical batches for identical np seeds

    graph_obj = graphed._update_graph
    for seed in (20, 21, 22, 23):
        np.random.seed(seed)
        m_twin = dict(twin.update())
        np.random.seed(seed)
        m_graph = dict(graphed.update())
        for key in ("critic_loss", "actor_loss", "alpha", "q1_mean", "entropy"):
            assert m_twin[key] == pytest.approx(m_graph[key], rel=2e-3, abs=2e-4), key
    assert graphed._update_graph is graph_obj, "graph reused, not recaptured"
    p_twin, p_graph = _params(twin), _params(graphed)
    for group in ("actor", "critic", "target"):
        for a, b in zip(p_twin[group], p_graph[group]):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(p_twin["log_alpha"], p_graph["log_alpha"], rtol=1e-4, atol=1e-5)


@cuda_only
@needs_headroom
def test_graph_is_invalidated_on_load_and_reused_otherwise(tmp_path):
    algo = SACAlgorithm(_cfg(cuda_graph_update=True), 5, 1)
    _fill(algo, np.random.default_rng(2))
    algo.update()
    first = algo._update_graph
    assert first is not None
    algo.update()
    assert algo._update_graph is first
    algo.save(str(tmp_path / "c.pth"))
    algo.load(str(tmp_path / "c.pth"))
    assert algo._update_graph is None, "load replaced optimizer state -> recapture"
    algo.update()
    assert algo._update_graph is not None and algo._update_graph is not first
    assert all(np.isfinite(v) for v in algo.training_metrics.values() if isinstance(v, float))


def test_graph_mode_needs_cuda_and_a_frozen_encoder():
    algo = SACAlgorithm(_cfg(cuda_graph_update=True, train_context_encoder=True), 5, 1)
    assert algo.cuda_graph_update is False      # trainable encoder -> eager
    _fill(algo, np.random.default_rng(3))
    assert "critic_loss" in algo.update()
    assert algo._update_graph is None


@cuda_only
@needs_headroom
def test_capturable_checkpoint_loads_into_eager_learner_and_back(tmp_path):
    a = SACAlgorithm(_cfg(cuda_graph_update=True), 5, 1)
    _fill(a, np.random.default_rng(4))
    a.update()
    a.save(str(tmp_path / "g.pth"))
    b = SACAlgorithm(_cfg(cuda_graph_update=False), 5, 1)
    b.load(str(tmp_path / "g.pth"))                 # must not raise; networks equal
    for pa, pb in zip(a.actor.parameters(), b.actor.parameters()):
        assert torch.equal(pa.detach(), pb.detach())
    _fill(b, np.random.default_rng(5))
    assert "critic_loss" in b.update()
    b.save(str(tmp_path / "e.pth"))
    c = SACAlgorithm(_cfg(cuda_graph_update=True), 5, 1)
    c.load(str(tmp_path / "e.pth"))
    _fill(c, np.random.default_rng(6))
    assert "critic_loss" in c.update() and c._update_graph is not None


@cuda_only
@needs_headroom
def test_capture_failure_is_raised_not_hidden(tmp_path):
    """A live autograd graph through the actor built outside update() (legacy
    stream) makes capture fail; torch then cannot restore the CUDA RNG state
    in-process, so update() must raise CudaGraphCaptureError instead of
    returning stale metrics forever. Run in a subprocess: the failure poisons
    the CUDA state of the process by design."""
    import subprocess
    import sys
    script = r'''
import torch, numpy as np
from opencda_marl.builtin.algorithms.sac import SACAlgorithm, CudaGraphCaptureError
cfg = {"warmup_steps": 0, "memory_size": 64, "batch_size": 8, "actor_hidden_dims": [16, 16],
       "critic_hidden_dims": [16, 16], "lstm_hidden_size": 8, "tensorboard": {"enabled": False},
       "metrics_store": {"enabled": False}, "cuda_graph_update": True}
algo = SACAlgorithm(cfg, 5, 1)
rng = np.random.default_rng(0)
for i in range(24):
    o = {f"v{j}": rng.normal(size=5).astype(np.float32) for j in range(3)}
    algo.store_transition(o, "v0", 30.0, 1.0, o, False)
live = algo.actor.sample_with_aux(torch.randn(4, 13, device="cuda"))   # legacy-stream graph, kept alive
try:
    algo.update()
    print("NO-RAISE")
except CudaGraphCaptureError as e:
    print("RAISED", "cuda_graph_update: false" in str(e))
'''
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, env=env)
    if not result.stdout.strip() and "out of memory" in result.stderr.lower():
        pytest.skip("subprocess could not allocate on the GPU (shared GPU); "
                    "the capture path was never reached")
    assert "RAISED True" in result.stdout, (result.stdout[-800:], result.stderr[-1500:])

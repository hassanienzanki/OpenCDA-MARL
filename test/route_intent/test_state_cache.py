"""MARLManager._extract_states identity cache: once per unique dict."""
from opencda_marl._kernel.policy.manager import MARLManager


def _manager():
    return MARLManager(
        {
            "algorithm": "masac",
            "action_space": {"mode": "speed_only"},
            "reward_profile": "cooperative_intersection_v1",
            "state_dim": 3,
            "action_dim": 1,
            "masac": {
                "features": {"rel_x": 1, "rel_y": 1, "speed": 1},
                "warmup_steps": 0,
                "memory_size": 16,
                "batch_size": 4,
                "actor_hidden_dims": [8, 8],
                "critic_hidden_dims": [8, 8],
                "lstm_hidden_size": 4,
                "tensorboard": {"enabled": False},
            },
        },
        "masac",
    )


def _obs(speed):
    return {
        "veh_a": {"relative_position_to_intersection": {"x": 1.0, "y": 2.0}, "speed": speed},
    }


def test_same_dict_extracted_once(monkeypatch):
    mgr = _manager()
    calls = {"n": 0}
    real = mgr.observation_extractor.extract

    def counting(observations):
        calls["n"] += 1
        return real(observations)

    monkeypatch.setattr(mgr.observation_extractor, "extract", counting)

    obs = _obs(10.0)
    s1 = mgr._extract_states(obs)
    s2 = mgr._extract_states(obs)   # compute_actions -> update(obs, ...) pattern
    s3 = mgr._extract_states(obs)
    assert calls["n"] == 1
    assert s1 is s2 is s3


def test_distinct_dicts_are_extracted_separately(monkeypatch):
    mgr = _manager()
    calls = {"n": 0}
    real = mgr.observation_extractor.extract

    def counting(observations):
        calls["n"] += 1
        return real(observations)

    monkeypatch.setattr(mgr.observation_extractor, "extract", counting)

    a, b = _obs(10.0), _obs(20.0)   # equal structure, different objects/values
    sa = mgr._extract_states(a)
    sb = mgr._extract_states(b)
    assert calls["n"] == 2
    assert sa["veh_a"]["ego_state"][2] != sb["veh_a"]["ego_state"][2]


def test_cache_holds_current_and_previous_only(monkeypatch):
    mgr = _manager()
    dicts = [_obs(float(i)) for i in range(4)]
    for d in dicts:
        mgr._extract_states(d)
    assert len(mgr._state_cache) == 2
    # The two most recent dicts are cached; older ones were evicted.
    assert id(dicts[-1]) in mgr._state_cache and id(dicts[-2]) in mgr._state_cache
    assert id(dicts[0]) not in mgr._state_cache


def test_id_reuse_cannot_alias_a_new_dict():
    """A strong ref is kept to the keyed dict, so a new dict cannot inherit a
    recycled id() and read a stale result."""
    mgr = _manager()
    first = _obs(1.0)
    s_first = mgr._extract_states(first)
    key = id(first)
    # Even if a different dict somehow had the same id, the identity check
    # (hit[0] is observations) must reject it.
    fake_hit = (first, s_first)
    assert mgr._state_cache[key][0] is first
    other = _obs(99.0)
    mgr._state_cache[id(other)] = fake_hit  # simulate id collision
    s_other = mgr._extract_states(other)
    assert s_other is not s_first
    assert s_other["veh_a"]["ego_state"][2] == 99.0


def test_reset_episode_clears_cache():
    mgr = _manager()
    mgr._extract_states(_obs(1.0))
    assert mgr._state_cache
    mgr.reset_episode()
    assert mgr._state_cache == {}

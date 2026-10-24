'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 18:29:51
FilePath     : /OpenCDA-MARL/opencda_marl/builtin/algorithms/base_algorithm.py
Description  : Base class for all reinforcement learning algorithms
Copyright (c) 2025-2026 Lihao Guo and the Radar Lab, University of Arizona
SPDX-License-Identifier: Apache-2.0
'''
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from .training_logger import TENSORBOARD_AVAILABLE, TrainingLogger  # noqa: F401  (re-export)


def _cap_cpu_threads(setting) -> int:
    """Bound torch's intra-op thread pool for the replica's small batches.

    With small acting batches the default pool is slower than a few
    threads -- it can even lose to the GPU -- because a tiny op pays to
    wake every thread. ``acting_threads`` sets the count; ``auto`` (the
    default) caps it at 8. Training on the GPU does not need the CPU
    pool, so the cap is process-wide.
    """
    import os

    if str(setting).lower() == "auto":
        threads = max(1, min(8, os.cpu_count() or 8))
    else:
        threads = max(1, int(setting))
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
        logger.info(f"acting replica: torch intra-op threads capped at {threads}")
    return threads


class ActingReplica:
    """Copies of the modules that act, on another device, refreshed from
    the training modules after every weight change.

    With few agents per step the GPU is launch-bound: a full acting
    step (context encode, actor, critic, transfers) is cheaper on the CPU
    than on CUDA at these small shapes, while a refresh of all acting
    weights through one pinned flat copy costs far less than the step.
    The PPO families refresh once per update, so acting on the replica is
    a net win; the off-policy learners refresh once per step and still
    come out ahead.

    The replica is refreshed lazily -- marked dirty by the learner after
    an optimiser step or a checkpoint load, copied on the next acting
    call -- so the values and log-probabilities stashed while acting are
    always those of the current weights. CPU and GPU arithmetic differ in
    the last bits, so a run's actions are not bit-identical to one that
    acted on the GPU; PPO's ratio at the first epoch is then 1 within
    float noise rather than exactly 1.
    """

    def __init__(self, modules: Dict[str, "nn.Module"], normalizers: Dict[str, Any], device) -> None:
        import copy

        self.device = torch.device(device)
        self._live = dict(modules)
        self.modules: Dict[str, nn.Module] = {}
        for name, module in self._live.items():
            replica = copy.deepcopy(module).to(self.device).eval()
            for parameter in replica.parameters():
                parameter.requires_grad_(False)
            self.modules[name] = replica
        self._live_normalizers = {n: v for n, v in normalizers.items() if v is not None}
        self.normalizers = {n: v.mirror(self.device) for n, v in self._live_normalizers.items()}
        self._flat = None
        parameters = self._live_parameters()
        if parameters and len({p.dtype for p in parameters}) == 1:
            pin = torch.cuda.is_available() and self.device.type == "cpu"
            self._flat = torch.empty(sum(p.numel() for p in parameters), dtype=parameters[0].dtype,
                                     device=self.device, pin_memory=pin)
        self.dirty = True
        self.refreshes = 0

    def _live_parameters(self):
        return [p for m in self._live.values() for p in m.parameters()]

    def _replica_parameters(self):
        return [p for m in self.modules.values() for p in m.parameters()]

    @torch.no_grad()
    def refresh(self) -> None:
        """Copy the training modules' weights, buffers and normaliser
        statistics into the replica: one flat transfer for the weights."""
        from torch.nn.utils import parameters_to_vector, vector_to_parameters

        live = self._live_parameters()
        if live:
            if self._flat is not None:
                self._flat.copy_(parameters_to_vector(live))
                vector_to_parameters(self._flat, self._replica_parameters())
            else:
                for target, source in zip(self._replica_parameters(), live):
                    target.copy_(source)
        for name, module in self._live.items():
            for target, source in zip(self.modules[name].buffers(), module.buffers()):
                target.copy_(source)
        for name, normalizer in self._live_normalizers.items():
            self.normalizers[name].copy_from(normalizer)
        self.dirty = False
        self.refreshes += 1


class BaseAlgorithm(TrainingLogger, ABC):
    """
    Abstract base class for all reinforcement learning algorithms.

    This class defines the interface that all RL algorithms must implement
    for use with the MARLAgent.
    """

    supports_vector_rewards = False
    #: Single-agent policies only: ``True`` when ``select_action`` returns an
    #: index into the policy's ``speed_actions`` list rather than the target
    #: speed. The manager resolves the index to a speed and hands the index
    #: back to ``store_transition``. Declared, never inferred: a policy can
    #: own a speed table and still return speeds (DQN does).
    returns_action_index = False

    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        """
        Initialize the algorithm.

        Args:
            config: Algorithm-specific configuration
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Common parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.discount_factor = config.get('discount_factor', 0.95)

        # Training state
        self.training_step = 0
        self.episode_count = 0

        # TensorBoard logging + convergence tracking (TrainingLogger mixin)
        self._init_training_logger(config)

        logger.success(
            f"Initialized {self.__class__.__name__} with state_dim={state_dim}, action_dim={action_dim}")

    def _build_context_encoder(self, input_size: int):
        """The neighbour-context encoder, resolved as a registrable kind.

        Every learner encodes the same ego-first, nearest-first sequence
        of neighbouring observations into one context vector; *how* is a
        configuration axis, not a rewrite. ``<algo>.context_encoder`` names the
        registered ``context_encoder`` component (default ``lstm``, the
        historical :class:`LSTMEncoder`). A replacement implements
        ``forward(sequences)`` and ``forward_padded(sequences, lengths)``
        with the same shapes; the attribute keeps its historical name so
        checkpoints and the update paths are untouched.
        """
        from opencda_marl.api.components import resolve_component

        key = str(self.config.get("context_encoder", "lstm") or "lstm")
        factory = resolve_component("context_encoder", key)
        return factory(
            input_size=int(input_size),
            hidden_size=int(self.lstm_hidden_size),
            num_layers=int(self.lstm_num_layers),
        ).to(self.device)

    @abstractmethod
    def select_action(self, multi_agent_obs: Dict[str, np.ndarray],
                      ego_agent_id: Any, training: bool = True) -> Any:
        """
        Select the ego agent's action from the multi-agent observation.

        Args:
            multi_agent_obs: ``{agent_id: state_vector}`` for every live agent
            ego_agent_id: Which agent to act for
            training: Whether in training mode (affects exploration)

        Returns:
            Target speed in km/h, or ``None`` to defer to the rule-based agent
            (warmup). See ``opencda_marl/api/policy.py`` for the full
            contract.
        """
        pass

    # ------------------------------------------------------------------ #
    # Batched action selection (Template Method)
    # ------------------------------------------------------------------ #
    # Subclasses opt in by setting ``supports_batched_actions = True`` and
    # implementing :meth:`_batched_forward`. The manager
    # (``opencda_marl/_kernel/policy/manager.py``) checks the flag and calls
    # :meth:`select_actions` once per simulation step instead of N
    # per-agent ``select_action`` calls, collapsing N actor/critic launches
    # into one GPU forward pass. Non-batched algorithms keep the
    # ``supports_batched_actions = False`` default and pay zero overhead.

    #: When ``True``, the manager dispatches via :meth:`select_actions`;
    #: when ``False`` (default), the manager falls back to per-agent
    #: :meth:`select_action`. Concrete algorithms flip this on the class
    #: line right next to their ``_batched_forward`` implementation so the
    #: pairing is visible at a glance.
    supports_batched_actions: bool = False

    def select_actions(
        self,
        multi_agent_obs: Dict[str, np.ndarray],
        ego_agent_ids: List[Any],
        training: bool = True,
    ) -> Dict[str, float]:
        """Batched action dispatcher (Template Method).

        Orchestrates the standard pipeline shared by PCMAPPO, MAPPO,
        SAC, and TD3:

        1. Empty-guard on ``ego_agent_ids``.
        2. Warmup-phase short-circuit (random actions, no forward pass).
        3. Stack inputs into ``[B, state_dim]`` ego tensor + LSTM context.
        4. One algorithm-specific batched actor/critic forward pass.
        5. Unpack per-agent action + bookkeeping, return ``{id: speed}``.
        6. Fallback to per-agent :meth:`select_action` on any exception
           so a batched-path bug never derails a training step.

        Subclasses customize via:

        * :meth:`_batched_forward` (required): the actual forward pass.
        * :meth:`_warmup_action_batch` (optional): override only if the
          algorithm needs to stash per-agent placeholders (PPO families
          need ``last_log_probs`` / ``last_values`` even during warmup
          so transition storage doesn't see ``KeyError``).
        * :meth:`_prepare_batch_inputs` (optional): override only if the
          algorithm's LSTM encoder takes non-standard inputs.

        Algorithms that do NOT opt in keep ``supports_batched_actions =
        False`` (the default); the manager then takes the per-agent
        :meth:`select_action` path and this method is never called.
        """
        if not ego_agent_ids:
            return {}

        if self._in_action_warmup(training):
            return self._warmup_action_batch(ego_agent_ids)

        try:
            ego_states, context, resolved_ids = self._prepare_batch_inputs(
                multi_agent_obs, ego_agent_ids,
            )
            if not resolved_ids:
                return {}
            return self._batched_forward(
                ego_states, context, resolved_ids, training=training,
            )
        except Exception as exc:
            logger.error(
                f"Batched action error in {type(self).__name__}: {exc}"
            )
            import traceback
            traceback.print_exc()
            return {
                str(aid): self._fallback_select_action(
                    multi_agent_obs, str(aid), training=training,
                )
                for aid in ego_agent_ids
            }

    def _in_action_warmup(self, training: bool) -> bool:
        """Return True iff the algorithm should emit random warmup actions.

        Default rule (works for MAPPO/PCMAPPO/PCMASAC): training mode,
        positive ``warmup_steps``, and ``steps_collected`` / replay-buffer
        size still under the warmup threshold. SAC tracks progress via
        ``len(self.memory)`` instead of ``steps_collected`` -- we probe
        for either, so the default works for both families.
        """
        warmup = getattr(self, "warmup_steps", 0)
        if warmup <= 0 or not training:
            return False
        if getattr(self, "_pretrained", False):
            return False
        progress = getattr(self, "steps_collected", None)
        if progress is None:
            buf = getattr(self, "memory", None)
            try:
                progress = len(buf) if buf is not None else 0
            except TypeError:
                progress = 0
        return progress < warmup

    def _warmup_action_batch(
        self, ego_agent_ids: List[Any],
    ) -> Dict[str, float]:
        """Default warmup batch: uniform random in ``[min_action, max_action]``.

        Algorithms that need to stash per-agent state during warmup
        (PPO ``last_log_probs`` / ``last_values`` so the transition
        storage path doesn't dereference missing keys) override this
        method and populate their own bookkeeping dicts.
        """
        return {
            str(aid): float(np.random.uniform(self.min_action, self.max_action))
            for aid in ego_agent_ids
        }

    # Number of recently seen observation tables whose sorted form is kept.
    # A step touches at most the current table, the next table and one
    # terminal-agent variant, so three slots cover it without thrashing.
    _CONTEXT_TABLE_CACHE_SLOTS = 3

    def _sorted_context_table(
        self, multi_agent_obs: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
        """Stack an observation table and sort it by distance to the
        intersection, memoised on the table object.

        Returns ``(row_of, table, order)``: ``row_of[agent_id]`` is the row
        of that agent in ``table`` (``[N, state_dim]`` float32, in dict
        order) and ``order`` the row indices sorted by
        ``sqrt(rel_x^2 + rel_y^2)`` of each row (float64, stable), which is
        the neighbour ordering every LSTM context in the family uses.

        The extractor hands every ego of a step the *same* table object, and
        the strategy passes that object once per ego to ``store_transition``
        — so the stack + sort is done once per table instead of once per
        (ego, table). The cache holds strong references to the last few
        tables (they are never mutated after extraction), so an id can't be
        recycled while its entry is alive.
        """
        cache = self.__dict__.setdefault("_context_table_cache", [])
        for i, (table_obj, entry) in enumerate(cache):
            if table_obj is multi_agent_obs and len(entry[0]) == len(multi_agent_obs):
                if i != len(cache) - 1:            # keep most-recent last
                    cache.append(cache.pop(i))
                return entry

        ids = list(multi_agent_obs.keys())
        table = np.asarray(list(multi_agent_obs.values()), dtype=np.float32)
        row_of: Dict[Any, int] = {}
        for i, agent_id in enumerate(ids):        # exact key, plus str alias
            row_of.setdefault(agent_id, i)
            row_of.setdefault(str(agent_id), i)
        rel_x = table[:, 0].astype(np.float64)
        rel_y = table[:, 1].astype(np.float64)
        order = np.argsort(np.sqrt(rel_x * rel_x + rel_y * rel_y), kind="stable")
        entry = (row_of, table, order)

        cache.append((multi_agent_obs, entry))
        if len(cache) > self._CONTEXT_TABLE_CACHE_SLOTS:
            del cache[0]
        return entry

    def _context_sequence(
        self, multi_agent_obs: Dict[str, np.ndarray], ego_agent_id: Any,
    ) -> Tuple[np.ndarray, int]:
        """Raw neighbour sequence for one ego: ``([n, state_dim], n)``.

        Row 0 is the ego observation, rows 1..n-1 the other agents sorted by
        distance to the intersection (nearest first) — the same order and
        the same optional ``max_context_agents`` truncation as the acting
        path (:meth:`_prepare_batch_inputs`), so encoding the stored rows
        reproduces the context the actor saw. The table is stacked and
        sorted once per table object (:meth:`_sorted_context_table`), so
        this is O(n) per ego. Learners store this sequence and encode it in
        one packed LSTM call at update time (:meth:`_encode_context_batch`).
        """
        if ego_agent_id not in multi_agent_obs:
            ego_agent_id = next(iter(multi_agent_obs.keys()))
        row_of, table, order = self._sorted_context_table(multi_agent_obs)
        row = row_of[ego_agent_id]
        others = order[order != row]
        cap = getattr(self, "max_context_agents", None)
        if cap is not None:
            others = others[:int(cap) - 1]
        seq = np.empty((1 + len(others), table.shape[1]), dtype=np.float32)
        seq[0] = table[row]
        seq[1:] = table[others]
        return seq, seq.shape[0]

    def _encode_context_batch(self, seqs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Batched context for zero-padded ``[B, W, state_dim]`` sequences.

        Rows with length 1 (ego only) get a zero context, matching the
        per-agent acting path; the rest are packed-encoded. ``lengths`` is
        a CPU int64 tensor. The whole batch is packed and encoded, then
        ego-only rows are zeroed by a mask — no boolean gather/scatter and
        no ``.any()`` device sync on the hot path. Requires
        ``self.lstm_encoder`` (the registered context encoder; see
        :meth:`_build_context_encoder`).
        """
        encoded = self.lstm_encoder.forward_padded(seqs, lengths)
        keep = (lengths > 1).to(device=encoded.device, dtype=encoded.dtype).unsqueeze(1)
        return encoded * keep

    # Replay-buffer layout shared by the off-policy learners (SAC, TD3):
    #   0 ego_state [state_dim]      1 seq [W, state_dim]   2 seq_len
    #   3 action                     4 reward
    #   5 next_ego_state [state_dim] 6 next_seq [W, state_dim] 7 next_seq_len
    #   8 done
    # W = self._seq_width grows with the widest sequence seen.
    _SEQUENCE_FIELDS = (1, 6)

    def _sync_seq_width(self) -> None:
        """Adopt the attached buffer's actual sequence width.

        The buffer outlives the learner object across CARLA world reloads
        (TrainingStateSnapshot hands the same buffer to a freshly built
        algorithm whose ``_seq_width`` starts at 1); padding or "widening"
        against a stale width would then reject every push. Call before
        padding a transition.
        """
        memory = getattr(self, "memory", None)
        field_shape = getattr(memory, "field_shape", None)
        if field_shape is None:
            return
        shape = field_shape(self._SEQUENCE_FIELDS[0])
        if shape:
            self._seq_width = int(shape[0])

    def _pad_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Zero-pad ``[n, state_dim]`` to the buffer's current width
        (``self._seq_width``)."""
        n = seq.shape[0]
        if n == self._seq_width:
            return seq
        padded = np.zeros((self._seq_width, seq.shape[1]), dtype=np.float32)
        padded[:n] = seq
        return padded

    def _widen_sequences(self, width: int) -> None:
        """Grow the stored sequence width (fields ``_SEQUENCE_FIELDS``) to
        ``width``.

        Called when a wider neighbour set than ever before appears; earlier
        transitions keep their rows and their true lengths, so packed
        encoding is unaffected. Runs a handful of times per training run
        (each new record of concurrent agents), copying the sequence fields
        once. Requires ``self.memory`` with ``widen_field``.
        """
        if width <= self._seq_width:
            return
        self._seq_width = int(width)
        for field in self._SEQUENCE_FIELDS:
            self.memory.widen_field(field, (self._seq_width, self.state_dim))
        logger.debug(f"{type(self).__name__} context width -> {self._seq_width} rows")

    @staticmethod
    def _pad_sequences(seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Zero-pad ragged ``[n_i, D]`` sequences to ``[B, max_n, D]``;
        returns ``(padded, lengths)`` with ``lengths`` int64."""
        lengths = np.fromiter((s.shape[0] for s in seqs), dtype=np.int64, count=len(seqs))
        width = int(lengths.max()) if len(seqs) else 1
        padded = np.zeros((len(seqs), width, seqs[0].shape[1] if seqs else 0), dtype=np.float32)
        for i, s in enumerate(seqs):
            padded[i, :s.shape[0]] = s
        return padded, lengths

    # ------------------------------------------------------------------ #
    # Acting replica: act on the CPU, train on the GPU
    # ------------------------------------------------------------------ #
    def _acting_modules(self) -> Dict[str, "nn.Module"]:
        """``name -> module`` for the modules action selection runs; an
        algorithm that returns none acts on its training device."""
        return {}

    def _acting_normalizers(self) -> Dict[str, Any]:
        """``name -> ValueNormalizer`` (or None) the acting critic needs."""
        return {}

    def _acting(self):
        """The replica to act with, refreshed if a weight changed since the
        last step, or ``None`` when acting on the training device."""
        replica = self.__dict__.get("_acting_replica")
        if replica is None:
            replica = self._build_acting_replica()
            self._acting_replica = replica
        if replica:
            if replica.dirty:
                replica.refresh()
            return replica
        return None

    def _build_acting_replica(self):
        modules = self._acting_modules()
        device = getattr(self, "device", None)
        setting = str(self.config.get("acting_device", "auto")).lower()
        if not modules or device is None or setting in ("train", "training", "same"):
            return False
        if setting == "auto":
            target = "cpu" if torch.device(device).type != "cpu" else None
        else:
            target = setting
        if target is None or torch.device(target) == torch.device(device):
            return False
        replica = ActingReplica(modules, self._acting_normalizers(), target)
        if replica.device.type == "cpu":
            _cap_cpu_threads(self.config.get("acting_threads", "auto"))
        return replica

    def _mark_acting_dirty(self) -> None:
        """Call after any change to the acting modules' weights or the
        normalisers' statistics (an optimiser step, a checkpoint load)."""
        replica = self.__dict__.get("_acting_replica")
        if replica:
            replica.dirty = True

    def _acting_device(self) -> torch.device:
        replica = self._acting()
        return replica.device if replica else self.device

    def _acting_module(self, name: str):
        replica = self._acting()
        return replica.modules[name] if replica else getattr(self, name)

    def _acting_normalizer(self, name: str, tensor: torch.Tensor):
        """The normaliser matching *tensor*'s device: the replica's mirror
        for tensors on the acting device, else the live one."""
        replica = self.__dict__.get("_acting_replica")
        live = self._acting_normalizers().get(name)
        if replica and tensor.device == replica.device:
            return replica.normalizers.get(name)
        return live

    def _prepare_batch_inputs(
        self,
        multi_agent_obs: Dict[str, np.ndarray],
        ego_agent_ids: List[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Shared LSTM-context batch prep.

        Produces:

        * ``ego_states``  - ``[B, state_dim]`` stacked ego observations.
        * ``context``     - ``[B, lstm_hidden_size]`` LSTM-encoded other-
          agent context (sequence sorted by relative distance to ego).
        * ``resolved_ids`` - the agent id list (str) the batch rows
          correspond to, with missing agents falling back to the first
          available agent id (matches the legacy per-agent select_action
          behaviour at the same boundary).

        Requires the subclass to expose ``self.lstm_encoder``,
        ``self.state_dim``, ``self.lstm_hidden_size``, and ``self.device``.
        Algorithms without an LSTM context (none in-tree today) override
        this method.

        The sort key is each *other* agent's own ``sqrt(rel_x^2 + rel_y^2)``
        — its distance to the intersection, independent of the ego — so the
        table is sorted once and every ego's sequence is that global order
        with the ego moved to the front (O(N log N) + O(N) per ego instead
        of a fresh O(N log N) sort per ego). Ordering and numerics are
        identical to the previous per-ego sort: same float64 key, stable
        tie-breaking on table order.

        If the algorithm defines ``max_context_agents`` (SAC), the sequence
        is truncated to the ego plus the nearest ``max_context_agents - 1``
        others — the same truncation its replay buffer stores — so acting
        and learning see the same context.

        Every ego of one table gets the same sequence length, so the batch
        needs no padding.
        """
        device = self._acting_device()
        encoder = self._acting_module("lstm_encoder")
        if not multi_agent_obs:
            empty_state = torch.empty(0, self.state_dim, device=device)
            empty_context = torch.empty(
                0, self.lstm_hidden_size, device=device,
            )
            return empty_state, empty_context, []

        fallback_id = next(iter(multi_agent_obs.keys()))
        resolved_ids = [
            str(aid) if str(aid) in multi_agent_obs else fallback_id
            for aid in ego_agent_ids
        ]

        table_pos, table, order = self._sorted_context_table(multi_agent_obs)
        cap = getattr(self, "max_context_agents", None)
        n_others = table.shape[0] - 1
        if cap is not None:
            n_others = min(n_others, max(int(cap) - 1, 0))
        seq_len = 1 + n_others

        ego_rows = np.fromiter(
            (table_pos[aid] for aid in resolved_ids), dtype=np.int64, count=len(resolved_ids),
        )
        if seq_len > 1:
            sequences = np.empty(
                (len(resolved_ids), seq_len, table.shape[1]), dtype=np.float32,
            )
            for i, row in enumerate(ego_rows):
                sequences[i, 0] = table[row]
                sequences[i, 1:] = table[order[order != row][:n_others]]
            sequence_tensor = torch.as_tensor(sequences, device=device)
            context = encoder(sequence_tensor)
            # The ego rows are the sequences' first slot: one host-to-device
            # copy per step instead of two (each is a launch and a sync on
            # a GPU).
            ego_states = sequence_tensor[:, 0]
        else:
            ego_states = torch.as_tensor(
                table[ego_rows], dtype=torch.float32, device=device,
            )
            context = torch.zeros(
                len(resolved_ids), self.lstm_hidden_size, device=device,
            )
        return ego_states, context, resolved_ids

    def _batched_forward(
        self,
        ego_states: torch.Tensor,
        context: torch.Tensor,
        resolved_ids: List[str],
        training: bool,
    ) -> Dict[str, float]:
        """Algorithm-specific batched actor/critic forward.

        Must return ``{agent_id (str): target_speed (float)}`` for every
        id in ``resolved_ids``, performing any per-agent bookkeeping
        (``last_log_probs`` / ``last_values`` / etc.) the algorithm
        needs for downstream transition storage.

        Subclasses that flip ``supports_batched_actions = True`` MUST
        override this; the base class raises ``NotImplementedError`` so
        a misconfigured opt-in is caught loudly (and then handled by
        the try/except in :meth:`select_actions`, which falls back to
        per-agent dispatch).
        """
        raise NotImplementedError(
            f"{type(self).__name__} declared supports_batched_actions=True "
            "but did not override _batched_forward(ego_states, context, "
            "resolved_ids, training)."
        )

    def _fallback_select_action(
        self,
        multi_agent_obs: Dict[str, np.ndarray],
        ego_agent_id: str,
        training: bool,
    ) -> Optional[float]:
        """Single-agent fallback called from the :meth:`select_actions`
        exception handler. Default forwards to the legacy
        ``select_action(multi_agent_obs, ego_agent_id, training=...)``
        signature used by all multi-agent algorithms. Algorithms whose
        :meth:`select_action` takes a different signature (e.g., single-
        agent algos that take a flat state) override here so the
        fallback path keeps working.
        """
        return self.select_action(
            multi_agent_obs, ego_agent_id, training=training,
        )

    @abstractmethod
    def store_transition(self, multi_agent_obs: Dict[str, np.ndarray],
                         ego_agent_id: Any, action: Any, reward: Any,
                         next_multi_agent_obs: Dict[str, np.ndarray],
                         done: bool):
        """
        Store one transition for learning.

        Args:
            multi_agent_obs: All agents' states before the action (S_t)
            ego_agent_id: The acting agent
            action: Action taken (target speed in km/h)
            reward: Reward signal (scalar, or vector when the algorithm sets
                ``supports_vector_rewards``)
            next_multi_agent_obs: All agents' states after the action (S_t+1)
            done: Whether this agent reached a terminal outcome
        """
        pass

    @abstractmethod
    def update(self) -> Dict[str, float]:
        """
        Update the algorithm (learning step).

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def reset_episode(self):
        """Reset algorithm state for new episode."""
        pass

    @abstractmethod
    def get_training_info(self) -> Dict[str, Any]:
        """
        Get information about training progress.
        Must be implemented by all algorithms.

        Returns:
            Dictionary with training statistics
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save algorithm state."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load algorithm state."""
        pass

"""Shared IntentPolicy implementation for speed controllers.

The one-shot route head is owned by the speed algorithm (it shares the
controller's device and, for PPO families, its optimizer), so every
intent-capable algorithm needs the same delegation surface. That surface was
copy-pasted three times; it lives here instead.

An algorithm becomes intent-capable by mixing this in and declaring which
head class it builds::

    class MyAlgorithm(RouteHeadMixin, BaseAlgorithm):
        route_head_cls = HybridRouteSACHead

PPO-family heads share the controller's optimizer; set
``route_head_shares_optimizer = True`` to pass it through.
"""
from __future__ import annotations

from typing import Any, Dict

from loguru import logger


class RouteHeadMixin:
    """Implements ``IntentPolicy`` by delegating to an owned route head."""

    #: Head class constructed by :meth:`enable_route_intent`. Subclasses set
    #: this; a string is resolved lazily from ``opencda_marl.builtin.algorithms.intent.hybrid_policy``
    #: to keep import cost off the module load path.
    route_head_cls: Any = None

    #: PPO-family heads train through the controller's optimizer.
    route_head_shares_optimizer: bool = False

    #: Present so ``__init__`` order never matters: the attribute exists
    #: before any subclass constructor runs.
    route_intent_head = None

    def enable_route_intent(
        self,
        config: Dict[str, Any],
        *,
        state_dim: int,
        num_actions: int = 4,
        pref_dim: int,
        reward_dim: int,
    ) -> None:
        """Attach the one-shot route head to this controller.

        The head is part of this algorithm object rather than a second
        independent learner, which is what makes the policy a hybrid
        discrete-route / continuous-speed controller.
        """
        label = type(self).__name__
        if self.route_intent_head is not None:
            logger.warning(
                f"{label} route-intent head already enabled; keeping existing head")
            return

        head_cls = self._resolve_route_head_cls()
        kwargs = dict(
            config=config,
            state_dim=state_dim,
            num_actions=num_actions,
            pref_dim=pref_dim,
            reward_dim=reward_dim,
            device=self.device,
        )
        if self.route_head_shares_optimizer:
            kwargs["owner_optimizer"] = self.optimizer

        self.route_intent_head = head_cls(**kwargs)
        logger.success(
            f"{label} hybrid route head enabled: state_dim={state_dim}, "
            f"pref_dim={pref_dim}, reward_dim={reward_dim}, "
            f"num_actions={num_actions}")

    @classmethod
    def _resolve_route_head_cls(cls):
        head_cls = cls.route_head_cls
        if head_cls is None:
            raise NotImplementedError(
                f"{cls.__name__} mixes in RouteHeadMixin but does not set "
                "route_head_cls")
        if isinstance(head_cls, str):
            import importlib

            module = importlib.import_module("opencda_marl.builtin.algorithms.intent.hybrid_policy")
            head_cls = getattr(module, head_cls)
        return head_cls

    def _require_route_head(self):
        if self.route_intent_head is None:
            raise RuntimeError(
                f"{type(self).__name__} route-intent head is not enabled")
        return self.route_intent_head

    @property
    def route_intent_buffer(self):
        return self._require_route_head().buffer

    @property
    def route_intent_batch_size(self) -> int:
        return int(self._require_route_head().batch_size)

    def select_intent(self, observation, action_mask, preference, *,
                      training: bool = True):
        return self._require_route_head().select_intent(
            observation, action_mask, preference, training=training,
        )

    def store_pending(self, **kwargs) -> None:
        self._require_route_head().store_pending(**kwargs)

    def has_pending(self, actor_id: Any) -> bool:
        return (self.route_intent_head is not None
                and self.route_intent_head.has_pending(actor_id))

    def finalize_terminal(self, **kwargs) -> None:
        self._require_route_head().finalize_terminal(**kwargs)

    def update_route_intent(self) -> Dict[str, float]:
        return self._require_route_head().update()


__all__ = ["RouteHeadMixin"]

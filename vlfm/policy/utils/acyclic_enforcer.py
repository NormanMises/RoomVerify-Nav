# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from __future__ import annotations

from typing import Any, Hashable, Set, Tuple

import numpy as np


class StateAction:
    def __init__(self, position: np.ndarray, action: Any, other: Any = None):
        self.position = position
        self.action = action
        self.other = other
        self._frozen = (self._freeze(position), self._freeze(action), self._freeze(other))

    @staticmethod
    def _freeze(value: Any) -> Hashable:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.ndarray):
            arr = value.astype(float).reshape(-1)
            return tuple(float(x) for x in arr)
        if isinstance(value, (list, tuple)):
            return tuple(StateAction._freeze(v) for v in value)
        if isinstance(value, dict):
            return tuple(sorted((StateAction._freeze(k), StateAction._freeze(v)) for k, v in value.items()))
        return str(value)

    def __hash__(self) -> int:
        return hash(self._frozen)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateAction):
            return False
        return self._frozen == other._frozen


class AcyclicEnforcer:
    def __init__(self) -> None:
        # Per-instance history; callers typically re-create the enforcer on reset.
        self.history: Set[StateAction] = set()

    def check_cyclic(self, position: np.ndarray, action: Any, other: Any = None) -> bool:
        state_action = StateAction(position, action, other)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, action: Any, other: Any = None) -> None:
        state_action = StateAction(position, action, other)
        self.history.add(state_action)


def make_acyclic_enforcer() -> Any:
    return AcyclicEnforcer()

"""
High-Performance Smart Replay Buffer for RL algorithms.
Uses numpy arrays for O(1) sampling instead of O(N) deque operations.
"""

import numpy as np
from typing import List, Tuple, Optional
from loguru import logger


class SmartReplayBuffer:
    """
    High-performance replay buffer with recency bias.
    Uses pre-allocated numpy arrays for O(1) random access and sampling.

    Performance: O(1) push, O(batch_size) sample (instead of O(N) with deque)
    """

    def __init__(self, capacity: int = 20000, recency_ratio: float = 0.5):
        """
        Initialize smart replay buffer.

        Args:
            capacity: Maximum buffer size
            recency_ratio: Fraction of samples from recent experiences (0.5 = 50% recent)
        """
        self.capacity = capacity
        self.recency_ratio = recency_ratio

        # Circular buffer state
        self.position = 0  # Next write position
        self.size = 0      # Current number of valid entries

        # Storage arrays - initialized lazily on first push
        self._initialized = False
        self._num_fields = 0
        self._field_arrays: List[np.ndarray] = []
        self._field_dtypes: List[np.dtype] = []
        self._field_shapes: List[Tuple] = []

        # Track statistics
        self.total_stored = 0
        self.sample_count = 0

        logger.info(f"SmartReplayBuffer initialized: capacity={capacity}, recency_ratio={recency_ratio}")

    def _initialize_storage(self, transition: Tuple):
        """Initialize numpy arrays based on first transition structure."""
        self._num_fields = len(transition)

        for i, field in enumerate(transition):
            if isinstance(field, np.ndarray):
                # For numpy arrays, store with original shape
                shape = (self.capacity,) + field.shape
                dtype = field.dtype
            elif isinstance(field, bool):
                shape = (self.capacity,)
                dtype = np.bool_
            elif isinstance(field, (int, np.integer)):
                shape = (self.capacity,)
                dtype = np.int64
            else:  # float
                shape = (self.capacity,)
                dtype = np.float32

            self._field_arrays.append(np.zeros(shape, dtype=dtype))
            self._field_dtypes.append(dtype)
            self._field_shapes.append(shape[1:] if len(shape) > 1 else ())

        self._initialized = True
        logger.debug(f"Buffer storage initialized with {self._num_fields} fields")

    def push(self, *args):
        """Store a transition using circular buffer. O(1) operation."""
        if not self._initialized:
            self._initialize_storage(args)

        # Store each field at current position
        for i, value in enumerate(args):
            if isinstance(value, np.ndarray):
                self._field_arrays[i][self.position] = value
            else:
                self._field_arrays[i][self.position] = value

        # Update circular buffer position
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_stored += 1

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample batch with recency bias using numpy vectorized operations.
        O(batch_size) operation instead of O(N).

        50% from most recent 20% of buffer (recent experiences)
        50% from entire buffer (diverse experiences)
        """
        return self._indices_to_transitions(self._draw_indices(batch_size))

    def sample_fields(self, batch_size: int) -> List[np.ndarray]:
        """Same draw as :meth:`sample`, returned as one stacked array per field.

        ``result[f]`` is ``field_arrays[f][indices]`` — shape
        ``(batch, *field_shape)`` — so a learner can build its batch tensors
        directly instead of re-stacking ``batch`` tuples of ``num_fields``
        scalars/arrays. Consumes the RNG exactly like :meth:`sample`, so both
        views of the buffer draw identical batches for identical state.
        """
        indices = self._draw_indices(batch_size)
        return [arr[indices] for arr in self._field_arrays]

    def _draw_indices(self, batch_size: int) -> np.ndarray:
        """Recency-biased index draw shared by :meth:`sample`/:meth:`sample_fields`."""
        if self.size < batch_size:
            # Return all available
            return np.arange(self.size)

        self.sample_count += 1

        # Calculate split
        recent_samples = int(batch_size * self.recency_ratio)
        random_samples = batch_size - recent_samples

        # Calculate recent window (last 20% of filled buffer)
        recent_window_size = max(1, int(self.size * 0.2))

        # Get valid indices for recent window
        # For circular buffer, recent entries are at positions before current position
        if self.size < self.capacity:
            # Buffer not full yet - recent is simply the last entries
            recent_start = max(0, self.size - recent_window_size)
            recent_indices = np.random.randint(recent_start, self.size, size=recent_samples)
            random_indices = np.random.randint(0, self.size, size=random_samples)
        else:
            # Buffer is full - handle circular wraparound
            # Recent entries are the ones written most recently before position
            recent_start = (self.position - recent_window_size) % self.capacity

            if recent_start < self.position:
                # No wraparound - recent window is contiguous
                recent_indices = np.random.randint(recent_start, self.position, size=recent_samples)
            else:
                # Wraparound - sample from two segments
                segment1_size = self.capacity - recent_start  # From recent_start to end
                segment2_size = self.position  # From 0 to position
                total_recent = segment1_size + segment2_size

                # Generate random positions within the conceptual recent window
                recent_positions = np.random.randint(0, total_recent, size=recent_samples)
                recent_indices = np.where(
                    recent_positions < segment1_size,
                    recent_start + recent_positions,  # In segment 1
                    recent_positions - segment1_size   # In segment 2 (wraps to 0)
                )

            random_indices = np.random.randint(0, self.capacity, size=random_samples)

        # Combine and shuffle indices
        all_indices = np.concatenate([recent_indices, random_indices])
        np.random.shuffle(all_indices)

        # Log statistics occasionally
        if self.sample_count % 100 == 0:
            logger.debug(f"Buffer stats: size={self.size}/{self.capacity}, "
                        f"total_stored={self.total_stored}, samples={self.sample_count}")

        return all_indices

    def _indices_to_transitions(self, indices: np.ndarray) -> List[Tuple]:
        """Convert array indices to list of transition tuples. O(batch_size) operation."""
        # Use numpy fancy indexing - O(batch_size) not O(N)
        batch = []
        for i in range(len(indices)):
            idx = indices[i]
            transition = tuple(
                self._field_arrays[f][idx] for f in range(self._num_fields)
            )
            batch.append(transition)
        return batch

    def _newest_indices(self, count: int) -> np.ndarray:
        """Storage indices of the ``count`` most recently written entries,
        oldest first, honouring circular wraparound."""
        start = (self.position - count) % self.capacity
        return (start + np.arange(count)) % self.capacity

    def field_shape(self, index: int) -> Optional[Tuple[int, ...]]:
        """Per-entry shape of an array field, or ``None`` before the first push."""
        if not self._initialized or index >= len(self._field_arrays):
            return None
        return tuple(self._field_arrays[index].shape[1:])

    def widen_field(self, index: int, new_shape: Tuple[int, ...]) -> None:
        """Grow one array field's per-entry shape in place, zero-filled.

        Existing entries keep their values in the leading sub-block; the
        write position and size are untouched. Used by learners whose
        per-transition arrays can legitimately get wider over time (e.g.
        a neighbour sequence when more agents than ever seen before appear)
        without forcing a fixed cap up front. Never shrinks.
        """
        if not self._initialized:
            return  # first push defines the shapes
        old = self._field_arrays[index]
        old_shape = old.shape[1:]
        new_shape = tuple(int(n) for n in new_shape)
        if len(new_shape) != len(old_shape) or any(n < o for n, o in zip(new_shape, old_shape)):
            raise ValueError(
                f"widen_field can only grow field {index}: {old_shape} -> {new_shape}")
        if new_shape == tuple(old_shape):
            return
        grown = np.zeros((self.capacity,) + new_shape, dtype=old.dtype)
        grown[(slice(None),) + tuple(slice(0, o) for o in old_shape)] = old
        self._field_arrays[index] = grown
        self._field_shapes[index] = new_shape

    def clear_old(self, keep_ratio: float = 0.7):
        """Drop the oldest experiences, keeping the newest ``keep_ratio``.

        The kept block is compacted to storage indices ``[0, size)`` and the
        write position reset to ``size``, so the sampler's "valid entries
        are ``[0, size)`` until the buffer wraps" invariant holds again.
        (Only shrinking ``size`` while leaving ``position`` alone — the
        previous behaviour — kept the *oldest* entries and hid the newest
        ones from sampling until the next wraparound.) O(capacity) copy,
        but this runs rarely (TD3 triggers it near capacity).

        Args:
            keep_ratio: Fraction of buffer to keep (0.7 = keep 70% newest)
        """
        if self.size < 1000:
            return  # Don't clear if buffer is small

        old_size = self.size
        keep_count = int(self.size * keep_ratio)
        keep = self._newest_indices(keep_count)
        for i, arr in enumerate(self._field_arrays):
            kept = arr[keep].copy()
            arr[:keep_count] = kept
            arr[keep_count:] = 0
            self._field_arrays[i] = arr
        self.position = keep_count % self.capacity
        self.size = keep_count

        logger.info(f"Cleared old experiences: {old_size} -> {self.size}")

    def __len__(self):
        return self.size

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        return {
            'current_size': self.size,
            'capacity': self.capacity,
            'total_stored': self.total_stored,
            'sample_count': self.sample_count,
            'recency_ratio': self.recency_ratio,
            'position': self.position,
            'initialized': self._initialized
        }


class PrioritizedReplayBuffer:
    """
    High-performance Prioritized Experience Replay buffer.
    Uses numpy arrays for O(1) sampling with cached priority normalization.
    Based on the PER paper: https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity: int = 20000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling weight (0 = no correction, 1 = full correction)
            beta_increment: Beta increment per sampling
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6  # Small constant to ensure non-zero priorities

        # Circular buffer state
        self.position = 0
        self.size = 0

        # Pre-allocated priority array (always available)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

        # Cached probability array (updated lazily)
        self._probabilities: Optional[np.ndarray] = None
        self._prob_dirty = True  # Flag to indicate if probabilities need recalc

        # Storage arrays - initialized lazily on first push
        self._initialized = False
        self._num_fields = 0
        self._field_arrays: List[np.ndarray] = []

        # Statistics
        self.total_stored = 0
        self.sample_count = 0

        logger.info(f"PrioritizedReplayBuffer initialized: capacity={capacity}, "
                   f"alpha={alpha}, beta={beta}")

    def _initialize_storage(self, transition: Tuple):
        """Initialize numpy arrays based on first transition structure."""
        self._num_fields = len(transition)

        for i, field in enumerate(transition):
            if isinstance(field, np.ndarray):
                shape = (self.capacity,) + field.shape
                dtype = field.dtype
            elif isinstance(field, bool):
                shape = (self.capacity,)
                dtype = np.bool_
            elif isinstance(field, (int, np.integer)):
                shape = (self.capacity,)
                dtype = np.int64
            else:
                shape = (self.capacity,)
                dtype = np.float32

            self._field_arrays.append(np.zeros(shape, dtype=dtype))

        self._initialized = True
        logger.debug(f"PER storage initialized with {self._num_fields} fields")

    def push(self, *args, td_error: float = None):
        """Store transition with priority. O(1) operation."""
        if not self._initialized:
            self._initialize_storage(args)

        # Store each field
        for i, value in enumerate(args):
            self._field_arrays[i][self.position] = value

        # Set priority
        if td_error is None:
            priority = self.max_priority
        else:
            priority = (abs(td_error) + self.epsilon) ** self.alpha

        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)

        # Mark probabilities as needing recalculation
        self._prob_dirty = True

        # Update circular buffer
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_stored += 1

    def _update_probabilities(self):
        """Update cached probability array. Only called when dirty."""
        if not self._prob_dirty:
            return

        # Only compute over valid entries
        valid_priorities = self.priorities[:self.size]
        total = valid_priorities.sum()
        if total > 0:
            self._probabilities = valid_priorities / total
        else:
            self._probabilities = np.ones(self.size) / self.size

        self._prob_dirty = False

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample based on priorities with importance sampling weights. O(batch_size) operation."""
        if self.size < batch_size:
            indices = np.arange(self.size)
            batch = self._indices_to_transitions(indices)
            weights = np.ones(self.size, dtype=np.float32)
            return batch, indices, weights

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.sample_count += 1

        # Update probabilities if needed (cached)
        self._update_probabilities()

        # Sample indices based on priorities - O(batch_size)
        indices = np.random.choice(self.size, batch_size, p=self._probabilities, replace=False)

        # Calculate importance sampling weights
        weights = (self.size * self._probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Get transitions
        batch = self._indices_to_transitions(indices)

        # Log statistics occasionally
        if self.sample_count % 100 == 0:
            avg_priority = self.priorities[:self.size].mean()
            logger.debug(f"PER stats: size={self.size}, avg_priority={avg_priority:.4f}, "
                        f"beta={self.beta:.3f}, samples={self.sample_count}")

        return batch, indices, weights.astype(np.float32)

    def _indices_to_transitions(self, indices: np.ndarray) -> List[Tuple]:
        """Convert indices to transitions. O(batch_size) operation."""
        batch = []
        for idx in indices:
            transition = tuple(self._field_arrays[f][idx] for f in range(self._num_fields))
            batch.append(transition)
        return batch

    def field_shape(self, index: int) -> Optional[Tuple[int, ...]]:
        """Per-entry shape of an array field, or ``None`` before the first push."""
        if not self._initialized or index >= len(self._field_arrays):
            return None
        return tuple(self._field_arrays[index].shape[1:])

    def widen_field(self, index: int, new_shape: Tuple[int, ...]) -> None:
        """Grow one array field's per-entry shape in place, zero-filled;
        see :meth:`SmartReplayBuffer.widen_field`."""
        if not self._initialized:
            return
        old = self._field_arrays[index]
        old_shape = old.shape[1:]
        new_shape = tuple(int(n) for n in new_shape)
        if len(new_shape) != len(old_shape) or any(n < o for n, o in zip(new_shape, old_shape)):
            raise ValueError(
                f"widen_field can only grow field {index}: {old_shape} -> {new_shape}")
        if new_shape == tuple(old_shape):
            return
        grown = np.zeros((self.capacity,) + new_shape, dtype=old.dtype)
        grown[(slice(None),) + tuple(slice(0, o) for o in old_shape)] = old
        self._field_arrays[index] = grown

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD-errors. Vectorized O(batch_size) operation."""
        # Filter valid indices
        valid_mask = (indices >= 0) & (indices < self.size)
        valid_indices = indices[valid_mask]
        valid_td_errors = td_errors[valid_mask]

        if len(valid_indices) == 0:
            return

        # Vectorized priority update - no loop
        new_priorities = (np.abs(valid_td_errors) + self.epsilon) ** self.alpha
        self.priorities[valid_indices] = new_priorities

        # Update max priority
        self.max_priority = max(self.max_priority, new_priorities.max())

        # Mark probabilities as needing recalculation
        self._prob_dirty = True

    def clear_old(self, keep_ratio: float = 0.5):
        """Drop the oldest experiences, keeping the newest ``keep_ratio``.

        Compacts the kept entries (and their priorities) to ``[0, size)``
        and resets the write position, so ``priorities[:size]`` and the
        sampled indices refer to the kept block. See
        :meth:`SmartReplayBuffer.clear_old` for why shrinking ``size``
        alone was wrong.
        """
        if self.size < 5000:
            return

        old_size = self.size
        keep_count = int(self.size * keep_ratio)
        start = (self.position - keep_count) % self.capacity
        keep = (start + np.arange(keep_count)) % self.capacity
        for i, arr in enumerate(self._field_arrays):
            kept = arr[keep].copy()
            arr[:keep_count] = kept
            arr[keep_count:] = 0
            self._field_arrays[i] = arr
        kept_priorities = self.priorities[keep].copy()
        self.priorities[:keep_count] = kept_priorities
        self.priorities[keep_count:] = 0.0
        self.position = keep_count % self.capacity
        self.size = keep_count
        self._prob_dirty = True

        logger.info(f"PER: Cleared old experiences: {old_size} -> {self.size}")

    def __len__(self):
        return self.size

    def get_stats(self) -> dict:
        """Get buffer statistics."""
        valid_priorities = self.priorities[:self.size] if self.size > 0 else np.array([0])
        return {
            'current_size': self.size,
            'capacity': self.capacity,
            'total_stored': self.total_stored,
            'sample_count': self.sample_count,
            'alpha': self.alpha,
            'beta': self.beta,
            'avg_priority': float(valid_priorities.mean()),
            'max_priority': self.max_priority
        }
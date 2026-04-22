# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A sum tree data structure without JAX, using NumPy for randomness."""

from bbf.replay_memory import sum_tree
import numpy as np


def _rng_uniform(rng, low=0.0, high=1.0, size=None):
    """Framework-agnostic uniform sampler using NumPy-style RNGs."""
    if rng is None:
        rng = np.random.default_rng()

    # NumPy Generator
    if hasattr(rng, "uniform"):
        return rng.uniform(low=low, high=high, size=size)

    # Fallback to legacy numpy.random module-like interface
    return np.random.uniform(low=low, high=high, size=size)


def step(query_value, index, nodes):
    """Single traversal step down the sum tree."""
    left_child = index * 2 + 1
    left_sum = nodes[left_child]
    if query_value < left_sum:
        index = left_child
    else:
        index = left_child + 1
        query_value = query_value - left_sum
    return query_value, index, nodes


def parallel_stratified_sample(rng, nodes, batch_size, depth):
    """Performs stratified sampling over the sum tree leaves."""
    total_priority = nodes[0]
    indices = np.empty(batch_size, dtype=np.int32)

    for i in range(batch_size):
        upper_bound = (i + 1) / batch_size
        lower_bound = i / batch_size
        query = _rng_uniform(rng, low=lower_bound, high=upper_bound)
        query_value = float(query) * total_priority
        index = 0
        for _ in range(depth):
            query_value, index, _ = step(query_value, index, nodes)
        indices[i] = index

    return indices


class DeterministicSumTree(sum_tree.SumTree):
    """A sum tree data structure for storing replay priorities.

    In contrast to the original implementation, this version does not use JAX.
    It uses NumPy-based randomness while preserving the same public API.
    """

    def __init__(self, capacity):
        """Creates the sum tree data structure for the given replay capacity.

        Args:
          capacity: int, the maximum number of elements that can be stored in
            this data structure.

        Raises:
          ValueError: If requested capacity is not positive.
        """
        assert isinstance(capacity, int)
        if capacity <= 0:
            raise ValueError(
                'Sum tree capacity should be positive. Got: {}'.format(
                    capacity))

        self.nodes = []
        self.depth = int(np.ceil(np.log2(capacity)))
        self.low_idx = (2**self.depth) - 1  # pri_idx + low_idx -> tree_idx
        self.high_idx = capacity + self.low_idx
        self.nodes = np.zeros(2**(self.depth + 1) - 1, dtype=np.float64)
        self.capacity = capacity

        self.highest_set = 0
        self.max_recorded_priority = 1.0

    def _total_priority(self):
        """Returns the sum of all priorities stored in this sum tree.

        Returns:
          float, sum of priorities stored in this sum tree.
        """
        return self.nodes[0]

    def sample(self, rng, query_value=None):
        """Samples an element from the sum tree.

        Args:
          rng: NumPy RNG-like object or None.
          query_value: Optional float in [0, 1). If None, sampled uniformly.

        Returns:
          Sampled leaf index clipped by highest_set.
        """
        query_value = (
            float(_rng_uniform(rng)) if query_value is None else float(query_value)
        )
        query_value *= self._total_priority()

        index = 0
        for _ in range(self.depth):
            query_value, index, _ = step(query_value, index, self.nodes)

        return np.minimum(index - self.low_idx, self.highest_set)

    def stratified_sample(self, batch_size, rng):
        """Performs stratified sampling using the sum tree.

        Args:
          batch_size: Number of samples.
          rng: NumPy RNG-like object or None.

        Returns:
          Array of sampled indices clipped by highest_set.
        """
        if self._total_priority() == 0.0:
            raise Exception('Cannot sample from an empty sum tree.')

        indices = parallel_stratified_sample(rng, self.nodes, batch_size,
                                             self.depth)
        return np.minimum(indices - self.low_idx, self.highest_set)

    def get(self, node_index):
        """Returns the value of the leaf node corresponding to the index.

        Args:
            node_index: The index of the leaf node.

        Returns:
            The value of the leaf node.
        """
        return self.nodes[node_index + self.low_idx]

    def reset_priorities(self):
        for i in range(self.highest_set):
            self.set(i, self.max_recorded_priority)

    def set(self, node_index, value):
        """Sets the value of a leaf node and updates internal nodes accordingly.

        This operation takes O(log(capacity)).

        Args:
            node_index: int, the index of the leaf node to be updated.
            value: float, the value which we assign to the node. This value must
              be nonnegative. Setting value = 0 will cause the element to never
              be sampled.

        Raises:
            ValueError: If the given value is negative.
        """
        if value < 0.0:
            raise ValueError(
                'Sum tree values should be nonnegative. Got {}'.format(value))
        self.highest_set = max(node_index, self.highest_set)
        node_index = node_index + self.low_idx
        self.max_recorded_priority = max(value, self.max_recorded_priority)

        delta_value = value - self.nodes[node_index]

        # Now traverse back the tree, adjusting all sums along the way.
        for _ in reversed(range(self.depth)):
            # Note: Adding a delta leads to some tolerable numerical inaccuracies.
            self.nodes[node_index] += delta_value
            node_index = (node_index - 1) // 2

        self.nodes[node_index] += delta_value
        assert node_index == 0, ('Sum tree traversal failed, final node index '
                                 'is not 0.')
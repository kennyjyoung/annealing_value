

from dataclasses import dataclass
from typing import Callable, List
import numpy as np
import pickle


def _make_piecewise_constant(bin_edges: np.ndarray, values: np.ndarray, counts: np.ndarray) -> Callable[[float], float]:
    """
    Build a scalar function f(x) that returns a piecewise-constant value per position bin.
    - bin_edges: shape (B+1,)
    - values: shape (B,)
    - counts: shape (B,) used to repair empty bins
    """
    assert bin_edges.ndim == 1 and values.ndim == 1 and counts.ndim == 1
    assert len(bin_edges) == len(values) + 1 == len(counts) + 1

    # Fill empty bins by nearest-neighbor interpolation over indices
    filled = values.astype(float).copy()
    valid = counts > 0
    if not np.any(valid):
        # Fallback: flat zero if no data at all
        filled[:] = 0.0
    else:
        idxs = np.arange(len(values))
        filled[~valid] = np.interp(idxs[~valid], idxs[valid], values[valid])

    def f(x: float) -> float:
        # Right-inclusive on upper edge except final bin
        i = int(np.searchsorted(bin_edges, x, side="right") - 1)
        i = 0 if i < 0 else (len(filled) - 1 if i >= len(filled) else i)
        return float(filled[i])

    return f


@dataclass
class HistogramEnergyFunctions:
    energy_functions: List[Callable[[float], float]]
    pos_edges: np.ndarray
    step_edges: np.ndarray  # integer step edges of length N+1
    initial_temperature: float
    cooling_rate: float

    def _temperature_to_step(self, temperature: float) -> int:
        """
        Map a temperature to an approximate step index using:
            T(step) = initial_temperature * cooling_rate ** step
        """
        # Guard against invalid temperatures
        if temperature <= 0:
            return int(self.step_edges[-2])  # map to last bin
        # Solve for step; note log(cooling_rate) < 0
        step_float = np.log(temperature / self.initial_temperature) / np.log(self.cooling_rate)
        # Lower temps → larger steps; clamp to valid range
        step_idx = int(np.floor(step_float))
        # Allow steps before start (for temps slightly above initial) and after end
        return max(int(self.step_edges[0]), min(step_idx, int(self.step_edges[-1] - 1)))

    def energy(self, temperature: float, position: float) -> float:
        # Convert temperature to step and locate the slice index by step_edges
        step_idx = self._temperature_to_step(temperature)
        # Find i s.t. step_idx in [step_edges[i], step_edges[i+1])
        i = int(np.searchsorted(self.step_edges, step_idx, side="right") - 1)
        i = 0 if i < 0 else (len(self.energy_functions) - 1 if i >= len(self.energy_functions) else i)
        return self.energy_functions[i](position)


def load_histogram_energy_functions(
    pickle_path: str = "value_histograms.pkl",
    initial_temperature: float = 10.0,
    cooling_rate: float = 0.99,
) -> HistogramEnergyFunctions:
    """
    Load histogram-derived value functions and return a callable adapter.
    - Energy per slice is represented as a piecewise-constant function over position bins
      whose heights equal the mean final cost in each bin.
    - Temperature selects the appropriate slice based on the geometric cooling schedule.
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    pos_edges = np.asarray(data["pos_edges"])
    slices = data["slices"]  # list of dicts with counts, sum_costs, mean_costs, and start/end steps
    step_edges = np.asarray(data["step_edges"])  # length = num_slices + 1

    energy_functions: List[Callable[[float], float]] = []
    for s in slices:
        counts = np.asarray(s["counts"])
        means = np.asarray(s["mean_costs"])
        fn = _make_piecewise_constant(pos_edges, means, counts)
        energy_functions.append(fn)

    return HistogramEnergyFunctions(
        energy_functions=energy_functions,
        pos_edges=pos_edges,
        step_edges=step_edges,
        initial_temperature=float(initial_temperature),
        cooling_rate=float(cooling_rate),
    )

def run_value_based_annealing_step(
    position: float,
    key: np.ndarray,
    temperature: float,
    value_function: HistogramEnergyFunctions,
) -> float:
    step_idx = value_function._temperature_to_step(temperature)
    return value_function.energy(temperature, position)
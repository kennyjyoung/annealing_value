
from dataclasses import dataclass
import functools
from typing import Callable, List, Optional
import jax
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sim import energy
import jax.numpy as jnp

# Visualization state (reused across calls)
_viz_fig = None
_viz_ax = None
_viz_energy_line = None
_viz_value_line = None
_viz_dot = None
_viz_x_range = (-5.0, 5.0)
_viz_y_range = None

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=True)
class HistogramEnergyFunctions:
    values: jnp.ndarray  # Shape: (num_slices, num_bins)
    pos_edges: jnp.ndarray # Shape: (num_bins + 1,)
    step_edges: jnp.ndarray  # integer step edges of length num_slices + 1
    initial_temperature: float
    cooling_rate: float

    def tree_flatten(self):
        children = (self.values, self.pos_edges, self.step_edges)
        aux_data = (self.initial_temperature, self.cooling_rate)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

    def _temperature_to_step(self, temperature: jnp.ndarray) -> jnp.ndarray:
        """
        Map a temperature to an approximate step index using:
            T(step) = initial_temperature * cooling_rate ** step
        """
        # Guard against invalid temperatures to avoid NaNs in log
        safe_temp = jnp.where(temperature <= 1e-10, 1e-10, temperature)
        
        # Solve for step; note log(cooling_rate) < 0
        step_float = jnp.log(safe_temp / self.initial_temperature) / jnp.log(self.cooling_rate)
        
        # Lower temps → larger steps; clamp to valid range
        step_idx = jnp.floor(step_float)
        
        # Allow steps before start (for temps slightly above initial) and after end
        step = jnp.clip(step_idx, self.step_edges[0], self.step_edges[-1] - 1)
        
        # If temp was effectively 0 or negative, ensure we map to the end
        return jnp.where(temperature <= 1e-10, self.step_edges[-1] - 1, step)

    def energy(self, temperature: jnp.ndarray, position: jnp.ndarray) -> jnp.ndarray:
        # Convert temperature to step and locate the slice index by step_edges
        step_idx = self._temperature_to_step(temperature)
        
        # Find i s.t. step_idx in [step_edges[i], step_edges[i+1])
        # step_edges is sorted
        slice_idx = jnp.searchsorted(self.step_edges, step_idx, side="right") - 1
        slice_idx = jnp.clip(slice_idx, 0, self.values.shape[0] - 1)
        
        # Find bin index for position
        # pos_edges is sorted
        bin_idx = jnp.searchsorted(self.pos_edges, position, side="right") - 1
        bin_idx = jnp.clip(bin_idx, 0, self.values.shape[1] - 1)
        
        return self.values[slice_idx, bin_idx]


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

    pos_edges = jnp.asarray(data["pos_edges"])
    slices = data["slices"]  # list of dicts with counts, sum_costs, mean_costs, and start/end steps
    step_edges = jnp.asarray(data["step_edges"])  # length = num_slices + 1

    all_values = []
    for s in slices:
        counts = np.asarray(s["counts"])
        values = np.asarray(s["mean_costs"])
        
        # Fill empty bins by nearest-neighbor interpolation over indices
        filled = values.astype(float).copy()
        valid = counts > 0
        if not np.any(valid):
            # Fallback: flat zero if no data at all
            filled[:] = 0.0
        else:
            idxs = np.arange(len(values))
            filled[~valid] = np.interp(idxs[~valid], idxs[valid], values[valid])
        
        all_values.append(filled)

    values_array = jnp.asarray(np.stack(all_values))

    return HistogramEnergyFunctions(
        values=values_array,
        pos_edges=pos_edges,
        step_edges=step_edges,
        initial_temperature=float(initial_temperature),
        cooling_rate=float(cooling_rate),
    )

# Remove static_argnums for value_function as it is now a valid PyTree
@jax.jit
def run_value_based_annealing_step(
    position: float,
    key: np.ndarray,
    temperature: float,
    last_cost: float,
    last_value: float,
    value_function: HistogramEnergyFunctions,
) -> tuple[float, float, float]:
    step_idx = value_function._temperature_to_step(temperature)
    key, subkey = jax.random.split(key)
    proposed_position = position + 0.01 * (jax.random.bernoulli(subkey, 0.5) - 0.5) * 2
    proposed_cost = energy(proposed_position)
    proposed_value = value_function.energy(temperature, proposed_position)
    
    # Need another split for acceptance probability? 
    # Original code reused 'key' which was split above.
    # key, subkey = jax.random.split(key) -> this was in original code
    key, subkey = jax.random.split(key)
    
    # better_value = proposed_value <= last_value
    # accept = jnp.where(better_value, False, True)
    accept = jax.random.bernoulli(subkey, jnp.clip(jnp.exp(-(proposed_value - last_value) / temperature), 0, 1))
    # accept = jax.random.bernoulli(subkey, jnp.clip(jnp.exp(-(proposed_cost - last_cost) / temperature), 0, 1))
    # accept = jnp.where(better_value, accept, True)
    return jnp.where(accept, proposed_position, position), jnp.where(accept, proposed_cost, last_cost), jnp.where(accept, proposed_value, last_value)

def run_multiple_value_based_annealing_steps(
    position: float,
    key: np.ndarray,
    temperature: float,
    last_cost: float,
    last_value: float,
    value_function: HistogramEnergyFunctions,
    num_steps: int,
) -> tuple[float, float, float, np.ndarray]:
    def scan_body(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        step_idx: jnp.ndarray,
    ) -> tuple[tuple[float, float, float, jnp.ndarray], None]:
        position, last_cost, last_value, key = carry
        key, step_key = jax.random.split(key)
        position, last_cost, last_value = run_value_based_annealing_step(position, step_key, temperature, last_cost, last_value, value_function)
        return (position, last_cost, last_value, key), None
    
    (position, last_cost, last_value, key), _ = jax.lax.scan(
        scan_body, 
        (position, last_cost, last_value, key), 
        jnp.arange(num_steps)
    )
    return position, last_cost, last_value, key

def _init_visualization(x_min: float, x_max: float) -> None:
    global _viz_fig, _viz_ax, _viz_energy_line, _viz_value_line, _viz_dot, _viz_x_range, _viz_y_range
    plt.ion()
    _viz_x_range = (x_min, x_max)
    x = jnp.linspace(x_min, x_max, 1000)
    y = energy(x)
    x_np = np.asarray(x)
    y_np = np.asarray(y)

    _viz_fig, _viz_ax = plt.subplots(figsize=(8, 4.5))
    (_viz_energy_line,) = _viz_ax.plot(x_np, y_np, color="black", linewidth=1.5, label="Energy")
    # Initialize value function line (empty for now)
    (_viz_value_line,) = _viz_ax.plot([], [], color="blue", linewidth=1.5, alpha=0.7, label="Value Function")
    (_viz_dot,) = _viz_ax.plot([], [], "o", color="green", markersize=8)
    _viz_ax.set_xlabel("position")
    _viz_ax.set_ylabel("energy")
    _viz_ax.set_title("Energy and Value Function")
    _viz_ax.set_xlim(float(x_np.min()), float(x_np.max()))
    y_margin = 0.1 * (float(y_np.max()) - float(y_np.min()))
    _viz_y_range = (float(y_np.min()) - y_margin, float(y_np.max()) + y_margin)
    _viz_ax.set_ylim(*_viz_y_range)
    _viz_ax.legend()
    _viz_fig.tight_layout()
    _viz_fig.canvas.draw_idle()
    _viz_fig.canvas.flush_events()

def visualize_position(
    position: jnp.ndarray, 
    value_function: Optional[HistogramEnergyFunctions] = None, 
    temperature: Optional[float] = None,
    x_min: float = -5.0, 
    x_max: float = 5.0, 
    delay_seconds: float = 0.03
) -> None:
    """
    Plot the energy curve and a green dot at the given position(s).
    - position can be a scalar or array-like; all points will be plotted.
    - If value_function and temperature are provided, plots the value function for that temperature.
    - Non-blocking: holds for delay_seconds, then returns.
    """
    global _viz_fig, _viz_ax, _viz_energy_line, _viz_value_line, _viz_dot, _viz_x_range, _viz_y_range
    # Initialize or reinitialize if x-range changed
    if _viz_fig is None or _viz_x_range != (x_min, x_max):
        _init_visualization(x_min, x_max)

    pos = jnp.asarray(position).reshape(-1)
    e_pos = energy(pos)

    pos_np = np.asarray(pos)
    e_pos_np = np.asarray(e_pos)

    # Update the dot
    _viz_dot.set_data(pos_np, e_pos_np)
    
    # Update value function line if available
    if value_function is not None and temperature is not None:
        # Use vectorized evaluation for plotting
        # Create a grid of points
        x_grid = jnp.linspace(x_min, x_max, 1000)
        # To vectorize the value function evaluation over positions, we can use vmap
        # But the energy method takes scalar/array inputs for position if implemented right.
        # Let's check implementation of energy...
        # It uses searchsorted which handles arrays.
        # So we can pass x_grid directly if temperature is scalar.
        
        v_grid = value_function.energy(temperature, x_grid)
        _viz_value_line.set_data(np.asarray(x_grid), np.asarray(v_grid))
    
    _viz_fig.canvas.draw_idle()
    _viz_fig.canvas.flush_events()
    plt.pause(delay_seconds)

def close_visualization() -> None:
    global _viz_fig, _viz_ax, _viz_energy_line, _viz_dot
    if _viz_fig is not None:
        plt.ioff()
        plt.close(_viz_fig)
    _viz_fig = None
    _viz_ax = None
    _viz_energy_line = None
    _viz_dot = None

if __name__ == "__main__":
    value_function = load_histogram_energy_functions()
    position = jnp.array(0.0)
    key = jax.random.PRNGKey(0)
    temperature = 100.0
    last_cost = energy(position)
    last_value = value_function.energy(temperature, position)

    # Only num_steps (arg 6) is static. value_function (arg 5) is a PyTree.
    jit_run_multiple_value_based_annealing_steps = jax.jit(
        run_multiple_value_based_annealing_steps, static_argnums=(6,)
    )

    num_steps = 1000
    for i in range(num_steps):
        # Need to update key as well
        position, last_cost, last_value, key = jit_run_multiple_value_based_annealing_steps(
            position, key, temperature, last_cost, last_value, value_function, 1
        )
        visualize_position(
            position,
            value_function=value_function,
            temperature=temperature,
            delay_seconds=0.01,
        )
        # Decay temperature so the value function slice and acceptance probability
        # evolve over time, following the same geometric schedule used to build it.
        temperature *= value_function.cooling_rate

    visualize_position(
        position,
        value_function=value_function,
        temperature=temperature,
        delay_seconds=0.5,
    )
    close_visualization()
    print(f"final position: {position}")
    print(f"final cost: {last_cost}")
    print(f"final value: {last_value}")

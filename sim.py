import functools
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Visualization state (reused across calls)
_viz_fig = None
_viz_ax = None
_viz_energy_line = None
_viz_dot = None
_viz_x_range = (-5.0, 5.0)
_viz_y_range = None

def _init_visualization(x_min: float, x_max: float) -> None:
    global _viz_fig, _viz_ax, _viz_energy_line, _viz_dot, _viz_x_range, _viz_y_range
    plt.ion()
    _viz_x_range = (x_min, x_max)
    x = jnp.linspace(x_min, x_max, 1000)
    y = energy(x)
    x_np = np.asarray(x)
    y_np = np.asarray(y)

    _viz_fig, _viz_ax = plt.subplots(figsize=(8, 4.5))
    (_viz_energy_line,) = _viz_ax.plot(x_np, y_np, color="black", linewidth=1.5)
    (_viz_dot,) = _viz_ax.plot([], [], "o", color="green", markersize=8)
    _viz_ax.set_xlabel("position")
    _viz_ax.set_ylabel("energy")
    _viz_ax.set_title("Energy Function with Position")
    _viz_ax.set_xlim(float(x_np.min()), float(x_np.max()))
    y_margin = 0.1 * (float(y_np.max()) - float(y_np.min()))
    _viz_y_range = (float(y_np.min()) - y_margin, float(y_np.max()) + y_margin)
    _viz_ax.set_ylim(*_viz_y_range)
    _viz_fig.tight_layout()
    _viz_fig.canvas.draw_idle()
    _viz_fig.canvas.flush_events()

def energy(position: jnp.ndarray) -> jnp.ndarray:
    """
    Energy function with:
    - Broad, shallow quadratic well centered near -2
    - Narrow, deep Gaussian well centered near +2
    - Light quartic confinement to prevent runaway at extremes
    """
    left_well = 0.12 * (position + 2.0) ** 2 - 0.8
    right_well = -2.5 * jnp.exp(-0.5 * ((position - 2.0) / 0.3) ** 2)
    confinement = 0.002 * position**4
    return left_well + right_well + confinement

def run_annealing_step(
    position: jnp.ndarray,
    key: jnp.ndarray,
    temperature: jnp.ndarray,
    last_cost: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    key, subkey = jax.random.split(key)
    proposed_position = position + 0.05 * (jax.random.bernoulli(subkey, 0.5) - 0.5) * 2
    proposed_cost = energy(proposed_position)
    key, subkey = jax.random.split(key)
    accept = jax.random.bernoulli(subkey, jnp.clip(jnp.exp(-(proposed_cost - last_cost) / temperature), 0, 1))
    return jnp.where(accept, proposed_position, position), jnp.where(accept, proposed_cost, last_cost)

def run_multiple_annealing_steps(
    position: jnp.ndarray, 
    key: jnp.ndarray, 
    temperature: jnp.ndarray, 
    last_cost: jnp.ndarray, 
    num_steps: int,
    cooling_rate: float,
    samples: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    key, subkey = jax.random.split(key)
    sampled_steps = jax.random.randint(subkey, (samples,), 0, num_steps)
    sampled_positions = jnp.zeros((samples,))
    def scan_body(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        step_idx: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        _ = step_idx
        position, last_cost, temperature, key, sampled_positions = carry
        key, step_key = jax.random.split(key)
        position, last_cost = run_annealing_step(position, step_key, temperature, last_cost)
        temperature = temperature * cooling_rate
        sampled_positions = jnp.where(step_idx == sampled_steps, position, sampled_positions)
        return (position, last_cost, temperature, key, sampled_positions), None
    (final_position, final_cost, final_temperature, final_key, final_sampled_positions), _ = jax.lax.scan(
        scan_body, (position, last_cost, temperature, key, sampled_positions), jnp.arange(num_steps)
    )
    return final_sampled_positions, sampled_steps, final_position, final_cost, final_temperature, final_key

def visualize_position(position: jnp.ndarray, x_min: float = -5.0, x_max: float = 5.0, delay_seconds: float = 0.03) -> None:
    """
    Plot the energy curve and a green dot at the given position(s).
    - position can be a scalar or array-like; all points will be plotted.
    - Non-blocking: holds for delay_seconds, then returns.
    """
    global _viz_fig, _viz_ax, _viz_energy_line, _viz_dot, _viz_x_range, _viz_y_range
    # Initialize or reinitialize if x-range changed
    if _viz_fig is None or _viz_x_range != (x_min, x_max):
        _init_visualization(x_min, x_max)

    pos = jnp.asarray(position).reshape(-1)
    e_pos = energy(pos)

    pos_np = np.asarray(pos)
    e_pos_np = np.asarray(e_pos)

    # Update the dot
    _viz_dot.set_data(pos_np, e_pos_np)
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
    position = jnp.array([0.0])
    key = jax.random.PRNGKey(0)
    temperature = jnp.array([1.0])
    last_cost = energy(position)
    num_steps = 1000
    visualize_position(position, delay_seconds=0.1)
    jit_run_multiple_annealing_steps = jax.jit(run_multiple_annealing_steps, static_argnums=(4,))
    for i in range(num_steps):
        sampled_positions, sampled_steps, position, last_cost, temperature, key = jit_run_multiple_annealing_steps(position, key, temperature, last_cost, 1, cooling_rate)
        temperature = temperature * cooling_rate
        visualize_position(position, delay_seconds=0.01)
    visualize_position(position, delay_seconds=0.5)
    close_visualization()
    print(f"final position: {position}")
    print(f"final cost: {last_cost}")
    print(f"final temperature: {temperature}")
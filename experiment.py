import jax
from sim import energy, run_multiple_annealing_steps, visualize_position
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pickle

num_steps = 500
cooling_rate = 0.95
initial_temperature = 1000.0
num_simulations = 100000

parallel_sim = jax.jit(jax.vmap(run_multiple_annealing_steps, in_axes=(0, 0, None, 0, None, None, None)), static_argnums=(4, 5, 6))

positions = jax.random.uniform(jax.random.PRNGKey(0), (num_simulations,), minval=-5.0, maxval=5.0)
initial_costs = jax.vmap(energy)(positions)

keys = jax.random.split(jax.random.PRNGKey(0), num_simulations)

sampled_positions, sampled_steps, final_positions, final_costs, final_temperatures, final_keys = parallel_sim(positions, keys, initial_temperature, initial_costs, num_steps,cooling_rate, 10)

mean_final_cost = jnp.mean(final_costs)
std_final_cost = jnp.std(final_costs)

print(f"mean final cost: {mean_final_cost}")
print(f"std final cost: {std_final_cost}")
print(f"minimum final cost: {jnp.min(final_costs)}")
print(f"final temperature: {jnp.mean(final_temperatures)}")

best_position = final_positions[jnp.argmin(final_costs)]
visualize_position(best_position, delay_seconds=0.5)

def plot_value_histograms(sampled_positions, sampled_steps, final_costs):
    """
    Plot 10 bar charts showing the expected final cost as a function of position.
    Each subplot aggregates samples from a 1/10th interval of the total steps.
    For each time slice, positions are binned along x, and the bar height is the
    mean final cost of trajectories whose sampled position falls in that bin.
    """
    # Convert to NumPy for plotting
    pos = np.asarray(sampled_positions)          # shape: (num_sims, samples)
    steps = np.asarray(sampled_steps)            # shape: (num_sims, samples)
    costs = np.asarray(final_costs)              # shape: (num_sims,)

    if pos.ndim != 2 or steps.ndim != 2 or costs.ndim != 1:
        raise ValueError("Expected shapes: sampled_positions (N,S), sampled_steps (N,S), final_costs (N,)")

    num_sims, samples = pos.shape
    # Tile costs to align with sampled positions per sample
    costs_tiled = np.repeat(costs[:, None], samples, axis=1)  # (num_sims, samples)

    # Choose plotting range from data percentiles to avoid lumping far-out values
    pos_flat = pos.reshape(-1)
    lo, hi = np.percentile(pos_flat, [0.5, 99.5])
    pad = 0.05 * (hi - lo + 1e-8)
    pos_min, pos_max = float(lo - pad), float(hi + pad)

    # Create subplots: 10 panels
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    # Partition steps into 10 equal intervals
    total_steps = int(num_steps)  # use module-level num_steps
    edges = [int(i * total_steps / 10) for i in range(11)]

    # Position binning for bar plots
    num_pos_bins = 100
    pos_edges = np.linspace(pos_min, pos_max, num_pos_bins + 1)
    bin_width = pos_edges[1] - pos_edges[0]
    bin_centers = 0.5 * (pos_edges[:-1] + pos_edges[1:])

    global_y_min = np.inf
    global_y_max = -np.inf

    # Precompute original energy curve over position range
    x_curve = np.linspace(pos_min, pos_max, 1000)
    y_curve = np.asarray(energy(jnp.array(x_curve)))
    energy_min = float(np.min(y_curve))
    energy_max = float(np.max(y_curve))

    # Collect histogram slice data for pickling
    hist_slices = []

    for i in range(10):
        start, end = edges[i], edges[i + 1]
        ax = axes[i]
        # Mask samples whose sampled step falls in [start, end)
        mask = (steps >= start) & (steps < end)
        pos_i = pos[mask]                 # (K,)
        costs_i = costs_tiled[mask]       # (K,)

        # Drop samples falling outside plotting range to avoid clipping into edge bins
        in_range = (pos_i >= pos_min) & (pos_i <= pos_max)
        pos_i = pos_i[in_range]
        costs_i = costs_i[in_range]

        if pos_i.size > 0:
            # Map positions to bin indices in [0, num_pos_bins-1]
            idx = np.floor((pos_i - pos_min) / bin_width).astype(int)
            idx = np.clip(idx, 0, num_pos_bins - 1)

            counts = np.zeros(num_pos_bins, dtype=int)
            sum_costs = np.zeros(num_pos_bins, dtype=float)
            np.add.at(counts, idx, 1)
            np.add.at(sum_costs, idx, costs_i)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_costs = sum_costs / np.maximum(counts, 1)

            valid = counts > 0
            ax.bar(bin_centers[valid], mean_costs[valid], width=bin_width * 0.9, color="steelblue")

            if np.any(valid):
                local_min = float(np.nanmin(mean_costs[valid]))
                local_max = float(np.nanmax(mean_costs[valid]))
                global_y_min = min(global_y_min, local_min)
                global_y_max = max(global_y_max, local_max)

            hist_slices.append({
                "start": start,
                "end": end,
                "counts": counts,
                "sum_costs": sum_costs,
                "mean_costs": mean_costs,
            })
        else:
            # Append empty slice data to preserve ordering
            hist_slices.append({
                "start": start,
                "end": end,
                "counts": np.zeros(num_pos_bins, dtype=int),
                "sum_costs": np.zeros(num_pos_bins, dtype=float),
                "mean_costs": np.full(num_pos_bins, np.nan),
            })

        # Overlay original energy function
        ax.plot(x_curve, y_curve, color="black", linewidth=1.5, alpha=0.8)

        ax.set_title(f"steps {start}-{end}")
        ax.set_xlabel("position")
        ax.set_ylabel("final cost")
        ax.set_xlim(pos_min, pos_max)

    # Harmonize y-limits across subplots
    # Include energy curve range to ensure visibility
    if not np.isfinite(global_y_min):
        global_y_min = energy_min
    else:
        global_y_min = min(global_y_min, energy_min)
    if not np.isfinite(global_y_max):
        global_y_max = energy_max
    else:
        global_y_max = max(global_y_max, energy_max)

    y_pad = 0.05 * (global_y_max - global_y_min + 1e-8)
    y_min = global_y_min - y_pad
    y_max = global_y_max + y_pad
    for ax in axes:
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig("value_histograms.png")
    plt.close()

    # Persist histogram data for further analysis
    histogram_data = {
        "pos_edges": pos_edges,
        "bin_centers": bin_centers,
        "pos_min": pos_min,
        "pos_max": pos_max,
        "step_edges": np.array(edges),
        "slices": hist_slices,
        "energy_curve": {"x": x_curve, "y": y_curve},
        "y_limits": (y_min, y_max),
        "num_pos_bins": num_pos_bins,
    }
    with open("value_histograms.pkl", "wb") as f:
        pickle.dump(histogram_data, f, protocol=pickle.HIGHEST_PROTOCOL)

plot_value_histograms(sampled_positions, sampled_steps, final_costs)

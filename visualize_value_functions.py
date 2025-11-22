from __future__ import annotations

import jax.numpy as jnp

from value_function_experiment import (
    HistogramEnergyFunctions,
    load_histogram_energy_functions,
    visualize_position,
    close_visualization,
)


def visualize_value_function_over_temperature(
    value_function: HistogramEnergyFunctions,
    num_frames: int = 50,
    x_min: float = -5.0,
    x_max: float = 5.0,
    delay_seconds: float = 0.05,
) -> None:
    """
    Produce the same style of visualization as in `value_function_experiment.py`,
    but driven purely by the *loaded* histogram-based value function.

    - Plots the fixed underlying energy landscape.
    - Overlays the value function (blue line) for a sequence of temperatures
      following the geometric cooling schedule implied by the histogram data.
    - Keeps the position fixed at 0.0; only the value function changes.
    """
    # Derive a reasonable temperature range from the histogram metadata.
    # We interpret the last valid step index as `max_step` and map it to temperature
    # using the same schedule: T(step) = initial_temperature * cooling_rate ** step.
    max_step = int(value_function.step_edges[-1] - 1)
    steps = jnp.linspace(0, max_step, num_frames)

    # Fixed position for the green dot; matches the visualization helper
    position = jnp.array(0.0)

    for step in steps:
        temperature = float(
            value_function.initial_temperature * (value_function.cooling_rate ** step)
        )
        visualize_position(
            position,
            value_function=value_function,
            temperature=temperature,
            x_min=x_min,
            x_max=x_max,
            delay_seconds=delay_seconds,
        )

    # Hold on the final frame a bit longer
    visualize_position(
        position,
        value_function=value_function,
        temperature=temperature,
        x_min=x_min,
        x_max=x_max,
        delay_seconds=0.5,
    )


if __name__ == "__main__":
    # Load the histogram-derived value function from disk and visualize it
    value_function = load_histogram_energy_functions()
    try:
        visualize_value_function_over_temperature(value_function)
    finally:
        close_visualization()



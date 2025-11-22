import jax
from sim import energy, run_multiple_annealing_steps, visualize_position
import jax.numpy as jnp

num_steps = 1000
cooling_rate = 0.99
initial_temperature = 10.0

parallel_sim = jax.jit(jax.vmap(run_multiple_annealing_steps, in_axes=(0, 0, None, 0, None)), static_argnums=(4,))

positions = jax.random.uniform(jax.random.PRNGKey(0), (100,), minval=-5.0, maxval=5.0)
initial_costs = jax.vmap(energy)(positions)

keys = jax.random.split(jax.random.PRNGKey(0), 100)

final_positions, final_costs, final_temperatures, final_keys = parallel_sim(positions, keys, initial_temperature, initial_costs, num_steps)

mean_final_cost = jnp.mean(final_costs)
std_final_cost = jnp.std(final_costs)

print(f"mean final cost: {mean_final_cost}")
print(f"std final cost: {std_final_cost}")
print(f"minimum final cost: {jnp.min(final_costs)}")

best_position = final_positions[jnp.argmin(final_costs)]
visualize_position(best_position, delay_seconds=0.5)

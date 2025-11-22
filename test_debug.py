
from value_function_experiment import load_histogram_energy_functions
import jax.numpy as jnp
import jax
import numpy as np

def test():
    print("Loading...")
    vf = load_histogram_energy_functions()
    print("Loaded.")
    print(f"Values shape: {vf.values.shape}")
    print(f"Pos edges shape: {vf.pos_edges.shape}")
    print(f"Step edges shape: {vf.step_edges.shape}")
    
    t = 10.0
    p = 0.0
    
    print("Calculating energy (eager)...")
    e = vf.energy(t, p)
    print(f"Energy: {e}")
    
    print("JIT compiling energy...")
    jit_energy = jax.jit(vf.energy)
    e_jit = jit_energy(t, p)
    print(f"JIT Energy: {e_jit}")

    print("Testing tree flatten...")
    leaves, aux = jax.tree_util.tree_flatten(vf)
    print(f"Leaves: {[l.shape for l in leaves]}")
    print(f"Aux: {aux}")
    
    print("Reconstructing...")
    vf2 = jax.tree_util.tree_unflatten(aux, leaves)
    print("Done.")

if __name__ == "__main__":
    try:
        test()
    except Exception as e:
        print(e)


import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax, orbax
from collections import Counter
from dataclasses import dataclass
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np
import tiktoken, time, wandb

def main():
    print(jax.devices())


if __name__ == "__main__":
    main()

import os
from functools import partial
import jax
from jax import vmap, pmap
import jax.numpy as jnp
import argparse

parser = argparse.ArgumentParser(
        description='Test vmap, pmap, etc..')
parser.add_argument(
       '--logdir', type=str, default='/tmp/jaxtest',
       help='Place to log profiles.')
parser.add_argument(
       '--method', type=str, default='vmap',
       choices=['pmap', 'vmap', 'pmap-of-vmap', 'xmap', 'none'],
       help='How to parallelize the function.')
parser.add_argument(
       '--parallelism', type=int, default=1000,
       help='Number of parallel calls to make.')
parser.add_argument(
        '--num_devices', type=int, default=None,
        help='Number of XLA devices specified via "xla_force_host_platform_device_count"')
parser.add_argument(
       '--dim', type=int, default=128,
       help='Dimension of the matrix and vector.')
parser.add_argument(
       '--num_muls', type=int, default=100,
       help='Number of times to multiply the matrix and vector.')
parser.add_argument(
       '--seed', type=int, default=0,
       help='PRNG seed.')

def multiply(key, dim, num_muls):
  k1, k2 = jax.random.split(key)
  A = jax.random.uniform(k1, [dim, dim])
  b = jax.random.uniform(k2, [dim])

  def for_fn(unused_i, x):
    return A @ x

  return jax.lax.fori_loop(0, num_muls, for_fn, b)

def main():
  args = parser.parse_args()
  if args.num_devices is not None:
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={args.num_devices}"
    print(f"Set number of XLA devices to {args.num_devices}, JAX now sees {jax.local_device_count()} devices.")
  with jax.profiler.trace(args.logdir):
    k = jax.random.PRNGKey(args.seed)
    f = jax.jit(partial(multiply, dim=args.dim, num_muls=args.num_muls))
    out = None
    if args.method == 'none':
      print("Running without parallelism")
      out = f(k)
    elif args.method == 'vmap':
      print(f"Running with vmap, parallelism {args.parallelism}.")
      keys = jax.random.split(k, num=args.parallelism)
      out = vmap(f)(keys)
    elif args.method == 'pmap':
      print(f"Running with pmap, parallelism {args.parallelism}, num_devices {jax.local_device_count()}.")
      keys = jax.random.split(k, num=args.parallelism)
      out = pmap(f)(keys)
    elif args.method == 'pmap-of-vmap':
      num_devices = jax.local_device_count()
      assert args.parallelism % num_devices == 0, \
              f"Num devices {num_devices} does not evenly divide parallelism {args.parallelism}"
      num_per_device = args.parallelism // num_devices
      keys = jax.random.split(k, num=args.parallelism)
      keys = jnp.reshape(keys, [num_devices, num_per_device, -1])
      print(f"Running pmap-of-vmap, parallelism {args.parallelism}," \
            f" num_devices {jax.local_device_count()}, num_per_device {num_per_device}.")
      out = pmap(vmap(f))(keys)

    assert out is not None
    out.block_until_ready()

if __name__ == '__main__':
  main()

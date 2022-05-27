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
       choices=['pmap', 'vmap', 'xmap', 'none'],
       help='How to parallelize the function.')
parser.add_argument(
       '--parallelism', type=int, default=1000,
       help='Number of parallel calls to make.')
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
  with jax.profiler.trace(args.logdir):
    k = jax.random.PRNGKey(args.seed)
    f = jax.jit(partial(multiply, dim=args.dim, num_muls=args.num_muls))
    out = None
    if args.method == 'none':
      out = f(k)
    elif args.method == 'vmap':
      keys = jax.random.split(k, num=args.parallelism)
      out = vmap(f)(keys)
    assert out is not None
    out.block_until_ready()

if __name__ == '__main__':
  main()

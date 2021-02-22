# Locally Optimal Block Preconditioned Conjugate Gradient

Algorithms in this module are from [Duersch _et. al_](https://epubs.siam.org/doi/abs/10.1137/17M1129830).

```bibtex
@article{duersch2018robust,
  title={A robust and efficient implementation of LOBPCG},
  author={Duersch, Jed A and Shao, Meiyue and Yang, Chao and Gu, Ming},
  journal={SIAM Journal on Scientific Computing},
  volume={40},
  number={5},
  pages={C655--C676},
  year={2018},
  publisher={SIAM}
}
```

## Notes

- `A_norm` and `B_norm` from original are calculated based on `A(x0)`, but the formula used is for Gaussian distributed inputs.
- Some tests only work due to fortuitous use of seeds / parameters. I'm mostly moving away from LOBPCG, so don't expect this to get more robust any time soon.

## TODO

- Ortho implementation
  - using python control flow + masking (un-`jit`able)
  - using `jax.lax` control flow + simulated masking (just use zeros?)
- fixed number of iterations implementation?
- Basic implementation with non-None `B` and `iK`

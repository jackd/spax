# spax.linalg

Contains sparse implementations of various linear algebra algorithms. Most take as argument a:

- callable (`spax.types.ArrayFun`) implementation a matrix-vector product;
- `spax.SparseArray`; or
- `jnp.ndarray`.

Currently ipmmlemented algorithms include:

- `eigh`: symmetric hermitian eigen decomposition implementations
  - `subspace_iteration.py`: subspace iteration implementations based on Saad 2011
  - `lobpcg`: locally optimal block preconditions conjugate gradient based on Duersch 2018
- `polynomials.py`: polynomial implementations e.g. Chebyshev

## References

### Saad 2011

- [linalg.subspace_iteration](spax/linalg/subspace_iteration.py)

```bibtex
@book{saad2011numerical,
  title={Numerical methods for large eigenvalue problems: revised edition},
  author={Saad, Yousef},
  year={2011},
  publisher={SIAM}
}
```

### Duersch 2018

- [linalg.lobpcg](spax/linalg/lobpcg/basic.py)

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

# [Spax](https://github.com/jackd/spax)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Sparse classes and algorithms for Jax.

## At a Glance

- [COO, CSR, BSR, ELL SparseArrays](./spax/sparse.py) based on [Jax WIP](https://github.com/google/jax/pull/4422/).
- [ops](./spax/ops/__init__.py) for minor additional functionality.
- [linalg](./spax/linalg/README.md) module for linear algebra algorithms.

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```

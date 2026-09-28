# Code for "The AHI family of sum of squares polynomials"

Numerical experiments accompanying the paper by Shaon Naskar, Kanchan Rajwar and Sujeet Kumar Singh
(Indian Statistical Institute, Hyderabad Unit). Paper: <!-- TODO: arXiv / journal link -->.

The repository contains the code behind the numerical section: benchmarks of the factored AHI certificate (Level B)
against generic SOS, sparse SOS and DSOS on products of AHI polynomials, and the Level C example with disjoint minimizers.

## Layout

```
matlab/ahi_benchmark.m              E1, E2, E3 benchmarks (YALMIP + MOSEK)
matlab/table1/run_ahi_benchmark_fast.m  Table 1: dense SOS vs AHI Gram basis (YALMIP + SDPT3)
python/levelc_check.py              Level C hierarchy on the disjoint-minimizer pair (CVXPY + Clarabel)
python/verify_disjoint_minimizers.py  symbolic / numerical checks of that example
python/requirements.txt
results/E2_console_log.txt          console output of the E2 run used in the paper
```

## Which script produces which part of the paper

| Paper item | Source |
|---|---|
| Table 1 (AHI vs dense SOS) | `matlab/table1/run_ahi_benchmark_fast.m` (SDPT3; see the note on Table 1 below) |
| Examples 5.1--5.3, worked example (5.5.5) | not included |
| E2 tables: `dsos-infeasible`, `dsos-time`, `levelb-k-scaling` (products of binomial squares, k = 2..7) | `matlab/ahi_benchmark.m`, experiment E2; log in `results/E2_console_log.txt` |
| E3 tables: `dsos-membership`, `dsos-boundary-cost` (random overlapping products, tightness parameter theta) | `matlab/ahi_benchmark.m`, experiment E3 |
| E1 (single AHI polynomial, cost comparison with SOS / sparse SOS / DSOS) | `matlab/ahi_benchmark.m`, experiment E1 |
| Table `levelc-counterexample` (Level C with and without localizers) | `python/levelc_check.py` |
| Minimizers, Level B bound and counterexample point of the disjoint-minimizer example | `python/verify_disjoint_minimizers.py` |

## Requirements

* MATLAB R2023b, [YALMIP](https://yalmip.github.io) (release 20250626 used), [MOSEK](https://www.mosek.com) (version: <!-- TODO -->).
* SDPT3 for `matlab/table1/run_ahi_benchmark_fast.m` (Table 1 only).
* Python 3 with `pip install -r python/requirements.txt`.

## Running

MATLAB (about 20--30 min on an Intel i7 2.8 GHz / 16 GB; almost all of it is E2 at k = 4 and k = 5):

```matlab
cd matlab
ahi_benchmark        % set RUN = [E1 E2 E3] and E2_k at the top of the file to run a subset
```

It prints one table per case, saves `ahi_benchmark_results.csv` (E1/E3 rows, one per case and method) and a checkpoint
`ALL_partial.mat` after every case, and ends with the E3 summary (number of DSOS-feasible products per theta).
The random instances use `rng(1)`; exact block sizes for E1/E3 depend on the MATLAB random-number generator.

Python:

```bash
pip install -r python/requirements.txt
python python/verify_disjoint_minimizers.py     # a few seconds
python python/levelc_check.py                   # about 20 s
```

## What the benchmark measures (please read before citing the numbers)

* **Task.** Every method certifies `f >= 0` (feasibility, gamma = 0) for the polynomial or product at hand.
* **Level B** solves one small SOS problem per factor on the factor's half-support; the product certificate is the product of the
  factor certificates. It is given the factored form.
* **SOS / Sparse SOS.** YALMIP `solvesos` on the *expanded* product with Newton-polytope reduction and congruence
  block-diagonalization. `Sparse SOS` additionally sets `sos.csp = 1`. In every instance tested the two gave identical block sizes:
  the correlative sparsity graph of these products is a single clique, so the reduction comes from the Newton polytope alone.
  Term sparsity (e.g. TSSOS) was **not** tested.
* **SOS+LevelA** is a generic SDP on the Minkowski-sum basis `B_1 + ... + B_k`. It is *not* the closed-form Level A certificate
  of the paper (a symbolic product of the factor certificates, which needs no solver).
* **DSOS** is not built into YALMIP; it is implemented by hand as an LP (`Q` diagonally dominant on a monomial basis), on the basis
  returned by `solvesos` (`DSOS red.`) and on the full homogeneous basis (`DSOS full`, tiny cases only). Infeasibility of the LP is
  reported as `no cert`.
* **Times.** `total` includes YALMIP model building (about 0.2--0.5 s per row, similar for all methods); `solver` is MOSEK time only.
  For instances with `K <= 200` the solver times are milliseconds and close to timing noise.
* **"not attempted" / "skipped"** means a guard in the `cfg` block of the script skipped the method because its basis would be too
  large. It does **not** mean the method crashed with an out-of-memory error.
* **Tightness parameter (E3).** `c = sum(lambda.^2) + theta*((sum lambda)^2 - sum(lambda.^2))`: theta = 0 gives factors with all
  coefficients nonnegative, theta = 1 puts them on the AM-HM boundary. E3 uses 3 seeds per cell, which is small.

## Table 1 (dense SOS vs AHI)

`matlab/table1/run_ahi_benchmark_fast.m` builds, for each `(n, 2d)`, the AHI polynomial with `a_i = x_i^(2d/n)` and uniform weights
`lambda_i = 1/n` (degree exactly `2d`), then solves two hand-built SDPs with SDPT3: (i) the dense baseline, a Gram matrix over all
monomials of degree at most `d` with no reduction of any kind, and (ii) the AHI method, a Gram matrix indexed by half of every
exponent in the support of `p` (this equals the half-support, of size `n(n-1)+1`, for this family).

* The dense solve is attempted **only if its basis has fewer than 600 monomials** (in Table 1 only `(4,16)`). For every larger case the
  script prints `OOM` without running the solve, so those entries mean "not attempted", not a measured out-of-memory failure.
* The polynomials here (pure powers of single variables) differ from the random-exponent polynomials of experiment E1.
* Table 1 uses SDPT3; experiments E1--E3 use MOSEK.

## Level C example (Python)

`f1 = (x1^2 - x2^4)^2/49`, `f2 = (x2^2 - x1^4)^2/49` on `K = [2,3]^2` (mapped to `[-1,1]^2`): the factors have disjoint minimizers,
the Level B bound is 1 and the true value is 20736/2401 = 8.6364. Without box localizers the hierarchy cannot exceed the Level B
bound (the point `x1 = x2 = 1.7868`, outside `K`, has `f1 = f2 = 49`); with localizers it is exact, and so is plain Putinar with no
`h_l` terms. Values slightly above 1 in the no-localizer rows are interior-point solver error (status `optimal_inaccurate`) and vary
from machine to machine. This example was solved with CVXPY/Clarabel because MOSEK reported numerical problems on the unscaled
box `[2,3]^2`.

## Known limitations

* Table 1 compares against an unreduced dense baseline. Newton-reduced SOS on the Table 1 family is not benchmarked here (experiment E1
  does it, but on random-exponent AHI polynomials).
* The Level C example is a single two-variable instance; the `h_l` terms bring no advantage over plain Putinar there.
* The benchmarks certify nonnegativity of products of AHI polynomials; they do not benchmark constrained lower bounds beyond the
  Level C example.
* Results are from one machine and one seed.

## Citation and license

<!-- TODO: BibTeX entry -->
License: <!-- TODO: choose a license (e.g. MIT) and add a LICENSE file -->

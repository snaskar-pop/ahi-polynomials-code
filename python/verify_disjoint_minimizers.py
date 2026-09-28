"""
verify_disjoint_minimizers.py -- symbolic/numerical checks behind the disjoint-minimizer example of the numerical section.

  1. f1 = (x1^2 - x2^4)^2 and f2 = (x2^2 - x1^4)^2 are AHI polynomials (n = 2, lambda = (1,1), c = 4, boundary case K = -2 of Thm 2.5).
  2. On K = [2,3]^2:  inf f1 = inf f2 = 49 at (3,2) and (2,3) (different points);  Level B bound = 2401;  true inf f1*f2 = 20736 at (2,2).
  3. The point x1 = x2 = sqrt((1+sqrt(29))/2) ~ 1.7868 lies OUTSIDE K and has f1 = f2 = 49, so the hierarchy without localizers is capped
     at the Level B bound (Proposition `prop:level-c-base`).
"""
import numpy as np
import sympy as sp
from scipy.optimize import minimize, differential_evolution

x1, x2 = sp.symbols('x1 x2', real=True)
f1_expr = (x1**2 - x2**4)**2
f2_expr = (x2**2 - x1**4)**2

# 1. AHI structure: (l1 a1 + l2 a2)(l1 a2 + l2 a1) - c a1 a2  with l1 = l2 = 1, c = 4  equals (a1 - a2)^2
a1, a2, l1, l2, c = sp.symbols('a1 a2 l1 l2 c')
ahi = sp.expand(((l1*a1 + l2*a2)*(l1*a2 + l2*a1) - c*a1*a2).subs({l1: 1, l2: 1, c: 4}))
assert sp.simplify(ahi - sp.expand((a1 - a2)**2)) == 0
assert sp.simplify(sp.expand(ahi.subs({a1: x1**2, a2: x2**4})) - sp.expand(f1_expr)) == 0
print("AHI structure verified: f1 = (x1^2 - x2^4)^2 (lambda=(1,1), c=4; K = -2 = -2*lambda1*lambda2, boundary of Thm 2.5)")

f1 = sp.lambdify((x1, x2), f1_expr, 'numpy')
f2 = sp.lambdify((x1, x2), f2_expr, 'numpy')
prod = sp.lambdify((x1, x2), sp.expand(f1_expr*f2_expr), 'numpy')
bounds = [(2, 3), (2, 3)]

def box_min(fun, n=400):
    xs = np.linspace(*bounds[0], n); ys = np.linspace(*bounds[1], n)
    X, Y = np.meshgrid(xs, ys); Z = fun(X, Y)
    i = np.unravel_index(np.argmin(Z), Z.shape)
    r1 = minimize(lambda v: fun(v[0], v[1]), [X[i], Y[i]], bounds=bounds, method='L-BFGS-B')
    r2 = differential_evolution(lambda v: fun(v[0], v[1]), bounds, tol=1e-14, seed=1, polish=True, maxiter=2000)
    best = r1 if r1.fun < r2.fun else r2
    return best.x, best.fun

xa, va = box_min(f1); xb, vb = box_min(f2); xc, vc = box_min(prod)
print(f"inf_K f1 = {va:.6g} at {np.round(xa, 6)}")
print(f"inf_K f2 = {vb:.6g} at {np.round(xb, 6)}")
print(f"Level B bound (inf f1)(inf f2) = {va*vb:.6g}")
print(f"true inf_K f1*f2 = {vc:.6g} at {np.round(xc, 6)}   (gap {vc - va*vb:.6g})")

# 3. counterexample point outside K
s = (1 + np.sqrt(29))/2; xh = np.sqrt(s)
print(f"\ncounterexample point x1 = x2 = {xh:.6f}: f1 = {f1(xh, xh):.6f}, f2 = {f2(xh, xh):.6f}, f = {f1(xh, xh)*f2(xh, xh):.6f}, inside K: {2 <= xh <= 3}")

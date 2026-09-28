"""
levelc_check.py -- Level C multiplier hierarchy on a disjoint-minimizer AHI pair (numerical section, Table `levelc-counterexample`).

Instance (after multiplying by 1/49, which keeps the polynomials AHI):
    f1 = (x1^2 - x2^4)^2 / 49,   f2 = (x2^2 - x1^4)^2 / 49,   K = [2,3]^2  (mapped to y in [-1,1]^2 for conditioning)
    inf_K f1 = inf_K f2 = 1 (at (3,2) and (2,3)),  Level B bound = 1,  true f*_K = 20736/2401 = 8.6364 (at (2,2)).

Hierarchy:  f - gamma = sigma_0 + sigma_1 h1 + sigma_2 h2 + sigma_12 h1 h2 + tau_1 g1 + tau_2 g2   (all multipliers SOS / nonnegative)
    with h_l = f_l - 1 and g_j the box constraints; 'use_g = False' drops the tau terms (eq. (23) without localizers).
    sigma_12 is a nonnegative constant, sigma_1, sigma_2 have degree dsig, tau_j have degree 14 (so tau_j g_j reaches deg f = 16).

Solved with CVXPY + Clarabel.  Expected: without localizers gamma ~ 1 for every dsig (values slightly above 1 are solver error);
with localizers gamma ~ 8.6364 for every dsig, including the plain Putinar baseline (dsig = None).
"""
import sympy as sp, numpy as np, cvxpy as cp, time, sys

y1, y2 = sp.symbols('y1 y2')
x1 = sp.Rational(5,2) + sp.Rational(1,2)*y1      # K=[2,3]^2  ->  y in [-1,1]^2
x2 = sp.Rational(5,2) + sp.Rational(1,2)*y2
f1 = sp.expand((x1**2 - x2**4)**2/49)            # /49: positive multiple of AHI, inf_K f1 = 1
f2 = sp.expand((x2**2 - x1**4)**2/49)
h1, h2 = f1-1, f2-1
f  = sp.expand(f1*f2)
g1, g2 = 1-y1**2, 1-y2**2

def pd(e):
    P = sp.Poly(sp.expand(e), y1, y2)
    return {m: float(c) for m, c in P.terms()}

D = 16
mons = [(a, b) for a in range(D+1) for b in range(D+1-a)]
midx = {m: i for i, m in enumerate(mons)}
def vec(d):
    v = np.zeros(len(mons))
    for m, c in d.items(): v[midx[m]] = c
    return v
def basis(k): return [(a, b) for a in range(k+1) for b in range(k+1-a)]

def gram_map(k, mult):
    B = basis(k); n = len(B)
    M = np.zeros((len(mons), n*n))
    for j in range(n):
        for l in range(n):
            for m, c in mult.items():
                t = (B[j][0]+B[l][0]+m[0], B[j][1]+B[l][1]+m[1])
                M[midx[t], l*n+j] += c            # column-major, matches cp.vec
    return M, n

def solve(use_g, dsig, solver='CLARABEL'):
    """dsig=None: no h terms; dsig=0: constant multipliers; dsig>=2: SOS multipliers of degree dsig.
       sigma_12 is a nonneg constant.  use_g: add K-localizers tau_j*g_j (tau degree 14)."""
    gam = cp.Variable()
    terms = []; cons = []
    def add(k, mult):
        M, n = gram_map(k, mult)
        Q = cp.Variable((n, n), PSD=True)
        terms.append(M @ cp.vec(Q, order='F'))
        return Q
    add(D//2, {(0,0): 1.0})                              # sigma_0 (residual), degree 16
    if use_g:
        add(7, pd(g1)); add(7, pd(g2))                   # tau_j g_j, tau degree 14
    if dsig is not None:
        if dsig == 0:
            s1 = cp.Variable(nonneg=True); s2 = cp.Variable(nonneg=True)
            terms += [s1*vec(pd(h1)), s2*vec(pd(h2))]
        else:
            add(dsig//2, pd(h1)); add(dsig//2, pd(h2))
        s12 = cp.Variable(nonneg=True)
        terms.append(s12*vec(pd(sp.expand(h1*h2))))
    e0 = np.zeros(len(mons)); e0[midx[(0,0)]] = 1
    cons.append(vec(pd(f)) - gam*e0 - sum(terms) == 0)
    prob = cp.Problem(cp.Maximize(gam), cons)
    t = time.time()
    try:
        prob.solve(solver=solver)
    except Exception as ex:
        return None, f"error {type(ex).__name__}", time.time()-t
    return gam.value, prob.status, time.time()-t

if __name__ == "__main__":
    print("Level B bound = 1, true f*_K =", 20736/2401)
    print("\n--- paper's eq.(23) as written: NO K-localizers (predicted cap = 1) ---")
    for d in [0, 2, 4, 8]:
        v, s, t = solve(False, d)
        print(f"  dsig={d}: gamma = {v}  [{s}, {t:.1f}s]")
    print("\n--- with K-localizers ---")
    for d in [None, 0, 2, 4, 8]:
        v, s, t = solve(True, d)
        print(f"  {'plain Putinar' if d is None else 'with h, dsig=%d'%d}: gamma = {v}  [{s}, {t:.1f}s]")

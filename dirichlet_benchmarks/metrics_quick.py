"""
metrics_quick.py

Distanzmasse zwischen empirischen Stichproben, optimiert fuer den Fall
"viele Chains, verschachtelte Prefixe, gemeinsame Referenz".

Einzelwerte (abwaertskompatibel zu metrics.py):
    empirical_w2          -- exakte empirische W_2 (POT, Netzwerk-Simplex)
    w2_assignment         -- exakte empirische W_2 fuer n == m via LAP
    sliced_wasserstein    -- SW_2 via Monte-Carlo-Projektion
    mmd_biased            -- MMD_b^2 (V-Statistik)
    mmd_unbiased          -- MMD_u^2 (U-Statistik)
    gaussian_kernel       -- volle Gram-Matrix (nur fuer kleine Stichproben)

Kurven ueber geometrisches n-Gitter:
    geometric_grid        -- aequidistantes Gitter in log(n)
    prepare_reference     -- Referenz permutieren, 1x pro Verteilung
    make_directions       -- SW-Richtungen, 1x pro Verteilung (common rn)
    mmd_reference_terms   -- MMD-Referenzterme, 1x pro Verteilung
    w2_curve / sw_curve / mmd_curve
    w2_curves_batch       -- W2-Kurven parallel, RAM-bewusst geplant

Hilfsmittel:
    idata_to_chains       -- ArviZ-idata -> dict chain -> (draw, d)
    calibrate             -- Laufzeit-Extrapolation vor dem Vollauf

Referenzen:
    Peyre & Cuturi, Computational Optimal Transport, arXiv:1803.00567
    Crouse 2016, doi:10.1109/TAES.2016.140952        (LAP-Loeser in SciPy)
    Bonneel et al. 2011, doi:10.1145/2024156.2024192 (Lemon-Simplex in POT)
    Muller 1959, doi:10.1145/377939.377946           (uniform auf S^{d-1})
    Gretton et al. 2012, JMLR 13:723-773             (MMD-Schaetzer)
"""
import os
import time

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from scipy.linalg import solve_triangular

__all__ = [
    # W2
    "empirical_w2", "w2_assignment", "w2_curve", "w2_curves_batch",
    # SW
    "sliced_wasserstein", "sw_curve", "make_directions",
    # MMD
    "gaussian_kernel", "whiten", "mmd_biased", "mmd_unbiased",
    "mmd_curve", "mmd_reference_terms",
    "analytic_bandwidth", "kernel_health",
    # Infrastruktur
    "geometric_grid", "prepare_reference", "idata_to_chains", "calibrate",
    "idata_to_array", "evaluate_distribution",
]

# ==========================================================================
# Kostenmatrix
# ==========================================================================
def _sqdist(X, Y):
    """Quadrierte euklidische Distanzen via GEMM. Identisch zu ot.dist(X, Y)."""
    xs = np.einsum('ij,ij->i', X, X)
    ys = np.einsum('ij,ij->i', Y, Y)
    M = X @ Y.T
    M *= -2.0
    M += xs[:, None]
    M += ys[None, :]
    np.maximum(M, 0.0, out=M)
    return M


# ==========================================================================
# Gitter, Referenz, idata
# ==========================================================================
def geometric_grid(n_max, n_min=250, ratio=2.0):
    """Aufsteigendes geometrisches Gitter, aequidistant in log(n).

    Bei Solver-Aufwand ~ n^3 kostet die gesamte Kurve nur den Faktor
    1/(1-ratio^-3) ~ 1.14 des obersten Punktes.
    """
    ns, n = [], int(n_max)
    while n >= n_min:
        ns.append(n)
        n = int(round(n / ratio))
    return np.array(sorted(set(ns)), dtype=int)


def prepare_reference(Y_gt, n_max, seed=0):
    """Ground truth einmal permutieren und auf n_max kuerzen.

    Pro Verteilung EINMAL aufrufen und fuer alle Chains und die iid-Baseline
    verwenden (common random numbers): die Referenzfehler sind dann ueber die
    Sampler hinweg identisch und fallen bei Differenzen weitgehend heraus.
    Die Permutation macht jedes Prefix Y[:n] selbst wieder iid.
    """
    Y = np.ascontiguousarray(Y_gt, dtype=np.float64)
    perm = np.random.default_rng(seed).permutation(len(Y))
    return np.ascontiguousarray(Y[perm[:n_max]])


def idata_to_chains(idata, var_names=None, group="posterior"):
    """ArviZ-idata -> dict chain_index -> (n_draws, d), Draw-Reihenfolge erhalten.

    Warmup muss bereits abgetrennt sein (liegt in idata.warmup_posterior).
    """
    ds = getattr(idata, group)
    if var_names is not None:
        ds = ds[var_names]
    stacked = ds.to_stacked_array("__d__", sample_dims=("chain", "draw"))
    arr = np.asarray(stacked)                      # (chain, draw, d)
    return {c: np.ascontiguousarray(arr[c], dtype=np.float64)
            for c in range(arr.shape[0])}


def idata_to_array(idata, var_names=None, group="posterior", flatten=True):
    """ArviZ-idata -> (chain*draw, d) bei flatten=True, sonst (chain, draw, d).

    Die Spaltenreihenfolge kommt aus to_stacked_array und haengt an der
    Variablenreihenfolge im Dataset. Ueber alle idata-Objekte einer
    Verteilung muss sie identisch sein -- var_names explizit setzen, wenn
    die Objekte aus verschiedenen Quellen stammen.
    """
    ds = getattr(idata, group)
    if var_names is not None:
        ds = ds[list(var_names)]
    arr = np.asarray(ds.to_stacked_array("__d__", sample_dims=("chain", "draw")))
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr.reshape(-1, arr.shape[-1]) if flatten else arr


# ==========================================================================
# W2 -- Einzelwerte
# ==========================================================================
def empirical_w2(X, Y, num_iter_max=10_000_000):
    """Exakte empirische W2 via POT (Netzwerk-Simplex).

    Allgemeiner Fall n != m. Bei n == m ist w2_assignment vorzuziehen.
    num_iter_max: Iterationsschranke. Wird sie erreicht, warnt POT und der
        Rueckgabewert ist nur eine obere Schranke -- Warnung nicht unterdruecken.
    Rueckgabe: W_2 (nicht W_2^2)
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    a = np.full(len(X), 1.0 / len(X))
    b = np.full(len(Y), 1.0 / len(Y))
    M = _sqdist(X, Y)
    return float(np.sqrt(ot.emd2(a, b, M, numItermax=num_iter_max)))


def w2_assignment(X, Y):
    """Exakte empirische W2 fuer n == m und uniforme Gewichte.

    Bei n == m und uniformen Gewichten ist der zulaessige Bereich das mit 1/n
    skalierte Birkhoff-Polytop, dessen Ecken die skalierten
    Permutationsmatrizen sind; die optimale Kopplung ist also eine Permutation
    (Peyre & Cuturi, arXiv:1803.00567, Prop. 2.1). Loesbar per LAP
    (scipy.optimize.linear_sum_assignment) oder per Netzwerk-Simplex
    (POT/Lemon, Bonneel et al. 2011, doi:10.1145/2024156.2024192) --
    identisches Ergebnis.

    Empirisch (d=2, Dirichlet-Posterior, n=4000): Simplex ~9x schneller als
    LAP, Skalierungsexponent 2.33 gegen 3.04. Der Simplex nutzt die
    geometrische Struktur niedrigdimensionaler Punktwolken aus. In hohen
    Dimensionen kann sich das umkehren -- solver="lap" ist dann die
    Alternative.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y, dtype=np.float64)
    n = len(X)
    if len(Y) != n:
        raise ValueError(f"w2_assignment verlangt n == m, hat {n} und {len(Y)}")
    a = np.full(n, 1.0 / n)
    M = _sqdist(X, Y)
    return float(np.sqrt(ot.emd2(a, a, M, numItermax=10_000_000)))


# ==========================================================================
# W2 -- Kurven
# ==========================================================================
def w2_curve(X, Y_ref, ns=None, n_min=250, ratio=2.0, n_blocks=1):
    """W2 der Prefixe X[:n] gegen Y_ref[:n] auf geometrischem Gitter.

    X        : (N, d) Chain-Samples, chronologisch (Prefix = Konvergenzverlauf)
    Y_ref    : (M, d) Referenz, vorher via prepare_reference permutiert
    n_blocks : > 1 mittelt ueber disjunkte Bloecke X[r*n:(r+1)*n] und liefert
               einen Standardfehler. Nur fuer stationaere Ketten sinnvoll --
               fuer die Konvergenzkurve n_blocks=1 lassen.

    Rueckgabe: dict mit ns, w2, se, n_blocks
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = np.ascontiguousarray(Y_ref, dtype=np.float64)
    n_max = min(len(X), len(Y))
    if ns is None:
        ns = geometric_grid(n_max, n_min, ratio)
    ns = np.asarray(sorted(n for n in ns if n <= n_max), dtype=int)

    w2 = np.empty(len(ns))
    se = np.zeros(len(ns))
    nb = np.ones(len(ns), dtype=int)

    for k, n in enumerate(ns):
        r_max = min(n_blocks, len(X) // n) if n_blocks > 1 else 1
        vals = [w2_assignment(X[r * n:(r + 1) * n], Y[:n]) for r in range(r_max)]
        w2[k] = float(np.mean(vals))
        nb[k] = r_max
        if r_max > 1:
            se[k] = float(np.std(vals, ddof=1) / np.sqrt(r_max))

    return {"ns": ns, "w2": w2, "se": se, "n_blocks": nb}


def _total_ram_gb():
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30
    except (ValueError, OSError, AttributeError):
        return 8.0


def w2_curves_batch(tasks, Y_ref, ns=None, n_min=250, ratio=2.0,
                    mem_budget_gb=None, n_cpu=None, backend="loky",
                    verbose=True):
    """Viele W2-Kurven gegen dieselbe Referenz auswerten.

    Geplant wird ueber die einzelnen (Kurve, n)-Jobs, absteigend nach n, mit
    einer von n abhaengigen Worker-Zahl: die Kostenmatrix belegt 8*n^2 Byte,
    das limitiert bei grossem n frueher als die Kernzahl.

    tasks : dict label -> (N, d) Array
    Rueckgabe: dict label -> Ergebnis wie w2_curve
    """
    from joblib import Parallel, delayed

    Y = np.ascontiguousarray(Y_ref, dtype=np.float64)
    n_cpu = n_cpu or os.cpu_count()
    if mem_budget_gb is None:
        mem_budget_gb = 0.5 * _total_ram_gb()

    prepared = {k: np.ascontiguousarray(v, dtype=np.float64)
                for k, v in tasks.items()}
    n_max = min(min(len(v) for v in prepared.values()), len(Y))
    if ns is None:
        ns = geometric_grid(n_max, n_min, ratio)
    ns = np.asarray(sorted(n for n in ns if n <= n_max), dtype=int)

    out = {k: {"ns": ns, "w2": np.full(len(ns), np.nan),
               "se": np.zeros(len(ns)), "n_blocks": np.ones(len(ns), int)}
           for k in prepared}

    labels = list(prepared)
    for k in range(len(ns) - 1, -1, -1):     # teuerste Stufe zuerst
        n = int(ns[k])
        gb = 8.0 * n * n / 2**30
        n_jobs = int(np.clip(mem_budget_gb // max(gb, 1e-9), 1, n_cpu))
        if verbose:
            print(f"n={n:6d}  Matrix {gb:5.2f} GB  -> {n_jobs} Worker  "
                  f"({len(labels)} Kurven)", flush=True)
        vals = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
            delayed(w2_assignment)(prepared[lab][:n], Y[:n]) for lab in labels)
        for lab, v in zip(labels, vals):
            out[lab]["w2"][k] = v
    return out


# ==========================================================================
# Sliced Wasserstein
# ==========================================================================
def make_directions(d, n_projections=500, seed=0, dtype=np.float32):
    """Uniform verteilte Richtungen auf S^{d-1} (Muller 1959).

    Pro Verteilung EINMAL ziehen und fuer alle Chains und alle Gitterpunkte
    wiederverwenden -- der Monte-Carlo-Fehler ist dann ueber die Sampler
    hinweg identisch (common random numbers).
    """
    rng = np.random.default_rng(seed)
    th = rng.standard_normal((n_projections, d))
    th /= np.linalg.norm(th, axis=1, keepdims=True)
    return np.ascontiguousarray(th, dtype=dtype)


def sliced_wasserstein(X, Y, n_projections=2000, seed=None,
                       dtype=np.float32, block=64, thetas=None):
    """SW_2 via Monte Carlo (Peyre & Cuturi, arXiv:1803.00567, Def. 10.13).

    Blockweise ueber die Richtungen, damit nie eine volle (L, n)-Matrix im
    Speicher steht. thetas optional vorgegeben (s. make_directions).
    Rueckgabe: SW_2 (nicht SW_2^2)
    """
    X = np.ascontiguousarray(X, dtype=dtype)
    Y = np.ascontiguousarray(Y, dtype=dtype)
    n, d = X.shape
    m = Y.shape[0]
    if Y.shape[1] != d:
        raise ValueError("X und Y muessen gleiche Dimension haben.")

    if thetas is None:
        thetas = make_directions(d, n_projections, seed, dtype)
    thetas = np.ascontiguousarray(thetas, dtype=dtype)
    L = len(thetas)

    XT = np.ascontiguousarray(X.T)          # C-contiguous -> schneller Sort
    YT = np.ascontiguousarray(Y.T)
    if n != m:
        k = max(n, m)
        quantiles = np.linspace(0.0, 1.0, k + 2)[1:-1]

    parts = []
    for i in range(0, L, block):
        th = thetas[i:i + block]
        Xp = th @ XT
        Yp = th @ YT
        if n == m:
            Xp.sort(axis=1)
            Yp.sort(axis=1)
            Xp -= Yp
            Xp *= Xp
            parts.append(Xp.mean(axis=1, dtype=np.float64))
        else:
            xq = np.quantile(Xp, quantiles, axis=1)
            yq = np.quantile(Yp, quantiles, axis=1)
            xq -= yq
            xq *= xq
            parts.append(xq.mean(axis=0, dtype=np.float64))
    return float(np.sqrt(max(np.concatenate(parts).mean(), 0.0)))


def sw_curve(X, Y_ref, ns, thetas, dtype=np.float32, block=64):
    """SW_2(X[:n], Y_ref[:n]) auf dem Gitter ns.

    Projektion wird einmal bei n_max gerechnet, danach nur noch die Prefixe
    sortiert. Da Chain und Referenz dasselbe Prefix bekommen, ist stets
    n == m -- der teure Quantil-Pfad entfaellt.
    """
    X = np.ascontiguousarray(X, dtype=dtype)
    Y = np.ascontiguousarray(Y_ref, dtype=dtype)
    ns = np.asarray(sorted(ns), dtype=int)
    N = int(ns[-1])
    if len(X) < N or len(Y) < N:
        raise ValueError("X und Y_ref muessen mindestens max(ns) Zeilen haben.")
    L = len(thetas)
    thetas = np.ascontiguousarray(thetas, dtype=dtype)

    XT = np.ascontiguousarray(X[:N].T)
    YT = np.ascontiguousarray(Y[:N].T)

    acc = np.zeros(len(ns))
    for i in range(0, L, block):
        th = thetas[i:i + block]
        Xp = th @ XT
        Yp = th @ YT
        for k, n in enumerate(ns):
            xs = np.sort(Xp[:, :n], axis=1)
            ys = np.sort(Yp[:, :n], axis=1)
            xs -= ys
            xs *= xs
            acc[k] += float(xs.mean(axis=1, dtype=np.float64).sum())
    return {"ns": ns, "sw": np.sqrt(np.maximum(acc / L, 0.0))}


# ==========================================================================
# MMD
# ==========================================================================
def whiten(X, sigma_chol):
    """X Sigma^{-1/2} via Dreiecksloesung; sigma_chol = untere Cholesky von Sigma."""
    return np.ascontiguousarray(
        solve_triangular(sigma_chol, np.asarray(X, dtype=float).T, lower=True).T)


def gaussian_kernel(X, Y, sigma):
    """Volle Gram-Matrix K[i,j] = exp(-0.5 (x_i-y_j)^T Sigma^{-1} (x_i-y_j)).

    Nur fuer kleine Stichproben -- fuer MMD stattdessen mmd_* / mmd_curve
    verwenden, die blockweise akkumulieren und keine (n, m)-Matrix anlegen.
    """
    L = np.linalg.cholesky(np.asarray(sigma, dtype=float))
    D = _sqdist(whiten(X, L), whiten(Y, L))
    D *= -0.5
    return np.exp(D, out=D)


def _kernel_sum(A, B, block=4096, dtype=np.float64):
    """sum_{i,j} exp(-0.5 ||a_i - b_j||^2), Akkumulation in float64.

    dtype=np.float32 halbiert die Bandbreite, die GEMM-Form der quadrierten
    Distanzen leidet dann aber unter Ausloeschung, wenn ||a||^2 gross gegen
    ||a-b||^2 ist. Die Zentrierung in _mmd / mmd_curve daempft das; bei
    Zweifeln float64 lassen.
    """
    A = np.ascontiguousarray(A, dtype=dtype)
    B = np.ascontiguousarray(B, dtype=dtype)
    BT = np.ascontiguousarray(B.T)
    b_sq = np.einsum('ij,ij->i', B, B)
    total = 0.0
    for i in range(0, len(A), block):
        Ai = A[i:i + block]
        a_sq = np.einsum('ij,ij->i', Ai, Ai)
        D = Ai @ BT
        D *= -2.0
        D += a_sq[:, None]
        D += b_sq[None, :]
        np.maximum(D, 0.0, out=D)
        D *= -0.5
        np.exp(D, out=D)
        total += float(D.sum(dtype=np.float64))
    return total


def _mmd(X, Y, sigma, unbiased, s_yy=None, block=4096):
    L = np.linalg.cholesky(np.asarray(sigma, dtype=float))
    Xw, Yw = whiten(X, L), whiten(Y, L)
    mu = Yw.mean(axis=0)          # verschiebt Distanzen nicht, daempft aber
    Xw = np.ascontiguousarray(Xw - mu)   # Ausloeschung in der GEMM-Form
    Yw = np.ascontiguousarray(Yw - mu)
    n, m = len(Xw), len(Yw)
    s_xx = _kernel_sum(Xw, Xw, block)
    if s_yy is None:
        s_yy = _kernel_sum(Yw, Yw, block)
    s_xy = _kernel_sum(Xw, Yw, block)
    if unbiased:
        # k(x,x) = 1 -> Diagonale analytisch abziehen
        return ((s_xx - n) / (n * (n - 1)) + (s_yy - m) / (m * (m - 1))
                - 2.0 * s_xy / (n * m))
    return s_xx / n**2 + s_yy / m**2 - 2.0 * s_xy / (n * m)


def mmd_biased(X, Y, sigma, s_yy=None, block=4096):
    """MMD_b^2 (V-Statistik). s_yy einmal berechnen und ueber Chains cachen."""
    return _mmd(X, Y, sigma, unbiased=False, s_yy=s_yy, block=block)


def mmd_unbiased(X, Y, sigma, s_yy=None, block=4096):
    """MMD_u^2 (U-Statistik), kann negativ werden."""
    return _mmd(X, Y, sigma, unbiased=True, s_yy=s_yy, block=block)


def _kernel_block_sums(A, B, ns, block=4096, dtype=np.float64):
    """S(n_k) = sum_{i<n_k, j<n_k} k(a_i, b_j) fuer alle k in einem Durchlauf.

    Zerlegung der Indexmenge in Segmente [n_{k-1}, n_k); jedes Segmentpaar
    gehoert fuer jedes n_k eindeutig dazu oder nicht, daher exakt via
    2D-Kumulation. Aufwand O(N^2 d) -- unabhaengig von der Zahl der
    Gitterpunkte, statt sum_k n_k^2 ~ 1.33 N^2.
    """
    ns = np.asarray(sorted(ns), dtype=int)
    starts = np.concatenate(([0], ns[:-1]))
    K = len(ns)
    seg = np.zeros((K, K))
    for p in range(K):
        Ap = A[starts[p]:ns[p]]
        if not len(Ap):
            continue
        for q in range(K):
            Bq = B[starts[q]:ns[q]]
            if not len(Bq):
                continue
            seg[p, q] = _kernel_sum(Ap, Bq, block=block, dtype=dtype)
    cum = np.cumsum(np.cumsum(seg, axis=0), axis=1)
    return np.diag(cum).copy()



def analytic_bandwidth(d):
    """Kernel-Bandbreite sigma_bw = sqrt(d).

    Unter exaktem Whitening mit der Zielkovarianz ist ||x-y||^2 ~ 2*chi^2_d
    fuer unabhaengige Ziehungen, also E||x-y||^2 = 2d. Mit sigma_bw^2 = d
    folgt E[k(x,y)] ~ exp(-1) ~ 0.37, unabhaengig von der Dimension.
    Ohne diese Skalierung waere k = exp(-d) -- bei d=100 numerisch null,
    die Gram-Matrix wird zur Identitaet und MMD_b kollabiert auf 1/n + 1/m.
    """
    return float(np.sqrt(d))


def kernel_health(Y, sigma=None, bandwidth=1.0, n_sub=1000, seed=0,
                  verbose=True):
    """Diagnose: mittlerer Kernwert zwischen unabhaengigen Punkten.

    Sollte grob in [0.05, 0.8] liegen. ~0 heisst degeneriert (Bandbreite zu
    klein), ~1 heisst zu breit (Kernel unterscheidet nichts mehr).
    """
    Y = np.asarray(Y, dtype=float)
    if sigma is not None:
        Y = whiten(Y, np.linalg.cholesky(np.asarray(sigma, float)) * bandwidth)
    rng = np.random.default_rng(seed)
    A = Y[rng.choice(len(Y), min(n_sub, len(Y)), replace=False)]
    D = _sqdist(A, A)
    off = D[~np.eye(len(A), dtype=bool)]
    med, mk = float(np.median(off)), float(np.exp(-0.5 * off).mean())
    if verbose:
        print(f"  med ||x-y||^2 = {med:8.2f}   mean k = {mk:.3e}")
    return {"median_sqdist": med, "mean_kernel": mk}

def mmd_reference_terms(Y_ref, sigma, ns, block=4096, bandwidth=None):
    """Praeprozessierung PRO VERTEILUNG, einmal fuer alle Sample-Sets.

    sigma     : Zielkovarianz (bei dir analytisch). Legt die FORM des Kernels
                fest -- welche Richtungen wie stark zaehlen.
    bandwidth : SKALA des Kernels, zweiter unabhaengiger Freiheitsgrad.
                None -> sqrt(d) via analytic_bandwidth. Muss ueber alle
                Sample-Sets einer Verteilung identisch sein, sonst werden
                verschiedene Metriken miteinander verglichen.

    Liefert geweisste/zentrierte Referenz und ihre Kernelsummen S_yy(n_k).
    Spart rund ein Drittel des Gesamtaufwands, weil s_yy nicht pro Chain
    neu berechnet wird.
    """
    sigma = np.asarray(sigma, dtype=float)
    if bandwidth is None:
        bandwidth = analytic_bandwidth(sigma.shape[0])
    # Bandbreite in den Cholesky-Faktor falten -- whiten teilt dann mit durch bw,
    # _kernel_sum und mmd_curve bleiben unveraendert
    L = np.linalg.cholesky(sigma) * bandwidth
    Yw = whiten(Y_ref, L)
    mu = Yw.mean(axis=0)
    Yw = np.ascontiguousarray(Yw - mu)
    ns = np.asarray(sorted(ns), dtype=int)
    return {"chol": L, "mu": mu, "Yw": Yw, "ns": ns, "bandwidth": bandwidth,
            "s_yy": _kernel_block_sums(Yw, Yw, ns, block=block)}


def mmd_curve(X, ref, unbiased=True, block=4096):
    """MMD^2(X[:n], Y_ref[:n]) auf dem Gitter. ref aus mmd_reference_terms.

    Hinweis: der unbiased Schaetzer kann negativ werden, wenn die
    Verteilungen nahe beieinander liegen. Fuer log-Plots entweder auf
    unbiased=False wechseln oder Vorzeichen explizit behandeln -- nicht
    stillschweigend clippen.
    """
    ns = ref["ns"]
    Xw = np.ascontiguousarray(whiten(X, ref["chol"]) - ref["mu"])
    if len(Xw) < int(ns[-1]):
        raise ValueError("X muss mindestens max(ns) Zeilen haben.")
    s_xx = _kernel_block_sums(Xw, Xw, ns, block=block)
    s_xy = _kernel_block_sums(Xw, ref["Yw"], ns, block=block)
    n = ns.astype(float)
    if unbiased:
        val = ((s_xx - n) / (n * (n - 1))
               + (ref["s_yy"] - n) / (n * (n - 1))
               - 2.0 * s_xy / n**2)
    else:
        val = (s_xx + ref["s_yy"] - 2.0 * s_xy) / n**2
    return {"ns": ns, "mmd2": val}


# ==========================================================================
# Kalibrierung
# ==========================================================================
def calibrate(X, Y, ns=(1000, 2000, 4000), n_curves=36, targets=(5000, 10000, 20000)):
    """Laufzeit von w2_assignment messen und auf n_max extrapolieren.

    Mit echten Daten aufrufen, nicht mit Gauss-Rauschen -- die Konstante ist
    geometrieabhaengig. Der Faktor 1.15 ist die Summe ueber das geometrische
    Gitter bei ratio=2. Extrapolation ist optimistisch, sobald die Matrix
    nicht mehr in den Cache passt: als untere Schranke lesen.
    """
    ts = []
    for n in ns:
        t0 = time.perf_counter()
        w2_assignment(X[:n], Y[:n])
        ts.append(time.perf_counter() - t0)
        print(f"n={n:6d}  {ts[-1]:8.2f}s", flush=True)

    p = float(np.polyfit(np.log(ns), np.log(ts), 1)[0])
    print(f"\nExponent p = {p:.2f}")
    for n_max in targets:
        t_max = ts[-1] * (n_max / ns[-1])**p
        print(f"  n_max={n_max:6d}:  T = {t_max:8.0f}s   "
              f"Matrix {8*n_max**2/2**30:5.2f} GB   "
              f"alle Kurven ~ {n_curves*1.15*t_max/3600:6.1f}h (seriell)")
    return p



def evaluate_distribution(gt_idata, baseline_idata, chain_idatas,
                          metrics=("w2", "sw", "mmd_b"),
                          n_min=250, ratio=2.0, n_projections=500,
                          var_names=None, sigma=None, seed=0,
                          backend="loky", verbose=True,
                          ns=None, n_cpu=None, mem_budget_gb=None):
    """Alle Kurven fuer EINE Verteilung.

    gt_idata        : idata der Ground Truth -> Referenz
    baseline_idata  : idata der iid-Baseline -> ein Sample-Set
    chain_idatas    : Liste von idata-Objekten mit je mehreren Chains
    metrics         : Teilmenge von {"w2", "sw", "mmd_b", "mmd_u"}
    sigma           : Kernel-Kovarianz; None -> aus der Ground Truth geschaetzt
    ns              : explizites n-Gitter; None -> geometric_grid(n_max, ...)
    n_cpu           : Worker-Obergrenze fuer w2_curves_batch
    mem_budget_gb   : RAM-Budget fuer w2_curves_batch

    Rueckgabe: Liste von dicts im Long-Format.
    """
    metrics = set(metrics)
    unknown = metrics - {"w2", "sw", "mmd_b", "mmd_u"}
    if unknown:
        raise ValueError(f"unbekannte Metriken: {sorted(unknown)}")

    gt = idata_to_array(gt_idata, var_names)

    tasks = {"iid_baseline": idata_to_array(baseline_idata, var_names)}
    for t, idata in enumerate(chain_idatas):
        tag = chr(ord("A") + t)
        arr = idata_to_array(idata, var_names, flatten=False)   # (chain, draw, d)
        for c in range(arr.shape[0]):
            tasks[f"{tag}_chain{c}"] = np.ascontiguousarray(arr[c])

    d = gt.shape[1]
    for lab, X in tasks.items():
        if X.shape[1] != d:
            raise ValueError(f"{lab}: Dimension {X.shape[1]} != GT {d}. "
                             "Vermutlich abweichende Variablenreihenfolge -- "
                             "var_names explizit setzen.")

    n_max = min(min(len(v) for v in tasks.values()), len(gt))
    if ns is None:
        ns = geometric_grid(n_max, n_min, ratio)
    ns = np.asarray(sorted(n for n in ns if n <= n_max), dtype=int)
    if not len(ns):
        raise ValueError(f"leeres Gitter: n_max={n_max}, n_min={n_min}")

    # Referenz nur so gross wie noetig -- spart beim Testlauf mit kleinem Gitter
    Y_ref = prepare_reference(gt, int(ns[-1]), seed=seed)
    if verbose:
        print(f"  {len(tasks)} Sample-Sets, d={d}, n_max={int(ns[-1])}, "
              f"Gitter {list(ns)}", flush=True)

    # --- nur berechnen, was angefordert ist --------------------------------
    w2 = w2_curves_batch(tasks, Y_ref, ns=ns, backend=backend, verbose=verbose,
                         n_cpu=n_cpu, mem_budget_gb=mem_budget_gb) \
        if "w2" in metrics else None

    thetas = make_directions(d, n_projections, seed=seed) \
        if "sw" in metrics else None

    ref = None
    if metrics & {"mmd_b", "mmd_u"}:
        if sigma is None:
            sigma = np.cov(gt, rowvar=False)
        ref = mmd_reference_terms(Y_ref, sigma, ns)
        if verbose:
            print(f"  MMD bandwidth = {ref['bandwidth']:.3f}", flush=True)
            kernel_health(ref["Yw"])

    rows = []
    for lab, X in tasks.items():
        out = {}
        if "sw" in metrics:
            out["sw"] = sw_curve(X, Y_ref, ns, thetas)["sw"]
        if "mmd_b" in metrics:
            out["mmd2_b"] = mmd_curve(X, ref, unbiased=False)["mmd2"]
        if "mmd_u" in metrics:
            out["mmd2_u"] = mmd_curve(X, ref, unbiased=True)["mmd2"]
        if w2 is not None:
            out["w2"] = w2[lab]["w2"]
        for k, n in enumerate(ns):
            rows.append({"sampler": lab, "n": int(n),
                         **{m: float(v[k]) for m, v in out.items()}})
    return rows
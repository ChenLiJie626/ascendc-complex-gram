#!/usr/bin/env python3
"""CPU golden/reference test for the requested AscendC complex Gram fusion op.

Requested math:
  Ar/Ai: [272, 256, 8*n]
  for each g in 0..16 and i in 0..15:
      P = A[g*16+i]^H @ A[g*16+i]
      B[g]    += norm(P)          # norm(a+bi)=a^2+b^2
      BPlu[g] += P, with lower triangle filled by conjugate of upper triangle
  B/BPlu /= 16
  Bsum = mean_g B[g]
  Csum[a,b] = mean_g B[g, a*8, b*8]
"""
import numpy as np

OUTER = 272
M = 256
GROUPS = 17
SLICES_PER_GROUP = 16


def golden(ar: np.ndarray, ai: np.ndarray, n: int):
    k = 8 * n
    assert ar.shape == (OUTER, M, k)
    assert ai.shape == (OUTER, M, k)
    B = np.zeros((GROUPS, k, k), dtype=np.float32)
    BPlur = np.zeros_like(B)
    BPlui = np.zeros_like(B)

    for g in range(GROUPS):
        for i in range(SLICES_PER_GROUP):
            s = g * SLICES_PER_GROUP + i
            A = ar[s].astype(np.float64) + 1j * ai[s].astype(np.float64)
            P = A.conj().T @ A
            # Explicitly implement "upper triangle uses original, lower triangle uses conjugate".
            # For A^H A this is mathematically Hermitian; this policy also removes small
            # numerical asymmetry if only upper triangle is trusted in the device epilogue.
            P_tri = P.copy()
            lo = np.tril_indices(k, -1)
            P_tri[lo] = np.conj(P_tri.T[lo])
            B[g] += (P_tri.real * P_tri.real + P_tri.imag * P_tri.imag).astype(np.float32)
            BPlur[g] += P_tri.real.astype(np.float32)
            BPlui[g] += P_tri.imag.astype(np.float32)
        B[g] /= SLICES_PER_GROUP
        BPlur[g] /= SLICES_PER_GROUP
        BPlui[g] /= SLICES_PER_GROUP

    Bsum = B.mean(axis=0)
    Csum = np.zeros((n, n), dtype=np.float32)
    for a in range(n):
        for b in range(n):
            Csum[a, b] = B[:, a * 8, b * 8].mean()
    return B, BPlur, BPlui, Bsum, Csum


def test_shapes_formula_and_hermitian_policy():
    rng = np.random.default_rng(20260515)
    n = 2
    k = 8 * n
    ar = rng.normal(size=(OUTER, M, k)).astype(np.float32)
    ai = rng.normal(size=(OUTER, M, k)).astype(np.float32)
    B, BPlur, BPlui, Bsum, Csum = golden(ar, ai, n)

    assert B.shape == (17, k, k)
    assert BPlur.shape == (17, k, k)
    assert BPlui.shape == (17, k, k)
    assert Bsum.shape == (k, k)
    assert Csum.shape == (n, n)

    g, row, col = 3, 5, 9
    direct_norm = direct_re = direct_im = 0.0
    for i in range(16):
        s = g * 16 + i
        re = 0.0
        im = 0.0
        for m in range(M):
            # conj(A[m,row]) * A[m,col]
            re += float(ar[s, m, row]) * float(ar[s, m, col]) + float(ai[s, m, row]) * float(ai[s, m, col])
            im += float(ar[s, m, row]) * float(ai[s, m, col]) - float(ai[s, m, row]) * float(ar[s, m, col])
        direct_re += re
        direct_im += im
        direct_norm += re * re + im * im

    assert np.allclose(B[g, row, col], direct_norm / 16, rtol=2e-4, atol=2e-3)
    assert np.allclose(BPlur[g, row, col], direct_re / 16, rtol=2e-4, atol=2e-3)
    assert np.allclose(BPlui[g, row, col], direct_im / 16, rtol=2e-4, atol=2e-3)

    assert np.allclose(BPlur, np.swapaxes(BPlur, 1, 2), rtol=1e-5, atol=1e-4)
    assert np.allclose(BPlui, -np.swapaxes(BPlui, 1, 2), rtol=1e-5, atol=1e-4)
    assert np.allclose(Csum[1, 0], B[:, 8, 0].mean(), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_shapes_formula_and_hermitian_policy()
    print("golden test passed")

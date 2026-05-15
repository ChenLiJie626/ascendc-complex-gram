#!/usr/bin/python3
# coding=utf-8

import argparse
import os

import numpy as np

GROUP_NUM = 17
AVG_NUM = 16
K_DIM = 256
USER_VEC = 8


def generate(n: int, seed: int) -> None:
    if n <= 0:
        raise ValueError("n must be greater than 0")

    rng = np.random.default_rng(seed)
    u = n * USER_VEC
    ar = rng.uniform(-0.2, 0.2, size=(GROUP_NUM, AVG_NUM, K_DIM, u)).astype(np.float16)
    ai = rng.uniform(-0.2, 0.2, size=(GROUP_NUM, AVG_NUM, K_DIM, u)).astype(np.float16)

    b = np.zeros((GROUP_NUM, u, u), dtype=np.float32)
    bplur = np.zeros((GROUP_NUM, u, u), dtype=np.float32)
    bplui = np.zeros((GROUP_NUM, u, u), dtype=np.float32)
    bsum = np.zeros((u, u), dtype=np.float32)
    csum = np.zeros((n, n), dtype=np.float32)
    lower = np.tril_indices(u, -1)

    for g in range(GROUP_NUM):
        bg = np.zeros((u, u), dtype=np.float32)
        prg = np.zeros((u, u), dtype=np.float32)
        pig = np.zeros((u, u), dtype=np.float32)
        for inner in range(AVG_NUM):
            ar_slice = ar[g, inner].astype(np.float32)
            ai_slice = ai[g, inner].astype(np.float32)
            rr = ar_slice.T @ ar_slice
            ii = ai_slice.T @ ai_slice
            ri = ar_slice.T @ ai_slice
            ir = ai_slice.T @ ar_slice
            real = (rr + ii).astype(np.float32)
            imag = (ri - ir).astype(np.float32)
            norm = (real * real + imag * imag).astype(np.float32)
            imag_plu = imag.copy()
            imag_plu[lower] *= -1.0
            bg += norm / AVG_NUM
            prg += real / AVG_NUM
            pig += imag_plu / AVG_NUM
        b[g] = bg
        bplur[g] = prg
        bplui[g] = pig
        bsum += bg / GROUP_NUM
        csum += bg[::USER_VEC, ::USER_VEC] / GROUP_NUM

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    ar.tofile("./input/ar.bin")
    ai.tofile("./input/ai.bin")
    b.tofile("./output/golden_b.bin")
    bplur.tofile("./output/golden_bplur.bin")
    bplui.tofile("./output/golden_bplui.bin")
    bsum.tofile("./output/golden_bsum.bin")
    csum.tofile("./output/golden_csum.bin")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=int(os.environ.get("USER_NUM", "1")))
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()
    generate(args.n, args.seed)


if __name__ == "__main__":
    main()

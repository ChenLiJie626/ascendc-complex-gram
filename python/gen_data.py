#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
from test_complex_gram_golden import golden, OUTER, M


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    p.add_argument('--seed', type=int, default=20260515)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    k = 8 * args.n
    rng = np.random.default_rng(args.seed)
    ar32 = rng.normal(0.0, 0.2, size=(OUTER, M, k)).astype(np.float32)
    ai32 = rng.normal(0.0, 0.2, size=(OUTER, M, k)).astype(np.float32)
    ar = ar32.astype(np.float16 if args.dtype == 'float16' else np.float32)
    ai = ai32.astype(np.float16 if args.dtype == 'float16' else np.float32)
    ar.tofile(args.out / 'ar.bin')
    ai.tofile(args.out / 'ai.bin')
    # Golden uses the actual stored dtype converted back to float32/float64.
    B, BPlur, BPlui, Bsum, Csum = golden(ar.astype(np.float32), ai.astype(np.float32), args.n)
    B.tofile(args.out / 'golden_b.bin')
    BPlur.tofile(args.out / 'golden_bplur.bin')
    BPlui.tofile(args.out / 'golden_bplui.bin')
    Bsum.tofile(args.out / 'golden_bsum.bin')
    Csum.tofile(args.out / 'golden_csum.bin')
    meta = f'n={args.n}\nK={k}\ndtype={args.dtype}\n'
    (args.out / 'meta.txt').write_text(meta)
    print(meta, end='')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

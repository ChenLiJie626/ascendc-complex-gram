#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def check(name, output_path, golden_path, shape, rtol, atol):
    out = np.fromfile(output_path, dtype=np.float32).reshape(shape)
    gold = np.fromfile(golden_path, dtype=np.float32).reshape(shape)
    diff = np.abs(out - gold)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(gold), atol)
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    ok = np.allclose(out, gold, rtol=rtol, atol=atol)
    print(f'{name:8s} max_abs={max_abs:.6g} max_rel={max_rel:.6g} {"PASS" if ok else "FAIL"}')
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--golden', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rtol', type=float, default=5e-2)
    p.add_argument('--atol', type=float, default=5e-2)
    args = p.parse_args()
    k = 8 * args.n
    oks = []
    oks.append(check('B', args.output/'b.bin', args.golden/'golden_b.bin', (17,k,k), args.rtol, args.atol))
    oks.append(check('BPlur', args.output/'bplur.bin', args.golden/'golden_bplur.bin', (17,k,k), args.rtol, args.atol))
    oks.append(check('BPlui', args.output/'bplui.bin', args.golden/'golden_bplui.bin', (17,k,k), args.rtol, args.atol))
    oks.append(check('Bsum', args.output/'bsum.bin', args.golden/'golden_bsum.bin', (k,k), args.rtol, args.atol))
    oks.append(check('Csum', args.output/'csum.bin', args.golden/'golden_csum.bin', (args.n,args.n), args.rtol, args.atol))
    if not all(oks):
        raise SystemExit(1)
    print('ALL PASS')


if __name__ == '__main__':
    main()

#!/usr/bin/python3
# coding=utf-8

import argparse
import os
import sys

import numpy as np

DEFAULT_PAIRS = (
    ("output/b.bin", "output/golden_b.bin"),
    ("output/bplur.bin", "output/golden_bplur.bin"),
    ("output/bplui.bin", "output/golden_bplui.bin"),
    ("output/bsum.bin", "output/golden_bsum.bin"),
    ("output/csum.bin", "output/golden_csum.bin"),
)


def verify_one(output_path: str, golden_path: str, rtol: float, atol: float, error_tol: float) -> bool:
    output = np.fromfile(output_path, dtype=np.float32).reshape(-1)
    golden = np.fromfile(golden_path, dtype=np.float32).reshape(-1)
    if output.shape != golden.shape:
        print(f"{output_path}: shape mismatch output={output.shape}, golden={golden.shape}")
        return False

    close = np.isclose(output, golden, rtol=rtol, atol=atol, equal_nan=True)
    diff = np.where(close == False)[0]
    for idx, real_index in enumerate(diff[:100]):
        expected = golden[real_index]
        actual = output[real_index]
        denom = max(abs(float(expected)), atol)
        print(
            "%s index: %06d, expected: %-.9f, actual: %-.9f, rdiff: %-.6f"
            % (output_path, real_index, expected, actual, abs(float(actual - expected)) / denom)
        )
        if idx == 99:
            break

    error_ratio = float(diff.size) / float(golden.size) if golden.size != 0 else 0.0
    print("%s error ratio: %.6f, tolerance: %.6f" % (output_path, error_ratio, error_tol))
    return error_ratio <= error_tol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--error-tol", type=float, default=1e-4)
    args = parser.parse_args()

    if args.paths:
        if len(args.paths) != 2:
            raise ValueError("manual mode expects: output_path golden_path")
        pairs = ((args.paths[0], args.paths[1]),)
    else:
        pairs = DEFAULT_PAIRS

    ok = True
    for output_path, golden_path in pairs:
        if not os.path.exists(output_path) or not os.path.exists(golden_path):
            print(f"missing file: {output_path} or {golden_path}")
            ok = False
            continue
        ok = verify_one(output_path, golden_path, args.rtol, args.atol, args.error_tol) and ok

    if not ok:
        raise ValueError("[ERROR] result error")
    print("test pass")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        print(err)
        sys.exit(1)

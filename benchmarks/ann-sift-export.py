#!/usr/bin/env python3
"""Export the ann-benchmarks `sift-128-euclidean` dataset to flat binary
files `crates/polypack-core/examples/ann_sift_bench.rs` can read directly,
without adding an hdf5 dependency to polypack-core for a one-off benchmark.

Usage:
    pip install h5py numpy
    curl -O http://ann-benchmarks.com/sift-128-euclidean.hdf5
    python3 benchmarks/ann-sift-export.py --hdf5 sift-128-euclidean.hdf5 --out-dir /path/to/dir [--queries 1000]

Writes, into --out-dir:
    sift_train.f32bin       (1_000_000, 128) float32, little-endian, flat
    sift_test.f32bin        (queries, 128) float32, little-endian, flat
    sift_gt_cosine.i32bin   (queries, 10) int32 — exact cosine top-10, brute
                            force over the full 1M base set
    sift_gt_euclidean.i32bin (queries, 10) int32 — the dataset's own official
                            (Euclidean) ground truth, truncated to top-10
    sift_meta.txt           "<train_count> <dims> <test_count>"

Why two ground truths: Polypack's HnswIndex historically only supported
cosine similarity (fixed in this same session to also support Euclidean —
see HnswConfig.distance). SIFT descriptor norms are nearly constant (mean
~508.66, std ~0.68 measured on this dataset), so cosine and Euclidean
rankings coincide almost exactly here (99.2% top-10 overlap, measured) —
but a fresh, exact cosine ground truth is still the "clean" recall target
if a cosine-configured index is what's under test.
"""
import argparse
import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hdf5", required=True, help="path to the downloaded sift-128-euclidean.hdf5")
    parser.add_argument("--out-dir", required=True, help="directory to write the exported binary files into")
    parser.add_argument("--queries", type=int, default=1000, help="number of test queries to export (max 10000)")
    args = parser.parse_args()

    f = h5py.File(args.hdf5, "r")
    train = f["train"][:]  # (1_000_000, 128) float32
    test = f["test"][: args.queries]  # (queries, 128) float32
    euclid_neighbors = f["neighbors"][: args.queries, :10]  # (queries, 10) int32, official ground truth

    print(f"train {train.shape} {train.dtype}, test {test.shape}", flush=True)

    train_unit = train / np.linalg.norm(train, axis=1, keepdims=True)
    test_unit = test / np.linalg.norm(test, axis=1, keepdims=True)
    batch = 200
    cosine_neighbors = np.zeros((args.queries, 10), dtype=np.int32)
    for start in range(0, args.queries, batch):
        end = min(start + batch, args.queries)
        sims = test_unit[start:end] @ train_unit.T
        cosine_neighbors[start:end] = np.argsort(-sims, axis=1)[:, :10]
        print(f"cosine ground truth {end}/{args.queries}", flush=True)

    train.astype("<f4").tofile(f"{args.out_dir}/sift_train.f32bin")
    test.astype("<f4").tofile(f"{args.out_dir}/sift_test.f32bin")
    cosine_neighbors.astype("<i4").tofile(f"{args.out_dir}/sift_gt_cosine.i32bin")
    euclid_neighbors.astype("<i4").tofile(f"{args.out_dir}/sift_gt_euclidean.i32bin")

    with open(f"{args.out_dir}/sift_meta.txt", "w") as fh:
        fh.write(f"{train.shape[0]} {train.shape[1]} {test.shape[0]}\n")

    print("done", flush=True)


if __name__ == "__main__":
    main()

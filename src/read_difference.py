"""Summarize a kl_out.csv-style file produced by three back-to-back
kl_compare.py sweeps over the same set of 5 datasets, in this fixed order:

    rows 1-5   ("full"):      base model  vs full aligned model
    rows 6-10  ("extracted"): base model  vs rank-extracted model
    rows 11-15 ("between"):   full aligned model  vs rank-extracted model

Prints a table with the extracted/full ratio per dataset, to see how much of
the full alignment's KL divergence a rank-k extraction retains.

Assumes exactly 5 datasets per block, run in the same order in each of the
three kl_compare.py invocations that produced the input CSV -- it does not
read the dataset name from each row to verify alignment, so a different
number/order of datasets will silently misalign the columns.

Usage: python src/read_difference.py [kl_out.csv]
"""

import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, default="kl_out.csv")
args = parser.parse_args()

def main():
    with open(args.input, "r") as f:
        lines = f.readlines()
        full = [float(line.split(",")[-1].strip()) for line in lines[1:6]]
        extracted = [float(line.split(",")[-1].strip()) for line in lines[6:11]]
        between = [float(line.split(",")[-1].strip()) for line in lines[11:16]]
        names = [line.split(",")[3].strip() for line in lines[1:6]]

    nparray_full = np.array(full)
    nparray_extracted = np.array(extracted)
    nparray_between = np.array(between)

    df = pd.DataFrame({
        "name": names,
        "full": nparray_full,
        "extracted": nparray_extracted,
        "between": nparray_between,
        "extracted/full": nparray_extracted / nparray_full
    })

    # for i, name in enumerate(names):
    #     print(f"{name}: {nparray_extracted[i] / nparray_full[i] }")
    print(df)

if __name__ == "__main__":
    main()
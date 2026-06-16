#!/usr/bin/env python3
"""
report24.py  --  analyse results from tb_hgh_cordic_24b_10k
Usage: python report24.py
"""
import math, os

ONE = 1048576  # 2^20

def signed24(v):
    return v - (1<<24) if v >= (1<<23) else v

def load(fname):
    with open(fname) as f:
        return [int(x) for x in f.read().split()]

def analyse(mode, inp_file, out_file, ref_fn):
    inputs  = load(inp_file)
    outputs = load(out_file)
    n = min(len(inputs), len(outputs))
    errs = []
    for i in range(n):
        q   = signed24(inputs[i]) / ONE
        got = signed24(outputs[i]) / ONE
        ref = ref_fn(q)
        err = abs(got - ref)
        errs.append(err)
    avg   = sum(errs) / len(errs)
    worst = max(errs)
    worst_q = [signed24(inputs[i])/ONE for i in range(n) if errs[i] == worst][0]
    print(f"\n{'='*45}")
    print(f"{mode}:  {n} samples")
    print(f"  Avg   error : {avg:.4e}  ({avg*ONE:.2f} LSB)")
    print(f"  Worst error : {worst:.4e}  ({worst*ONE:.2f} LSB)")
    print(f"  Worst at Q  : {worst_q:.5f}")
    ok = sum(1 for e in errs if e < 4/ONE)
    print(f"  <4 LSB      : {ok}/{n} = {100*ok/n:.2f}%")

# Vectoring: log2(Q)
analyse(
    "VECTORING  log2(Q)",
    "vec_inputs24.txt",
    "results_vec24.txt",
    lambda q: math.log2(q)
)

# Rotation: 2^Q
analyse(
    "ROTATION   2^Q",
    "rot_inputs24.txt",
    "results_rot24.txt",
    lambda q: math.pow(2, q)
)

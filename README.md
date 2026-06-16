# Variable-Radix HGH-CORDIC

**A Variable-Radix HGH-CORDIC Architecture for Fast Convergence of Exponential and Logarithmic Functions**

Open hardware (Verilog) and validation code for a **Hybrid-Radix High-Radix
Generalized Hyperbolic CORDIC** that evaluates `2^x` (rotation) and `log₂(x)`
(vectoring) using a multiplier-free, shift-and-add datapath.

The design combines **four radix-8 stages** with **two radix-4 stages** in a
single six-stage framework: the high-radix stages give rapid initial
convergence, the low-radix stages preserve late-stage refinement. The result
reaches target precision in fewer effective stages than fixed-radix
HGH-CORDIC, while staying hardware-friendly for FPGA and ASIC.

---

## Repository layout

```
.
├── Validation/                  # BERT inference validation suite
├── fig7_hybrid.py               # convergence study: radix-2/4/8 vs hybrid (8-4-2)
├── hghr24.v                     # DESIGN FILE — rotation mode, 2^Q
├── hghv24.v                     # DESIGN FILE — vectoring mode, log2(Q)
├── tb_hgh_cordic_24b_10k.v      # testbench — 10k random vectors, both modes
└── report24.py                  # post-simulation error analysis
```

- **`hghr24.v`, `hghv24.v`** — the two synthesizable design files.
- **`tb_hgh_cordic_24b_10k.v`** — testbench driving both cores over 10,000
  random vectors and dumping inputs/outputs.
- **`report24.py`** — reads the dumps and reports average / worst-case error in
  LSB and a `< 4 LSB` hit rate.
- **`Validation/`** — Python scripts that swap the CORDIC `2^x` core into BERT
  Softmax and measure downstream accuracy on NLP benchmarks.
- **`fig7_hybrid.py`** — reproduces the convergence-vs-iteration figure
  comparing the hybrid schedule against fixed-radix baselines.

---

## Architecture

Both cores share a common skeleton and number format, differing only in
update direction, digit selection, and output post-processing.

| Property | Value |
|----------|-------|
| Number format | 24-bit signed **Q4.20** (`ONE = 2²⁰ = 1,048,576`) |
| Schedule | `R = {8, 8, 8, 8, 4, 4}` (six stages) |
| Rotation domain | `Q ∈ [-0.5, 0.5]` (full range via `2^x = 2ⁿ · 2^f`) |
| Vectoring domain | `Q ∈ [0.5, 2.5]` |
| End-to-end accuracy | ~`2⁻¹⁵` after six stages |
| Rotation latency | 9 clocks (init + 6 stages + sum + scale-compensation) |
| Vectoring latency | 8 clocks (no sum / scale-compensation stage) |

**Hardware simplifications.** All coordinate updates are shift-add only (no
multipliers in the iterative stages); digit selection uses comparator
thresholds rather than division; and angle storage is compressed — the first
two stages use small LUTs for the largest hyperbolic angle constants, while
from the 3rd stage onward angles are derived by shifting a single stored anchor
constant.

**Rotation mode (`2^Q`).** Initialized `x₀ = ONE, y₀ = 0, z₀ = Q, k₀ = ONE`.
At each stage `|zᵢ|` is compared against stored thresholds to pick a digit
`dᵢ ∈ {-4,…,4}`. The accumulated hyperbolic scale factor is compensated by
propagating an inverse coefficient and applying it once at the output:
`2^Q = (x_N + y_N) · K⁻¹`.

**Vectoring mode (`log₂Q`).** Initialized `x₀ = Q + ONE, y₀ = Q − ONE, z₀ = 0`.
The recurrence drives `y → 0` while accumulating the result in `z`; the signed
digit already carries the sign of `yᵢ`, so the cross term is subtracted
consistently for `yᵢ < 0` (avoiding the double-negation error near the
midpoint of the interval). The output is recovered by a single doubling:
`log₂(Q) = 2·z_N`. No scale-factor compensation is needed, so the inverse-scale
lookup and tail multiplier are removed from this datapath.

---

## Results

### Numerical accuracy (10,000 random vectors)

| Function | Proposed (6 stages) Avg / Max | Reference [5] (8 stages) Avg / Max |
|----------|-------------------------------|------------------------------------|
| `log₂(Q)` | 1.09×10⁻⁵ / 2.24×10⁻⁵ | 6.31×10⁻⁶ / 1.45×10⁻⁵ |
| `2^Q` | 1.16×10⁻⁵ / 5.35×10⁻⁵ | 4.36×10⁻⁶ / 1.62×10⁻⁵ |

Comparable accuracy with two fewer stages.

### Post-synthesis (45 nm, 1.2 ns clock period)

| Metric | Rotation (HGHR) | Vectoring (HGHV) | Rotation [5] | Vectoring [5] |
|--------|-----------------|------------------|--------------|----------------|
| Area (µm²) | 10223 | 14162 | 8375 | 11187 |
| Power (mW) | 6.44 | 9.41 | 6.18 | 8.87 |

The design improves latency by two clock cycles over [5], benefiting overall
throughput.

### Application-level validation (BERT, FP32 `eˣ` baseline vs HGH-CORDIC)

The CORDIC `2^x` core replaces the Softmax exponential in BERT-Base/Large
**without retraining**.

| Dataset | Model | Baseline (%) | HGH-CORDIC (%) |
|---------|-------|--------------|-----------------|
| SemEval-14 | BERT-Base | 80.36 | 80.62 |
| SemEval-14 | BERT-Large | 83.12 | 82.50 |
| SWAG | BERT-Base | 81.19 | 79.87 |
| SWAG | BERT-Large | 83.68 | 82.07 |
| SQuAD | BERT-Base | 78.90 | 76.24 |
| SQuAD | BERT-Large | 81.50 | 79.46 |

Over 96% relative accuracy across tasks, exceeding 99% on classification.

---

## Getting started

### RTL simulation (Icarus Verilog)

```bash
rm -f sim24k          # remove stale artifacts to avoid false debugging
iverilog -g2012 hghv24.v hghr24.v tb_hgh_cordic_24b_10k.v -o sim24k
vvp sim24k
python report24.py
```

### Convergence figure

```bash
pip install numpy matplotlib
python fig7_hybrid.py
```

### BERT validation

The `Validation/` scripts contain a bit-exact NumPy model of the RTL `2^x` core
(`pow2_cordic_np`) that monkey-patches `F.softmax` for each evaluation phase
(ideal base-e, simple base-2 shift, and HGH-CORDIC).

```bash
pip install torch transformers datasets scikit-learn tqdm numpy
# point MODEL_PATH at your local fine-tuned BERT checkpoint, then e.g.:
python Validation/semeval_base_all3_hgh.py
python Validation/swag_eval_hgh.py
```


---

## Implementation notes

- **Signed-multiply width trap.** In Verilog the *declared* operand widths set
  the multiply evaluation width. Declare CORDIC registers and digit operands at
  full signed width and sign-extend explicitly, e.g.
  `$signed(di) * $signed({{24{op[23]}}, op})` into a `[47:0]` product.
- **Vectoring x-update sign.** `xn = x − dᵢ·|y|>>S` subtracts the magnitude; if
  the product keeps the sign of `y`, double-negation occurs for `y < 0`.
- **Range vs schedule.** Leading R8 stages fix the valid input range; ≥2 R8
  stages locks vectoring to roughly `[0.39, 2.54]`.

---

## Citation

> *A Variable-Radix HGH-CORDIC Architecture for Fast Convergence of Exponential
> and Logarithmic Functions.*

Builds on H. Chen, L. Quan, K. Chen, W. Liu, "High-radix generalized hyperbolic
CORDIC and its hardware implementation," *IEEE Transactions on Computers*,
vol. 74, no. 3, pp. 983–995, 2024 (reference [5]).

## License

All design files are made freely available for research and design use. Add
your chosen license (e.g. MIT) before publishing.

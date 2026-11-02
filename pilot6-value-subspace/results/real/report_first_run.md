# Pilot 6 results

## P1. A value circuit or a generic comparator?

| task | pairs | half-transfer layer | top-20 value writers | random 20 |
|---|---|---|---|---|
| value_test | 1440 | 28 | +0.64 [+0.63, +0.65] | +0.00 [-0.00, +0.00] |
| quantity | 216 | 28 | +0.54 [+0.53, +0.56] | -0.00 [-0.00, +0.00] |
| weight | 261 | 28 | +0.39 [+0.37, +0.40] | +0.00 [-0.00, +0.00] |

**P1: generic comparator: the non-value comparisons use the same stage and the same pieces.**

Whole-state transfer by layer:

- value_test: 4: 0.00, 8: 0.00, 12: 0.00, 16: 0.01, 20: 0.01, 24: 0.03, 28: 0.77, 32: 0.94, 36: 0.97, 40: 1.00
- quantity: 4: -0.00, 8: -0.00, 12: -0.00, 16: -0.00, 20: -0.00, 24: 0.00, 28: 0.79, 32: 0.93, 36: 0.95, 40: 1.00
- weight: 4: -0.00, 8: -0.00, 12: -0.00, 16: -0.00, 20: -0.00, 24: 0.01, 28: 0.73, 32: 0.91, 36: 0.94, 40: 1.00

## P2. How few directions carry A's value into the decision? (held-out pairs)

| layer | run | of the whole-state swap | of the donor's answer |
|---|---|---|---|
| 28 | whole-state swap itself | 1 | +0.77 |
| 28 | probe direction (k=1) | +0.01 | +0.01 |
| 28 | random direction (k=1) | -0.00 | -0.00 |
| 28 | learned, whole_k1 | +0.99 [+0.98, +0.99] | +0.77 |
| 28 | learned, whole_k2 | +0.99 [+0.99, +1.00] | +0.77 |
| 28 | learned, whole_k4 | +0.99 [+0.99, +1.00] | +0.77 |
| 28 | learned, whole_k8 | +1.01 [+1.01, +1.02] | +0.79 |
| 28 | learned, whole_k16 | +1.01 [+1.01, +1.02] | +0.79 |
| 28 | learned, donor_k1 | +1.24 [+1.23, +1.25] | +0.99 |
| 28 | learned, donor_k4 | +1.23 [+1.21, +1.24] | +0.98 |
| 32 | whole-state swap itself | 1 | +0.94 |
| 32 | probe direction (k=1) | +0.01 | +0.01 |
| 32 | random direction (k=1) | +0.00 | +0.00 |
| 32 | learned, whole_k1 | +0.99 [+0.98, +0.99] | +0.93 |
| 32 | learned, whole_k2 | +1.00 [+1.00, +1.00] | +0.94 |
| 32 | learned, whole_k4 | +0.99 [+0.99, +0.99] | +0.93 |
| 32 | learned, whole_k8 | +0.97 [+0.97, +0.97] | +0.91 |
| 32 | learned, whole_k16 | +0.98 [+0.97, +0.98] | +0.92 |
| 32 | learned, donor_k1 | +1.05 [+1.04, +1.05] | +0.99 |
| 32 | learned, donor_k4 | +1.05 [+1.04, +1.05] | +0.99 |

**P2 (layer 28, k = 1, whole-swap target): 0.99 of the whole-state swap; pass (needs 0.5).**

## P4. Is the learned direction real (not an illusion)?

Learned direction at layer 28: its coordinate correlates +0.52 with A's value on held-out prompts (probe: +0.85), and +0.67 with the difference V_A - V_B (probe: +0.60); 32.3% of it lies in the top writers' 64-dimensional output space (25.8 x a random direction; probe: 2.7 x); cosine with the probe +0.09.

**P4: pass.**

## P3. Does pushing the learned direction move the choice?

300 held-out prompts, layer 28; each push calibrated to one natural value unit. The model's own sensitivity: 3.22 logit difference per value unit.

| direction | logit difference per unit (95%) | log-odds per unit | push length |
|---|---|---|---|
| learned | -0.516 [-0.521, -0.510] | -0.516 | 1.06 |
| probe | +0.060 [+0.055, +0.064] | +0.060 | 1.53 |
| random | +0.002 [-0.003, +0.007] | +0.002 | 1.06 |

**P3: fail.**

import numpy as np
import math
import matplotlib.pyplot as plt

# ==========================================
# 1. BASELINE FUNCTIONS (From Paper)
# ==========================================

def radix2_ghv_cordic(Q, total_cycles):
    x, y, z = Q + 1.0, Q - 1.0, 0.0
    ideal_ans = math.log2(Q)
    errors = []
    i, repeated = 1, False
    for cycle in range(total_cycles):
        di = -1 if y >= 0 else 1
        x_next = x + di * y * (2**-i)
        y_next = y + di * x * (2**-i)
        z_next = z - di * (math.atanh(2**-i) / math.log(2))
        x, y, z = x_next, y_next, z_next
        err = abs(z * 2 - ideal_ans)
        errors.append(math.log10(err) if err > 1e-16 else -16)
        if i in [4, 13, 40] and not repeated: repeated = True 
        else: i += 1; repeated = False
    return errors

def high_radix_hghv_cordic(Q, r, b, total_cycles):
    x, w, z = Q + 1.0, r * (Q - 1.0), 0.0
    ideal_ans = math.log(Q, b)
    errors = []
    max_d = r // 2
    for i in range(1, total_cycles + 1):
        di = max(-max_d, min(max_d, round(w / x)))
        x_next = x - di * (r**(-2*i)) * w
        w_next = r * (w - di * x)
        z_next = z + (math.atanh(di * (r**-i)) / math.log(b)) if di != 0 else z
        x, w, z = x_next, w_next, z_next
        err = abs(z * 2 - ideal_ans)
        errors.append(math.log10(err) if err > 1e-16 else -16)
    return errors

def radix2_ghr_cordic(Q, total_cycles):
    x, y, z = 1.0, 0.0, Q
    ideal_ans = math.pow(2, Q)
    errors = []
    i, repeated = 1, False
    current_k_inv = 1.0
    for cycle in range(total_cycles):
        di = 1 if z >= 0 else -1
        x_next = x + di * y * (2**-i)
        y_next = y + di * x * (2**-i)
        z_next = z - di * (math.atanh(2**-i) / math.log(2))
        current_k_inv *= 1.0 / math.sqrt(1 - (2**-i)**2)
        x, y, z = x_next, y_next, z_next
        err = abs((x + y) * current_k_inv - ideal_ans)
        errors.append(math.log10(err) if err > 1e-16 else -16)
        if i in [4, 13, 40] and not repeated: repeated = True 
        else: i += 1; repeated = False
    return errors

def high_radix_hghr_cordic(Q, r, b, total_cycles):
    x, y, z = 1.0, 0.0, Q
    ideal_ans = math.pow(b, Q)
    errors = []
    max_d = r // 2
    current_k_inv = 1.0
    for i in range(1, total_cycles + 1):
        best_di = 0
        min_diff = float('inf')
        for d_cand in range(-max_d, max_d + 1):
            diff = abs(z - (math.atanh(d_cand * (r**-i)) / math.log(b)))
            if diff < min_diff:
                min_diff = diff
                best_di = d_cand
        di = best_di
        x_next = x + di * y * (r**-i)
        y_next = y + di * x * (r**-i)
        z_next = z - (math.atanh(di * (r**-i)) / math.log(b))
        if di != 0:
            current_k_inv *= 1.0 / math.sqrt(1 - (di * (r**-i))**2)
        x, y, z = x_next, y_next, z_next
        err = abs((x + y) * current_k_inv - ideal_ans)
        errors.append(math.log10(err) if err > 1e-16 else -16)
    return errors

# ==========================================
# 2. YOUR HYBRID APPROACH (Customizable Schedule)
# ==========================================


# 4 Stages of Radix-8, 6 Stages of Radix-4, then Radix-2 tail
RADIX_SCHEDULE = [3, 3, 3, 3, 2, 2, 2, 2, 2, 2] 

def hybrid_hghv_cordic(Q, total_cycles):
    """Vectoring (Log2): Scheduled Hybrid"""
    x, y, z = Q + 1.0, Q - 1.0, 0.0
    ideal_ans = math.log2(Q)
    errors = []
    cum_shift = 0
    
    for i in range(1, total_cycles + 1):
        # Read from the schedule, default to Radix-2 (1 bit) if we run out of schedule
        r_bits = RADIX_SCHEDULE[i-1] if i <= len(RADIX_SCHEDULE) else 1
        
        if r_bits == 3:   d_max = 4 # Radix-8
        elif r_bits == 2: d_max = 2 # Radix-4
        else:             d_max = 1 # Radix-2
        
        cum_shift += r_bits
        s = cum_shift
        
        di = round((y / x) * (2**s))
        di = max(-d_max, min(d_max, di))
        
        shift_val = di * (2.0**-s)
        x_next = x - shift_val * y
        y_next = y - shift_val * x
        
        arg = di * (2.0**-s)
        angle = (math.atanh(arg) / math.log(2)) if abs(arg) < 1 else 0
        z_next = z + angle
        
        x, y, z = x_next, y_next, z_next
        err = abs(z * 2 - ideal_ans)
        errors.append(math.log10(err) if err > 1e-16 else -16)
    return errors

def hybrid_hghr_cordic(Q, total_cycles):
    """Rotation (Exp2): Scheduled Hybrid"""
    x, y, z = 1.0, 0.0, Q
    ideal_ans = math.pow(2, Q)
    errors = []
    current_k_inv = 1.0
    cum_shift = 0
    
    for i in range(1, total_cycles + 1):
        r_bits = RADIX_SCHEDULE[i-1] if i <= len(RADIX_SCHEDULE) else 1
        
        if r_bits == 3:   d_max = 4
        elif r_bits == 2: d_max = 2
        else:             d_max = 1
        
        cum_shift += r_bits
        s = cum_shift
        
        best_di = 0
        min_diff = float('inf')
        for d_cand in range(-d_max, d_max + 1):
            arg = d_cand * (2.0**-s)
            if abs(arg) < 1:
                angle = math.atanh(arg) / math.log(2)
                diff = abs(z - angle)
                if diff < min_diff:
                    min_diff = diff
                    best_di = d_cand
                    
        di = best_di
        shift_val = di * (2.0**-s)
        x_next = x + shift_val * y
        y_next = y + shift_val * x
        
        arg = di * (2.0**-s)
        angle = (math.atanh(arg) / math.log(2)) if abs(arg) < 1 else 0
        z_next = z - angle
        
        if di != 0:
            current_k_inv *= 1.0 / math.sqrt(1 - (shift_val)**2)
            
        x, y, z = x_next, y_next, z_next
        err = abs((x + y) * current_k_inv - ideal_ans)
        errors.append(math.log10(err) if err > 1e-16 else -16)
    return errors

# ==========================================
# 3. MAIN TESTBENCH & PLOTTING
# ==========================================

def run_full_testbench():
    NUM_TESTS = 10000
    TOTAL_CYCLES = 60
    
    # 1. Generate constraint-compliant test vectors
    vec_q_vals = np.random.uniform(0.5, 3.0, NUM_TESTS) 
    rot_q_vals = np.random.uniform(-0.5, 0.5, NUM_TESTS)
    
    # Storage arrays
    errs_v = {k: np.zeros(TOTAL_CYCLES) for k in ['r2', 'r4', 'r8', 'hybrid']}
    errs_r = {k: np.zeros(TOTAL_CYCLES) for k in ['r2', 'r4', 'r8', 'hybrid']}
    
    print(f"Simulating {NUM_TESTS} Vectors for Logarithm (Vector Mode)...")
    for Q in vec_q_vals:
        errs_v['r2']     += np.array(radix2_ghv_cordic(Q, TOTAL_CYCLES))
        errs_v['r4']     += np.array(high_radix_hghv_cordic(Q, 4, 2, TOTAL_CYCLES))
        errs_v['r8']     += np.array(high_radix_hghv_cordic(Q, 8, 2, TOTAL_CYCLES))
        errs_v['hybrid'] += np.array(hybrid_hghv_cordic(Q, TOTAL_CYCLES))
        
    print(f"Simulating {NUM_TESTS} Vectors for Exponential (Rotation Mode)...")
    for Q in rot_q_vals:
        errs_r['r2']     += np.array(radix2_ghr_cordic(Q, TOTAL_CYCLES))
        errs_r['r4']     += np.array(high_radix_hghr_cordic(Q, 4, 2, TOTAL_CYCLES))
        errs_r['r8']     += np.array(high_radix_hghr_cordic(Q, 8, 2, TOTAL_CYCLES))
        errs_r['hybrid'] += np.array(hybrid_hghr_cordic(Q, TOTAL_CYCLES))

    # Average the errors
    for k in errs_v:
        errs_v[k] /= NUM_TESTS
        errs_r[k] /= NUM_TESTS

    # --- Plotting the Results ---
    cycles = np.arange(1, TOTAL_CYCLES + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))
    
    # Subplot (a): Vector Mode
    ax1.plot(cycles, errs_v['r2'], color='dimgray', label='Radix-2 Baseline')
    ax1.plot(cycles, errs_v['r4'], color='firebrick', label='Radix-4 (Paper)')
    ax1.plot(cycles, errs_v['r8'], color='mediumseagreen', label='Radix-8 (Paper)')
    ax1.plot(cycles, errs_v['hybrid'], 'b-o', markersize=4, label='Hybrid (8-4-2) [Ours]')
    ax1.axhline(y=-6, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlim(0, 40); ax1.set_ylim(-16, 1)
    ax1.set_title('(a) Vector Mode', y=-0.15)
    ax1.set_xlabel('Iteration'); ax1.set_ylabel('log10(Precision)')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Subplot (b): Rotation Mode 
    mark_spacing = 2 
    ax2.plot(cycles, errs_r['r2'], 'k-s', markevery=mark_spacing, label='Radix-2 Baseline', alpha=0.5)
    ax2.plot(cycles, errs_r['r4'], 'm-o', color='firebrick', markevery=mark_spacing, label='Radix-4 (Paper)')
    ax2.plot(cycles, errs_r['r8'], 'g-^', color='mediumseagreen', markevery=mark_spacing, label='Radix-8 (Paper)')
    ax2.plot(cycles, errs_r['hybrid'], 'b-*', markersize=6, markevery=mark_spacing, label='Hybrid (8-4-2) [Ours]')
    ax2.axhline(y=-6, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlim(0, 40); ax2.set_ylim(-16, 1)
    ax2.set_title('(b) Rotation Mode', y=-0.15)
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('log10(Precision)')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_full_testbench()
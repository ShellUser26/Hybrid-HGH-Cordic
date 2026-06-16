import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from datasets import load_dataset

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Pointing to your successfully trained BERT-Large baseline model!
MODEL_PATH = "./swag_large_e_model/best_hardware_baseline"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# ==============================================================================
# HGH-CORDIC ROTATION-MODE 2^x  —  BIT-EXACT MODEL OF hghr24.v (Q4.20)
# 6 CORDIC stages (R8x4 + R4x2), 24-bit signed wrap, arithmetic-shift product
# slices, 25-entry kinv LUT. Native domain Q in [-0.5,0.5]; full range via
# 2^x = 2^n * 2^f (n=floor(x+0.5), f in [-0.5,0.5)). In softmax the argument
# (x-max) is always <= 0, so it stays in range.
# ==============================================================================
_ONE   = 1 << 20            # 1.0 in Q4.20
_WMASK = (1 << 24) - 1
_WSIGN = 1 << 23

def _s24(v):
    v &= _WMASK
    return v - (1 << 24) if (v & _WSIGN) else v

_ANC = 2955
_KINV = {
 (4,4):1213163,(4,3):1212123,(4,2):1211382,(4,1):1210939,(4,0):1210791,
 (3,4):1133335,(3,3):1132364,(3,2):1131672,(3,1):1131257,(3,0):1131119,
 (2,4):1085086,(2,3):1084156,(2,2):1083493,(2,1):1083096,(2,0):1082964,
 (1,4):1058935,(1,3):1058028,(1,2):1057381,(1,1):1056994,(1,0):1056865,
 (0,4):1050630,(0,3):1049729,(0,2):1049088,(0,1):1048704,
}
_SCHED = [
    (3,  [(713678,4),(491380,3),(288236,2),(95045,1)], 'S1'),
    (6,  [(82817,4),(59126,3),(35464,2),(11819,1)],    'S2'),
    (9,  [(10341,4),(7386,3),(4431,2),(1477,1)],        'ANC0'),
    (12, [(1292,4),(923,3),(553,2),(184,1)],            'ANC3'),
    (14, [(138,2),(46,1)],                              'ANC5'),
    (16, [(34,2),(11,1)],                               'ANC7'),
]
_APMAP = {
    'S1': {1:190091,2:386382,3:596379,4:830977},
    'S2': {1:23639,2:47289,3:70963,4:94672},
}
for _tag, _b in (('ANC0',_ANC),('ANC3',_ANC>>3),('ANC5',_ANC>>5),('ANC7',_ANC>>7)):
    _APMAP[_tag] = {1:_b, 2:_s24(_b<<1), 3:_s24(_b+(_b<<1)), 4:_s24(_b<<2)}

def _s24v(v):
    v = v & _WMASK
    return np.where(v & _WSIGN, v - (1 << 24), v)

def _cordic_frac_vec(Qf):
    """Bit-exact 2^Q for Q in [-0.5,0.5]; Qf Q4.20 int64 ndarray -> Q4.20 int."""
    x = np.full_like(Qf, _ONE); y = np.zeros_like(Qf); z = _s24v(Qf)
    d1 = None; k = np.full_like(Qf, _ONE)
    for idx, (S, thr, tag) in enumerate(_SCHED):
        za = np.abs(z); da = np.zeros_like(z)
        for t, d in thr:
            da = np.where((da == 0) & (za >= t), d, da)
        di = np.where(z < 0, -da, da)
        ap = np.zeros_like(z)
        for d, val in _APMAP[tag].items():
            ap = np.where(da == d, val, ap)
        ang = np.where(di < 0, -ap, ap)
        xn = _s24v(x + (di * y >> S))
        yn = _s24v(y + (di * x >> S))
        zn = _s24v(z - ang)
        if tag == 'S2':
            kk = np.full_like(Qf, _ONE)
            for (a, b), val in _KINV.items():
                kk = np.where((d1 == a) & (da == b), val, kk)
            k = _s24v(kk)
        if idx == 0: d1 = da
        x, y, z = xn, yn, zn
    sr = _s24v(x + y)
    return _s24v((sr * k >> 20) & _WMASK)

def pow2_cordic_np(x):
    """Bit-exact HGH-CORDIC 2^x over a float ndarray, any range."""
    n  = np.floor(x + 0.5).astype(np.int64)
    f  = x - n
    Qf = np.clip(np.rint(f * _ONE).astype(np.int64), -(_ONE // 2), _ONE // 2)
    frac = _cordic_frac_vec(Qf).astype(np.float64) / _ONE
    return np.ldexp(frac, n)

def hgh_cordic_softmax_override(input, dim=None, _stacklevel=3, dtype=None):
    """Normalized base-2 softmax using the bit-exact HGH-CORDIC 2^x core."""
    if dim is None: dim = -1
    max_x = torch.max(input, dim=dim, keepdim=True)[0]
    arg_np = (input - max_x).detach().to(torch.float32).cpu().numpy()
    num = torch.from_numpy(pow2_cordic_np(arg_np)).to(input.device).to(input.dtype)
    den = torch.sum(num, dim=dim, keepdim=True).clamp(min=1e-9)
    return num / den

# 1. SIMPLE BASE-2 OVERRIDE (Shift Approximation)
# ==============================================================================
def base2_softmax_override(input, dim=None, _stacklevel=3, dtype=None):
    if dim is None: dim = -1
    max_x = torch.max(input, dim=dim, keepdim=True)[0]
    
    # Use 2^x instead of e^x
    num = torch.pow(2.0, input - max_x)
    den = torch.sum(num, dim=dim, keepdim=True).clamp(min=1e-9)
    return num / den

# ==============================================================================
# 2. THE 16-BIT HARDWARE DIGITAL TWIN (True RTL LNS)
# ==============================================================================
def rtl_matmul_override(tensor1, tensor2, *args, **kwargs):
    if tensor1.dim() == 4 and tensor2.dim() == 4:
        
        # 1. Is it a square matrix? (Seq_Q == Seq_K)
        is_square = (tensor1.shape[-1] == tensor1.shape[-2])
        
        # 2. THE FIX: Is it a Probability matrix?
        is_probability_matrix = (torch.min(tensor1) >= 0.0) 
        
        if is_square and (tensor1.shape[-1] == tensor2.shape[-2]) and is_probability_matrix:
            
            # --- YOUR 16-BIT MULTIPLIER-FREE RTL LOGIC ---
            P_lns = torch.log2(tensor1.clamp(min=1e-9))
            V_sign = torch.sign(tensor2)
            V_log = torch.log2(torch.abs(tensor2).clamp(min=1e-9))
            
            L_total = P_lns.unsqueeze(-1) + V_log.unsqueeze(-3)
            product_sign = V_sign.unsqueeze(-3)
            
            del P_lns, V_sign, V_log
            
            attn_out_32 = torch.pow(2.0, L_total.to(torch.float32)) * product_sign.to(torch.float32)
            del L_total, product_sign
            
            context_layer_32 = torch.sum(attn_out_32, dim=-2)
            return context_layer_32.to(torch.float16)
            
    # If it is QxK^T, or shapes don't match, do standard math
    return _orig_matmul(tensor1, tensor2, *args, **kwargs)


# ==============================================================================
# EVALUATION RUNNER
# ==============================================================================
def run_final_benchmark():
    print("\n" + "="*70)
    print("      FINAL HARDWARE BENCHMARK: SWAG (BERT-LARGE, 3-PHASE)")
    print("="*70)
    
    print(f"Loading Model: {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Base Model (Standard Math)
    model_base = AutoModelForMultipleChoice.from_pretrained(
        MODEL_PATH, 
        attn_implementation="eager"
    ).to(DEVICE)
    model_base.eval()

    # Hardware Model (Forced 16-bit memory environment)
    model_hw = AutoModelForMultipleChoice.from_pretrained(
        MODEL_PATH, 
        attn_implementation="eager",
        torch_dtype=torch.float16
    ).to(DEVICE)
    model_hw.eval()

    print("Loading SWAG Validation Dataset...")
    dataset = load_dataset("swag", "regular", split="validation")
    total_questions = len(dataset)

    # ---------------------------------------------------------
    # TEST 1: Baseline (Infinite Precision Base-e)
    # ---------------------------------------------------------
    print("\n[Phase 1/3] Testing Baseline (Infinite Precision Base-e)...")
    correct_base = 0
    with torch.no_grad():
        for i in tqdm(range(total_questions), desc="Eval Baseline"):
            item = dataset[i]
            first_sentences = [item["sent1"]] * 4
            second_sentences = [f"{item['sent2']} {item[f'ending{j}']}" for j in range(4)]
            
            inputs = tokenizer(first_sentences, second_sentences, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in inputs.items()}
            
            logits = model_base(**inputs).logits[0]
            if torch.argmax(logits, dim=-1).item() == item['label']:
                correct_base += 1
                
    acc_base = (correct_base / total_questions) * 100
    torch.cuda.empty_cache() # Clear VRAM for BERT-Large

    # ---------------------------------------------------------
    # TEST 2: Simple Base-2 (Shift Approximation)
    # ---------------------------------------------------------
    print("\n[Phase 2/3] Testing HGH-CORDIC 2^x (Bit-Exact RTL Model)...")
    
    # Override Softmax temporarily
    global _orig_softmax
    _orig_softmax = F.softmax
    F.softmax = hgh_cordic_softmax_override
    
    correct_base2 = 0
    with torch.no_grad():
        for i in tqdm(range(total_questions), desc="Eval Base-2"):
            item = dataset[i]
            first_sentences = [item["sent1"]] * 4
            second_sentences = [f"{item['sent2']} {item[f'ending{j}']}" for j in range(4)]
            
            inputs = tokenizer(first_sentences, second_sentences, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in inputs.items()}
            
            logits = model_base(**inputs).logits[0]
            if torch.argmax(logits, dim=-1).item() == item['label']:
                correct_base2 += 1
                
    acc_base2 = (correct_base2 / total_questions) * 100
    
    # Cleanup Softmax
    F.softmax = _orig_softmax
    torch.cuda.empty_cache() # Clear VRAM for BERT-Large

    # ---------------------------------------------------------
    # TEST 3: Custom Attention Unit (True 16-bit RTL)
    # ---------------------------------------------------------
    print("\n[Phase 3/3] Testing Custom Attention Unit (True 16-bit RTL)...")
    
    # Override Matmul temporarily
    global _orig_matmul
    _orig_matmul = torch.matmul
    torch.matmul = rtl_matmul_override
    
    correct_hw = 0
    with torch.no_grad():
        for i in tqdm(range(total_questions), desc="Eval True RTL"):
            item = dataset[i]
            first_sentences = [item["sent1"]] * 4
            second_sentences = [f"{item['sent2']} {item[f'ending{j}']}" for j in range(4)]
            
            inputs = tokenizer(first_sentences, second_sentences, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in inputs.items()}
            
            logits = model_hw(**inputs).logits[0]
            if torch.argmax(logits, dim=-1).item() == item['label']:
                correct_hw += 1
                
    acc_hw = (correct_hw / total_questions) * 100
    
    # Cleanup Matmul
    torch.matmul = _orig_matmul
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # FINAL RESULTS
    # ---------------------------------------------------------
    print("\n\n" + "="*70)
    print("      FINAL HARDWARE BENCHMARK RESULTS: SWAG (BERT-LARGE)")
    print("="*70)
    print(f"1. Baseline (Infinite Precision Base-e):     {acc_base:.2f}%")
    print(f"2. HGH-CORDIC 2^x (Bit-Exact RTL Model):     {acc_base2:.2f}%")
    print(f"3. Custom Attention Unit (Multiplier-Free):  {acc_hw:.2f}%")
    print("="*70)
    
    retention = (acc_hw / acc_base) * 100
    print(f"Hardware Accuracy Retention (Phase 3 vs 1):  {retention:.2f}%")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_final_benchmark()

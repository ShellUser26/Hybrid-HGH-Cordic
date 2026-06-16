import torch
import collections
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    DataCollatorWithPadding
)

# ==============================================================================
# 1. MASTER CONFIGURATION (BERT-LARGE)
# ==============================================================================
# Set to our new SQuAD Large model
ACTIVE_TASK = "squad" 

# Pointing to your trained Baseline folders
MODEL_PATHS = {
    "semeval":     "./semeval_base_e_model_bert_base/best_hardware_baseline",
    "squad":       "./squad_base_e_model/best_hardware_baseline",
    "squad_large": "./squad_large_e_model/best_hardware_baseline", # Added BERT-Large path
    "swag":        "./swag_base_e_model/best_hardware_baseline"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save PyTorch originals for resetting between phases
_orig_softmax = F.softmax 
_orig_matmul = torch.matmul

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

# 2. HARDWARE OVERRIDES (The Universal RTL Patch)
# ==============================================================================
def base2_hardware_override(input, dim=None, _stacklevel=3, dtype=None):
    if dim is None: dim = -1
    max_x = torch.max(input, dim=dim, keepdim=True)[0]
    x_shifted = input - max_x
    y = x_shifted * 1.44269504 
    y_int = torch.floor(y)
    y_frac = y - y_int
    exps = torch.pow(2.0, x_shifted) 
    return exps / torch.sum(exps, dim=dim, keepdim=True)

# ==============================================================================
# FUSED HARDWARE BUS (Simulating a direct wire between Softmax and MatMul)
# ==============================================================================
FUSED_HW_BUS = {}

# ==============================================================================
# 2. CUSTOM ATTENTION: MBS SOFTMAX (With Log-Domain Extraction)
# ==============================================================================
def mbs_hardware_override_fused(input, dim=None, _stacklevel=3, dtype=None):
    if dim is None: dim = -1
    max_x = torch.max(input, dim=dim, keepdim=True)[0]
    
    r1 = torch.ceil(max_x - input)
    r2 = torch.ceil(2.0*max_x - 2.0*input)
    frac1 = r1 + input - max_x
    l1 = torch.pow(2.0, frac1) * torch.pow(2.0, -r1) 
    frac2 = r2 + 2.0*input - 2.0*max_x
    l2 = torch.pow(2.0, frac2) * torch.pow(2.0, -r2)
    
    sum1 = torch.sum(l1, dim=dim, keepdim=True)
    sum2 = torch.sum(l2, dim=dim, keepdim=True)
    sum_N = torch.clamp(sum1 + sum2, min=1e-9)
    
    p = torch.floor(torch.log2(sum_N))
    q = sum_N * torch.pow(2.0, -p)
    l3_N = torch.log2(q) + p
    
    u1 = input - max_x - l3_N
    u2 = 2.0*input - 2.0*max_x - l3_N
    
    FUSED_HW_BUS['u1_log_P'] = u1
    
    r_u1 = torch.ceil(-u1)
    frac_u1 = r_u1 + u1 
    l4 = torch.pow(2.0, frac_u1) * torch.pow(2.0, -r_u1)
    r_u2 = torch.ceil(-u2)
    frac_u2 = r_u2 + u2 
    l5 = torch.pow(2.0, frac_u2) * torch.pow(2.0, -r_u2)
    
    return l4 + l5

# ==============================================================================
# 3. CUSTOM ATTENTION: TRUE RTL VALUE MATRIX (Fused Pipeline - BULLETPROOF)
# ==============================================================================
def rtl_matmul_override_fused(tensor1, tensor2, *args, **kwargs):
    if tensor1.dim() == 4 and tensor2.dim() == 4:
        
        is_square = (tensor1.shape[-1] == tensor1.shape[-2])
        is_probability_matrix = (torch.min(tensor1) >= 0.0) 
        
        if is_square and (tensor1.shape[-1] == tensor2.shape[-2]) and is_probability_matrix:
            
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
            
    return _orig_matmul(tensor1, tensor2, *args, **kwargs)

# ==============================================================================
# 3. TASK-SPECIFIC DATA PREP & EVALUATION LOGIC
# ==============================================================================
def get_semeval_data(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset("xinyixiuxiu/semeval2014_restaurant", split="test")
    
    label_map = {"negative": 0, "neutral": 1, "positive": 2, "0": 0, "1": 1, "2": 2}
    
    def eval_fn(model):
        correct, total = 0, 0
        for item in tqdm(dataset, desc="Eval SemEval"):
            inputs = tokenizer(str(item['text']), text_pair=str(item['aspect']), return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=-1).item()
            
            raw_label = item['label']
            if isinstance(raw_label, list): raw_label = raw_label[0]
            gold = label_map.get(str(raw_label).lower(), 1)
            
            if pred == gold: correct += 1
            total += 1
        return (correct / total) * 100
        
    return AutoModelForSequenceClassification, eval_fn

def get_squad_data(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset("squad", split="validation")
    
    def eval_fn(model):
        total_f1 = 0.0
        total = 0
        for item in tqdm(dataset, desc=f"Eval SQuAD (F1 Score)"):
            inputs = tokenizer(item["question"], item["context"], return_tensors="pt", truncation="only_second", max_length=384).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs)
                start_pred = torch.argmax(outputs.start_logits, dim=-1).item()
                end_pred = torch.argmax(outputs.end_logits, dim=-1).item()
            
            pred_answer = tokenizer.decode(inputs["input_ids"][0][start_pred:end_pred+1]).strip().lower()
            gold_answers = [ans.lower().strip() for ans in item["answers"]["text"]]
            
            max_f1 = 0.0
            for gold in gold_answers:
                gold_toks = gold.split()
                pred_toks = pred_answer.split()
                common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
                num_same = sum(common.values())
                
                if len(gold_toks) == 0 or len(pred_toks) == 0:
                    f1 = int(gold_toks == pred_toks)
                elif num_same == 0:
                    f1 = 0.0
                else:
                    precision = 1.0 * num_same / len(pred_toks)
                    recall = 1.0 * num_same / len(gold_toks)
                    f1 = (2 * precision * recall) / (precision + recall)
                
                max_f1 = max(max_f1, f1)
                
            total_f1 += max_f1
            total += 1
            
        return (total_f1 / total) * 100

    return AutoModelForQuestionAnswering, eval_fn

def get_swag_data(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = load_dataset("swag", "regular", split="validation").select(range(1000))
    
    def eval_fn(model):
        correct, total = 0, 0
        for item in tqdm(dataset, desc="Eval SWAG"):
            first_sentences = [item["sent1"]] * 4
            second_sentences = [f"{item['sent2']} {item[f'ending{i}']}" for i in range(4)]
            
            inputs = tokenizer(first_sentences, second_sentences, return_tensors="pt", truncation=True, max_length=128, padding=True)
            inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
                pred = torch.argmax(logits, dim=-1).item()
                
            if pred == item["label"]: correct += 1
            total += 1
        return (correct / total) * 100

    return AutoModelForMultipleChoice, eval_fn

# ==============================================================================
# 4. MAIN BENCHMARK RUNNER
# ==============================================================================
def run_master_benchmark():
    model_path = MODEL_PATHS[ACTIVE_TASK]
    print(f"\n========================================================")
    print(f"  INITIALIZING MASTER BENCHMARK FOR: {ACTIVE_TASK.upper()}")
    print(f"========================================================")
    
    if ACTIVE_TASK == "semeval":                      model_class, eval_fn = get_semeval_data(model_path)
    elif ACTIVE_TASK in ["squad", "squad_large"]:     model_class, eval_fn = get_squad_data(model_path)
    elif ACTIVE_TASK == "swag":                       model_class, eval_fn = get_swag_data(model_path)
    else: raise ValueError("Invalid ACTIVE_TASK configured.")

    torch.cuda.empty_cache()
    
    print(f"Loading Model Weights from: {model_path}...")
    model = model_class.from_pretrained(
        model_path, 
        attn_implementation="eager",
        torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()

    # --- PHASE 1 ---
    print("\n[Phase 1/3] Testing Ideal Float16 (Base-e)...")
    F.softmax = _orig_softmax
    torch.matmul = _orig_matmul
    acc_e = eval_fn(model)
    print(f"--> Phase 1 Score: {acc_e:.2f}%")
    torch.cuda.empty_cache() # Clear Memory for Large Model

    # --- PHASE 2 ---
    print("\n[Phase 2/3] Testing Simple Base-2 (Shift Approximation)...")
    F.softmax = base2_hardware_override
    torch.matmul = _orig_matmul
    acc_2 = eval_fn(model)
    print(f"--> Phase 2 Score: {acc_2:.2f}%")
    torch.cuda.empty_cache() # Clear Memory for Large Model

    # --- PHASE 3 ---
    print("\n[Phase 3/3] Testing HGH-CORDIC 2^x (Bit-Exact RTL Model)...")
    F.softmax = hgh_cordic_softmax_override
    torch.matmul = _orig_matmul
    acc_full = eval_fn(model)
    print(f"--> Phase 3 Score: {acc_full:.2f}%")
    
    # Clean up overrides
    F.softmax = _orig_softmax
    torch.matmul = _orig_matmul
    torch.cuda.empty_cache()

    print("\n" + "="*70)
    print(f"      FINAL HARDWARE BENCHMARK RESULTS: {ACTIVE_TASK.upper()}")
    print("="*70)
    print(f"1. Baseline (Infinite Precision Base-e):     {acc_e:.2f}%")
    print(f"2. Simple Base-2 (Shift Approximation):      {acc_2:.2f}%")
    print(f"3. HGH-CORDIC 2^x (Bit-Exact RTL Model):     {acc_full:.2f}%")
    print("="*70)

if __name__ == "__main__":
    run_master_benchmark()

# ==============================================================================
# HGH-CORDIC ROTATION-MODE 2^x  —  BIT-EXACT MODEL OF hghr24.v (Q4.20)
# Drop-in replacement for the MBS / base-2 softmax override.
#
# Mirrors the RTL exactly: 6 CORDIC stages (R8x4 + R4x2), Q4.20 fixed-point,
# 24-bit signed wrap, arithmetic-shift product slices, and the 25-entry kinv LUT.
# Native RTL domain is Q in [-0.5, 0.5]; full-range 2^x is obtained by
# range reduction 2^x = 2^n * 2^f  (n via floor(x+0.5), f in [-0.5,0.5)).
# Inside softmax the argument (x - max) is always <= 0, so it stays in range.
# ==============================================================================
import numpy as np
import torch

ONE   = 1 << 20            # 1.0 in Q4.20
WMASK = (1 << 24) - 1
WSIGN = 1 << 23

def _s24(v):               # scalar signed-24 wrap (for LUT precompute)
    v &= WMASK
    return v - (1 << 24) if (v & WSIGN) else v

_ANC = 2955
_KINV = {
 (4,4):1213163,(4,3):1212123,(4,2):1211382,(4,1):1210939,(4,0):1210791,
 (3,4):1133335,(3,3):1132364,(3,2):1131672,(3,1):1131257,(3,0):1131119,
 (2,4):1085086,(2,3):1084156,(2,2):1083493,(2,1):1083096,(2,0):1082964,
 (1,4):1058935,(1,3):1058028,(1,2):1057381,(1,1):1056994,(1,0):1056865,
 (0,4):1050630,(0,3):1049729,(0,2):1049088,(0,1):1048704,
}
# (shift S, [(threshold, digit)...] high->low, angle-table tag)
_SCHED = [
    (3,  [(713678,4),(491380,3),(288236,2),(95045,1)], 'S1'),
    (6,  [(82817,4),(59126,3),(35464,2),(11819,1)],   'S2'),
    (9,  [(10341,4),(7386,3),(4431,2),(1477,1)],       'ANC0'),
    (12, [(1292,4),(923,3),(553,2),(184,1)],           'ANC3'),
    (14, [(138,2),(46,1)],                             'ANC5'),
    (16, [(34,2),(11,1)],                              'ANC7'),
]
_APMAP = {
    'S1':   {1:190091,2:386382,3:596379,4:830977},
    'S2':   {1:23639,2:47289,3:70963,4:94672},
}
for _tag, _b in (('ANC0',_ANC),('ANC3',_ANC>>3),('ANC5',_ANC>>5),('ANC7',_ANC>>7)):
    _APMAP[_tag] = {1:_b, 2:_s24(_b<<1), 3:_s24(_b+(_b<<1)), 4:_s24(_b<<2)}


def _s24v(v):
    v = v & WMASK
    return np.where(v & WSIGN, v - (1 << 24), v)


def _cordic_frac_vec(Qf):
    """Bit-exact 2^Q for Q in [-0.5,0.5]; Qf is a Q4.20 int64 ndarray. Returns Q4.20 int."""
    x = np.full_like(Qf, ONE); y = np.zeros_like(Qf); z = _s24v(Qf)
    d1 = None; k = np.full_like(Qf, ONE)
    for idx, (S, thr, tag) in enumerate(_SCHED):
        za = np.abs(z)
        da = np.zeros_like(z)
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
            kk = np.full_like(Qf, ONE)
            for (a, b), val in _KINV.items():
                kk = np.where((d1 == a) & (da == b), val, kk)
            k = _s24v(kk)
        if idx == 0:
            d1 = da
        x, y, z = xn, yn, zn
    sr = _s24v(x + y)
    return _s24v((sr * k >> 20) & WMASK)


def pow2_cordic_np(x):
    """Bit-exact HGH-CORDIC 2^x over a float ndarray, any range."""
    n  = np.floor(x + 0.5).astype(np.int64)
    f  = x - n
    Qf = np.clip(np.rint(f * ONE).astype(np.int64), -(ONE // 2), ONE // 2)
    frac = _cordic_frac_vec(Qf).astype(np.float64) / ONE
    return np.ldexp(frac, n)


# ==============================================================================
# DROP-IN OVERRIDE  — replaces mbs_hardware_override / base2_softmax_override
# ==============================================================================
def cordic_softmax_override(input, dim=None, _stacklevel=3, dtype=None):
    """Normalized base-2 softmax using the bit-exact HGH-CORDIC 2^x rotation core."""
    if dim is None:
        dim = -1
    max_x = torch.max(input, dim=dim, keepdim=True)[0]
    x_shift = (input - max_x)                       # always <= 0  -> in CORDIC range

    arg_np = x_shift.detach().to(torch.float32).cpu().numpy()
    num_np = pow2_cordic_np(arg_np)                 # bit-exact 2^x
    num = torch.from_numpy(num_np).to(input.device).to(input.dtype)

    den = torch.sum(num, dim=dim, keepdim=True).clamp(min=1e-9)
    return num / den

import time, sys
from datetime import datetime
from typing import List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ────────────────────────────  OpenFHE (Boolean)  ────────────────────────────
from openfhe import *  # BinFHEContext, gate enums, …

# ----------------------------- Bit‑helpers -----------------------------------

def twos_complement_bits(val: int, nbits: int) -> List[bool]:
    """Return *nbits* MSB‑first Boolean list for signed *val*."""
    return [bool((val >> (nbits - 1 - i)) & 1) for i in range(nbits)]


def bits_to_int(bits: List[int]) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | b
    return out


def twos_comp_val(val: int, bits: int) -> int:
    if val & (1 << bits - 1):
        val -= 1 << bits
    return val

# --------------------- Gate‑level arithmetic (copied from LR) ----------------

def subtractBits(r, a, b, carry):
    t1 = cc.EvalBinGate(XOR, a, b)
    r[0] = cc.EvalBinGate(XOR, t1, carry)
    acomp = cc.EvalNOT(a)
    abcomp = cc.EvalNOT(t1)
    t2 = cc.EvalBinGate(AND, acomp, b)
    t3 = cc.EvalBinGate(AND, abcomp, carry)
    r[1] = cc.EvalBinGate(OR, t2, t3)
    return r


def subtractNumbers(ctA, ctB, nBits):
    ctRes = [False] * nBits
    bitResult = [False] * 2
    ctRes[0] = cc.EvalBinGate(XOR, ctA[0], ctB[0])
    t1 = cc.EvalNOT(ctA[0])
    carry = cc.EvalBinGate(AND, t1, ctB[0])
    for i in range(1, nBits):
        bitResult = subtractBits(bitResult, ctA[i], ctB[i], carry)
        ctRes[i] = bitResult[0]
        carry = bitResult[1]
    return ctRes


def approaddBits(r, a, b, carry):
    r[1] = cc.EvalBinGate(OR, a, b)
    temp = cc.EvalBinGate(AND, a, b)
    r[0] = cc.EvalBinGate(OR, carry, temp)
    return r


def addBits(r, a, b, carry):
    t1 = cc.EvalBinGate(XOR, a, b)
    r[0] = cc.EvalBinGate(XOR, t1, carry)
    t2 = cc.EvalBinGate(AND, a, carry)
    t3 = cc.EvalBinGate(AND, b, carry)
    t4 = cc.EvalBinGate(AND, a, b)
    t5 = cc.EvalBinGate(OR, t2, t3)
    r[1] = cc.EvalBinGate(OR, t5, t4)
    return r


def addNumbers(ctA, ctB, nBits):
    ctRes = [False] * nBits
    bitResult = [False] * 2
    ctRes[0] = cc.EvalBinGate(XOR, ctA[0], ctB[0])
    carry = cc.EvalNOT(cc.EvalBinGate(NAND, ctA[0], ctB[0]))
    for i in range(1, nBits):
        if i > 4:  # exact adder beyond bit‑4
            bitResult = addBits(bitResult, ctA[i], ctB[i], carry)
        else:      # low‑cost approximate adder
            bitResult = approaddBits(bitResult, ctA[i], ctB[i], carry)
        ctRes[i] = bitResult[0]
        carry = bitResult[1]
    return ctRes


def mux(c_s, true, false):
    temp1 = cc.EvalBinGate(NAND, c_s, true)
    temp1_and = cc.EvalNOT(temp1)
    temp2 = cc.EvalBinGate(NAND, cc.EvalNOT(c_s), false)
    temp2_and = cc.EvalNOT(temp2)
    return cc.EvalBinGate(OR, temp1_and, temp2_and)


def make_neg(bits, nbits):
    one = cc.Encrypt(sk, True)
    one_vec = [cc.Encrypt(sk, False) if i else one for i in range(nbits)]
    not_bits = [cc.EvalNOT(b) for b in bits]
    return addNumbers(not_bits, one_vec, nbits)


def mulNumbers(ctA, ctB, sk, in_bits, out_bits):
    # classic shift‑and‑add multiplier (same as LR code)
    result = [cc.Encrypt(sk, False) for _ in range(out_bits)]
    for i in range(in_bits):
        andRes = [cc.Encrypt(sk, False) for _ in range(out_bits)]
        for j in range(in_bits):
            if j + i < out_bits:
                andRes[j + i] = cc.EvalBinGate(AND, ctA[j], ctB[i])
        result = addNumbers(andRes, result, out_bits)
    return result[:in_bits]

# ----------------------- Encryption / Decryption helpers ---------------------

def int_to_cipher_bits(val: int, nbits: int) -> List[Ciphertext]:
    return [cc.Encrypt(sk, bit) for bit in twos_complement_bits(val, nbits)]


def cipher_bits_to_int(bits: List[Ciphertext]) -> int:
    plain = [int(cc.Decrypt(sk, b)) for b in bits]
    return twos_comp_val(bits_to_int(plain), len(bits))


def main(sentence: str):
    print("[INFO] Plaintext BERT‑Tiny embedding…")
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    model.eval()
    with torch.no_grad():
        cls = model(**tokenizer(sentence, return_tensors="pt")).last_hidden_state[0, 0].cpu().numpy()

    SCALE = 128
    x_int = np.round(cls * SCALE).astype(np.int16)
    dim = x_int.size

    # random small weights & bias for demo
    rng = np.random.default_rng(0)
    W_int = rng.integers(-64, 64, size=dim, dtype=np.int16)
    b_int = int(rng.integers(-256, 256))

    nbits = 16  # per‑element bit‑width

    print("[INFO] Setting up BinFHE…")
    global cc, sk
    cc = BinFHEContext()
    cc.GenerateBinFHEContext(STD128)
    sk = cc.KeyGen()
    cc.BTKeyGen(sk)

    enc_x = [int_to_cipher_bits(int(v), nbits) for v in x_int]
    enc_W = [int_to_cipher_bits(int(w), nbits) for w in W_int]
    enc_b = int_to_cipher_bits(b_int, nbits)

    print("[INFO] Homomorphic dot‑product (", dim, "dimensions)…")
    t0 = time.time()
    products = [mulNumbers(enc_x[i], enc_W[i], sk, nbits, nbits * 2) for i in range(dim)]
    # tree‑sum to keep depth down
    while len(products) > 1:
        a = products.pop()
        b = products.pop()
        products.append(addNumbers(a, b, nbits))
    y_bits = addNumbers(products[0], enc_b, nbits)
    enc_time = time.time() - t0

    y_int = cipher_bits_to_int(y_bits)
    y_float = y_int / (SCALE ** 2)

    print(f"Encrypted output int : {y_int}")
    print(f"De‑quantised value  : {y_float:.4f}")
    print(f"Prediction          : {'positive' if y_int > 0 else 'negative'}")
    print(f"FHE compute time    : {enc_time:.2f}s")


if __name__ == "__main__":
    sentence = " ".join(sys.argv[1:]) or "This is a test sentence."
    main(sentence)

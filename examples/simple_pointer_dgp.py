"""Simple memory + pointer DGP for transformer pedagogy.

A fallback / companion to bit_dgp.py. Where bit_dgp.py uses a recursive
prefix-coded bit stream with instruction-stream addressing (genuinely
hard for small models to learn from scratch), this file generates a
fixed-format task that small transformers can learn cleanly. The
pedagogical point is the same: attention is content-addressable memory.

Three task variants of increasing pedagogical depth:

  Variant 1 (single lookup) - the headline "attention is lookup" demo.
      Format: [M memory bits] [A address bits] [1 target = memory[addr]]
      Total length M + A + 1. With M=8, A=3 that's 12 bits per example.
      A single-head transformer can solve this. An MLP at the same
      parameter count cannot.

  Variant 2 (two lookups) - motivates multi-head attention.
      Format: [M] [A=addr1] [A=addr2] [1 = memory[addr1] XOR memory[addr2]]
      A single-head transformer has to do two lookups and a XOR in one
      head; two heads can specialize, one per lookup.

  Variant 3 (pointer to pointer) - motivates depth.
      Format: [M] [A=addr] [1 = memory[memory[addr] reinterpreted as
      a binary integer mod 2^A]]
      A 1-layer transformer can resolve one hop. A 2-layer transformer
      can resolve two.

Each variant returns (X, Y) NumPy arrays: X is shape (N, sequence_len)
of integer bit ids, Y is shape (N,) with the target bit at position
sequence_len - 1.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Variant 1: single lookup
# ---------------------------------------------------------------------------

def make_variant1(num_examples, M=8, A=3, seed=0):
    """[M memory bits] [A address bits] [1 target = memory[addr]]."""
    assert 2 ** A >= M  # address must be able to index any memory cell
    rng = np.random.default_rng(seed)
    n_addr_codes = 2 ** A

    memory = rng.integers(0, 2, size=(num_examples, M), dtype=np.int64)
    address_int = rng.integers(0, M, size=num_examples)
    # Encode address_int as A binary bits (most significant first).
    address_bits = np.zeros((num_examples, A), dtype=np.int64)
    for j in range(A):
        address_bits[:, j] = (address_int >> (A - 1 - j)) & 1
    target = memory[np.arange(num_examples), address_int][:, None]

    sequences = np.concatenate([memory, address_bits, target], axis=1)
    return sequences, target.squeeze(-1)


# ---------------------------------------------------------------------------
# Variant 2: two lookups + XOR (multi-head)
# ---------------------------------------------------------------------------

def make_variant2(num_examples, M=8, A=3, seed=0):
    """[M] [A=addr1] [A=addr2] [1 target = mem[a1] XOR mem[a2]]."""
    rng = np.random.default_rng(seed)
    memory = rng.integers(0, 2, size=(num_examples, M), dtype=np.int64)
    addr1 = rng.integers(0, M, size=num_examples)
    addr2 = rng.integers(0, M, size=num_examples)
    bits1 = np.zeros((num_examples, A), dtype=np.int64)
    bits2 = np.zeros((num_examples, A), dtype=np.int64)
    for j in range(A):
        bits1[:, j] = (addr1 >> (A - 1 - j)) & 1
        bits2[:, j] = (addr2 >> (A - 1 - j)) & 1
    v1 = memory[np.arange(num_examples), addr1]
    v2 = memory[np.arange(num_examples), addr2]
    target = (v1 ^ v2)[:, None]
    sequences = np.concatenate([memory, bits1, bits2, target], axis=1)
    return sequences, target.squeeze(-1)


# ---------------------------------------------------------------------------
# Variant 3: pointer to pointer (multi-hop)
# ---------------------------------------------------------------------------

def make_variant3(num_examples, M=8, A=3, seed=0):
    """[M] [A=addr] [1 target = mem[mem[addr] reinterpreted as small idx]].

    The first lookup retrieves a single bit. We turn that bit into a
    new address by combining it with a separate, fixed-position bit of
    memory (the LSB of the address) so that the "pointer value" is a
    sensible new index. Concretely, the second address used is
    constructed from the bit retrieved by the first lookup placed at
    position 0 (MSB), with the original address bits 1..A-1 placed at
    positions 1..A-1. The model has to chain the two lookups.
    """
    assert 2 ** A >= M
    rng = np.random.default_rng(seed)
    memory = rng.integers(0, 2, size=(num_examples, M), dtype=np.int64)
    addr = rng.integers(0, M, size=num_examples)
    addr_bits = np.zeros((num_examples, A), dtype=np.int64)
    for j in range(A):
        addr_bits[:, j] = (addr >> (A - 1 - j)) & 1
    # First hop: read memory[addr]
    first_value = memory[np.arange(num_examples), addr]
    # Build new address: high bit = first_value, low bits = addr's low bits.
    new_addr = (first_value.astype(np.int64) << (A - 1)) | (addr & ((1 << (A - 1)) - 1))
    new_addr = new_addr % M  # safety
    # Second hop: read memory[new_addr]
    target = memory[np.arange(num_examples), new_addr][:, None]
    sequences = np.concatenate([memory, addr_bits, target], axis=1)
    return sequences, target.squeeze(-1)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Variant 1 (single lookup): 5 examples")
    X, Y = make_variant1(5, M=8, A=3, seed=0)
    for i in range(5):
        print(f"  {X[i].tolist()}  target={Y[i]}")
    print()
    print("Variant 2 (two lookups + XOR): 3 examples")
    X, Y = make_variant2(3, M=8, A=3, seed=0)
    for i in range(3):
        print(f"  {X[i].tolist()}  target={Y[i]}")
    print()
    print("Variant 3 (pointer to pointer): 3 examples")
    X, Y = make_variant3(3, M=8, A=3, seed=0)
    for i in range(3):
        print(f"  {X[i].tolist()}  target={Y[i]}")

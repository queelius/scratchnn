"""A recursive pointer-dereferencing data-generating process over a bit stream.

The DGP emits a continuous bit stream by repeatedly sampling one of three
instructions and writing its prefix-coded representation. Each instruction
has a single-bit semantic value; for derefs, the value is computed
recursively by looking up the addressed past instruction's value.

The prefix code is:
    0    -> literal value 0
    10   -> literal value 1
    11   -> deref starts here

After `11`, the encoding is:
    11 + gamma(k) + literal_codeword(value)

where:
    gamma(k) is the Elias gamma encoding of the offset k >= 1, in
        *instruction* units (so k counts back through the instruction
        stream, not the bit stream).
    value = value(I_{i-k}) computed recursively.
    literal_codeword(0) = "0", literal_codeword(1) = "10".

A deref's bit cost is therefore 2 + |gamma(k)| + (1 or 2), with no nested
deref instructions ever appearing in the value position. Multi-hop chains
arise in the semantics (when a deref's target happens to be another
deref) but the bit-stream encoding stays shallow per instruction.

Each bit in the stream is tagged by its role for per-position evaluation:
"lit_0", "lit_1_a"/"lit_1_b", "marker_a"/"marker_b", "addr",
"value_lit0", "value_lit1_a"/"value_lit1_b". The deref instructions are
also annotated with their hop depth (1 if the target is a literal, 2 if
the target is a 1-hop deref, etc.).
"""
import math
import random


# ---------------------------------------------------------------------------
# Elias gamma encoding
# ---------------------------------------------------------------------------

def encode_gamma(n):
    """Encode a positive integer n as Elias gamma bits.

    Returns a list[int] of bits. For n=1 the code is "1"; for n=2,3 it
    is "010"/"011"; for n=4..7 it is "00100".."00111"; and so on.
    """
    if n < 1:
        raise ValueError(f"gamma encodes positive integers; got {n}")
    binary = []
    m = n
    while m > 0:
        binary.append(m & 1)
        m >>= 1
    binary.reverse()  # MSB first
    L = len(binary) - 1
    return [0] * L + binary


def decode_gamma(bits, start):
    """Decode an Elias gamma codeword from bits[start:].

    Returns (n, end_position) where end_position is the index just past
    the codeword.
    """
    L = 0
    pos = start
    while pos < len(bits) and bits[pos] == 0:
        L += 1
        pos += 1
    if pos >= len(bits):
        raise ValueError("incomplete gamma code")
    n = 0
    for _ in range(L + 1):
        if pos >= len(bits):
            raise ValueError("incomplete gamma code")
        n = (n << 1) | bits[pos]
        pos += 1
    return n, pos


# ---------------------------------------------------------------------------
# Stream generation
# ---------------------------------------------------------------------------

def _sample_offset(max_offset, mean, rng):
    """Sample a positive integer offset k <= max_offset from a geometric
    distribution with given mean. Falls back to uniform if mean is too large
    relative to max_offset.
    """
    if max_offset < 1:
        raise ValueError("no past instructions to point at")
    rate = 1.0 / max(mean, 1.001)
    for _ in range(64):
        u = rng.random()
        # Inverse CDF of geometric (1-based)
        try:
            k = int(math.log(max(u, 1e-12)) / math.log(1.0 - rate)) + 1
        except ValueError:
            k = 1
        if 1 <= k <= max_offset:
            return k
    return rng.randint(1, max_offset)


def generate_stream(num_instructions, p_deref=0.20, offset_mean=4.0, seed=0):
    """Generate a bit stream of `num_instructions` instructions.

    Parameters:
        num_instructions: how many instructions to emit.
        p_deref:          probability that any given instruction is a deref
                          (when there is at least one past instruction to
                          point at). The first instruction is always a literal.
        offset_mean:      target mean of the geometric offset distribution.
                          Smaller values bias toward recent lookups.
        seed:             RNG seed for reproducibility.

    Returns:
        bits:         list[int] of the raw bit stream.
        instructions: list[dict], one per emitted instruction. Each dict
                      carries 'index' (1-based), 'kind' ('lit0' / 'lit1' /
                      'deref'), 'value' (0 or 1), 'bit_start', 'bit_end',
                      and for derefs also 'offset' (k), 'target_index', and
                      'hop_depth' (1 = points to literal; 2 = to a 1-hop
                      deref; etc.).
        tags:         list[str] of length len(bits). Per-bit role tags for
                      evaluation, one of:
                          "lit_0"
                          "lit_1_a", "lit_1_b"
                          "marker_a", "marker_b"
                          "addr"
                          "value_lit0"
                          "value_lit1_a", "value_lit1_b"
        hop_depths:   list[int|None] of length len(bits). For value bits of
                      a deref, the hop depth of the surrounding deref; None
                      otherwise. Lets you slice the value-position loss by
                      depth at evaluation time.
    """
    rng = random.Random(seed)
    bits = []
    instructions = []
    tags = []
    hop_depths = []

    for i in range(1, num_instructions + 1):
        n_past = i - 1
        make_deref = (n_past >= 1) and (rng.random() < p_deref)
        bit_start = len(bits)

        if make_deref:
            k = _sample_offset(n_past, offset_mean, rng)
            target = instructions[n_past - k]  # 0-indexed lookup
            value = target['value']

            # Emit the bit pattern.
            bits.extend([1, 1])
            tags.extend(['marker_a', 'marker_b'])
            hop_depths.extend([None, None])

            gamma_bits = encode_gamma(k)
            bits.extend(gamma_bits)
            tags.extend(['addr'] * len(gamma_bits))
            hop_depths.extend([None] * len(gamma_bits))

            if target['kind'] == 'deref':
                hop_depth = target['hop_depth'] + 1
            else:
                hop_depth = 1

            if value == 0:
                bits.append(0)
                tags.append('value_lit0')
                hop_depths.append(hop_depth)
            else:
                bits.extend([1, 0])
                tags.extend(['value_lit1_a', 'value_lit1_b'])
                hop_depths.extend([hop_depth, hop_depth])

            bit_end = len(bits)
            instructions.append({
                'index': i,
                'kind': 'deref',
                'value': value,
                'bit_start': bit_start,
                'bit_end': bit_end,
                'offset': k,
                'target_index': target['index'],
                'hop_depth': hop_depth,
            })
        else:
            value = rng.randint(0, 1)
            if value == 0:
                bits.append(0)
                tags.append('lit_0')
                hop_depths.append(None)
                kind = 'lit0'
            else:
                bits.extend([1, 0])
                tags.extend(['lit_1_a', 'lit_1_b'])
                hop_depths.extend([None, None])
                kind = 'lit1'
            bit_end = len(bits)
            instructions.append({
                'index': i,
                'kind': kind,
                'value': value,
                'bit_start': bit_start,
                'bit_end': bit_end,
                'offset': None,
                'target_index': None,
                'hop_depth': 0,
            })

    assert len(tags) == len(bits)
    assert len(hop_depths) == len(bits)
    return bits, instructions, tags, hop_depths


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def parse_stream(bits):
    """Parse a bit stream back into instructions, verifying the encoding.

    Re-derives the values by recursive lookup and returns the parsed
    instruction list. Useful as a sanity check on the generator.
    """
    instructions = []
    pos = 0
    while pos < len(bits):
        bit_start = pos
        if bits[pos] == 0:
            instructions.append({'kind': 'lit0', 'value': 0,
                                 'bit_start': bit_start, 'bit_end': pos + 1})
            pos += 1
        elif bits[pos] == 1 and pos + 1 < len(bits) and bits[pos + 1] == 0:
            instructions.append({'kind': 'lit1', 'value': 1,
                                 'bit_start': bit_start,
                                 'bit_end': pos + 2})
            pos += 2
        elif bits[pos] == 1 and pos + 1 < len(bits) and bits[pos + 1] == 1:
            # deref
            k, after_addr = decode_gamma(bits, pos + 2)
            target = instructions[len(instructions) - k]
            value = target['value']
            if value == 0:
                if bits[after_addr] != 0:
                    raise ValueError(
                        f"deref at {pos}: expected value codeword 0, "
                        f"saw bit {bits[after_addr]}")
                end = after_addr + 1
            else:
                if bits[after_addr] != 1 or bits[after_addr + 1] != 0:
                    raise ValueError(
                        f"deref at {pos}: expected value codeword 10, "
                        f"saw {bits[after_addr:after_addr+2]}")
                end = after_addr + 2
            instructions.append({'kind': 'deref', 'value': value,
                                 'offset': k, 'bit_start': bit_start,
                                 'bit_end': end})
            pos = end
        else:
            raise ValueError(f"unexpected bits at position {pos}")
    return instructions


if __name__ == "__main__":
    # Smoke test: generate a small stream and parse it back.
    bits, instrs, tags, hops = generate_stream(
        num_instructions=30, p_deref=0.30, offset_mean=3.0, seed=0)
    print(f"emitted {len(instrs)} instructions in {len(bits)} bits")
    bit_str = "".join(str(b) for b in bits)
    print(f"stream (first 60 bits): {bit_str[:60]}")
    parsed = parse_stream(bits)
    assert len(parsed) == len(instrs)
    for a, b in zip(parsed, instrs):
        assert a['value'] == b['value'], f"value mismatch: {a} vs {b}"
        assert a['kind'] == b['kind'], f"kind mismatch: {a} vs {b}"
    print("parse round-trip ok")

    n_deref = sum(1 for i in instrs if i['kind'] == 'deref')
    print(f"  literals: {len(instrs) - n_deref}")
    print(f"  derefs:   {n_deref}")
    max_hop = max((i['hop_depth'] for i in instrs if i['kind'] == 'deref'),
                  default=0)
    print(f"  max hop depth: {max_hop}")
    hop_counts = {}
    for i in instrs:
        if i['kind'] == 'deref':
            hop_counts[i['hop_depth']] = hop_counts.get(i['hop_depth'], 0) + 1
    print(f"  hop-depth distribution: {dict(sorted(hop_counts.items()))}")

    # Show a worked example matching the post's narrative.
    print()
    print("instruction-by-instruction breakdown (first 8):")
    for instr in instrs[:8]:
        slice_bits = "".join(str(b) for b in bits[instr['bit_start']:instr['bit_end']])
        print(f"  I_{instr['index']:>2}  {instr['kind']:<6}  value={instr['value']}  "
              f"bits[{instr['bit_start']:>3}:{instr['bit_end']:>3}]='{slice_bits}'"
              + (f"  offset={instr['offset']} hop={instr['hop_depth']}"
                 if instr['kind'] == 'deref' else ""))

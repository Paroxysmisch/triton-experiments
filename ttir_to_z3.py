import sys
import re
import z3

def triton_to_z3(ir_text):
    regs = {}

    def get_reg(name, preferred_type='bv'):
        """Helper to get or create a symbolic register with type awareness."""
        if name not in regs:
            if preferred_type == 'fp':
                regs[name] = z3.FP(name, z3.Float32())
            else:
                regs[name] = z3.BitVec(name, 32 if 'ptr' not in name else 64)

        # If we have a BitVec but need an FP (or vice versa), cast it
        val = regs[name]
        if preferred_type == 'fp' and z3.is_bv(val):
            return z3.fpToFP(val, z3.Float32())
        if preferred_type == 'bv' and z3.is_fp(val):
            return z3.fpToIEEEBV(val)
        return val

    patterns = {
        'arg': re.compile(r'%(?P<name>\w+):'),
        'constant': re.compile(r'%(?P<res>\w+) = arith\.constant (?P<val>[\d\.e+-]+|dense<[^>]+>)'),
        'loop': re.compile(r'scf\.for %(?P<index>\w+) ='),
        'make_range': re.compile(r'%(?P<res>\w+) = tt\.make_range'),
        'arith_2op': re.compile(r'%(?P<res>\w+) = (arith|math)\.(?P<op>\w+) %(?P<lhs>\w+), %(?P<rhs>\w+)'),
        'arith_1op': re.compile(r'%(?P<res>\w+) = math\.(?P<op>exp) %(?P<arg>\w+)'),
        'addptr': re.compile(r'%(?P<res>\w+) = tt\.addptr %(?P<ptr>\w+), %(?P<off>\w+)'),
        'splat': re.compile(r'%(?P<res>\w+) = tt\.splat %(?P<src>\w+)'),
        'load': re.compile(r'%(?P<res>\w+) = tt\.load %(?P<ptr>\w+), %(?P<mask>\w+), %(?P<other>\w+)'),
        'reduce_block': re.compile(r'\^bb0\(%(?P<a>\w+): f32.*, %(?P<b>\w+): f32'),
    }

    for line in ir_text.splitlines():
        line = line.strip()

        # 0. Handle Reduction Block Inputs (Floating Point)
        m = patterns['reduce_block'].search(line)
        if m:
            regs[m.group('a')] = z3.FP(m.group('a'), z3.Float32())
            regs[m.group('b')] = z3.FP(m.group('b'), z3.Float32())
            continue

        # 1. Function Arguments
        if "@softmax_kernel" in line:
            for arg in patterns['arg'].findall(line):
                get_reg(arg, 'bv') # Arguments are pointers or sizes (integers)
            continue

        # 2. Constants & Loops
        m = patterns['constant'].search(line)
        if m:
            res, val = m.group('res'), m.group('val')
            if 'dense' in val:
                # Handle hex constants like dense<0xFF800000> (-Infinity)
                if '0xFF800000' in val:
                    # Correct Z3 Python API name is fpMinusInfinity
                    regs[res] = z3.fpMinusInfinity(z3.Float32())
                else:
                    regs[res] = z3.FP(f"const_{res}", z3.Float32())
            else:
                # Check if it's an integer or float constant
                if val.isdigit() or (val.startswith('-') and val[1:].isdigit()):
                    regs[res] = z3.BitVecVal(int(val), 32)
                else:
                    regs[res] = z3.FPVal(float(val), z3.Float32())
            continue

        # 3. Arithmetic (The critical fix for maxnumf)
        m = patterns['arith_2op'].search(line)
        if m:
            res, op, lhs, rhs = m.group('res'), m.group('op'), m.group('lhs'), m.group('rhs')

            # Decide if we need FP or BV logic based on the operator
            is_fp_op = op in ['subf', 'divf', 'addf', 'maxnumf']
            l = get_reg(lhs, 'fp' if is_fp_op else 'bv')
            r = get_reg(rhs, 'fp' if is_fp_op else 'bv')

            if op == 'muli': regs[res] = l * r
            elif op == 'addi': regs[res] = l + r
            elif op == 'subf': regs[res] = z3.fpSub(z3.RNE(), l, r)
            elif op == 'divf': regs[res] = z3.fpDiv(z3.RNE(), l, r)
            elif op == 'addf': regs[res] = z3.fpAdd(z3.RNE(), l, r)
            elif op == 'maxnumf': regs[res] = z3.If(z3.fpGT(l, r), l, r)
            elif op == 'cmpi': regs[res] = z3.Bool(res) # Abstracted for POC
            continue

        m = patterns['arith_1op'].search(line)
        if m:
            res, op, arg = m.group('res'), m.group('op'), m.group('arg')
            if op == 'exp':
                exp_func = z3.Function('exp', z3.Float32(), z3.Float32())
                regs[res] = exp_func(get_reg(arg, 'fp'))
            continue

        # 4. Pointers & Splat
        m = patterns['addptr'].search(line)
        if m:
            p = get_reg(m.group('ptr'), 'bv')
            o = get_reg(m.group('off'), 'bv')
            # Pointers are 64-bit
            p_64 = p if p.size() == 64 else z3.ZeroExt(32, p)
            o_64 = o if o.size() == 64 else z3.ZeroExt(32, o)
            regs[m.group('res')] = p_64 + o_64
            continue

        m = patterns['splat'].search(line)
        if m:
            regs[m.group('res')] = regs.get(m.group('src'), get_reg(m.group('src')))
            continue

        m = patterns['load'].search(line)
        if m:
            regs[m.group('res')] = z3.FP(m.group('res'), z3.Float32())
            continue

        if '"tt.reduce"' in line:
            res_name = line.split('=')[0].strip().replace('%', '')
            regs[res_name] = z3.FP(res_name, z3.Float32())

    return regs

if __name__ == "__main__":
    ir_input = sys.stdin.read()
    try:
        sym_regs = triton_to_z3(ir_input)
        print(";; Translation Successful\n")
        targets = ['row_minus_max', 'numerator', 'denominator', 'softmax_output_5']
        for t in targets:
            if t in sym_regs:
                print(f"%{t} = {sym_regs[t]}")
    except Exception as e:
        print(f";; Error: {e}")

import sys
import re
import z3

def triton_to_z3(ir_text):
    regs = {}
    terminals = set()

    def get_reg(name, preferred_type='fp'):
        if name not in regs:
            if preferred_type == 'fp':
                regs[name] = z3.FP(name, z3.Float32())
            else:
                regs[name] = z3.BitVec(name, 32 if 'ptr' not in name else 64)
        return regs[name]

    # Flexible patterns to catch both JIT and Manual IR styles
    patterns = {
        'arg': re.compile(r'%(?P<name>\w+):'),
        'constant': re.compile(r'%(?P<res>\w+) = arith\.constant (?P<val>.*)'),
        'arith_2op': re.compile(r'%(?P<res>\w+) = (arith|math)\.(?P<op>\w+) %(?P<lhs>\w+), %(?P<rhs>\w+)'),
        # Handles math.exp %arg
        'arith_1op': re.compile(r'%(?P<res>\w+) = math\.(?P<op>exp) %(?P<arg>\w+)'),
        # Handles tt.extern_elementwise %arg {symbol = "__nv_expf"}
        'extern': re.compile(r'%(?P<res>\w+) = tt\.extern_elementwise %(?P<arg>\w+).*symbol = "(?P<sym>[^"]+)"'),
        'alias': re.compile(r'%(?P<res>\w+) = tt\.(splat|broadcast|expand_dims) %(?P<src>\w+)'),
        'load': re.compile(r'%(?P<res>\w+) = tt\.load %(?P<ptr>\w+)'),
        'reduce': re.compile(r'%(?P<res>\w+) = "tt\.reduce"'),
    }

    for line in ir_text.splitlines():
        line = line.strip()
        if not line or "=" not in line and "@" not in line: continue

        # 1. Capture Args
        if "@" in line:
            for a in patterns['arg'].findall(line):
                terminals.add(a)
                get_reg(a, 'bv')
            continue

        # 2. Constants
        m = patterns['constant'].search(line)
        if m:
            res, val = m.group('res'), m.group('val')
            if '0xFF800000' in val: regs[res] = z3.fpMinusInfinity(z3.Float32())
            elif 'dense' in val: regs[res] = z3.FP(f"const_{res}", z3.Float32())
            else:
                clean_val = val.split(':')[0].strip()
                try: regs[res] = z3.FPVal(float(clean_val), z3.Float32())
                except: regs[res] = z3.BitVecVal(int(clean_val), 32)
            continue

        # 3. Aliases (Splat/Broadcast/Expand)
        m = patterns['alias'].search(line)
        if m:
            regs[m.group('res')] = get_reg(m.group('src'))
            continue

        # 4. Math Ops
        m = patterns['arith_2op'].search(line)
        if m:
            res, op, lhs, rhs = m.group('res'), m.group('op'), m.group('lhs'), m.group('rhs')
            l, r = get_reg(lhs, 'fp'), get_reg(rhs, 'fp')
            if 'sub' in op: regs[res] = z3.fpSub(z3.RNE(), l, r)
            elif 'div' in op: regs[res] = z3.fpDiv(z3.RNE(), l, r)
            elif 'add' in op: regs[res] = z3.fpAdd(z3.RNE(), l, r)
            elif 'max' in op: regs[res] = z3.If(z3.fpGT(l, r), l, r)
            continue

        # 5. Exp (Both variants)
        m_exp = patterns['arith_1op'].search(line) or patterns['extern'].search(line)
        if m_exp:
            res = m_exp.group('res')
            arg = m_exp.group('arg')
            f = z3.Function('exp', z3.Float32(), z3.Float32())
            regs[res] = f(get_reg(arg, 'fp'))
            continue

        # 6. Terminals (Loads and Reduction results)
        m_load = patterns['load'].search(line)
        if m_load:
            res = m_load.group('res')
            regs[res] = z3.FP(f"load_{res}", z3.Float32())
            terminals.add(res)
            continue

        if '"tt.reduce"' in line:
            res_name = line.split('=')[0].strip().replace('%', '')
            regs[res_name] = z3.FP(res_name, z3.Float32())
            terminals.add(res_name)

    return regs, terminals

def deep_chase(expr, regs, terminals):
    # Expand until we hit terminals
    current_expr = expr
    for _ in range(10): # Max depth
        # Get all symbols in current expression
        # Substitute anything that isn't a terminal
        subs = []
        for reg_name, val in regs.items():
            if reg_name not in terminals:
                # We need to match the specific Z3 object used in the expression
                sym = z3.FP(reg_name, z3.Float32()) if z3.is_fp(val) else z3.BitVec(reg_name, val.size())
                subs.append((sym, val))

        new_expr = z3.substitute(current_expr, *subs)
        if z3.eq(new_expr, current_expr):
            break
        current_expr = new_expr
    return current_expr

if __name__ == "__main__":
    ir_input = sys.stdin.read()
    regs, terminals = triton_to_z3(ir_input)

    store_match = re.search(r'tt\.store .*?, %(?P<val>\w+),', ir_input)
    if store_match:
        target = store_match.group('val')
        print(f";; Final Formula for %{target}:\n")
        print(deep_chase(regs[target], regs, terminals))

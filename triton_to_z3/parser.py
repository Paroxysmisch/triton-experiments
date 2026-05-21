"""TTIR / MLIR textual-format parser.

Produces a lightweight AST of *Operation* objects that the interpreter can walk.
Handles multi-line region-bearing ops (tt.reduce, scf.for, scf.if, scf.while).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .types import MLIRType, parse_type


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass
class Region:
    """A region body (e.g. the combiner inside tt.reduce, or a loop body)."""

    args: list[tuple[str, MLIRType]]  # block arguments: (name, type)
    body: list[Operation] = field(default_factory=list)


@dataclass
class Operation:
    results: list[str]  # SSA result names (without %)
    op: str  # fully-qualified op name, e.g. "arith.addf"
    operands: list[str]  # SSA operand names (without %)
    attributes: dict[str, str]  # parsed inline attributes
    type_str: str  # raw result/operand type string
    result_types: list[MLIRType]
    regions: list[Region] = field(default_factory=list)
    raw: str = ""  # original text for debugging


@dataclass
class Function:
    name: str
    args: list[tuple[str, MLIRType]]  # (name, type)
    body: list[Operation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Location stripping
# ---------------------------------------------------------------------------


def strip_locations(text: str) -> str:
    """Remove all ``loc(...)`` annotations from MLIR text (handles nesting)."""
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i : i + 5] == " loc(":
            depth = 1
            j = i + 5
            while j < n and depth > 0:
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                j += 1
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

_LOC_DEF = re.compile(r"^#loc\d*\s*=")


def preprocess(text: str) -> list[str]:
    """Strip locations, metadata lines, and return cleaned non-empty lines."""
    text = strip_locations(text)
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if _LOC_DEF.match(stripped):
            continue
        lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# Brace-aware block collector
# ---------------------------------------------------------------------------


def _net_braces(line: str) -> int:
    """Count unbalanced ``{`` / ``}`` ignoring attribute-dict ``<{…}>``."""
    depth = 0
    in_angle = 0
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "<" and i + 1 < len(line) and line[i + 1] == "{":
            in_angle += 1
            i += 2
            continue
        if ch == "}" and i + 1 < len(line) and line[i + 1] == ">" and in_angle > 0:
            in_angle -= 1
            i += 2
            continue
        if in_angle == 0:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        i += 1
    return depth


def collect_blocks(lines: list[str]) -> list[str]:
    """Group *lines* into multi-line operation blocks by brace depth."""
    blocks: list[str] = []
    current: list[str] = []
    depth = 0

    for line in lines:
        current.append(line)
        depth += _net_braces(line)
        if depth <= 0:
            blocks.append("\n".join(current))
            current = []
            depth = 0

    if current:
        blocks.append("\n".join(current))
    return blocks


# ---------------------------------------------------------------------------
# Single-operation line parser
# ---------------------------------------------------------------------------

# Matches: %res = op ...  or  %r1, %r2 = op ...  or  %res:3 = op ... (multi-result)
_RESULT_RE = re.compile(
    r"^(?P<results>%[\w]+(?::\d+)?(?:\s*,\s*%[\w]+(?::\d+)?)*)\s*=\s*(?P<rest>.*)$"
)
# Matches quoted op: "tt.reduce"(...)
_QUOTED_OP_RE = re.compile(r'^"(?P<op>[^"]+)"\((?P<operands>[^)]*)\)\s*(?P<rest>.*)$')
# Matches normal op: arith.addf %a, %b ...
_NORMAL_OP_RE = re.compile(r"^(?P<op>[\w.]+)\s*(?P<rest>.*)$")
# Operand references: %name or %name#N (multi-result index)
_OPERAND_RE = re.compile(r"%(?P<name>\w+(?:#\d+)?)")
# Block argument:  %name: type
_BLOCK_ARG_RE = re.compile(r"%(?P<name>\w+)\s*:\s*(?P<type>[^,)]+)")
# Attribute dict <{...}>
_ATTR_DICT_RE = re.compile(r"<\{(?P<body>[^}]*)\}>")
# Inline {key = value, ...}
_INLINE_ATTR_RE = re.compile(r"\{(?P<body>[^}]*)\}")
# Type annotation  : type  or  : (type) -> type
_TYPE_RE = re.compile(r":\s*(?P<type>.+)$")


def _parse_attributes(text: str) -> dict[str, str]:
    """Extract key=value pairs from attribute-dict text."""
    attrs: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            attrs[k.strip()] = v.strip()
    return attrs


def _strip_attr_dicts(text: str) -> tuple[str, dict[str, str]]:
    """Remove ``<{…}>`` and ``{…}`` attribute dicts, returning cleaned text and attrs."""
    attrs: dict[str, str] = {}

    for m in _ATTR_DICT_RE.finditer(text):
        attrs.update(_parse_attributes(m.group("body")))
    text = _ATTR_DICT_RE.sub("", text)

    for m in _INLINE_ATTR_RE.finditer(text):
        body = m.group("body")
        if "=" in body and not body.strip().startswith("tt."):
            attrs.update(_parse_attributes(body))

    # Only strip inline attrs that look like key=value (not region bodies)
    def _replace_inline(m: re.Match[str]) -> str:
        body = m.group("body")
        if "=" in body and not body.strip().startswith("tt.") and "%" not in body:
            return ""
        return m.group(0)

    text = _INLINE_ATTR_RE.sub(_replace_inline, text)

    return text.strip(), attrs


def _parse_type_annotation(text: str) -> tuple[str, list[MLIRType]]:
    """Extract the trailing ``: type`` annotation and parse it."""
    # Handle  : (inputtype) -> resulttype
    m = re.search(r":\s*\(([^)]*)\)\s*->\s*(.+)$", text)
    if m:
        result_type_str = m.group(2).strip()
        prefix = text[: m.start()].strip()
        return prefix, [parse_type(result_type_str)]

    # Handle  : type -> type  (for splat / broadcast / expand_dims)
    m = re.search(r":\s*(\S.*?)\s*->\s*(\S.*)$", text)
    if m:
        result_type_str = m.group(2).strip()
        prefix = text[: m.start()].strip()
        return prefix, [parse_type(result_type_str)]

    # Handle  : type1, type2  (for select: condition type, value type)
    m = re.search(r":\s*(.+)$", text)
    if m:
        type_str = m.group(1).strip()
        prefix = text[: m.start()].strip()
        # If comma-separated types, take the last one as result type
        # But be careful: tensor<1x1024xi32> contains commas in some contexts
        types = _split_type_list(type_str)
        return prefix, [parse_type(t) for t in types]

    return text.strip(), []


def _split_type_list(type_str: str) -> list[str]:
    """Split a comma-separated type list, respecting angle brackets."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in type_str:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _extract_operands(text: str) -> list[str]:
    """Pull all %name references from *text*."""
    return _OPERAND_RE.findall(text)


# ---------------------------------------------------------------------------
# Predicate / special-syntax handling
# ---------------------------------------------------------------------------

# arith.cmpi slt, %a, %b  or  arith.cmpf ogt, %a, %b
_CMP_RE = re.compile(r"^(?P<pred>\w+)\s*,\s*(?P<rest>.*)$")
# scf.for %iv = %lb to %ub step %st [iter_args(...)]
_FOR_RE = re.compile(
    r"^%(?P<iv>\w+)\s*=\s*%(?P<lb>\w+)\s+to\s+%(?P<ub>\w+)\s+step\s+%(?P<step>\w+)"
)
# iter_args(%block_arg = %init, ...)
_ITER_ARGS_RE = re.compile(r"iter_args\((?P<body>[^)]+)\)")
# arith.constant value : type
_CONST_RE = re.compile(r"^(?P<value>.+)$")


# ---------------------------------------------------------------------------
# Main per-block parser
# ---------------------------------------------------------------------------


def parse_block_text(text: str) -> list[Operation]:
    """Parse a multi-line block of MLIR text into a list of Operations."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return []

    blocks = collect_blocks(lines)
    ops: list[Operation] = []
    for blk in blocks:
        op = _parse_operation_block(blk)
        if op is not None:
            ops.append(op)
    return ops


def _parse_operation_block(text: str) -> Operation | None:
    """Parse one (possibly multi-line) operation block."""
    lines = text.splitlines()
    first = lines[0].strip()

    # Skip metadata / wrappers
    if first.startswith("#loc") or first == "module {" or first == "}":
        return None
    if first.startswith("^bb"):
        # Block label – skip it, parse remaining lines
        if len(lines) > 1:
            sub_text = "\n".join(lines[1:])
            sub_ops = parse_block_text(sub_text)
            return sub_ops[0] if sub_ops else None
        return None

    # --- Extract regions (inner bodies delimited by braces) ---
    regions: list[Region] = []
    header_line = first
    body_lines: list[str] = []

    if len(lines) > 1:
        # Multi-line: first line is header, middle lines are body, last closes
        body_lines = lines[1:]
        # The closing line may be  })..., } ..., } else { ...
        # Separate body from closing tokens
        regions, body_lines = _extract_regions(body_lines)

    # --- Parse the header line ---
    # 1. Check for results
    results: list[str] = []
    rest = header_line
    m = _RESULT_RE.match(rest)
    if m:
        for r in m.group("results").split(","):
            r = r.strip().lstrip("%")
            # Handle %name:N multi-result syntax → name#0, name#1, ..., name#(N-1)
            if ":" in r:
                name, count_str = r.rsplit(":", 1)
                if count_str.isdigit():
                    for i in range(int(count_str)):
                        results.append(f"{name}#{i}")
                else:
                    results.append(r)
            else:
                results.append(r)
        rest = m.group("rest")

    # 2. Extract op name
    op_name = ""
    m_q = _QUOTED_OP_RE.match(rest)
    m_n = _NORMAL_OP_RE.match(rest)
    if m_q:
        op_name = m_q.group("op")
        rest = m_q.group("operands") + " " + m_q.group("rest")
    elif m_n:
        op_name = m_n.group("op")
        rest = m_n.group("rest")
    else:
        return None

    # Skip certain wrapper ops
    if op_name in ("module", "tt.func"):
        return None

    # 3. Strip attribute dicts and extract attributes
    rest, attrs = _strip_attr_dicts(rest)

    # 4. Handle predicate syntax for cmp ops
    if op_name in ("arith.cmpi", "arith.cmpf"):
        m_cmp = _CMP_RE.match(rest)
        if m_cmp:
            attrs["predicate"] = m_cmp.group("pred")
            rest = m_cmp.group("rest")

    # 5. Handle scf.for special syntax
    if op_name == "scf.for":
        m_for = _FOR_RE.match(rest)
        if m_for:
            attrs["iv"] = m_for.group("iv")
            attrs["lb"] = m_for.group("lb")
            attrs["ub"] = m_for.group("ub")
            attrs["step"] = m_for.group("step")
            rest = rest[m_for.end() :]
        # Parse iter_args(%block_arg = %init, ...)
        m_iter = _ITER_ARGS_RE.search(rest)
        if m_iter:
            iter_pairs: list[tuple[str, str]] = []
            for part in m_iter.group("body").split(","):
                part = part.strip()
                if "=" in part:
                    lhs, rhs = part.split("=", 1)
                    arg_name = lhs.strip().lstrip("%")
                    init_name = rhs.strip().lstrip("%")
                    iter_pairs.append((arg_name, init_name))
            attrs["iter_args"] = iter_pairs
            rest = rest[: m_iter.start()] + rest[m_iter.end() :]

    # 6. Parse type annotation
    rest_no_type, result_types = _parse_type_annotation(rest)
    type_str = rest.strip()

    # 7. Extract operands
    operands = _extract_operands(rest_no_type)

    # 8. Handle arith.constant: the operand text IS the value
    if op_name == "arith.constant":
        # Value is between op name and type annotation
        val_text = rest_no_type.strip()
        if val_text:
            attrs["value"] = val_text
        # Also store full type string for constant parsing
        if result_types:
            attrs["const_type"] = type_str

    return Operation(
        results=results,
        op=op_name,
        operands=operands,
        attributes=attrs,
        type_str=type_str,
        result_types=result_types,
        regions=regions,
        raw=text,
    )


def _extract_regions(body_lines: list[str]) -> tuple[list[Region], list[str]]:
    """Extract Region objects from body lines of a multi-line operation.

    Properly tracks brace depth so that nested regions (e.g. tt.reduce
    inside scf.for) don't prematurely close the outer region.
    """
    regions: list[Region] = []
    inner_lines: list[str] = []
    depth = 0  # nesting depth within this region

    for line in body_lines:
        stripped = line.strip()
        braces = _net_braces(stripped)

        # Check if this line closes the current top-level region
        # (depth would go negative → we're exiting our scope)
        if depth + braces < 0:
            # Region closer at our level
            if "else" in stripped:
                # "} else {" – finish current region, start a new one
                if inner_lines:
                    regions.append(_build_region(inner_lines))
                    inner_lines = []
                depth = 0
                continue
            # Normal close: }) ... or } ...
            if inner_lines:
                regions.append(_build_region(inner_lines))
                inner_lines = []
            depth = 0
            continue

        # If depth is 0 and braces == 0 and the line is just "}" or "})"
        # this also closes the region
        if depth == 0 and braces == 0 and stripped in ("}", "})"):
            if inner_lines:
                regions.append(_build_region(inner_lines))
                inner_lines = []
            continue

        inner_lines.append(stripped)
        depth += braces

    if inner_lines:
        regions.append(_build_region(inner_lines))

    return regions, []


def _build_region(lines: list[str]) -> Region:
    """Build a Region from its inner lines."""
    args: list[tuple[str, MLIRType]] = []

    start = 0
    if lines and lines[0].startswith("^"):
        # Block label with arguments:  ^bb0(%a: f32, %b: f32):
        label_line = lines[0]
        for m in _BLOCK_ARG_RE.finditer(label_line):
            name = m.group("name")
            t = parse_type(m.group("type").strip().rstrip("):"))
            args.append((name, t))
        start = 1

    body_text = "\n".join(lines[start:])
    body_ops = parse_block_text(body_text) if body_text.strip() else []
    return Region(args=args, body=body_ops)


# ---------------------------------------------------------------------------
# Function header parser
# ---------------------------------------------------------------------------

_FUNC_RE = re.compile(r"tt\.func\s+(?:public\s+)?@(?P<name>\w+)\((?P<args>.*?)\)")


def parse_function_header(line: str) -> tuple[str, list[tuple[str, MLIRType]]]:
    """Parse a ``tt.func`` declaration and return (name, args)."""
    m = _FUNC_RE.search(line)
    if not m:
        return "", []

    name = m.group("name")
    args_str = m.group("args")

    args: list[tuple[str, MLIRType]] = []
    # Split on top-level commas (respecting angle brackets)
    for part in _split_type_list(args_str):
        part = part.strip()
        am = re.match(r"%(\w+)\s*:\s*(.+?)(?:\s*\{.*\})?$", part)
        if am:
            arg_name = am.group(1)
            arg_type = parse_type(am.group(2).strip())
            args.append((arg_name, arg_type))

    return name, args


# ---------------------------------------------------------------------------
# Top-level parse entry point
# ---------------------------------------------------------------------------


def parse_ttir(text: str) -> Function | None:
    """Parse a TTIR module, returning the first ``tt.func`` as a Function."""
    lines = preprocess(text)
    if not lines:
        return None

    func_name = ""
    func_args: list[tuple[str, MLIRType]] = []
    body_start = 0

    # Find the function header
    for i, line in enumerate(lines):
        if "tt.func" in line:
            func_name, func_args = parse_function_header(line)
            body_start = i + 1
            break
    else:
        # No function found – treat entire text as body
        body_start = 0

    # Find matching closing brace for the function
    depth = 0
    body_end = len(lines)
    for i in range(body_start, len(lines)):
        depth += _net_braces(lines[i])
        # Once we return to depth 0 (or go negative), the function body ends
        if depth <= -1:
            body_end = i
            break

    body_lines = lines[body_start:body_end]
    body_text = "\n".join(body_lines)
    blocks = collect_blocks(body_lines)

    body_ops: list[Operation] = []
    for blk in blocks:
        op = _parse_operation_block(blk)
        if op is not None:
            body_ops.append(op)

    return Function(name=func_name, args=func_args, body=body_ops)

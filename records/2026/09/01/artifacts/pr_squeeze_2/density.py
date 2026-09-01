import ast, io, sys, tokenize
from pathlib import Path

def classify(path):
    src = Path(path).read_text()
    lines = src.splitlines()
    n = len(lines)
    kind = ["code"] * n
    for i, l in enumerate(lines):
        if not l.strip():
            kind[i] = "blank"
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                for i in range(b[0].lineno - 1, b[0].end_lineno):
                    kind[i] = "doc"
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            kind[tok.start[0] - 1] = "comment"
    c = {k: kind.count(k) for k in ("code", "doc", "comment", "blank")}
    c["total"] = n
    return c

rows = []
for p in sys.argv[1:]:
    c = classify(p)
    rows.append((p, c))
tot = {k: 0 for k in ("code", "doc", "comment", "blank", "total")}
print(f"{'file':<58}{'total':>7}{'code':>7}{'doc':>7}{'cmnt':>6}{'blank':>7}{'d+c/code':>10}")
for p, c in rows:
    for k in tot: tot[k] += c[k]
    r = (c["doc"] + c["comment"]) / c["code"] if c["code"] else 0
    print(f"{p.split('lmcache/')[-1].split('tests/')[-1]:<58}{c['total']:>7}{c['code']:>7}{c['doc']:>7}{c['comment']:>6}{c['blank']:>7}{r:>10.2f}")
r = (tot["doc"] + tot["comment"]) / tot["code"]
print(f"{'TOTAL':<58}{tot['total']:>7}{tot['code']:>7}{tot['doc']:>7}{tot['comment']:>6}{tot['blank']:>7}{r:>10.2f}")

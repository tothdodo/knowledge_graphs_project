"""Datalog engine and reasoner"""

from __future__ import annotations
import ast
import operator
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


class DatalogError(ValueError): """Raised for syntax errors and unsafe rules."""

# Terms

class Var:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Var({self.name})"

@dataclass
class Atom:
    pred: str
    terms: tuple

@dataclass
class Builtin:
    kind: str
    left: Any
    op: str | None
    right: Any
    vars: set = field(default_factory=set)


@dataclass
class Rule:
    head: Atom
    atoms: list
    builtins: list
    text: str = ""


# Parsing

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")
_ATOM = re.compile(r"^([a-z][A-Za-z0-9_]*)\s*\((.*)\)$", re.S)
_CMP = re.compile(r"(>=|<=|==|!=|>|<)")

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
_CMPOPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq,
           "!=": operator.ne, ">": operator.gt, "<": operator.lt}

# Removes comments (%, //)
def _strip_comments(text: str) -> str:
    out_lines = []
    for line in text.splitlines():
        buf, in_q, i = [], False, 0
        while i < len(line):
            ch = line[i]
            if ch == '"':
                in_q = not in_q
            if not in_q and (ch == "%" or line.startswith("//", i)):
                break
            buf.append(ch)
            i += 1
        out_lines.append("".join(buf))
    return "\n".join(out_lines)

# Split statements on top-level '.' 
def _split_statements(text: str) -> list[str]:
    stmts, buf, in_q = [], [], False
    for i, ch in enumerate(text):
        if ch == '"':
            in_q = not in_q
        if ch == "." and not in_q:
            prev_digit = i > 0 and text[i - 1].isdigit()
            next_digit = i + 1 < len(text) and text[i + 1].isdigit()
            if not (prev_digit and next_digit):
                stmt = "".join(buf).strip()
                if stmt:
                    stmts.append(stmt)
                buf = []
                continue
        buf.append(ch)
    if "".join(buf).strip():
        raise DatalogError(f"Missing '.' at end of statement: {''.join(buf).strip()!r}")
    return stmts

# Split a top level statement on a separator
def _split_top_level(s: str, sep: str = ",") -> list[str]:
    parts, buf, depth, in_q = [], [], 0, False
    for ch in s:
        if ch == '"':
            in_q = not in_q
        if not in_q:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == sep and depth == 0:
                parts.append("".join(buf).strip())
                buf = []
                continue
        buf.append(ch)
    if "".join(buf).strip():
        parts.append("".join(buf).strip())
    return parts


_anon = [0]

# Decides what one argument is
def _parse_term(tok: str):
    tok = tok.strip()
    if len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
        return tok[1:-1]
    if _NUMBER.match(tok):
        return float(tok) if "." in tok else int(tok)
    if tok == "_":
        _anon[0] += 1
        return Var(f"_anon{_anon[0]}")
    if _IDENT.match(tok):
        return Var(tok) if (tok[0].isupper() or tok[0] == "_") else tok
    raise DatalogError(f"Cannot parse term {tok!r}")

# Parses an atom (predicate + terms)
def _parse_atom(text: str) -> Atom:
    m = _ATOM.match(text.strip())
    if not m:
        raise DatalogError(f"Cannot parse atom {text!r}")
    args = _split_top_level(m.group(2))
    return Atom(m.group(1), tuple(_parse_term(a) for a in args))

# Parse an expression (arithmetic or comparison)
def _parse_expr(text: str):
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise DatalogError(f"Bad expression {text!r}") from exc
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                   ast.Load, ast.USub, *tuple(_BINOPS))):
            raise DatalogError(f"Unsupported syntax in expression {text!r}")
    return tree.body, names

# Eval logic
def _eval_expr(node, env: dict):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp):
        return -_eval_expr(node.operand, env)
    if isinstance(node, ast.BinOp):
        return _BINOPS[type(node.op)](_eval_expr(node.left, env), _eval_expr(node.right, env))
    raise DatalogError("Unsupported expression node")

# Parse a builtin (comparison or assignment)
def _parse_builtin(text: str) -> Builtin:
    m = _CMP.search(text)
    if m:
        left, l_names = _parse_expr(text[:m.start()])
        right, r_names = _parse_expr(text[m.end():])
        return Builtin("cmp", left, m.group(1), right, l_names | r_names)
    if "=" in text:
        name, expr = text.split("=", 1)
        name = name.strip()
        if not _IDENT.match(name) or not name[0].isupper():
            raise DatalogError(f"Assignment target must be a variable: {text!r}")
        right, names = _parse_expr(expr)
        return Builtin("assign", name, None, right, names)
    raise DatalogError(f"Cannot parse body literal {text!r}")

# Check if a literal is an atom
def _is_atom_literal(text: str) -> bool:
    return bool(_ATOM.match(text.strip())) and not _CMP.search(text.split("(", 1)[0])

# Reject rules where a variable is used before it's bound
def _check_safety(rule: Rule):
    bound = {t.name for a in rule.atoms for t in a.terms if isinstance(t, Var)}
    for b in rule.builtins:
        if not b.vars <= bound:
            raise DatalogError(f"Unbound variable(s) {b.vars - bound} in rule: {rule.text}")
        if b.kind == "assign":
            bound.add(b.left)
    for t in rule.head.terms:
        if isinstance(t, Var) and t.name not in bound:
            raise DatalogError(f"Head variable {t.name} is not bound in rule: {rule.text}")


# Parse a Datalog program (rules + ground facts + output predicates)
def parse_program(text: str):
    """Return (rules, ground_facts, output_predicates)."""
    outputs, kept = set(), []
    for line in _strip_comments(text).splitlines():
        s = line.strip()
        if s.startswith(".output"):
            outputs.update(p.strip() for p in s[len(".output"):].split(",") if p.strip())
        else:
            kept.append(line)
    rules, facts = [], []
    for stmt in _split_statements("\n".join(kept)):
        if ":-" in stmt:
            head_txt, body_txt = stmt.split(":-", 1)
            head = _parse_atom(head_txt)
            atoms, builtins = [], []
            for lit in _split_top_level(body_txt):
                if lit.startswith("not ") or lit.startswith("!"):
                    raise DatalogError("Negation is not supported (monotone Datalog only)")
                if _is_atom_literal(lit):
                    atoms.append(_parse_atom(lit))
                else:
                    builtins.append(_parse_builtin(lit))
            if not atoms:
                raise DatalogError(f"Rule without body atoms: {stmt}")
            rule = Rule(head, atoms, builtins, " ".join(stmt.split()))
            _check_safety(rule)
            rules.append(rule)
        else:
            atom = _parse_atom(stmt)
            if any(isinstance(t, Var) for t in atom.terms):
                raise DatalogError(f"Facts must be ground: {stmt}")
            facts.append((atom.pred, atom.terms))
    return rules, facts, outputs


# Datalog Engine
class DatalogEngine:
    def __init__(self):
        self.rules: list[Rule] = []
        self.facts: dict[str, set] = defaultdict(set)
        self._index: dict[str, dict[int, dict[Any, set]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self._edb: set = set()
        self.outputs: set[str] = set()
        self.rounds = 0

    # Loading the .dl program and read its text
    def load_program(self, text: str) -> list[Rule]:
        rules, facts, outputs = parse_program(text)
        self.rules.extend(rules)
        self.outputs |= outputs
        for pred, tup in facts:
            self._insert(pred, tup, edb=True)
        return rules

    # Insert a fact
    def _insert(self, pred: str, tup: tuple, edb: bool = False) -> bool:
        if tup in self.facts[pred]:
            return False
        self.facts[pred].add(tup)
        for pos, val in enumerate(tup):
            self._index[pred][pos][val].add(tup)
        if edb:
            self._edb.add((pred, tup))
        return True

    # Add new facts
    def add_facts(self, facts: Iterable[tuple[str, tuple]]) -> dict[str, set]:
        delta: dict[str, set] = defaultdict(set)
        for pred, tup in facts:
            if self._insert(pred, tuple(tup), edb=True):
                delta[pred].add(tuple(tup))
        return delta

    # Add new rules
    def add_rules(self, text: str) -> tuple[list[Rule], dict[str, set]]:
        n_before = len(self.rules)
        before_facts = {p: set(s) for p, s in self.facts.items()}
        self.load_program(text)
        delta = {p: s - before_facts.get(p, set()) for p, s in self.facts.items()}
        return self.rules[n_before:], {p: s for p, s in delta.items() if s}

    # Evaluation
    def evaluate(self, delta: dict[str, set] | None = None, first_round_rules: list[Rule] | None = None) -> int:
        """Passing only new facts as delta allows incremental evaluation."""
        def all_facts():
            return {p: set(s) for p, s in self.facts.items() if s}

        if delta is None:
            delta = all_facts()
        plan = [(r, delta) for r in self.rules]
        if first_round_rules is not None:
            full, fresh = all_facts(), {id(r) for r in first_round_rules}
            plan = [(r, full if id(r) in fresh else delta) for r in self.rules]

        self.rounds, inferred = 0, 0
        while plan:
            self.rounds += 1
            new: dict[str, set] = defaultdict(set)
            for rule, rule_delta in plan:
                for i, atom in enumerate(rule.atoms):
                    if atom.pred not in rule_delta:
                        continue
                    for env in self._solve(rule, i, rule_delta):
                        head = tuple(env[t.name] if isinstance(t, Var) else t for t in rule.head.terms)
                        if head not in self.facts[rule.head.pred]:
                            new[rule.head.pred].add(head)
            delta = {}
            for pred, tuples in new.items():
                for tup in tuples:
                    if self._insert(pred, tup):
                        delta.setdefault(pred, set()).add(tup)
                        inferred += 1
            plan = [(r, delta) for r in self.rules] if delta else []
        return inferred

    # Solving a rule recursively on atoms
    def _solve(self, rule: Rule, delta_idx: int, delta: dict[str, set]) -> Iterator[dict]:
        order = [delta_idx] + [j for j in range(len(rule.atoms)) if j != delta_idx]

        def rec(k: int, env: dict):
            if k == len(order):
                final = self._apply_builtins(rule, env)
                if final is not None:
                    yield final
                return
            j = order[k]
            atom = rule.atoms[j]
            for tup in self._candidates(atom, env, delta if j == delta_idx else None):
                new_env = self._unify(atom, tup, env)
                if new_env is not None:
                    yield from rec(k + 1, new_env)

        yield from rec(0, {})

    # Candidates for an atom, given the current environment and delta
    def _candidates(self, atom: Atom, env: dict, delta: dict | None):
        if delta is not None:
            return list(delta.get(atom.pred, ()))
        for pos, t in enumerate(atom.terms):
            if isinstance(t, Var):
                if t.name in env:
                    return list(self._index[atom.pred][pos].get(env[t.name], ()))
            else:
                return list(self._index[atom.pred][pos].get(t, ()))
        return list(self.facts.get(atom.pred, ()))

    # Unification of an atom with a tuple
    @staticmethod
    def _unify(atom: Atom, tup: tuple, env: dict):
        if len(tup) != len(atom.terms):
            return None
        new_env = dict(env)
        for t, v in zip(atom.terms, tup):
            if isinstance(t, Var):
                if t.name in new_env:
                    if new_env[t.name] != v:
                        return None
                else:
                    new_env[t.name] = v
            elif t != v:
                return None
        return new_env

    # Apply builtins (comparisons and assignments)
    @staticmethod
    def _apply_builtins(rule: Rule, env: dict):
        env = dict(env)
        for b in rule.builtins:
            try:
                if b.kind == "assign":
                    env[b.left] = _eval_expr(b.right, env)
                else:
                    if not _CMPOPS[b.op](_eval_expr(b.left, env), _eval_expr(b.right, env)):
                        return None
            except TypeError:
                return None
        return env

    # Results
    # Derived facts (excludes externally supplied ones)
    def derived(self, preds: Iterable[str] | None = None) -> dict[str, set]:
        preds = set(preds) if preds is not None else set(self.outputs)
        return {p: {t for t in self.facts[p] if (p, t) not in self._edb} for p in preds if p in self.facts}

    # Derived facts as triples (subject, predicate, object)
    def derived_triples(self, preds: Iterable[str] | None = None) -> list[tuple]:
        out = []
        for pred, tuples in self.derived(preds).items():
            for tup in tuples:
                if len(tup) == 2:
                    out.append((tup[0], pred, tup[1]))
        return sorted(out, key=lambda x: (str(x[0]), x[1], str(x[2])))

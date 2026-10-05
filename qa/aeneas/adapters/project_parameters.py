"""Erase an unused dictionary back edge and unused sqrt callback field.

The retained arithmetic bodies are unchanged. The projection fails if one of
them uses the erased parent dictionary or callback, so this is only suitable
for an extraction whose call graph does not include square roots. Transparent
callback functions can be retained as independent proof roots; their bodies
must not refer to either erased dictionary component.
"""

import argparse
import copy
import json
from pathlib import Path

if not __debug__:
    raise RuntimeError("This translation adapter requires Python assertions")


def project(document, retain_callback_functions=False):
    t = document["translated"]
    original = copy.deepcopy(t)
    name = [{"Ident": [x, 0]} for x in
            ["zakura_udon", "field", "pasta", "parameters", "sealed", "Parameters"]]
    pid = next(d["def_id"] for d in t["trait_decls"]
               if d and d["item_meta"]["name"] == name)
    parameters = t["trait_decls"][pid]
    assert len(parameters["implied_clauses"]) == 1
    clause = parameters["implied_clauses"][0]
    assert clause["clause_id"] == 0
    prime_id = clause["trait_"]["skip_binder"]["id"]
    assert t["trait_decls"][prime_id]["item_meta"]["name"][-1] == {"Ident": ["PrimeModulus", 0]}
    assert len(parameters["methods"]) == 1
    assert parameters["methods"][0]["skip_binder"]["name"] == "pow_sqrt_exponent"
    implementations = {i for i, d in enumerate(t["trait_impls"])
                       if d and d["impl_trait"]["id"] == pid}
    callbacks = set()
    for i in implementations:
        d = t["trait_impls"][i]
        assert len(d["implied_trait_refs"]) == 1 and len(d["methods"]) == 1
        callbacks.add(d["methods"][0]["skip_binder"]["id"])
    assert len(callbacks) == 2
    for i in callbacks:
        if retain_callback_functions:
            assert set(t["fun_decls"][i]["body"]) == {"Structured"}
        else:
            assert t["fun_decls"][i]["body"] == "Opaque"

    def check(x):
        if isinstance(x, dict):
            if "ParentClause" in x:
                parent, _ = x["ParentClause"]
                assert parent["Untagged"]["trait_decl_ref"]["skip_binder"]["id"] != pid
            if "Fun" in x and isinstance(x["Fun"], int):
                assert x["Fun"] not in callbacks
            if "TraitMethod" in x and isinstance(x["TraitMethod"], dict):
                assert x["TraitMethod"].get("trait_id") != pid
            for v in x.values():
                check(v)
        elif isinstance(x, list):
            for v in x:
                check(v)

    for i, f in enumerate(t["fun_decls"]):
        if f and (i not in callbacks or retain_callback_functions):
            check(f["body"])
            check(f["signature"])
    for g in t["global_decls"]:
        if g:
            check(g["value"])
    parameters["implied_clauses"] = []
    parameters["methods"] = []
    if not retain_callback_functions:
        t["assoc_item_names"][pid]["methods"] = []
    for i in implementations:
        t["trait_impls"][i]["implied_trait_refs"] = []
        t["trait_impls"][i]["methods"] = []
    for i in callbacks:
        if retain_callback_functions:
            # Aeneas otherwise drops these now-unreferenced dependency functions.
            t["fun_decls"][i]["item_meta"]["is_local"] = True
        else:
            t["fun_decls"][i] = None

    order = []
    for declaration in t["ordered_decls"]:
        kind, group = next(iter(declaration.items()))
        if kind == "Fun" and group.get("NonRec") in callbacks and not retain_callback_functions:
            continue
        if kind == "TraitDecl" and "Rec" in group and pid in group["Rec"]:
            assert set(group["Rec"]) == {pid, prime_id}
            order.extend([{"TraitDecl": {"NonRec": pid}},
                          {"TraitDecl": {"NonRec": prime_id}}])
        elif kind == "Mixed" and retain_callback_functions:
            members = group["Rec"]
            assert len(members) == 3
            impls = {m["TraitImpl"] for m in members if "TraitImpl" in m}
            funs = {m["Fun"] for m in members if "Fun" in m}
            assert len(impls) == 2 and len(funs) == 1
            assert funs <= callbacks
            first = next(i for i in impls if i in implementations)
            second = next(i for i in impls if i not in implementations)
            assert t["trait_impls"][second]["impl_trait"]["id"] == prime_id
            assert original["trait_impls"][first]["methods"][0]["skip_binder"]["id"] in funs
            order.extend([{"TraitImpl": {"NonRec": first}},
                          {"TraitImpl": {"NonRec": second}},
                          {"Fun": {"NonRec": next(iter(funs))}}])
        elif kind == "TraitImpl" and "Rec" in group and implementations.intersection(group["Rec"]):
            assert len(group["Rec"]) == 2
            first = next(i for i in group["Rec"] if i in implementations)
            second = next(i for i in group["Rec"] if i not in implementations)
            assert t["trait_impls"][second]["impl_trait"]["id"] == prime_id
            order.extend([{"TraitImpl": {"NonRec": first}},
                          {"TraitImpl": {"NonRec": second}}])
        else:
            order.append(declaration)
    t["ordered_decls"] = order
    for i, f in enumerate(original["fun_decls"]):
        if i in callbacks and retain_callback_functions:
            expected = copy.deepcopy(f)
            expected["item_meta"]["is_local"] = True
            assert t["fun_decls"][i] == expected
        elif i not in callbacks:
            assert t["fun_decls"][i] == f
    assert original["type_decls"] == t["type_decls"]
    assert original["global_decls"] == t["global_decls"]
    return {"erased_unused_parent": pid, "erased_unused_callback_fields": sorted(callbacks),
            "retained_callback_functions": sorted(callbacks) if retain_callback_functions else [],
            "retained_function_bodies_unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--retain-callback-functions", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.source.read_text())
    report = project(data, args.retain_callback_functions)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(report, indent=2))

/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include ZZ_Prelude_hh
#include "DeriveGenealogy.hh"

namespace ZZ {
using namespace std;
using namespace ENUM;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


// Does 'expr' match the expression of 'S' rooted at 'idx'?
static bool match(uint idx, State const& S, Expr const& expr_, Pool pool, SymTable<uint>& tab)
{
    Expr const* new_expr;
    if (expr_.kind == expr_RecDef){
        assert(expr_[0].kind == expr_Sym);
        if (expr_[0].targs.psize() != 0){
            wrLn("ERROR! Cannot handle recursive definitions with type variables: %_", expr_[0]); exit(1); }
        tab.addQ(expr_[0].name, idx);    // -- add symbol for recursive call
        new_expr = &expr_[1];
    }else
        new_expr = &expr_;

    Expr const& expr = *new_expr;


    //**/Dump(idx); Dump(S); Dump(expr); tab.dumpVal(); newLn();
    auto poolSym = [&](Expr const& e) {
        for (uint i = 0; i < pool.size(); i++)
            if (pool.sym(i) == e)
                return i;
        return UINT_MAX;    // -- no symbol
    };

    GExpr ge = S[idx];
    if (ge.type() != expr.type)
        return false;

    if (expr.kind == expr_Sym && tab.has(expr.name) && tab[expr.name] == idx)
        return true;

    switch (ge.kind){
    case g_Pool:
        return poolSym(expr) == ~ge.ins[0];
    case g_PI:
        if (expr.kind != expr_Sym || expr.targs.psize() != 0) return false;
        return tab[expr] == idx;
    case g_Id:
        return match(ge.ins[0], S, expr, pool, tab);
    case g_Appl:
        return expr.kind == expr_Appl
            && match(ge.ins[0], S, expr[0], pool, tab)
            && match(ge.ins[1], S, expr[1], pool, tab);
    case g_Tuple:
        if (expr.kind != expr_Tuple || ge.ins.psize() != expr.size()) return false;
        for (uint i = 0; i < ge.ins.psize(); i++)
            if (!match(ge.ins[i], S, expr[i], pool, tab)) return false;
        return true;
    case g_Sel:
        return expr.kind == expr_Sel && ~ge.ins[0] == stringToUInt64(expr.name)
            && match(ge.ins[1], S, expr[0], pool, tab);
    case g_Lamb:{
        if (expr.kind != expr_Lamb) return false;
        // Add formal argument to sub-symboltable 'stab':
        SymTable<uint> stab(tab);
        Array<Expr const> formals = tupleSlice(expr[0]);
        GExpr const& g_tup = S[ge.ins[0]]; assert(g_tup.kind == g_Tuple);
        if (g_tup.ins.psize() != formals.size()) return false;
        for (uint i = 0; i < formals.size(); i++){
            if (formals[i].kind != expr_Sym || formals[i].targs.psize() != 0) return false;
            stab.addQ(formals[i].name, g_tup.ins[i]);   // <<== except for '_'
        }
        // Match body:
        if (expr[1].kind != expr_Block || expr[1].size() != 1) return false;
        return match(ge.ins[1], S, expr[1][0], pool, stab); }
    case g_Obl:
        if (ge.ins.psize() == 0)
            return true;
        else{
            Vec<State> Q{S};
            for (uint q = 0; q < Q.size(); q++){
                if (Q[q][idx].kind == g_Obl && Q[q][idx].ins.psize() > 0){
                    auto enqueue = [&](State const& S_new) -> void { Q.push(S_new); };
                    expandOne(pool, Q[q], idx, enqueue, Params_SynthEnum());
                }else{
                    if (match(idx, Q[q], expr, pool, tab))
                        return true;
                }
            }
            return false;
        }
    default: assert(false); }
    return false;
}


static bool match(State const& S, Expr const& expr, Pool pool)
{
    SymTable<uint> tab;
    return match(0, S, expr, pool, tab);
}


static void addObligationsToPool(Expr const& expr, Vec<CExpr>& out_syms)
{
    static Atom a_obl("obl");
    if (expr.kind == expr_Sym && expr.name == a_obl){
        if (!has(out_syms, CExpr(1.0, expr)))
            out_syms.push(CExpr(1.0, expr));
    }else{
        for (Expr const& e : expr)
            addObligationsToPool(e, out_syms);
    }
}


// <<== NOTE! This may generate 'Obl's in incorrect positions (need mechanism to prevent using symbols from different scope)
static State withProperStateObls(State S, Pool pool)
{
    static Atom a_obl("obl");
    for (uint i = 0; i < S.size(); i++){
        GExpr const& ge = S[i];
        if (ge.kind == g_Pool){
            Expr const& e = pool.sym(~ge.ins[0]);
            if (e.kind == expr_Sym && e.name == a_obl)
                S = S.set(i, GExpr(g_Obl, {}, e.type));
        }
    }
    return S;
}


static void deriveGenealogy(State const& S0, Expr const& target, Pool pool, Params_SynthEnum const& P_enum, Vec<State>& out_hist)
{
    State S_match;
    auto enqueue = [&](State const& S) -> void {
        //**/wrLn("//====================================================================");
        //**/wrLn("// Trying match...\n");
        //**/wrLn("%x\n", S);
        //**/wrLn("Target:\n%_\n", ppFmtI(target, 4));
        if (match(S, target, pool)){
            if (S_match.size() != 0){
                //**/wrLn("HMM! Ambiguous match. Can this happen?");
                //**/wrLn("First match : %x", S_match);
                //**/wrLn("Second match: %x", S);
                //**/exit(1);
            }else
                S_match = S;
        }
    };

    uint tgt_i;
    if (!S0.getLast(ENUM::g_Obl, tgt_i))
        return;     // -- DONE!

    expandOne(pool, S0, tgt_i,  enqueue, P_enum);

    if (S_match.size() != 0){
        out_hist.push(withProperStateObls(S_match, pool));
        deriveGenealogy(S_match, target, pool, P_enum, out_hist);
    }else{
        wrLn("ERROR! Failed to construct genealogy.");
        wrLn("TARGET:\n%_", ppFmtI(target, 4));
        wrLn("FAILURE POINT:\n%_", ppFmtI(S0.expr(pool), 4));
        wrLn("S0=%x", S0);
        exit(1);
    }
}


Vec<State> deriveGenealogy(Expr const& target, Pool pool0)
{
    // Add 'obl<T>'s occuring in 'target' to pool:
    Vec<CExpr> new_pool(copy_, pool0);
    addObligationsToPool(target, new_pool);
    Pool pool(new_pool, pool0.costAppl(), pool0.costSel(), pool0.costLamb(), pool0.costTuple());

    Vec<State> result;      // <<== inline all 'let' expressions, creating a DAG for 'target' (won't solve the problem properly)
    deriveGenealogy(initialEnumState(target.type), target, pool, Params_SynthEnum(), result);
    return result;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

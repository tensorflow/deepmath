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
#include "SynthPrune.hh"
#include "zz/Generics/Sort.hh"


/*-------------------------------------------------------------------------------------------------
There are three rule constructs:
  - unordered(<rule>)
  - appl(<pool-sym>, <args>)
  - wild <atom>

The "unordered" construct can only be applied at top-level. Example syntax and their meaning:
    unordered(appl(add_, (wild"a", wild"b")))       ~>  unordered add_(%0, %1)
    appl(sub_, (wild "a", wild "a"))                ~>  sub_(0, %0)
    appl(mul_, (appl(neg_, wild "b"), wild "a"))    ~>  mul_(neg_(%1), %0)
-------------------------------------------------------------------------------------------------*/


namespace ZZ {
using namespace std;
using namespace ENUM;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Globals:


cchar* PruneKind_name[PruneKind_size] = {
    "<null>",
    "Appl",
    "Const",
    "Tuple",
    "Wild",
    "Unordered",
};


Arr<ushort> p_rules_empty({});      // -- 'p_rules_empty{}' or 'p_rules_empty = {}' will call default constructor rather than initializer_list constructor


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


static void checkNoInternalUnordered(Prune const& prune) {
    if (prune.kind == prune_Unordered){ wrLn("ERROR! Ill-formed pruning expression: 'unordered' can only be used at the top-level."); exit(1); }
    else for (Prune const& p : prune.exprs) checkNoInternalUnordered(p); }

static void collectWildcards(Prune const& prune, Vec<Atom>& out){
    if (prune.kind == prune_Wild) out += Atom(prune.num);
    else for (Prune const& p : prune.exprs) collectWildcards(p, out); }

static void remapWildcards(Prune& prune, IntTmpMap<Atom,uint> const& atom2idx){
    if (prune.kind == prune_Wild) prune.num = ~atom2idx[prune.num];
    else for (Prune& p : prune.exprs) remapWildcards(p, atom2idx); }


// Returns 'l_True' if state 'S' matches a rule and hence should be pruned (discarded),
// 'l_False' if the rule does not hold, and 'l_Undef' if expression is incoplete wrt. the rule.
//
// NOTE! For efficiency, a stronger equality (node index) is used for non-"unordered" patterns
// than for "unordered" ones (which use 'State::order', a recursively defined subexpression order).
static lbool ruleApplies(Prune const& rule, State const& S, uint idx, IntMap<uint,uint>& wild_binds)
{
    for(;;){
        GExpr const& g = S[idx];
        if (g.kind == g_Id)
            idx = g.ins[0];
        else if (g.kind == g_Sel && S[g.ins[1]].kind == g_Tuple)
            /**/wrLn(">> Sel/tuple redundancy: %x", S),
            idx = S[g.ins[1]].ins[~g.ins[0]];
        else
            break;
    }
    GExpr const& g = S[idx];
    assert(g.kind != g_Scope && g.kind != g_Begin && g.kind != g_End);

    if (g.kind == g_Obl) return l_Undef;     // -- expression is incomplete; cannot check rule

    switch (rule.kind){
    case prune_Wild:{
        while (S[idx].kind == g_Id) idx = S[idx].ins[0];
        uint n = ~rule.num;    // -- '~' for normalized indicies, otherwise 'Atom(rule.num)' is the text of the wildcard.
        if      (wild_binds[n] == UIND_MAX) wild_binds(n) = idx;
        else if (wild_binds[n] != idx)      return l_False;     // <<== could experiment with weaker equality here using 'order()'; NOTE! Need de-bruijn normalization of lambda variables
        return l_True; }

    case prune_Const:
        return lbool_lift(g.kind == g_Pool && rule.num == ~g.ins[0]);

    case prune_Unordered:{
        lbool ret = ruleApplies(rule.exprs[0], S, idx, wild_binds);
        if (ret == l_True){
            bool incomplete = false;
            for (uint i = 1; i < wild_binds.size(); i++){
                int ord = S.order(wild_binds[i-1], wild_binds[i]);
                if (ord == 1) return l_True;
                else if (ord == INT_MIN) incomplete = true;
            }
            return incomplete ? l_Undef : l_False;
        }
        return ret; }

    case prune_Tuple:
        if (g.kind == g_Tuple && g.ins.psize() == rule.exprs.psize()){
            bool incomplete = false;
            for (uint i = 0; i < g.ins.psize(); i++){
                lbool ret = ruleApplies(rule.exprs[i], S, g.ins[i], wild_binds);
                if (ret == l_False) return l_False;
                else if (ret == l_Undef) incomplete = true;
            }
            return incomplete ? l_Undef : l_True;
        }
        return l_False;

    case prune_Appl:
        if (g.kind == g_Appl){
            GExpr const& g_fun = S[g.ins[0]];
            if (g_fun.kind != g_Pool) return l_False;
            if (~g_fun.ins[0] != rule.num) return l_False;
            return ruleApplies(rule.exprs[0], S, g.ins[1], wild_binds);
        }
        return l_False;

    default: assert(false); }
}


bool Pruner::shouldPrune(State& S) const
{
    Vec<ushort> p_rules;
    IntMap<uint,uint> wild_binds(UINT_MAX);
    for (uint i = 0; i < S.size(); i++){
        GExpr g = S[i];
        if (g.kind == g_Appl){
            if (g.p_rules == p_rules_empty) continue;

            // Set up which rules to try:
            if (g.p_rules)
                vecCopy(g.p_rules, p_rules);
            else{
                p_rules.clear();
                for (uint r = 0; r < rules.size(); r++)
                    p_rules.push(r);
            }

            uint j = 0;
            for (uint r : p_rules){     // <<== do experiment where all rules are tried and excluded ones must be false
                Prune const& rule = rules[r];
                wild_binds.clear();
                lbool ret = ruleApplies(rule, S, i, wild_binds);
                if (ret == l_Undef)
                    p_rules[j++] = r;
                else if (ret == l_True)
                    return true;
            }
            p_rules.shrinkTo(j);

            if (p_rules.size() == 0){
                g.p_rules = p_rules_empty;
                S = S.set(i, g);
            }else if (!g.p_rules || p_rules.size() < g.p_rules.size()){
                g.p_rules = Arr<ushort>(p_rules);
                S = S.set(i, g);
            }
        }
    }
    return false;
}


//=================================================================================================
// -- Constructor:


static Prune buildPruneRule(Expr const& rule, Pool const& pool)
{
    static Atom a_appl("appl");
    static Atom a_wild("wild");
    static Atom a_unordered("unordered");

    auto poolSym = [&](Expr const& e) {
        for (uint i = 0; i < pool.size(); i++)
            if (pool.sym(i) == e)
                return i;
        throw Excp_ParseError(fmt("Pruning expression contains symbol not in symbol pool: %_", e));
    };

    if (rule.kind == expr_Tuple){
        return Prune(prune_Tuple, map(rule, [&](Expr const& e){ return buildPruneRule(e, pool); }));

    }else if (rule.kind == expr_Appl){
        if (rule[0].kind == expr_Sym){
            if (rule[0].name == a_appl){
                if (rule[1].kind != expr_Tuple || rule[1].size() != 2) wrLn("ERROR! Ill-formed pruning expression: %_\n  - expected two arguments after 'appl'"), exit(1);
                return Prune(prune_Appl, poolSym(rule[1][0]), {buildPruneRule(rule[1][1], pool)});

            }else if (rule[0].name == a_wild){
                if (rule[1].kind != expr_Lit || rule[1].type.name != a_Atom) wrLn("ERROR! Ill-formed pruning expression: %_\n  - 'wild' must be followed by an Atom", rule), exit(1);
                return Prune(prune_Wild, +rule[1].name, {});

            }else if (rule[0].name == a_unordered){
                return Prune(prune_Unordered, {buildPruneRule(rule[1], pool)});
            }
        }
        wrLn("ERROR! Ill-formed pruning expression: %_\n  - expected symbol: appl, wild, unordered", rule), exit(1);

    }else
        return Prune(prune_Const, poolSym(rule), {});
}


static Vec<Prune> buildPruneRules(Arr<Expr> const& pruning, Pool const& pool, bool verbose)
{
    Vec<Prune> result;
    for (Expr const rule : pruning){
        try{
            Prune p = buildPruneRule(rule, pool);
            result.push(p);
        }catch (Excp_ParseError err){
            if (verbose) wrLn("NOTE! %_", err.msg);
        }
    }

    // Wildcards are sorted alphabetically, and for 'unordered' rules, this is the order which is allowed.
    IntTmpMap<Atom,uint> atom2idx;
    for (Prune& prune : result){
        if (prune.kind == prune_Unordered){
            checkNoInternalUnordered(prune.exprs[0]);
            if (prune.exprs[0].kind != prune_Appl){ wrLn("ERROR! Ill-formed pruning expression; pattern must start with 'appl'."); exit(1); }
        }else
            if (prune.kind != prune_Appl){ wrLn("ERROR! Ill-formed pruning expression; pattern must start with 'appl'."); exit(1); }

        Vec<Atom> wilds;
        collectWildcards(prune, wilds);
        sortUnique(wilds, [](Atom x, Atom y){ return strcmp(x.c_str(), y.c_str()) < 0; });

        atom2idx.clear();
        for (uint i = 0; i < wilds.size(); i++)
            atom2idx(wilds[i]) = i;
        remapWildcards(prune, atom2idx);
    }
    if (verbose && result.size() > 0){
        wrLn("\a*PRUNING RULES:\a*");
        for (Prune const& prune : result) wrLn("  %_", PruneFmt{prune, pool});
    }

    return result;
}


void Pruner::init(Arr<Expr> const& pruning, Pool const& pool_, bool verbose)
{
    pool = pool_;
    rules = buildPruneRules(pruning, pool, verbose);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

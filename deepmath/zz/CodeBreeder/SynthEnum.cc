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
#include "SynthEnum.hh"

#include "zz/Generics/IntSet.hh"


namespace ZZ {
using namespace std;
using namespace ENUM;


ZZ_PTimer_Add(synth_enum_State_expr);
ZZ_PTimer_Add(synth_enum_expandOne);
ZZ_PTimer_Add(synth_enum_prepareEnqueue);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


// Can you extract something of type 't' from something of type 't0' (possibly by providing a function argument)?
static bool canProduce(Type const& t0, Type const& t) {
  if (t == t0) return true;
  if (t0.name == a_Fun) return canProduce(t0[1], t);
  if (t0.name == a_Tuple)
    return !trueForAll(t0, [t](Type const& s) { return !canProduce(s, t); });
  return false;
}


// Make sure 'e' is of type 'expr_Block' so that it fits the body of a lambda.
static Expr wrapBlock(Expr const& e) {
    if (e.kind == expr_Block) return e;
    Vec<Expr> es(1, e);
    return Expr::Block(es).setType(Type(e.type));
}


namespace ENUM {
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Generalized expression (for enumeration):


cchar* GExprKind_name[GExprKind_size] = {
    "NULL",
    "Pool",
    "PI",
    "Id",
    "Appl",
    "Tuple",
    "Sel",
    "Lamb",
    "Obl",
    "Scope",
    "Begin",
    "End",
};


// For space efficiency, map types to integers. Types are necessarily big because they store locations
// (for error messages), which we don't need here.
Vec<Type>       id2type;
Map<Type, uint> type2id;


uint getId(Type const& type) {
    uint *id;
    if (!type2id.get(type, id)){
        if (id2type.size() == 0) id2type.push();
        *id = id2type.size();
        Type ty = type; ty.loc = Loc();
        id2type.push(ty);
    }
    return *id;
}


}
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Search state:


bool State::getLast(GExprKind kind, uint& tgt_i) const
{
    PArr<GExpr> const& me = *this;
    return !me.enumAllCondRev([&](uind i, GExpr const& g){
        if (g.kind == kind){
            tgt_i = i;
            return false;   // -- break
        }else
            return true;    // -- continue
    });
}


// Returns range '[tgt_i0, tgt_i1[' of legal obligations to expand, or returns FALSE if
// no outstanding obligations.
bool State::getOblRange(uint& tgt_i0, uint& tgt_i1) const
{
    PArr<GExpr> const& me = *this;
    tgt_i0 = UINT_MAX;
    tgt_i1 = UINT_MAX;
    for (uint i = me.size(); i > 0;){ i--;      // <<== replace this by 'enumAllCondRev()'
        if (tgt_i1 != UINT_MAX && me[i].kind == g_Scope) break;
        if (me[i].kind == g_Obl){
            if (tgt_i1 == UINT_MAX) tgt_i1 = i+1;
            tgt_i0 = i;
        }
    }
    return tgt_i0 != UINT_MAX;
}


double State::cost() const
{
    PArr<GExpr> const& me = *this;
    double sum = 0;
    me.forAll([&](GExpr const& g) { sum += g.cost; });
    return sum;
}


Expr State::expr(Pool const& P, bool obl_as_fail) const
{
    ZZ_PTimer_Scope(synth_enum_State_expr);

    PArr<GExpr> const& me = *this;

    // Compute fanouts and begin/end of scope:
    Vec<uint> n_fanouts(me.size(), 0);
    n_fanouts(0)++; // -- top-level expression has an external reference not accounted for
    me.forAll([&](GExpr const& g) {
        for (uint in : +g.ins) if ((int)in >= 0) n_fanouts(in)++; });

    Vec<uint> stack(1, 0);
    Vec<uint> begin_idx(me.size());
    me.enumAll([&](uind i, GExpr const& g){
        if      (g.kind == g_Begin || g.kind == g_Scope) stack.push(i);
        else if (g.kind == g_End)                        stack.pop();
        begin_idx[i] = stack.last();
    });

    for (uint& s : stack) s = me.size();    // -- essentially push enough 'g_End' to match the extra 'g_Begin/g_Scope's
    Vec<uint> end_idx(me.size());
    me.enumAllRev([&](uind i, GExpr const& g){
        if      (g.kind == g_End)                        stack.push(i);
        else if (g.kind == g_Begin || g.kind == g_Scope) stack.pop();
        end_idx[i] = stack.last();
    });

    // Create expression:
    Vec<Expr> memo(me.size());

    function<Expr(uint)> build;

    function<Expr(uint)> buildMemo = [&](uint i) {
        if (!memo[i]) memo[i] = build(i);
        return memo[i];
    };

    function<Expr(uint)> buildTop = [&](uint i) {
        if (memo[i]) return memo[i];

        Vec<Expr> block;
        Vec<uint> shared;
        // Add names of multi-fanout symbols (rec-defs):
        for (uint j = begin_idx[i]; j < end_idx[i]; j++){
            if (begin_idx[j] != begin_idx[i]) continue; // -- skip nested scopes
            GExpr g = me[j];
            if (n_fanouts[j] > 1 && g.kind != g_PI && g.kind != g_Id && g.kind != g_Pool){
                shared.push(j);
                memo(j) = Expr::Sym(fmt("t%_", j), {}, Type(g.type()));
            }
        }
        // Create block expression:
        for (uint j : shared)
            block.push(Expr::RecDef(Expr(memo[j]), Type(me[j].type()), build(j)));
        block.push(buildMemo(i));
        return (block.size() == 1) ? block[0] : Expr::Block(block).setType(Type(block[LAST].type));
    };

    Atom a_obl("?");
    Atom a_adapt("?<");
    build = [&](uint i) -> Expr {
        GExpr g = me[i];
        switch (g.kind){
        case g_Pool : return *P.syms[~g.ins[0]];
        case g_PI   : return Expr::Sym(fmt("a%_", i), {}, Type(g.type()));
        case g_Id   : return buildMemo(g.ins[0]);
        case g_Appl : return Expr::Appl(buildMemo(g.ins[0]), buildMemo(g.ins[1])).setType(Type(g.type()));
        case g_Sel  : return Expr::Sel(buildMemo(g.ins[1]), fmt("%_", ~g.ins[0]).slice()).setType(Type(g.type()));
        case g_Tuple: return Expr::Tuple(map(g.ins, [&](uint j){ return buildMemo(j); })).setType(Type(g.type()));
        case g_Lamb : return Expr::Lamb(build(g.ins[0]), wrapBlock(buildTop(g.ins[1])), Type(g.type()));
        case g_Obl  :
            if (obl_as_fail){
                return Expr::Sym(a_fail, {g.type()}, Type(g.type()));
            }else{
                if (g.ins.psize() == 0) return Expr::Sym(a_obl, {}, Type(g.type()));
                else return Expr::Appl(Expr::Sym(a_adapt, {}, Type(/*no-type*/)), buildMemo(g.ins[0])).setType(Type(g.type()));
            }
        default: assert(false); }
    };

    return buildTop(0);
}


//=================================================================================================
// -- prepare state for enqueue:


// parent: 0=unvisited, 1..n=parent_idx+1 (UINT_MAX=no_parent, UINT_MAX-1 visited and proven safe)
static bool checkSimpleLeftRecursion(State const& S, uint i, IntMap<uint,uint>& parent, bool& was_recursive, uint from = UINT_MAX-1)
{
    if (parent[i] == 0){
        // Recurse:
        assert(from != UINT_MAX);
        parent(i) = from+1;
        for (uint in : +S[i].ins){
            if ((int)in >= 0 && !checkSimpleLeftRecursion(S, in, parent, was_recursive, i))
                return false;
        }
        parent(i) = UINT_MAX-1; // -- mark

    }else if (parent[i] != UINT_MAX-1){
        uint p = from;
        bool safe = false;
        while (p != i){
            if (S[p].kind == g_Lamb){ safe = true; break; }
            p = parent[p]-1;
        }
        if (!safe)
            return false;
        was_recursive = true;
    }

    return true;
}


// Will modify 'S' if scope has been finished (no outstanding obligations). Returns TRUE if resulting
// state is good and shuold be enqueued, FALSE if state should be discarded (e.g. contains left recursion).
static bool prepareEnqueue(State& S, IntMap<uint,uint>& used, Pool const& pool, bool must_use_formals, bool force_recursion)
{
    ZZ_PTimer_Scope(synth_enum_prepareEnqueue);

    // Resolve obligations of type 'Void' (only one meaningful option: '()')
    S.enumAll([&S](uind i, GExpr const& ge) {
        if (ge.kind == g_Obl && ge.ins.psize() == 0 && ge.type().name == a_Void)
            S = S.set(i, GExpr(g_Tuple, {}, Type(a_Void), 0.0));
    });

    // Check that if scope was closed, all function arguments were used:
    uint scope_i;
    while (S.getLast(g_Scope, scope_i)){
        uint tgt_i;
        if (!S.getLast(g_Obl, tgt_i) || tgt_i < scope_i){
            // Scope has been finished:
            S = S.set(scope_i, GExpr(g_Begin));
            S = S.push(GExpr(g_End));

            used.clear();
            for (uint i = scope_i+1; i < S.size(); i++){
                for (uint in : +S[i].ins)
                    if ((int)in >= 0) used(in)++;
            }
            for (uint i = scope_i+1; i < S.size(); i++){
                GExpr g = S[i];
                if (must_use_formals && g.kind == g_PI && used[i] < 2)
                    return false;    // -- ABORT
                g = S[i];
                g.internal = true;
                S = S.set(i, g);
             }
        }else
            break;
    }

    // Make sure there is no trivial left-recursion (reference to parent lambda not under a lambda itself):
    used.clear();
    bool was_recursive = false;
    if (!checkSimpleLeftRecursion(S, 0, used, was_recursive))
        return false;     // -- ABORT

    if (force_recursion){   // -- if no more outstanding obligations, there must be at least one recursive call
        uint tgt_i;
        if (!was_recursive && !S.getLast(g_Obl, tgt_i))
            return false;
    }

    return true;
}


//=================================================================================================
// -- expand one obligation:

static bool inFanin(uint tgt, const State& S, uint n, IntSeen<uint>& seen) {
  if (n == tgt) return true;
  if (!seen.add(n)) {
    for (uint m : +S[n].ins)
      if ((int)m >= 0 && inFanin(tgt, S, m, seen)) return true;
  }
  return false;
}


// Is 'tgt' in the transitive fanin of 'n'?
static bool inFanin(uint tgt, State S, uint n) {
    IntSeen<uint> seen;
    return inFanin(tgt, S, n, seen); }


inline bool reusableExpr(GExprKind kind) {
    // These are the nodes that if created as a subexpresson for one obligaton can be used by another
    // obligation (unless marked "internal"). PIs have to be included for technical reasons.
    return kind == g_Appl || kind == g_Tuple || kind == g_Sel || kind == g_Lamb || kind == g_PI; }


// NOTE! It is 'enqueue's responsibility to check that a closed scope meets the desired
// constraints and to remove the scope delimiter.
void expandOne(Pool const& pool, State S, uint tgt_i, function<void(State)> enqueue0, Params_SynthEnum const& P_, uint64* expansion_attempts)
{
    ZZ_PTimer_Scope(synth_enum_expandOne);

    IntMap<uint,uint> tmp_used;
    auto enqueue = [&](State S) {
        if (expansion_attempts) *expansion_attempts += 1;
        if (prepareEnqueue(S, tmp_used, pool, P_.must_use_formals, P_.force_recursion))
            enqueue0(S);
    };

    double min_adapt_cost = min_(pool.cost_Sel, pool.cost_Appl);

    GExpr f = S[tgt_i];     // -- f=final=target
    if (f.ins.psize() > 0){
        // Adaptor obligation:
        uint in_i = f.ins[0];
        GExpr e = S[in_i];
        if (e.type().name == a_Tuple){
            for (uint k = 0; k < e.type().size(); k++){
                if (e.type()[k] == f.type())
                    enqueue(S.set(tgt_i, GExpr(g_Sel, {~k, in_i}, e.type()[k], pool.cost_Sel)));
                else if (canProduce(e.type()[k], f.type())){
                    State S1 = S.push(GExpr(g_Sel, {~k, in_i}, e.type()[k]));
                    enqueue(S1.set(tgt_i, GExpr(g_Obl, {S.size()}, f.type(), min_adapt_cost)));
                }
            }

        }else{ assert(e.type().name == a_Fun);
            State S1 = S.push(GExpr(g_Obl, {}, e.type()[0], 0.0));    // <<== type-based lower bound computation goes here
            if (e.type()[1] == f.type()){
                State S2 = S1.set(tgt_i, GExpr(g_Appl, {in_i, S.size()}, e.type()[1], pool.cost_Appl));
                enqueue(S2); }
            else{
                State S2 = S1.push(GExpr(g_Appl, {in_i, S.size()}, e.type()[1], pool.cost_Appl));
                enqueue(S2.set(tgt_i, GExpr(g_Obl, {S1.size()}, f.type(), min_adapt_cost))); }
        }

    }else{
        // General obligation, resolved from symbol pool:
        for (uint i = 0; i < pool.syms.size(); i++){
            Type const& type = pool.syms[i]->type;
            if (type == f.type())
                enqueue(S.set(tgt_i, GExpr(g_Pool, {~i}, type, pool.syms[i].cost, /*internal*/true)));
            else if (canProduce(type, f.type())){
                State S1 = S.push(GExpr(g_Pool, {~i}, type, pool.syms[i].cost, /*internal*/true));
                enqueue(S1.set(tgt_i, GExpr(g_Obl, {S.size()}, f.type())));
            }
        }

        // General obligation, resolved from previous expression:
        for (uint i = 0; i < S.size(); i++){
            GExpr e = S[i];
            if (e.internal || !reusableExpr(e.kind)) continue;
            if (e.type().name != a_Fun && inFanin(tgt_i, S, i)) continue;     // -- if target is of non-function type and 'tgt_i' is in transitive fanin, skip (otherwise creating cycle)

            if (e.type() == f.type())
                enqueue(S.set(tgt_i, GExpr(g_Id, {i}, f.type())));
            else if (canProduce(e.type(), f.type())){
                enqueue(S.set(tgt_i, GExpr(g_Obl, {i}, f.type(), min_adapt_cost))); }
        }

        // General obligation, resolved by composition:
        if (f.type().name == a_Fun){
            S = S.push(GExpr(g_Scope, {}, Type()));
            Vec<uint> tup_elems;
            for (Type const& t : tupleSlice(f.type()[0])){
                tup_elems.push(S.size());
                S = S.push(GExpr(g_PI, {}, t, 0.0));
            }
            uint head_i = S.size(); S = S.push(GExpr(g_Tuple, tup_elems, f.type()[0], 0.0, /*internal*/true));    // -- note: we allow tuples of size one here; will be removed by expression construction
            uint body_i = S.size(); S = S.push(GExpr(g_Obl, {}, f.type()[1], 0.0, /*internal*/true));
            enqueue(S.set(tgt_i, GExpr(g_Lamb, {head_i, body_i}, f.type(), pool.cost_Lamb, /*internal*/P_.ban_recursion)));

        }else if (f.type().name == a_Tuple){
            Vec<uint> obls;
          #if 0
            for (Type const& t : f.type()){
                obls.push(S.size());
                S = S.push(GExpr(g_Obl, {}, t, 0.0, /*internal*/true)); // <<== type-based lower bound computation goes here
            }
          #else
            for (uint i = f.type().size(); i > 0;){ i--; Type t = f.type()[i];
                obls.push(S.size());
                S = S.push(GExpr(g_Obl, {}, t, 0.0, /*internal*/true)); // <<== type-based lower bound computation goes here
            }
            reverse(obls);
          #endif
            enqueue(S.set(tgt_i, GExpr(g_Tuple, obls, f.type(), pool.cost_Tuple)));
        }
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

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
  if (t0.name == a_Tuple) return !trueForAll(t0, [t](Type const& s) { return !canProduce(s, t); });
  return false;
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


Expr State::expr(Pool const& P, bool obl_as_fail, uint* n_cov_points) const
{
    ZZ_PTimer_Scope(synth_enum_State_expr);

    static Atom a_obl("?");
    static Atom a_adapt("??");
    static Atom a_coverage("\"coverage\"");
    static Atom a_x("x");

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
    if (n_cov_points) *n_cov_points = 0;

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
        for (uint j : shared){
            Expr e_def = build(j);
            if (e_def.type.name == a_Fun && e_def.kind != expr_Lamb){
                // If RHS is function type but not a 'Lamb' expression, apply reverse eta-reduction.
                //     e_def :A->B   ~>  \:A->B a { e_def a }
                // (Evo currently has limited support for mutual recursion in rec-lets)
                Expr e_x = mxSym(a_x, e_def.type[0]);
                e_def = mxLamb(e_x, mxAppl(e_def, e_x));
            }
            assert(e_def.type == me[j].type());
            block.push(Expr::RecDef(Expr(memo[j]), Type(me[j].type()), move(e_def)));
        }
        block.push(buildMemo(i));
        return (block.size() == 1) ? block[0] : Expr::Block(block).setType(Type(block[LAST].type));
    };

    build = [&](uint i) -> Expr {
        GExpr g = me[i];
        switch (g.kind){
        case g_Pool : return P.sym(~g.ins[0]);
        case g_PI   : return Expr::Sym(fmt("a%_", i), {}, Type(g.type()));
        case g_Id   : return buildMemo(g.ins[0]);
        case g_Appl : return Expr::Appl(buildMemo(g.ins[0]), buildMemo(g.ins[1])).setType(Type(g.type()));
        case g_Sel  : return Expr::Sel(buildMemo(g.ins[1]), fmt("%_", ~g.ins[0]).slice()).setType(Type(g.type()));
        case g_Tuple: return Expr::Tuple(map(g.ins, [&](uint j){ return buildMemo(j); })).setType(Type(g.type()));
        case g_Lamb : return Expr::Lamb(build(g.ins[0]), wrapBlock(buildTop(g.ins[1])), Type(g.type()));
        case g_Obl  :
            if (n_cov_points){
                // Experimental coverage conversion:
                //     Adapt(x) : Int   ~~>  x; fail<Int>
                //     Obl      : A->B  ~~>  \a{tthrow_<B,Int>("coverage", #n)}
                //     (other obligations not allowed)
                if (g.ins.psize() == 0){
                    // Create coverage point:
                    assert(g.type().name == a_Fun);
                    Expr e_tthrow = Expr::Sym(a_tthrow, {Type(g.type()[1]), Type(a_Int)}, Type(a_Fun, Type(a_Tuple, {Type(a_Atom), Type(a_Int)}), Type(g.type()[1])));  // -- type is: (Atom, Int) -> B
                    Expr e_body   = mxAppl(e_tthrow, mxTuple({mxLit_Atom(a_coverage), mxLit_Int(*n_cov_points)}));
                    Expr e_head   = Expr::Sym(a_underscore, {}, Type(g.type()[0]));
                    *n_cov_points += 1;
                    return mxLamb(e_head, wrapBlock(e_body));
                }else{
                    Expr e_sub = buildMemo(g.ins[0]);
                    Expr e_fail = Expr::Sym(a_fail, {g.type()}, Type(g.type()));
                    return mxBlock({e_sub, e_fail});
                }
            }else if (obl_as_fail){
                return Expr::Sym(a_fail, {g.type()}, Type(g.type()));
            }else{
                if (g.ins.psize() == 0) return Expr::Sym(a_obl, {}, Type(g.type()));
                else return Expr::Appl(Expr::Sym(a_adapt, {}, Type(/*no-type*/)), buildMemo(g.ins[0])).setType(Type(g.type()));
            }
        default: assert(false); }
    };

    return buildTop(0);
}


// S[i] <  S[j]  ~>  -1
// S[i] == S[j]  ~>  0
// S[i] >  S[j]  ~>  +1
// incomparable  ~>  INT_MIN
int State::order_(uint i, uint j, uint& count) const
{
    if (count == 0) return INT_MIN;
    count--;
    if (i == j) return 0;

    PArr<GExpr> const& me = *this; assert(me.size() <= 64);
    GExpr const& g = me[i];
    GExpr const& h = me[j];

    if (g.kind == g_Obl || h.kind == g_Obl) return INT_MIN;
    if (g.kind < h.kind) return -1;
    if (g.kind > h.kind) return +1;
    if (g.ins.psize() < h.ins.psize()) return -1;
    if (g.ins.psize() > h.ins.psize()) return +1;

    for (uint n = 0; n < g.ins.psize(); n++){
        if ((int)g.ins[n] < 0){
            assert((int)h.ins[n] < 0);
            if (g.ins[n] > h.ins[n]) return -1;     // -- intentionally '>' because negative encoding
            if (g.ins[n] < h.ins[n]) return +1;
        }else{
            int result = order_(g.ins[n], h.ins[n], count);
            if (result != 0) return result;
        }
    }
    return 0;
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


/*
<<==
Selecting grom a constructed tuple seems redundant, maybe ban this too?

  n16:Int$1=*Sel[#0, 19];
    n17:Int$1=*Appl[18, 19];
    n18:(Int, Int)->Int$1=*Pool[#3];
  n19:(Int, Int)$1=*Tuple[21, 20];
    n20:Int$0=*Id[3];
    n21:Int$0=*Id[3];
    n22:<null-type>$0=*End[];
*/


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

#if 1   // EXPERIMENTAL
    for (uint i = 0; i < S.size(); i++){
        GExpr const& ge = S[i];
        if (ge.kind == g_Sel && S[ge.ins[1]].kind == g_Tuple)
            return false;
    }
#endif

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


bool hasRecursion(State const& S)
{
    IntMap<uint,uint> used;
    bool was_recursive = false;
    checkSimpleLeftRecursion(S, 0, used, was_recursive);
    return was_recursive;
}


bool usesToplevelFormals(State const& S)
{
    GExpr const& g_top = S[0];
    if (g_top.kind != g_Lamb) return false;

    uint i_form = g_top.ins[0];
    GExpr const& g_form = S[i_form]; assert(g_form.kind == g_Tuple);

    for (uint pi : g_form.ins){
        assert(S[pi].kind == g_PI);
        for (uint i = 0; i < S.size(); i++){
            GExpr const& g = S[i];
            if (g.kind == g_Id && g.ins[0] == pi) goto Found;
            if (g.kind == g_Appl && (g.ins[0] == pi || g.ins[1] == pi)) goto Found;
            if (g.kind == g_Tuple && i != i_form){
                for (uint j : +g.ins)
                    if (j == pi) goto Found;
            }
        }
        return false;
      Found:;
    }
    return true;
}


//=================================================================================================
// -- expand one obligation:


static bool inFanin(uint tgt, State const& S, uint n, IntSeen<uint>& seen) {
  if (n == tgt) return true;
  if (!seen.add(n)) {
    for (uint m : +S[n].ins)
      if ((int)m >= 0 && inFanin(tgt, S, m, seen)) return true;
  }
  return false;
}


// Is 'tgt' in the transitive fanin of 'n'?
static bool inFanin(uint tgt, State const &S, uint n) {
    IntSeen<uint> seen;
    return inFanin(tgt, S, n, seen); }


inline bool reusableExpr(GExprKind kind) {
    // These are the nodes that if created as a subexpresson for one obligaton can be used by another
    // obligation (unless marked "internal"). PIs have to be included for technical reasons.
    return kind == g_Appl || kind == g_Tuple || kind == g_Sel || kind == g_Lamb || kind == g_PI; }


void expandOne(Pool const& pool, State S, uint tgt_i, function<void(State)> enqueue0, Params_SynthEnum const& P_, uint64* expansion_attempts)
{
    ZZ_PTimer_Scope(synth_enum_expandOne);

    IntMap<uint,uint> tmp_used;
    auto enqueue = [&](State S) {
        if (expansion_attempts) *expansion_attempts += 1;
        if (prepareEnqueue(S, tmp_used, pool, P_.must_use_formals, P_.force_recursion))
            enqueue0(S);
    };

    double min_adapt_cost = min_(pool.costSel(), pool.costAppl());

    GExpr f = S[tgt_i];     // -- f=final=target
    if (f.ins.psize() > 0){
        // Adaptor obligation:
        uint in_i = f.ins[0];
        GExpr e = S[in_i];
        if (e.type().name == a_Tuple){
            for (uint k = 0; k < e.type().size(); k++){
                if (e.type()[k] == f.type())
                    enqueue(S.set(tgt_i, GExpr(g_Sel, {~k, in_i}, e.type()[k], pool.costSel())));
                else if (canProduce(e.type()[k], f.type())){
                    State S1 = S.push(GExpr(g_Sel, {~k, in_i}, e.type()[k]));
                    enqueue(S1.set(tgt_i, GExpr(g_Obl, {S.size()}, f.type(), min_adapt_cost)));
                }
            }

        }else{ assert(e.type().name == a_Fun);
            State S1 = S.push(GExpr(g_Obl, {}, e.type()[0], 0.0));    // <<== type-based lower bound computation goes here
            if (e.type()[1] == f.type()){
                State S2 = S1.set(tgt_i, GExpr(g_Appl, {in_i, S.size()}, e.type()[1], pool.costAppl()));
                enqueue(S2); }
            else{
                State S2 = S1.push(GExpr(g_Appl, {in_i, S.size()}, e.type()[1], pool.costAppl()));
                enqueue(S2.set(tgt_i, GExpr(g_Obl, {S1.size()}, f.type(), min_adapt_cost))); }
        }

    }else{
        // General obligation, resolved from symbol pool:
        for (uint i = 0; i < pool.size(); i++){
            Type const& type = pool.sym(i).type;
            if (type == f.type())
                enqueue(S.set(tgt_i, GExpr(g_Pool, {~i}, type, pool.cost(i), /*internal*/true)));
            else if (canProduce(type, f.type())){
                State S1 = S.push(GExpr(g_Pool, {~i}, type, pool.cost(i), /*internal*/true));
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
            enqueue(S.set(tgt_i, GExpr(g_Lamb, {head_i, body_i}, f.type(), pool.costLamb(), /*internal*/P_.ban_recursion)));

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
            enqueue(S.set(tgt_i, GExpr(g_Tuple, obls, f.type(), pool.costTuple())));
        }
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Export as DOT:


static Map<Atom,Atom> type_abbrev;
ZZ_Initializer(type_abbrev, 0) {
    type_abbrev.set(Atom("Void" ), Atom("V"));
    type_abbrev.set(Atom("Bool" ), Atom("B"));
    type_abbrev.set(Atom("Int"  ), Atom("I"));
    type_abbrev.set(Atom("Float"), Atom("F"));
    type_abbrev.set(Atom("Atom" ), Atom("A"));
    type_abbrev.set(Atom("List" ), Atom("L"));
    type_abbrev.set(Atom("Maybe"), Atom("M"));
}


static
Type abbrevTypes(Type type)
{
    Atom res;
    if (type_abbrev.peek(type.name, res))
        type.name = res;
    if (type.size() > 0)
        type.args = map(type, abbrevTypes);
    return type;
}

static
Expr abbrevTypes(Expr const& expr)
{
    Arr<Expr> es; if (expr.exprs) es = map(expr.exprs, [&](Expr const& e){ return abbrevTypes(e); });
    Arr<Type> ts; if (expr.targs) ts = map(expr.targs, [&](Type const& t){ return abbrevTypes(t); });
    return Expr(expr.kind, expr.name, move(es), abbrevTypes(expr.type), move(ts), expr.loc);
}


static
void exportDot(Pool const& pool, State const& S, uint idx, Out& out, IntSeen<uint>& seen, IntSeen<uint>& parents, uint n_scopes)
{
    if (seen.has(idx)) return;

    seen.add(idx),
    parents.add(idx);

    // Print this node:
    static Vec<cchar*> appl_args{"fun", "arg"};
    static Vec<cchar*> lamb_args{"head", "body"};
    static const Array<cchar*> enum_args(empty_);

    GExpr const& ge = S[idx];

    String        node_qualifier;
    Array<cchar*> arg_names;    // -- if unset, no edge labels; if set to 'enum_args', "0, 1, 2..."; else index into given array.
    String        fill_color = "#FFFFFF";
    switch (ge.kind){
    case g_Pool : wr(node_qualifier, "[%_]", abbrevTypes(pool.sym(~ge.ins[0]))); break;
    case g_PI   : break;
    case g_Id   : break;
    case g_Appl : arg_names = appl_args.slice(); break;
    case g_Tuple: arg_names = enum_args; break;
    case g_Sel  : wr(node_qualifier, "[%_]", ~ge.ins[0]); break;
    case g_Lamb : arg_names = lamb_args.slice(); fill_color = "#FFEEBB"; break;
    case g_Obl  : break;
    default: wrLn("INTERNAL ERROR! Unexpeced GExpr kind in 'exportDot()': %_", GExprKind_name[ge.kind]), assert(false); }

    wrLn(out, "n%_ [shape=box style=filled fillcolor=\"%_\" label=\"%.r n%_ %C%_%_\\n%_\"]", idx, fill_color, n_scopes, idx, ge.internal?'*':0, GExprKind_name[ge.kind], node_qualifier, abbrevTypes(ge.type()));

    for (uint i = 0; i < ge.ins.psize(); i++){
        if ((int)ge.ins[i] >= 0){
            if (parents.has(ge.ins[i])){
                wrLn(out, "r%__%_ [label=\"n%_\" shape=plaintext]", idx, ge.ins[i], ge.ins[i]);
                wr(out, "n%_->r%__%_ [style=dotted]", idx, idx, ge.ins[i]);
            }else
                wr(out, "n%_->n%_", idx, ge.ins[i]);
            if (arg_names){
                if (arg_names == enum_args) wr(out, " [label=\"%_\" fontcolor=\"#0000FF\"]", i);
                else                        wr(out, " [label=\"%_\" fontcolor=\"#0000FF\"]", arg_names[i]);
            }
            out += "\n";
        }
    }

    // Recurse:
    for (uint child : +ge.ins)
        if ((int)child >= 0)
            exportDot(pool, S, child, out, seen, parents, n_scopes + (ge.kind == g_Lamb));

    parents.exclude(idx);
}


void exportDot(Pool const& pool, State const& S, String const& filename)
{
    OutFile out(filename);
    if (!out){ wrLn("ERROR! Could not open file for writing: %_", filename); exit(1); }

    wrLn(out, "digraph \"State\" {");
    IntSeen<uint> seen;
    IntSeen<uint> parents;
    exportDot(pool, S, 0, out, seen, parents, 0);
    wrLn(out, "}");
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

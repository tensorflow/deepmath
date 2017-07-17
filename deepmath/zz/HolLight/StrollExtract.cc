/* Copyright 2016 Google Inc. All Rights Reserved.

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
#include "zz/Generics/IntSet.hh"
#include "Kernel.hh"
#include "ProofStore.hh"
#include "Printing.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Bottom-up proof traverser:


struct SeenNode {
    IntSeen<Type> seen_ty;
    IntSeen<Term> seen_tm;
    IntSeen<Thm>  seen_th;
    bool add(Type ty) { return seen_ty.add(ty); }
    bool add(Term tm) { return seen_tm.add(tm); }
    bool add(Thm  th) { return seen_th.add(th); }
};


template<class F>
void visit(Type ty, F& f, SeenNode& seen)
{
    if (seen.add(ty)) return;

    if (ty.is_tapp()){
        for (List<Type> it = ty.targs(); it; ++it)
            visit(*it, f, seen);
    }

    f.type(ty);
}


template<class F>
void visit(Term tm, F& f, SeenNode& seen)
{
    if (seen.add(tm)) return;

    switch (tm.kind()){
    case Term::VAR : break;
    case Term::CNST: break;
    case Term::COMB: visit(tm.fun (), f, seen); visit(tm.arg  (), f, seen); break;
    case Term::ABS : visit(tm.avar(), f, seen); visit(tm.aterm(), f, seen); break;
    default: assert(false); }

    visit(tm.type(), f, seen);
    f.term(tm);
}


template<class F>
void visit(Thm th, F& f, SeenNode& seen)
{
    if (seen.add(th)) return;

    for (List<IdBase> it = th.proof().tail(); it; ++it){   // -- 0th element of list encodes RuleKind
        PArgKind kind;
        id_t     id;
        l_tuple(kind, id) = decodeArg(*it);

        switch (kind){
        case parg_Term : visit(Term(id), f, seen); break;
        case parg_Thm  : visit(Thm (id), f, seen); break;
        case parg_Type : visit(Type(id), f, seen); break;
        case parg_Terms: break;
        case parg_Types: break;
        case parg_Cnst : break;
        case parg_TCon : break;
        case parg_Axiom: break;
        default: assert(false); }
    }

    for (List<Term> it = th.hyps(); it; ++it)
        visit(*it, f, seen);
    visit(th.concl(), f, seen);
    f.thm(th);
}


// Launch function:
template<class F> void visit(Vec<Thm> const& thms, F& f) {
    SeenNode seen;
    for (Thm th : thms)
        visit(th, f, seen); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


struct FixVisitor {
    typedef Trip<RuleKind, uint, id_t> Key;    // -- triplet is: (rule, arg_no, arg_value)
    Map<Key, uint64> occur;

    void inc(Key key) { uint64* count; occur.getI(key, count); *count += 1; }

    void type(Type ty) {}   // wrLn("Visiting type: %_", ty);

    void term(Term tm) {
        if (tm.is_comb()){
            inc(Key(rule_App, 0, +encodeArg(tm.fun())));
            inc(Key(rule_App, 1, +encodeArg(tm.arg())));
        }else if (tm.is_abs()){
            inc(Key(rule_Abs, 0, +encodeArg(tm.avar ())));
            inc(Key(rule_Abs, 1, +encodeArg(tm.aterm())));
        }
    }

    void thm (Thm  th) {
        Key key(RuleKind(+th.proof()[0]), 0, id_NULL);
        for (List<IdBase> it = th.proof().tail(); it; ++it){
            key.trd = +*it;
            inc(key);
            key.snd++;  // -- increase 'arg_no'
        }
    }
};


// Will update 'thms'
void fixExtract(Vec<Thm>& thms, uint occur_lim)
{
    FixVisitor vis;
    visit(thms, vis);

    Vec<Pair<uint64, FixVisitor::Key>> result;
    For_Map(vis.occur){
        uint64 count = Map_Value(vis.occur);
        if (count > occur_lim)
            result.push(tuple(count, Map_Key(vis.occur)));
    }

    sort(result);
    //sort_reverse(result);
    for (auto&& p : result){
        uint64 count = p.fst;
        FixVisitor::Key key = p.snd;

        PArgKind kind;
        id_t     id;
        l_tuple(kind, id) = decodeArg(IdBase(key.trd));

        if (kind == parg_Thm)
            wrLn("%>6%_: %<8%_@%_ = THM  %_", count, key.fst, key.snd, Thm(id));
        else if (kind == parg_Term)
            wrLn("%>6%_: %<8%_@%_ = TERM %_", count, key.fst, key.snd, Term(id));
        else if (kind == parg_Type)
            wrLn("%>6%_: %<8%_@%_ = TYPE %_", count, key.fst, key.snd, Type(id));
        else
            wrLn("%>6%_: %<8%_@%_ = {kind=%_, id=%_}", count, key.fst, key.snd, kind, id);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


// input_cost, ruleapp_cost (+ uniquify input set when strolls are growing big)

struct PairVisitor {
    typedef Trip<RuleKind, uint, RuleKind> Key;    // -- triplet is: (rule, arg_no, sub_rule)
    Map<Key, uint64> occur;

    void type(Type ty) {}
    void term(Term tm) {}
    void thm (Thm  th) {
        Key key(RuleKind(+th.proof()[0]), 0, rule_NULL);
        for (List<IdBase> it = th.proof().tail(); it; ++it){
            PArgKind kind;
            id_t     id;
            l_tuple(kind, id) = decodeArg(*it);
            if (kind != parg_Thm) continue;   // <<== should really consider strolls including term and type building

            key.trd = RuleKind(+Thm(id).proof()[0]);

            uint64* count;
            occur.getI(key, count);
            *count += 1;
            key.snd++;  // -- increase 'arg_no'
        }
    }
};


void pairExtract(Vec<Thm> const& thms, uint n_strolls)
{
    PairVisitor vis;
    visit(thms, vis);

    Vec<Pair<uint64, PairVisitor::Key>> result;
    For_Map(vis.occur){
        uint64 count = Map_Value(vis.occur);
        /**/if (count > 100)
        result.push(tuple(count, Map_Key(vis.occur)));
    }

    sort(result);
    //sort_reverse(result);
    for (auto&& p : result){
        uint64 count = p.fst;
        PairVisitor::Key key = p.snd;

        wrLn("%>6%_: %_(@%_ %_)", count, key.fst, key.snd, key.trd);
    }
}


void strollExtract(String input_file, uint occur_lim, uint n_strolls)
{
    Vec<Thm> thms;
    {
        ProofStore P(input_file);
        for (line_t n = 0; n < P.size(); n++){
            if (n % 16384 == 0)
                wr("\rEvaluating theorems:  %.1f %% done.\f", 100.0 * n / P.size());
            if (P.is_humanThm(n))
                thms.push(P.evalThm(P.line2idx(n)));
        }
        wrLn("\rEvaluating theorems:  Complete!");
    }

    fixExtract(thms, occur_lim);
    //pairExtract(thms, n_strolls);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

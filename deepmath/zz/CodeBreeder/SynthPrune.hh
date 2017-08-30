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

#ifndef ZZ__CodeBreeder__SynthPrune_h
#define ZZ__CodeBreeder__SynthPrune_h

#include "zz/Generics/Arr.hh"
#include "SynthSpec.hh"
#include "SynthEnum.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


enum PruneKind {
    prune_NULL,
    prune_Appl,         // symbol# pexpr
    prune_Const,        // symbol#
    prune_Tuple,        // [pexpr]
    prune_Wild,         // wildcard#
    prune_Unordered,    // pexpr
    PruneKind_size,
};

extern cchar* PruneKind_name[PruneKind_size];


struct Prune {   // -- pruning rules
    PruneKind  kind;
    Arr<Prune> exprs;
    uint       num;
    Prune const& operator[](uint i) const { return exprs[i]; }

    Prune() : kind(prune_NULL), num(0) {}
    Prune(PruneKind kind, Vec<Prune> const& exprs) : Prune(kind, 0, exprs) {}
    Prune(PruneKind kind, uint num, Vec<Prune> const& exprs) : kind(kind), exprs(exprs), num(num) {}
};


template<> fts_macro void write_(Out& out, Prune const& v) {
    if (v.kind == prune_NULL) wr(out, "Prune_NULL");
    else wr(out, "Prune{kind=%_ num=%_ exprs=%_}", PruneKind_name[v.kind], v.num, v.exprs); }


// More readable formatting:
struct PruneFmt { Prune const& prune; Pool const& pool; };

template<> fts_macro void write_(Out& out, PruneFmt const& v) {
    switch (v.prune.kind){
    case prune_NULL:      wr(out, "-"); break;
    case prune_Tuple:     wr(out, "(%_)", join(", ", map(v.prune.exprs, [&](Prune const& prune){ return PruneFmt{prune, v.pool}; }))); break;
    case prune_Unordered: wr(out, "unordered %_", PruneFmt{v.prune.exprs[0], v.pool}); break;
    case prune_Const:     wr(out, "%_", v.pool.sym(v.prune.num)); break;
    case prune_Appl:{
        bool tup = (v.prune.exprs[0].kind == prune_Tuple);
        wr(out, "%_%C%_%C", v.pool.sym(v.prune.num), tup?0:'(', PruneFmt{v.prune.exprs[0], v.pool}, tup?0:')');
        break; }
    case prune_Wild:
        if ((int)v.prune.num >= 0) wr(out, "%_", Atom(v.prune.num));
        else                       wr(out, "%%%_", ~v.prune.num);
        break;
    default: assert(false); }
}


//=================================================================================================
// -- Main class:


class Pruner {
    Pool pool;
    Vec<Prune>  rules;
public:
    void init(Arr<Expr> const& pruning, Pool const& pool, bool verbose = true);
    bool shouldPrune(ENUM::State& S) const;     // -- will update 'p_rules' field
    bool isEmpty() const { return rules.size() == 0; }
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

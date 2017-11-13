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

#ifndef ZZ__CodeBreeder__SynthEnum_hh
#define ZZ__CodeBreeder__SynthEnum_hh

#include "zz/Generics/PArr.hh"
#include "zz/Generics/IntMap.hh"
#include "Types.hh"
#include "SynthSpec.hh"

namespace ZZ {
using namespace std;


namespace ENUM {
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Internal representation:


enum GExprKind : uchar {        // # = integer (represented as negative numbers), otherwise GExpr index
    g_NULL ,
    g_Pool ,    // pool_sym#
    g_PI   ,    // -
    g_Id   ,    // in
    g_Appl ,    // fun, arg
    g_Tuple,    // arg0, arg1, ..., arg[N-1]
    g_Sel  ,    // k#, tuple
    g_Lamb ,    // formal_tuple, body
    g_Obl  ,    // [in] (if input given, then adaptor)
    g_Scope,    // - (open scope: nodes with higher indices are inside scope)
    g_Begin,    // -
    g_End  ,    // -
    GExprKind_size
};


extern cchar* GExprKind_name[GExprKind_size];

extern Vec<Type>       id2type;
extern Map<Type, uint> type2id;
uint getId(Type const& type);


struct GExpr {
    GExprKind   kind;
    bool        internal;       // -- gates inside a closed scope (cannot be used anymore)
    uint        type_id;
    double      cost;           // -- for obligations, lower-bound on cost for completing obligation
    Arr<uint>   ins;            // <<== could be optimized
    Arr<ushort> p_rules;        // -- pruning rules left to try for this node; NULL means all of them

    GExpr(GExprKind kind = g_NULL) : kind(kind), internal(true), type_id(0), cost(0.0), ins() {}
    GExpr(GExprKind kind, Arr<uint> ins, Type type, double cost = 0.0 , bool internal = false) :
        kind(kind), internal(internal), type_id(getId(type)), cost(cost), ins(ins) {}

    void toProto(::CodeBreeder::NodeProto* node_proto) const;
    Type type() const { return id2type[type_id]; }
};


// 'State' -- represents a partial expression (with obligations = typed holes).
struct State : PArr<GExpr> {
    State() : PArr<GExpr>() {}
    State(PArr<GExpr>&& s) : PArr<GExpr>(move(s)) {}
    bool   getLast(GExprKind kind, uint& tgt_i) const;
    bool   getOblRange(uint& tgt_i0, uint& tgt_i1) const;
    double cost() const;
    Expr   expr(Pool const& P, bool obl_as_fail = false, uint* n_cov_points = nullptr) const;
    void   toProto(uint64 id, ::CodeBreeder::StateProto* proto) const;
    int    order(uint i, uint j) const { uint count = 100; return order_(i, j, count); }
private:
    int    order_(uint i, uint j, uint& count) const;
};


}
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Interface:


struct Params_SynthEnum {
    bool must_use_formals = false;
    bool ban_recursion    = false;
    bool force_recursion  = false;      // -- cannot be used together with 'ban_recursion'
};


inline ENUM::State initialEnumState(Type const& type) { return ENUM::State().push(ENUM::GExpr(ENUM::g_Obl, {}, type)); }
    // -- Create a state with a single obligation of type 'type'.

void expandOne(Pool const& pool, ENUM::State S, uint tgt_i, function<void(ENUM::State)> enqueue, Params_SynthEnum const& P, uint64* expansion_attempts = nullptr);
    // -- Expand node 'tgt_i' in 'S' (which must be an obligation) in every possible way, calling 'enqueue'
    // on the result. If 'expansion_attempts' is provided, each attempt at expanding an obligation will
    // increase its value by 1 (for resource management/progress output).


bool hasRecursion(ENUM::State const& S);
    // -- does state contain a recursive call anywhere? (inefficient implementation)

bool usesToplevelFormals(ENUM::State const& S);
    // -- true if 'S[0]' is of type 'g_Lamb' and all formal arguments are used inside 'S'

template<> fts_macro void write_(Out& out, ENUM::GExpr const& g)
{
    wr(out, "\a/%_\a/\a_$%_\a_=%C\a/\a*%_\a0[", g.type(), g.cost, g.internal?'*':'\0', ENUM::GExprKind_name[g.kind]);
    for (uint j = 0; j < (+g.ins).size(); j++){
        if (j != 0) wr(out, ", ");
        if ((int)g.ins[j] >= 0) out += g.ins[j];
        else wr(out, "#%_", ~g.ins[j]);
    }
    out += "]";
}


template<> fts_macro void write_(Out& out, ENUM::State const& v, Str flags)
{
    out += "State{", (flags ? "\n" : "");
    for (uint i = 0; i < v.size(); i++){
        ENUM::GExpr g = v[i];
        wr(out, "%_\a*n%_\a*:\a/%_\a/\a_$%_\a_=%C\a/\a*%_\a0[", flags?"    ":" ", i, g.type(), g.cost, g.internal?'*':'\0', ENUM::GExprKind_name[g.kind]);
        for (uint j = 0; j < (+g.ins).size(); j++){
            if (j != 0) wr(out, ", ");
            if ((int)g.ins[j] >= 0) out += g.ins[j];
            else wr(out, "#%_", ~g.ins[j]);
        }
        out += "];", (flags ? "\n" : " ");
    }
    out += '}';
}
template<> fts_macro void write_(Out& out, ENUM::State const& v) {
    write_(out, v, Str()); }


void exportDot(Pool const& pool, ENUM::State const& S, String const& filename);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

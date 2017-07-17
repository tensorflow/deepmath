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

// Forward declare protobuf.
namespace CodeBreeder { struct NodeProto; }
namespace CodeBreeder { struct StateProto; }

namespace ZZ {
using namespace std;


namespace ENUM {
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Internal representation:


enum GExprKind : uchar {        // # = integer (represented as negative numbers), otherwise GExpr index
    g_NULL,
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
    g_End,      // -
    GExprKind_size
};


extern cchar* GExprKind_name[GExprKind_size];

extern Vec<Type>       id2type;
extern Map<Type, uint> type2id;
uint getId(Type const& type);


struct GExpr {
    GExprKind   kind;
    bool        internal;       // -- gates inside a closed scope (cannot be used anymore)
    double      cost;           // -- for obligations, lower-bound on cost for completing obligation
    Arr<uint>   ins;            // <<== could be optimized
    uint        type_id;

    GExpr(GExprKind kind = g_NULL) : kind(kind), internal(true), cost(0.0), ins(), type_id(0) {}
    GExpr(GExprKind kind, Arr<uint> ins, Type type, double cost = 0.0 , bool internal = false) :
        kind(kind), internal(internal), cost(cost), ins(ins), type_id(getId(type)) {}

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
    Expr   expr(Pool const& P, bool obl_as_fail = false) const;
    void   toProto(::CodeBreeder::StateProto* proto) const;
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


template<> fts_macro void write_(Out& out, ENUM::State const& v)
{
    out += "State{ ";
    for (uint i = 0; i < v.size(); i++){
        ENUM::GExpr g = v[i];
        wr(out, "\a*n%_\a*:\a/%_\a/\a_$%_\a_=%C\a/\a*%_\a0[", i, g.type(), g.cost, g.internal?'*':'\0', ENUM::GExprKind_name[g.kind]);
        for (uint j = 0; j < (+g.ins).size(); j++){
            if (j != 0) wr(out, ", ");
            if ((int)g.ins[j] >= 0) out += g.ins[j];
            else wr(out, "#%_", ~g.ins[j]);
        }
        out += "]; ";
    }
    out += '}';
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

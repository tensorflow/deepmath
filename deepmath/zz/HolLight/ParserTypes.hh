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

#ifndef ZZ__HolLight__ParserTypes_hh
#define ZZ__HolLight__ParserTypes_hh

#include "Types.hh"

namespace ZZ {
using namespace std;

//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


enum RuleKind : uchar {
    rule_NULL,

    rule_TVar,
    rule_Cnst,
    rule_Var,
    rule_TAppl,
    rule_Abs,
    rule_App,

    rule_REFL,
    rule_TRANS,
    rule_MK_COMB,
    rule_ABS,
    rule_BETA,
    rule_ASSUME,
    rule_EQ_MP,
    rule_DEDUCT,
    rule_INST,
    rule_INST_T,

    rule_New_Ax,
    rule_New_Def,
    rule_New_TDef,
    rule_TDef_Ex1,
    rule_TDef_Ex2,

    rule_TThm,

    RuleKind_size
};

extern cchar* RuleKind_name[RuleKind_size];

template<> fts_macro void write_(Out& out, RuleKind const& v) {
    out += RuleKind_name[v]; }


enum ArgKind : uchar {
    arg_NULL,

    // Atomic:
    arg_CNST,
    arg_VAR,
    arg_TCON,
    arg_TVAR,
    arg_AXIOM,
    arg_TTHM,

    // Composite:
    arg_TYPE,
    arg_TERM,
    arg_THM,

    // Composite, proof-log index:
    arg_TYPE_IDX,
    arg_TERM_IDX,
    arg_THM_IDX,

    ArgKind_size
};

extern cchar* ArgKind_name[ArgKind_size];

template<> fts_macro void write_(Out& out, ArgKind const& v) {
    out += ArgKind_name[v]; }

inline bool isComposite(ArgKind t) {
    return t >= arg_TYPE; }

inline ArgKind makeIndex(ArgKind kind) {    // -- move from 'arg_TYPE -> arg_TYPE_IDX' etc
    assert(kind >= arg_TYPE && kind <= arg_THM);
    return ArgKind((int)kind - (int)arg_TYPE + (int)arg_TYPE_IDX); }

inline ArgKind makeNoIndex(ArgKind kind) {
    return (kind >= arg_TYPE_IDX && kind <= arg_THM_IDX) ? ArgKind((int)kind - (int)arg_TYPE_IDX + (int)arg_TYPE) : kind; }


// Union type for parse-rule argument. All elements of union must be derived from 'IdBase' without adding any state variables.
struct Arg {
    ArgKind kind;
    uint64  id : 56;

    Arg()            : kind(arg_NULL ), id(id_NULL) {}
    Arg(Cnst  cnst ) : kind(arg_CNST ), id(+cnst  ) {}
    Arg(Var   var  ) : kind(arg_VAR  ), id(+var   ) {}
    Arg(TCon  tcon ) : kind(arg_TCON ), id(+tcon  ) {}
    Arg(TVar  tvar ) : kind(arg_TVAR ), id(+tvar  ) {}
    Arg(Axiom axiom) : kind(arg_AXIOM), id(+axiom ) {}
    Arg(TThm  tthm ) : kind(arg_TTHM ), id(+tthm  ) {}
    Arg(Type  type ) : kind(arg_TYPE ), id(+type  ) {}
    Arg(Term  term ) : kind(arg_TERM ), id(+term  ) {}
    Arg(Thm   thm  ) : kind(arg_THM  ), id(+thm   ) {}

    Arg(ArgKind kind, uint64 idx) : kind(kind), id(idx) { assert(kind >= arg_TYPE_IDX); }

    // Atomic:
    Cnst  cnst () const { assert(kind == arg_CNST ); return Cnst (id); }
    Var   var  () const { assert(kind == arg_VAR  ); return Var  (id); }
    TCon  tcon () const { assert(kind == arg_TCON ); return TCon (id); }
    TVar  tvar () const { assert(kind == arg_TVAR ); return TVar (id); }
    Axiom axiom() const { assert(kind == arg_AXIOM); return Axiom(id); }
    TThm  tthm () const { assert(kind == arg_TTHM ); return TThm (id); }
    // Composite:
    Type  type () const { assert(kind == arg_TYPE ); return Type (id); }
    Term  term () const { assert(kind == arg_TERM ); return Term (id); }
    Thm   thm  () const { assert(kind == arg_THM  ); return Thm  (id); }

    bool isAtomic() const { return kind <  arg_TYPE; }
    Str str() const;
};


inline Str Arg::str() const {
    assert(isAtomic());
    switch (kind){
    case arg_NULL : return Str  ();
    case arg_CNST : return cnst ();
    case arg_VAR  : return var  ();
    case arg_TCON : return tcon ();
    case arg_TVAR : return tvar ();
    case arg_AXIOM: return axiom();
    case arg_TTHM : return tthm ();
    default: assert(false); }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

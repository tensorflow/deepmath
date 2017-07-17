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

#ifndef ZZ__HolLight__ProofLogging_h
#define ZZ__HolLight__ProofLogging_h

#include "Types.hh"
#include "ParserTypes.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


extern bool kernel_proof_logging;       // -- defined in 'Kernel.cc'; false by default


/*
Top-bits encode kind:
  0      -- Term
  10     -- Thm
  1100   -- Type
  1101   -- List<Term>
  1110   -- List<Type>
  111101 -- Cnst   }
  111110 -- TCon   }- these are hashed strings
  111111 -- Axiom  }
Var, TVar, TThm do not occur
*/


enum PArgKind : uchar {   // -- proof-step argument
    parg_NULL,

    parg_Term,
    parg_Thm,
    parg_Type,
    parg_Terms,
    parg_Types,
    parg_Cnst,
    parg_TCon,
    parg_Axiom,

    PArgKind_size
};

extern cchar* PArgKind_name[PArgKind_size];

template<> fts_macro void write_(Out& out, PArgKind const& v) {
    out += PArgKind_name[v]; }


template<uint bits, uint tag> inline IdBase tagArg(IdBase arg) {
  assert(+arg < (id_t(1) << (32 - bits)));
  return IdBase(+arg | (id_t(tag) << (32 - bits))); }

template<uint bits>           inline uint getTag(IdBase arg) { return +arg >> (32 - bits); }
template<uint bits>           inline id_t noTag (IdBase arg) { return +arg & ((id_t(1) << (32 - bits)) - 1); }
template<uint bits, uint tag> inline bool hasTag(IdBase arg) { return getTag<bits>(arg) == tag; }


inline IdBase encodeArg(Term       tm ) { return tagArg<1, 0>(tm); }
inline IdBase encodeArg(Thm        th ) { return tagArg<2, 2>(th); }
inline IdBase encodeArg(Type       ty ) { return tagArg<4,12>(ty); }
inline IdBase encodeArg(List<Term> tms) { return tagArg<4,13>(tms); }
inline IdBase encodeArg(List<Type> tys) { return tagArg<4,14>(tys); }
inline IdBase encodeArg(Cnst       cn ) { return tagArg<6,61>(cn); }
inline IdBase encodeArg(TCon       tc ) { return tagArg<6,62>(tc); }
inline IdBase encodeArg(Axiom      ax ) { return tagArg<6,63>(ax); }


inline Pair<PArgKind, id_t> decodeArg(IdBase arg) {
    if (hasTag<1, 0>(arg)) return tuple(parg_Term, noTag<1>(arg));
    if (hasTag<2, 2>(arg)) return tuple(parg_Thm , noTag<2>(arg));
    if (!hasTag<4, 15>(arg)) return tuple(PArgKind((uint)parg_Type + getTag<4>(arg) - 12), noTag<4>(arg));
    return tuple(PArgKind((uint)parg_Cnst + getTag<6>(arg) - 61), noTag<6>(arg));
}


//=================================================================================================
// -- Log step:


inline List<IdBase> logStep_helper(List<IdBase> acc) { return acc; }

template<class IdType, typename... Args>
inline List<IdBase> logStep_helper(List<IdBase> acc, IdType arg0, Args const&... args) {
    return cons(encodeArg(arg0), logStep_helper(acc, args...)); }


template<typename... Args>
inline List<IdBase> logStep(RuleKind step_kind, Args const&... args)
{
    if (!kernel_proof_logging) return nil_;

    List<IdBase> ret = logStep_helper(nil_, args...);
    ret = cons(IdBase(step_kind), ret);   // <<== encode rule as well (to support strolls)?
    return ret;
}


inline List<Term> mkList_substs(Vec<Subst> const& subs)
{
    if (!kernel_proof_logging) return nil_;

    List<Term> ret;
    for (uint i = subs.size(); i > 0;){ i--;
        ret = cons(subs[i].tm, ret);
        ret = cons(subs[i].x , ret);
    }
    return ret;
}


inline List<Type> mkList_tsubsts(Vec<TSubst> const& subs)
{
    if (!kernel_proof_logging) return nil_;


    List<Type> ret;
    for (uint i = subs.size(); i > 0;){ i--;
        ret = cons(subs[i].ty, ret);
        ret = cons(subs[i].a , ret);
    }
    return ret;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

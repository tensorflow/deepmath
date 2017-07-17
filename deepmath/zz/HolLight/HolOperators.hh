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

#ifndef ZZ__HolLight__HolOperators_hh
#define ZZ__HolLight__HolOperators_hh

#include "zz/Generics/IntMap.hh"
#include "Types.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


enum HolOpKind : ushort {
    op_NULL,
    op_BINDER,
    op_INFIXL,
    op_INFIXR,
    op_PREFIX,
};


constexpr static uchar prec_BINDER  = 1;
constexpr static uchar prec_PREFIX  = 253;
constexpr static uchar prec_COMB    = 254;
constexpr static uchar prec_ATOMIC  = 255;


struct HolOp {
    ushort      prec;   // -- higher precedence evaluates first
    HolOpKind   kind;

    constexpr HolOp(uchar prec = 0, HolOpKind kind = op_NULL) : prec(prec), kind(kind) {}
    explicit operator bool() const { return prec != 0; }
};


extern IntMap<Cnst, HolOp> hol_ops;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

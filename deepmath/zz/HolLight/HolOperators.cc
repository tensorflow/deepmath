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
#include "HolOperators.hh"

namespace ZZ {
using namespace std;

/*
To deal with (maybe):
  let unspaced_binops = ref [","; ".."; "$"];;

  // Binary operators to print at start of line when breaking.
  let prebroken_binops = ref ["==>"];;
*/


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


IntMap<Cnst, HolOp> hol_ops;


ZZ_Initializer(hol_ops, 5)
{
    auto addOp = [](cchar* str, HolOpKind kind, uchar prec) {
        assert(prec > 0);
        hol_ops(Cnst(str)) = HolOp(prec, kind);
    };

    auto parse_as_binder = [&](cchar* str) { addOp(str, op_BINDER, prec_BINDER); };
    auto parse_as_prefix = [&](cchar* str) { addOp(str, op_PREFIX, prec_PREFIX); };
    auto parse_as_infix  = [&](cchar* str, uint prec, cchar* assoc) {
        HolOpKind kind = (assoc[0] == 'l') ? op_INFIXL : op_INFIXR;
        addOp(str, kind, prec);
    };

    // Grepped from HOL-light directory, '*.ml':
    parse_as_infix("<", 12, "right");
    parse_as_infix("<=", 12, "right");
    parse_as_infix(">", 12, "right");
    parse_as_infix(">=", 12, "right");
    parse_as_infix("+", 16, "right");
    parse_as_infix("-", 18, "left");
    parse_as_infix("*", 20, "right");
    parse_as_infix("EXP", 24, "left");
    parse_as_infix("DIV", 22, "left");
    parse_as_infix("MOD", 22, "left");
    parse_as_binder("minimal");
    parse_as_prefix("~");
    parse_as_binder("\\");
    parse_as_binder("!");
    parse_as_binder("?");
    parse_as_binder("?!");
    parse_as_infix("==>", 4, "right");
    parse_as_infix("\\/", 6, "right");
    parse_as_infix("/\\", 8, "right");
    parse_as_infix("<=>", 2, "right");
    parse_as_infix("=", 12, "right");
    parse_as_infix("$", 25, "left");
    parse_as_binder("lambda");
    parse_as_infix("PCROSS", 22, "right");
    parse_as_binder("@");
    parse_as_infix("div", 22, "left");
    parse_as_infix("rem", 22, "left");
    parse_as_infix("==", 10, "right");
    parse_as_infix("divides", 12, "right");
    parse_as_prefix("mod");
    parse_as_infix(",", 14, "right");
    parse_as_infix("..", 15, "right");
    parse_as_infix(", ", 14, "right");
    parse_as_infix("++", 16, "right");
    parse_as_infix("**", 20, "right");
    parse_as_infix("<<=", 12, "right");
    parse_as_infix("===", 10, "right");
    parse_as_infix("treal_mul", 20, "right");
    parse_as_infix("treal_add", 16, "right");
    parse_as_infix("treal_le", 12, "right");
    parse_as_infix("treal_eq", 10, "right");
    parse_as_prefix("--");
    parse_as_infix("/", 22, "left");
    parse_as_infix("pow", 24, "left");
    parse_as_infix("IN", 11, "right");
    parse_as_infix("SUBSET", 12, "right");
    parse_as_infix("PSUBSET", 12, "right");
    parse_as_infix("INTER", 20, "right");
    parse_as_infix("UNION", 16, "right");
    parse_as_infix("DIFF", 18, "left");
    parse_as_infix("INSERT", 21, "right");
    parse_as_infix("DELETE", 21, "left");
    parse_as_infix("HAS_SIZE", 12, "right");
    parse_as_infix("<=_c", 12, "right");
    parse_as_infix("<_c", 12, "right");
    parse_as_infix(">=_c", 12, "right");
    parse_as_infix(">_c", 12, "right");
    parse_as_infix("=_c", 12, "right");
    parse_as_infix("CROSS", 22, "right");
    parse_as_infix("UNION_OF", 20, "right");
    parse_as_infix("INTERSECTION_OF", 20, "right");
    parse_as_infix("o", 26, "right");
    parse_as_infix("<<", 12, "right");
    parse_as_infix("<<<", 12, "right");
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

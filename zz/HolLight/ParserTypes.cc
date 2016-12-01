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
#include "ParserTypes.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


cchar* RuleKind_name[RuleKind_size] = {
    "<null>",
    "TVar",
    "Cnst",
    "Var",
    "TAppl",
    "Abs",
    "App",
    "REFL",
    "TRANS",
    "MK_COMB",
    "ABS",
    "BETA",
    "ASSUME",
    "EQ_MP",
    "DEDUCT",
    "INST",
    "INST_T",
    "New_Ax",
    "New_Def",
    "New_TDef",
    "TDef_Ex1",
    "TDef_Ex2",
    "TThm",
};


cchar* ArgKind_name[ArgKind_size] = {
    "<null>",
    "CNST",
    "VAR",
    "TCON",
    "TVAR",
    "AXIOM",
    "TTHM",
    "TYPE",
    "TERM",
    "THM",
    "TYPE_IDX",
    "TERM_IDX",
    "THM_IDX",
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

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

#ifndef ZZ__CodeBreeder__SynthSpec_hh
#define ZZ__CodeBreeder__SynthSpec_hh

#include "Types.hh"

// Forward declare protobuf.
namespace CodeBreeder { struct PoolProto; }

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Specification:


extern cchar* typevar_suffix;   // -- types ending in this represent type-variables in the list of symbols
extern cchar* cost_tag;         // -- symbols can be wrapped with 'cost_( <cost>, <symbol> )' to assign a non-unit cost


struct Pool {
    Vec<CExpr>  syms;
    double  cost_Appl  = 1;
    double  cost_Sel   = 1;
    double  cost_Lamb  = 1;
    double  cost_Tuple = 1;

    Pool() {}
    Pool(Pool&& s) { s.syms.moveTo(syms); cost_Appl = s.cost_Appl; cost_Sel = s.cost_Sel; cost_Lamb = s.cost_Lamb; cost_Tuple = s.cost_Tuple; }
    Pool& operator=(Pool&& s) { s.syms.moveTo(syms); cost_Appl = s.cost_Appl; cost_Sel = s.cost_Sel; cost_Lamb = s.cost_Lamb; cost_Tuple = s.cost_Tuple; return *this; }

    void toProto(::CodeBreeder::PoolProto* proto) const;  // -- will not store costs
};


struct Spec {
    Expr    prog;

    Atom    name;       // -- a short name for the task (from 'spec_name'), defaults to filename without '.evo'.
    Atom    descr;      // -- a longer description of the task (from 'spec_descr'), defaults to empty string.

    Pool    pool;       // -- primitive symbols to use for synthesis
    Type    target;     // -- synthesis target, a function from "output of 'in_wrap_'" to "input of 'out_wrap_'".

    // Subexpressions of 'prog' (the whole let/rec definition, not just the RHS)
    Expr    io_pairs;       // 'io_pairs_', must be specified
    Expr    runner;         // 'runner_'  , must be specified
    Expr    checker;        // 'checker_' , must be specified
    Expr    in_wrap;        // 'in_wrap_' , defaults to identity function of type '#0 io_pairs'
    Expr    out_wrap;       // 'out_wrap_', defaults to identity function of type '#1 io_pairs'

    uint    n_io_pairs;     // -- size of 'io_pairs' vector
};


Spec readSpec(String spec_file, bool spec_file_is_text, bool just_syms = false);
    // -- Normally 'spec_file' is a filename, but if 'spec_file_is_text' is TRUE then it is
    // the content of the file itself. If 'just_syms' is TRUE, only the symbol pool is
    // parsed and return (no target or subexpressions).


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

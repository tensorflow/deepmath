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

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Specification:


extern cchar* typevar_suffix;   // -- types ending in this represent type-variables in the list of symbols
extern cchar* cost_tag;         // -- symbols can be wrapped with 'cost_( <cost>, <symbol> )' to assign a non-unit cost


struct Pool {
    Arr<CExpr>  syms;   // -- don't access directly; only public for hash function

    Pool() {}
    Pool(Pool const& s) { syms = s.syms; }
    Pool& operator=(Pool const& s) { syms = s.syms; return *this; }
    Pool(Pool&& s) { syms = move(s.syms); }
    Pool& operator=(Pool&& s) { syms = move(s.syms); return *this; }

    Pool(Vec<CExpr> const& cs, double cost_Appl, double cost_Sel, double cost_Lamb, double cost_Tuple) :
        syms(reserve_, cs.size() + 4)
    {
        for (uint i = 0; i < cs.size(); i++) syms[i] = cs[i];
        syms[LAST-0].cost = cost_Appl;
        syms[LAST-1].cost = cost_Sel;
        syms[LAST-2].cost = cost_Lamb;
        syms[LAST-3].cost = cost_Tuple;
    }

    explicit operator bool() const { return (bool)syms; }

    void toProto(::CodeBreeder::PoolProto* proto) const;  // -- will not store costs

    // Read:
    uint         size      ()       const { return syms.size() - 4; }
    CExpr const& operator[](uint i) const { return syms[i]; }
    Expr const&  sym       (uint i) const { return *syms[i]; }
    double       cost      (uint i) const { return syms[i].cost; }

    double costAppl () const { return syms[LAST-0].cost; }
    double costSel  () const { return syms[LAST-1].cost; }
    double costLamb () const { return syms[LAST-2].cost; }
    double costTuple() const { return syms[LAST-3].cost; }
};

template <> struct Hash_default<Pool> {
    uint64 hash (Pool const& key) const { return defaultHash(key.syms); }
    bool   equal(Pool const& key1, const Pool& key2) const { return defaultEqual(key1.syms, key2.syms); }
};


struct Spec {
    Expr    prog;

    Atom    name;       // -- a short name for the task (from 'spec_name'), defaults to filename without '.evo'.
    Atom    descr;      // -- a longer description of the task (from 'spec_descr'), defaults to empty string.

    Pool    pool;       // -- primitive symbols to use for synthesis
    Type    target;     // -- synthesis target, determined by 'wrapper' function

    // Subexpressions of 'prog' (the whole let/rec definition, not just the RHS)
    Expr    io_pairs;   // -- 'io_pairs_', must be specified
    uint    n_io_pairs; // -- size of 'io_pairs' vector
    Expr    runner;     // -- 'runner_'  , defaults to 'default_runner_<IN,OUT>'
    Expr    checker;    // -- 'checker_' , defaults to 'default_checker_<IN,OUT>'
    Expr    wrapper;    // -- 'wrapper_' , defaults to 'default_wrapper_<IN,OUT>'

    // Used for random function generation:
    Expr    test_vec;   // -- 'test_vec_', type: (IN->OUT) -> [Maybe<OUT>]
    Expr    test_hash;  // -- 'test_hash_', type: (IN->OUT) -> [Int] (with special value -1 for "discard function")

    Expr    init_state; // -- 'init_state_', optional.

    Arr<Expr> pruning;  // -- pruning rules, from let definition of 'pruning'

    Arr<Arr<Expr>> synth_subst; // -- substitution rules for random function synthesis; the first element can be substituted for any of the non-first elements
};


Spec readSpec(String spec_file, bool spec_file_is_text, bool just_syms = false);
    // -- Normally 'spec_file' is a filename, but if 'spec_file_is_text' is TRUE then it is
    // the content of the file itself. If 'just_syms' is TRUE, only the symbol pool is
    // parsed and return (no target or subexpressions). if 'quick' is TRUE, then the parsed code
    // is not compiled an run, which implies 'n_io_pairs' is unset (and some errors are not caught).


//=================================================================================================
// -- Enum from 'new_spec.evo':


enum RunResult {
    res_NULL   ,
    res_RIGHT  ,
    res_WRONG  ,
    res_ABSTAIN,
    res_CRASH  ,
    res_CPU_LIM,
    res_MEM_LIM,
    res_REC_LIM,
    res_SIZE   ,
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

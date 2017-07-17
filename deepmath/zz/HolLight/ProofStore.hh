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

#ifndef ZZ__HolLight__ProofStore_hh
#define ZZ__HolLight__ProofStore_hh

#include "zz/Generics/IntSet.hh"
#include "ParserTypes.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Integer types:


/*
There are three types of integer handles referring to types, terms and theorems:

  - 'line_t'. [PROOF LINE]. Line numbers start at 1 and refer to the line in the proof-log where
  the object gets created. The class 'ProofStore' stores the entire proof-log in memory and its
  data structures are indexed on this handle type.

  - 'ty_t, tm_t, th_t'. [INDIVIDUAL INDEX]. Each of the three HOL-types 'Type', 'Term' and 'Thm'
  are individually numbered in the proof-log, starting at 1 (so "theorem 42" is different from
  "type 42"). We need these indices because the proof-log refers to objects through them.

  - 'Type, Term, Thm'. [HOL-OBJ / ID]. We refer to these three types as HOL objects. Internally
  they are represented as integers (called "IDs") indexing into a three separate global arrays. To
  get the underlying ID, use unary '+'.
*/


// We make these typedefs for implicit documentation.
typedef uind line_t;

typedef uind index_t;
typedef index_t ty_t;
typedef index_t tm_t;
typedef index_t th_t;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helper types:


// Convenience iterator over the theorem subset of an argument list:
struct ThmIt {
    Arg const* p;
    Arg const* end;
    IntMap<th_t,line_t> const& th_idx2line;
    ThmIt(Arg const* p, Arg const* end, IntMap<th_t,line_t> const& th_idx2line) : p(p), end(end), th_idx2line(th_idx2line) {}

    ThmIt& operator++() { do ++p; while (p != end && p->kind != arg_THM_IDX); return *this; }
    line_t operator*() const { return th_idx2line[p->id]; }

    bool operator==(ThmIt const& other) const { return p == other.p; }
};


struct ThmRange {
    Array<Arg const> args;
    IntMap<th_t,line_t> const& th_idx2line;
    ThmRange(Array<Arg const> args, IntMap<th_t,line_t> const& th_idx2line) : args(args), th_idx2line(th_idx2line) {}

    ThmIt begin() const { ThmIt it(args.begin() - 1, args.end(), th_idx2line); ++it; return it; }
    ThmIt end()   const { return ThmIt(args.end(), args.end(), th_idx2line); }
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Proof Store:


class ProofStore {
    Vec<Arg> args_data;

    // Indexed by 'line':
    Vec<ArgKind>    ret_kinds;
    Vec<RuleKind>   rule_kinds;
    Vec<uind>       args_offset;            // -- returns index into 'args_data'
    Vec<uint>       args_size;
    Vec<uint>       n_fanouts;

    // Temporaries:
    Vec<Arg> eval_tmp;

    // Internal helpers:
    void countFanouts();
    Array<Arg const> translateArgs(line_t n);

public:
  //________________________________________
  //  Read-only member variables:

    // Implicit dependencies:
    IntMap<Cnst, line_t> new_def_line;      // -- where is the 'New_Def' introduction of a constant-constructor?
    IntMap<TCon, line_t> new_tdef_line;     // -- where is the 'New_TDef' introduction of a type-constructor?

    // Cached evaluations:
    IntMap<ty_t, Type> idx2type;
    IntMap<tm_t, Term> idx2term;
    IntMap<th_t, Thm>  idx2thm;

    IntMap<line_t, index_t> line2idx;       // -- returned index is specific for the object kind produced by the given proof-line.
    IntMap<ty_t, line_t>    ty_idx2line;    // }
    IntMap<tm_t, line_t>    tm_idx2line;    // }- reverse maps for 'line2idx'
    IntMap<th_t, line_t>    th_idx2line;    // }

    IntMap<Thm, line_t>  thm2line;          // -- for theorems, we may need to get at the rule that produced them (not so for types and terms)

  //________________________________________
  //  Public member functions:

    ProofStore(String filename, bool quiet = false);

    // Proof-log traversal:
    line_t   size()         const { return ret_kinds.size(); }      // -- number of lines in proof
    ArgKind  ret (line_t n) const { return ret_kinds [n]; }         // -- one of: arg_TYPE, arg_TERM, arg_THM (or arg_NULL for n=0)
    RuleKind rule(line_t n) const { return rule_kinds[n]; }

    Array<Arg const> args(line_t n) const { return Array<Arg const>(&args_data[args_offset[n]], args_size[n]); }
    ThmRange         thms(line_t n) const { return ThmRange(args(n), th_idx2line); }

    bool is_thm      (line_t n) const { return ret(n) == arg_THM; }
    bool is_humanThm (line_t n) const { return n+1 < size() && rule(n+1) == rule_TThm && Str(args(n+1)[0].tthm())[0] != '!'; }
    bool is_markedThm(line_t n) const { return n+1 < size() && rule(n+1) == rule_TThm; }
    bool is_fanoutThm(line_t n) const;
    uint nFanouts    (line_t n) const { return n_fanouts[n]; }
    TThm thmName     (line_t n) const { return is_markedThm(n) ? args(n+1)[0].tthm() : TThm(); }

    uind nTypes() const { return ty_idx2line.size() - 1; }
    uind nTerms() const { return tm_idx2line.size() - 1; }
    uind nThms () const { return th_idx2line.size() - 1; }

    void evalLine(line_t n) { translateArgs(n); }
    Type evalType(ty_t idx);
    Term evalTerm(tm_t idx);
    Thm  evalThm (th_t idx);
};


inline bool ProofStore::is_fanoutThm(line_t n) const {
    auto nonTrivial = [&](line_t m) { return rule(m) != rule_BETA && rule(m) != rule_REFL && rule(m) != rule_ASSUME; };
    return n+1 < size() && (rule(n+1) == rule_TThm || (nFanouts(n) > 1 && nonTrivial(n))); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

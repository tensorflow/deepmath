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
#include "ProofStore.hh"
#include "Parser.hh"
#include "RuleApply.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Initialization:


ProofStore::ProofStore(String filename, bool quiet)
{
    // Reserve line 0:
    ret_kinds  .push();
    rule_kinds .push();
    args_offset.push();
    args_size  .push();
    n_fanouts  .push();

    // Read and store proof:
    index_t n_types = 1;
    index_t n_terms = 1;
    index_t n_thms  = 1;
    index_t n_human_thms = 0;

    auto callback = [&](ArgKind ret_kind, RuleKind rule_kind, Vec<Arg>& args, double progress) -> void
    {
        line_t line_no = size();
        if (ret_kind == arg_TYPE){ line2idx(line_no) = n_types; ty_idx2line(n_types) = line_no; n_types++; }
        if (ret_kind == arg_TERM){ line2idx(line_no) = n_terms; tm_idx2line(n_terms) = line_no; n_terms++; }
        if (ret_kind == arg_THM ){ line2idx(line_no) = n_thms ; th_idx2line(n_thms ) = line_no; n_thms ++; }

        if (rule_kind == rule_New_Def)
            new_def_line(args[0].cnst()) = line_no;

        if (rule_kind == rule_New_TDef){
            new_tdef_line(args[0].tcon()) = line_no;
            new_def_line (args[1].cnst()) = line_no;
            new_def_line (args[2].cnst()) = line_no;
        }

        ret_kinds  .push(ret_kind);
        rule_kinds .push(rule_kind);
        args_offset.push(args_data.size());
        args_size  .push(args.size());
        append(args_data, args);

        // Progress output:
        if (rule_kind == rule_TThm){
            Str name = Str(args[0].tthm());
            if (name[0] != '!')
                n_human_thms++;
        }
        if (!quiet && line_no % 100000 == 0)
            Write_ "\rReading proof-log:  %.1f %% done.  %,d named theorems.\f", progress, n_human_thms;
    };

    parseProofLog(filename, callback);
    if (!quiet) Write_ "\rReading proof-log:  Complete!  %,d named theorems.\n", n_human_thms;

    // Finalize data:
    countFanouts();
}


void ProofStore::countFanouts()
{
    n_fanouts.clear();
    n_fanouts.growTo(size(), 0);
    for (uind n = 0; n < size(); n++)
        if (is_thm(n))
            for (uind m : thms(n))
                n_fanouts[m]++;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Lazy evaluation:


Type ProofStore::evalType(ty_t idx)
{
    if (!idx2type[idx]){
        line_t n = ty_idx2line[idx];
        idx2type(idx) = produceType(rule(n), translateArgs(n));
    }
    return idx2type[idx];
}


Term ProofStore::evalTerm(th_t idx)
{
    if (!idx2term[idx]){
        line_t n = tm_idx2line[idx];
        idx2term(idx) = produceTerm(rule(n), translateArgs(n));
    }
    return idx2term[idx];
}


Thm ProofStore::evalThm(th_t idx)
{
    if (!idx2thm[idx]){
        line_t n = th_idx2line[idx];
        idx2thm(idx) = produceThm(rule(n), translateArgs(n));
        if (!thm2line[idx2thm[idx]])   // -- need this check since the same theorem can be generated multiple times (and only the first is safe to use if we want to avoid cycles)
          thm2line(idx2thm[idx]) = n;
    }
    return idx2thm[idx];
}


// NOTE! Returns a slice into a temporary class variable, so each call invalidates the result from
// the previous call.
Array<Arg const> ProofStore::translateArgs(line_t n)
{
    // Populate 'idx2type/term/thm' for transitive fanin:
    for (Arg const& a : args(n)){
        if      (a.kind == arg_TYPE_IDX) evalType(a.id);
        else if (a.kind == arg_TERM_IDX) evalTerm(a.id);
        else if (a.kind == arg_THM_IDX ) evalThm (a.id);
        else if (a.kind == arg_CNST){
            line_t m = new_def_line[a.cnst()]; assert(m <= n);
            if (m && m != n) evalThm(line2idx[m]); }
        else if (a.kind == arg_TCON){
            line_t m = new_tdef_line[a.tcon()]; assert(m <= n);
            if (m && m != n) evalThm(line2idx[m]); }
    }

    // Translate arguments for this node 'n':
    vecCopy(args(n), eval_tmp);
    for (Arg& a : eval_tmp){
        if      (a.kind == arg_TYPE_IDX) a = Type(idx2type[a.id]);
        else if (a.kind == arg_TERM_IDX) a = Term(idx2term[a.id]);
        else if (a.kind == arg_THM_IDX ) a = Thm (idx2thm [a.id]);
    }
    return eval_tmp.slice();
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

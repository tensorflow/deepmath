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
#include "Checker.hh"
#include "Parser.hh"
#include "RuleApply.hh"
#include "Printing.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


void checkProof(String filename, CheckMode mode, bool show_progress)
{
    // Produced objects storage:
    Vec<Type> types; types.push();   // -- first element is the null type/term/theorem (default constructed)
    Vec<Term> terms; terms.push();
    Vec<Thm>  thms;  thms .push();

    // Statistics
    uint64 ruleC[RuleKind_size] = {0};
    uint64 retC [ArgKind_size ] = {0};
    uint64 real_thmC = 0;
    uint64 argsC = 0;

    uint64 stats_cc = 0;
    auto printStats = [&](double progress) {
        // Hackish. Revert to printf() to flush output without newline.     // <<== fix this!
        printf("\rMemory used: %.2f MB     #types=%.0f  #terms=%.0f  #thms=%.0f     (%.1f %%)",
               (double)memUsedNow() / 1024 / 1024,
               (double)types.size()-1, (double)terms.size()-1, (double)thms.size()-1,
               progress);
        if (mode != CheckMode::CHECK)
            printf("\n");
        fflush(stdout);
    };

    auto callback = [&](ArgKind ret_kind, RuleKind rule_kind, Vec<Arg>& args, double progress) -> void
    {
        // Collect statistics:
        ruleC[rule_kind]++;
        retC [ret_kind]++;
        argsC += args.size();

        if (rule_kind == rule_TThm){
            Str name = Str(args[0].tthm());
            if (name[0] != '!'){
                if (mode == CheckMode::THMS)
                    // Write human theorems:
                    WriteLn "%_", name;
                real_thmC++;
            }
        }

        if (mode == CheckMode::APPL || mode == CheckMode::FULL){
            // Rule application output:
            if (ret_kind == arg_NULL) Write_ "`` %_(", rule_kind;
            else                      Write_ "`` %_[%_] = %_(", ret_kind, retC[ret_kind], rule_kind;
            for (uint i = 0; i < args.size(); i++){
                if (args[i].isAtomic()) Write_ " %_[\"%_\"]", makeNoIndex(args[i].kind), args[i].str();
                else                    Write_ " %_[%_]"    , makeNoIndex(args[i].kind), args[i].id;
            }
            WriteLn " )";
        }

        if (mode != CheckMode::STATS){
            // Construct Types, Terms, Thms (translate from proof-index to object ID):
            for (Arg& a : args){
                if      (a.kind == arg_TYPE_IDX) a = Type(types[a.id]);
                else if (a.kind == arg_TERM_IDX) a = Term(terms[a.id]);
                else if (a.kind == arg_THM_IDX ) a = Thm (thms [a.id]);
            }

            // Execute rule:
            switch (ret_kind){
            case arg_TYPE: types.push(produceType(rule_kind, args.slice())); break;
            case arg_TERM: terms.push(produceTerm(rule_kind, args.slice())); break;
            case arg_THM:  thms .push(produceThm (rule_kind, args.slice())); break;
            case arg_NULL: break;   // -- top-level theorem
            default: assert(false); }
        }

        if (mode == CheckMode::FULL){
            // More debug output:
            switch (ret_kind){
            case arg_TYPE: WriteLn "``   :%_", types.last(); break;
            case arg_TERM: WriteLn "``   %_ : %_", terms.last(), terms.last().type(); break;
            case arg_THM:
                if (rule_kind == rule_New_TDef) WriteLn "`` <no-theorem>";
                else WriteLn "``   %_", thms .last();
                break;
            case arg_NULL: WriteLn "`` ------------------------------------------------------------ %_", args[0].tthm(); break;
            default: assert(false); }
        }

        if (show_progress){
            if (++stats_cc % 50000 == 0)
                printStats(progress);
        }
    };

    // Read proof:
    parseProofLog(filename, callback, true);

    if (mode == CheckMode::THMS)
        return;     // -- no more output

    if (show_progress){
        printStats(100);
        NewLine;
        NewLine;
    }

    if (mode != CheckMode::STATS){
        // Uniqueness statistics:
        uind pfl_types = types.size() - 1;
        uind pfl_terms = terms.size() - 1;
        uind pfl_thms  = thms .size() - 1;
        sortUnique(types);
        sortUnique(terms);
        sortUnique(thms);
        WriteLn "Uniqueness statistics for top-level objects:";
        WriteLn "  - Types: %>11%,d -> %>11%,d", pfl_types, types.size() - 1;
        WriteLn "  - Terms: %>11%,d -> %>11%,d", pfl_terms, terms.size() - 1;
        WriteLn "  - Thms : %>11%,d -> %>11%,d", pfl_thms , thms .size() - 1;
        NewLine;

        // Memory statistics:
        #define AtomicStats(T) WriteLn "%<5%_: %>11%,d  %>11%,d", #T, T::count(), T::strAlloc();
        WriteLn "             COUNT     STR-DATA";
        AtomicStats(Cnst);
        AtomicStats(Var );
        AtomicStats(TCon);
        AtomicStats(TVar);
        AtomicStats(Axiom);
        AtomicStats(TThm);
        NewLine;

        typedef List<BaseList> Lst;
        #define CompositeStats(T) WriteLn "%<5%_: %>11%,d    %>2%_ bytes   (set efficiency = %.1f %%)", #T, T::count(), sizeof(T::Data), 100.0*T::count() / T::alloc();
        WriteLn "             COUNT   TYPE-SIZE";
        CompositeStats(Type);
        CompositeStats(Term);
        CompositeStats(Thm);
        CompositeStats(Lst);
        NewLine;
    }

    // Output statistics:
    WriteLn "Real theorems:   %,d", real_thmC;
    WriteLn "Total arguments: %,d", argsC;
    NewLine;

    WriteLn "Return types:";
    for (uint i = 0; i < ArgKind_size; i++)
        if (retC[i] != 0)
            WriteLn "  - %<8%_: %>11%,d", ArgKind(i), retC[i];
    NewLine;

    WriteLn "Rule applications:";
    for (uint i = 0; i < RuleKind_size; i++)
        if (ruleC[i] != 0)
            WriteLn "  - %<8%_: %>11%,d", RuleKind(i), ruleC[i];
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

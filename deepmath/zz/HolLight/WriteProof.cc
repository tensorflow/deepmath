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
#include "WriteProof.hh"
#include "Kernel.hh"
#include "Printing.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


// Translate reserved '`' to '&' (which is also unused in variable names in our proof-logs)
struct NoTick {
    Str str;
    NoTick(Str str) : str(str) {}
};


template<> fts_macro void write_(Out& out, NoTick const& v) {
    for (char c : v.str){
        if (c == '`') out += '&';
        else out += c;
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


class WriteProof {
    String filename;
    ProofStore* P;
    Vec<Pair<Thm,Str>> const& named_thms;
    OutFile out;

    uint64 typeC = 0;
    uint64 termC = 0;
    uint64 thmC  = 0;
    IntMap<Type, ty_t> type2idx;
    IntMap<Term, tm_t> term2idx;
    IntMap<Thm , th_t> thm2idx;

    ty_t emit(Type ty);
    tm_t emit(Term tm);
    th_t emit(Thm  th);

    void run();

public:
    WriteProof(String filename, ProofStore& P, Vec<Pair<Thm,Str>> const& named_thms) :
        filename(filename),
        P(&P),
        named_thms(named_thms),
        out(filename)
    {
        if (!out){
            shoutLn("ERROR! Could not open proof-file for writing: %_", filename);
            exit(1); }
        run();
    }
};


ty_t WriteProof::emit(Type ty)
{
    if (!type2idx[ty]){
        if (ty.is_tvar()){
            wrLn(out, "t%_", ty.tvar());

        }else{ assert(ty.is_tapp());
            for (List<Type> it = ty.targs(); it; ++it)
                emit(*it);
            emit(getTDef(ty.tcon()).th);
            wr(out, "a%_", ty.tcon());
            for (List<Type> it = ty.targs(); it; ++it)
                wr(out, " %_", type2idx[*it]);
            newLn(out);
        }

        type2idx(ty) = ++typeC;
    }
    return type2idx[ty];
}


tm_t WriteProof::emit(Term tm)
{
    if (!term2idx[tm]){
        emit(tm.type());
        if (tm.is_var()){
            wrLn(out, "v%_ %_", NoTick(Str(tm.var())), type2idx[tm.type()]);

        }else if (tm.is_cnst()){
            emit(getDef(tm.cnst()).th);
            wrLn(out, "c%_ %_", tm.cnst(), type2idx[tm.type()]);

        }else if (tm.is_comb()){
            emit(tm.fun());
            emit(tm.arg());
            wrLn(out, "f%_ %_", term2idx[tm.fun()], term2idx[tm.arg()]);

        }else{ assert(tm.is_abs());
            emit(tm.avar());
            emit(tm.aterm());
            wrLn(out, "l%_ %_", term2idx[tm.avar()], term2idx[tm.aterm()]);
        }

        term2idx(tm) = ++termC;
    }
    return term2idx[tm];
}


th_t WriteProof::emit(Thm th)
{
    if (!th) return 0;      // -- we can get here for builtin definitions (not backed up by any theorem)

    if (!thm2idx[th]){
        line_t n = P->thm2line[th]; assert(P->ret(n) == arg_THM);
        Array<Arg const> args = P->args(n);

        auto type = [&](uint i){ return emit(P->evalType(args[i].id)); };
        auto term = [&](uint i){ return emit(P->evalTerm(args[i].id)); };
        auto thm  = [&](uint i){ return emit(P->evalThm (args[i].id)); };

        switch (P->rule(n)){
        case rule_REFL    : wrLn(out, "R%_", term(0)); break;
        case rule_TRANS   : wrLn(out, "T%_ %_", thm(0), thm(1)); break;
        case rule_MK_COMB : wrLn(out, "C%_ %_", thm(0), thm(1)); break;
        case rule_ABS     : wrLn(out, "L%_ %_", term(0), thm(1)); break;
        case rule_BETA    : wrLn(out, "B%_", term(0)); break;
        case rule_ASSUME  : wrLn(out, "H%_", term(0)); break;
        case rule_EQ_MP   : wrLn(out, "E%_ %_", thm(0), thm(1)); break;
        case rule_DEDUCT  : wrLn(out, "D%_ %_", thm(0), thm(1)); break;
        case rule_New_Ax  : wrLn(out, "A%_ %_", args[0].axiom(), term(1)); break;
        case rule_New_Def : wrLn(out, "F%_ %_", args[0].cnst(), term(1)); break;
        case rule_New_TDef: wrLn(out, "Y%_ %_ %_ %_ %_ %_", args[0].tcon(), args[1].cnst(), args[2].cnst(), term(3), term(4), thm(5)); break;
        case rule_TDef_Ex1: wrLn(out, "1%_", thm(0)); break;
        case rule_TDef_Ex2: wrLn(out, "2%_", thm(0)); break;

        case rule_INST:{
            Vec<tm_t> subst;
            for (uint i = 0; i < args.size()-1; i++) subst.push(term(i));
            wrLn(out, "S%_ %_", join(' ', subst), thm(args.size()-1));
            break; }
        case rule_INST_T:{
            Vec<ty_t> tsubst;
            for (uint i = 0; i < args.size()-1; i++) tsubst.push(type(i));
            wrLn(out, "Q%_ %_", join(' ', tsubst), thm(args.size()-1));
            break; }

        default: assert(false); }

        thm2idx(th) = ++thmC;
    }
    return thm2idx[th];
}


void WriteProof::run()
{
    for (auto&& p : named_thms){
        emit(p.fst);
        wrLn(out, "+%_", p.snd);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Wrapper functions:


void writeProof(ProofStore& P, Vec<Pair<Thm,Str>> const& named_thms, String output_file)
{
    WriteProof wr(output_file, P, named_thms);
}


void writeProof(String input_file, String output_file)
{
    ProofStore P(input_file);
    IntMap<Thm,TThm> processed;

    WriteLn "Evaluating proof...";
    Vec<Pair<Thm,Str>> named_thms;
    for (line_t n = 0; n < P.size(); n++){
        if (P.is_humanThm(n)){
            Thm th = P.evalThm(P.line2idx[n]);
            if (processed[th])
                wrLn("  - NOTE! Skipping repeated theorem: %_ (= %_)", P.thmName(n), processed[th]);
                    // <<== add as vacuous INST...
            else{
                processed(th) = P.thmName(n);
                named_thms.push(tuple(th, processed[th]));
            }
        }
    }

    WriteLn "Writing new proof to '%_' (%_ distinct theorems)...", output_file, named_thms.size();
    writeProof(P, named_thms, output_file);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

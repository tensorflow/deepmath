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
#include "Train_PremiseGraph.hh"
#include "zz/Generics/Sort.hh"
#include "ProofStore.hh"
#include "Printing.hh"
#include "HolFormat.hh"

namespace ZZ {
using namespace std;

String fmtHolStep(Term tm, bool include_types);     // -- forward-declaration


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


static
Term stripForall(Term tm)
{
    static Cnst cnst_forall("!");
    while (tm.is_comb() && tm.fun().is_cnst() && tm.arg().is_abs() && tm.fun().cnst() == cnst_forall)
        tm = tm.arg().aterm();
    return tm;
}


static void listConstants(Vec<Term>& out, Term tm)
{
    if (tm.is_cnst())
        out.push(tm);
    else if (tm.is_comb()){
        listConstants(out, tm.fun());
        listConstants(out, tm.arg());
    }else if (tm.is_abs())
        listConstants(out, tm.aterm());
}
static String listConstants(Term tm)    // -- call this one
{
    Vec<Term> out;
    listConstants(out, tm);
    sortUnique(out);
    return fmt("%_", join(" ", out));
}


static void listTypecons(Vec<TCon>& out, Type ty)
{
    if (ty.is_tapp()){
        out.push(ty.tcon());
        for (List<Type> it = ty.targs(); it; ++it)
            listTypecons(out, *it);
    }
}
static void listTypecons(Vec<TCon>& out, Term tm)
{
    listTypecons(out, tm.type());
    if (tm.is_comb()){
        listTypecons(out, tm.fun());
        listTypecons(out, tm.arg());
    }else if (tm.is_abs()){
        listTypecons(out, tm.avar());
        listTypecons(out, tm.aterm());
    }
}
static String listTypecons(Term tm)     // -- call this one
{
    Vec<TCon> out;
    listTypecons(out, tm);
    sortUnique(out, [](TCon x, TCon y){ return x < y; });
    return fmt("%_", join(" ", out));
}


template<typename... Args>
void protoWrLn(Out& out, cchar* fmt, Args const&... args)
{
    String text;
    wr(text, fmt, args...);

    for (char c : text)
        if (c == '\\' || c == '\'') out += '\\', c;
        else out += c;
    out += NL;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


/*
message TThmInfo {
  uint64 id = 1;
  string name = 2;
  uint32 type = 3;    // 0=regular theorem, 1=constant definition, 2=type definition

  repeated uint64 premise_thms  = 4;   // list of theorems needed to prove this one ("inputs")
  repeated bytes  premise_types = 5;   // list of {0,1,2}s of same length as 'premise_thms'
  repeated uint64 user_thms     = 6;   // list of theorems using this theorem ("outputs")

  string text_hol = 7;
  string text_cez = 8;
  string text_std = 9;
  string constants;    // space separated list of constant symbols used in theorem (occurs in text representations)
  string typecons;     // space separated list of type constructors used in theorem (NOT in text representations)
};


message PremiseGraph {
  repeated TThmInfo tthms = 1;
};


tthms {
  id: 123
  name: "gurka"
  type: 4711
  premise_thms: 7
  premise_thms: 8
  premise_thms: 9
  premise_types: 0
  premise_types: 1
  premise_types: 2
  user_thms: 111
  user_thms: 222
  text_hol: "hol"
  text_cez: "cez"
  text_std: "std"
  constants: "= + empty"
  typecons: "bool fun"
}

*/


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Compute premises ("inputs") and users ("outputs"):


void getGraph(ProofStore& P, Vec<Vec<line_t>>& out_prems, Vec<Vec<line_t>>& out_users)
{
    Vec<line_t>    prems;
    IntZet<line_t> seen;

    function<void(line_t)> getPremises = [&](line_t n) {
        assert(P.is_thm(n));
        for (line_t m : P.thms(n)){
            if (seen.add(m)) continue;

            if (P.is_humanThm(m))
                prems.push(m);
            else
                getPremises(m);
        }
    };

    for (line_t n = 0; n < P.size(); n++){
        if (P.is_humanThm(n)){
            getPremises(n);
            for (line_t m : prems)
                out_users(m).push(n);
            out_prems(n) = move(prems);
            prems.clear();
            seen.clear();
        }
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


void writePremiseGraph(String input_file, String output_file)
{
    ProofStore P(input_file);

    pp_full_internal    = false;
    pp_use_ansi         = false;
    pp_show_types       = false;
    pp_show_bound_types = false;
    pp_readable_eqs     = false;

    // Establish local serial numbers for top-level theorems in premise graph:
    IntMap<line_t,uind> line2serial;
    line2serial(0) = 0;
    uind tthm_count = 0;
    for (line_t n = 0; n < P.size(); n++)
        if (P.is_humanThm(n))
            line2serial(n) = ++tthm_count;

    // Compute graph in terms of human named theorems:
    Vec<Vec<line_t>> prems;
    Vec<Vec<line_t>> users;
    getGraph(P, prems, users);

    // Write ASCII protobuf file:
    OutFile out(output_file);
    if (!out) shoutLn("ERROR! Could not open file for writing: %_", output_file), exit(1);
    for (line_t n = 0; n < P.size(); n++){
        if (n % 16384 == 0)
            wr("\rWriting premise graph:  %.1f %% done.\f", 100.0 * n / P.size());

        if (P.is_humanThm(n)){
            auto thmType = [&P](line_t n) {
                return (P.rule(n) == rule_New_Def ) ? 1 :
                       (P.rule(n) == rule_New_TDef) ? 2 :
                       /*otherwise*/                  0 ;
            };

            Thm  th = P.evalThm(P.line2idx[n]);
            if (!th.hyps().empty())
              wrLn("\rNOTE! Theorem '%_' has %_ hypothesis not stored in the proof-graph information.", P.thmName(n), th.hyps().size());
            //assert(th.hyps().empty());
            Term tm = stripForall(th.concl());

            protoWrLn(out, "tthms {");
            protoWrLn(out, "  serial: %_", line2serial[n]);
            protoWrLn(out, "  name: \"%_\"", P.thmName(n));
            protoWrLn(out, "  type: %_", thmType(n));

            for (line_t m : prems[n]) protoWrLn(out, "  premise_thms: %_", line2serial[m]);
            for (line_t m : prems[n]) protoWrLn(out, "  premise_types: %_", thmType(m));
            for (line_t m : users[n]) protoWrLn(out, "  user_thms: %_", line2serial[m]);

            protoWrLn(out, "  text_hol: \"%_\"", fmtTerm(tm));
            protoWrLn(out, "  text_cez: \"%_\"", fmtHolStep(tm, false));
            protoWrLn(out, "  text_std: \"%_\"", tm);
            protoWrLn(out, "  constants: \"%_\"", listConstants(tm));
            protoWrLn(out, "  typecons: \"%_\"", listTypecons(tm));

            protoWrLn(out, "}");
        }
    }
    wrLn("\rWriting premises graph:  Complete!\n");
    wrLn("Wrote: %_", output_file);
}


// 'test_fraction' is in [0,1]; 0.1 means 10% of all human named theorems will be written to output file as a test set.
void computeTestSet(String input_file, String output_file, double test_fraction, uint64 seed)
{
    assert(test_fraction >= 0 && test_fraction <= 1);

    // Read proof:
    ProofStore P(input_file);

    Vec<line_t> tthms;
    for (line_t n = 0; n < P.size(); n++)
        if (P.is_humanThm(n))
            tthms.push(n);

    Vec<Vec<line_t>> prems(P.size());
    Vec<Vec<line_t>> users(P.size());
    getGraph(P, prems, users);

    // Generate test set:
    uind target_sz = tthms.size() * test_fraction;
    Vec<line_t> test_set;
    while (test_set.size() < target_sz){
        wr("\rPicking theorem %_ of %_\f", test_set.size()+1, target_sz);
        uind i = irand(seed, tthms.size());
        line_t n = tthms[i];

        while (users[n].size() > 0){
            uind j = irand(seed, users[n].size());
            n = users[n][j];
        }

        for (line_t m : prems[n])
            revPullOut(users[m], n);
        revPullOut(tthms, n);

        test_set.push(n);
    }
    wrLn("\rPicked %_ theorems.", test_set.size());

    sort(test_set);
    OutFile out(output_file);
    if (!out) shoutLn("ERROR! Could not open file for writing: %_", output_file), exit(1);
    for (line_t n : test_set)
        wrLn(out, "%_", P.thmName(n));
    wrLn("Wrote: %_", output_file);

#if 0
{
    // Verify that all users are present in testset:
    Vec<Vec<line_t>> prems(P.size());
    Vec<Vec<line_t>> users(P.size());
    getGraph(P, prems, users);

    for (line_t n : test_set){
        if (users[n].size() > 0){
            wrLn("%_: %_ users", P.thmName(n), users[n].size());
            for (line_t m : users[n])
                assert(has(test_set, m));
        }
    }
}
#endif
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// HOL-step printing (translated from 'deepmath/hol/printer.cc'):


// NOTE! The original code will use de-Bruijn index for bound variable; the below code use
// de-Bruijn level.


static
void print_training_type(Out& out, Type ty)
{
    if (ty.is_tvar()){
        out += " VT";
    }else{ assert(ty.is_tapp());
        TCon i = ty.tcon();
        List<Type> l = ty.targs();
        out += " t", i;
        if (!l.empty()){
            for (List<Type> it = l; it; ++it)
                print_training_type(out, *it);
        }
    }
}


static
void print_training_tokens_parenthesize(Out& out, Term tm)
{
    if (tm.is_cnst()){
        out += "c", tm.cnst();
        print_training_type(out, tm.type());
    }else if (tm.is_var()){
        out += "v";
        print_training_type(out, tm.type());
    }else if (tm.is_comb()){
        Term tml = tm.fun();
        Term tmr = tm.arg();
        out += "(";
        print_training_tokens_parenthesize(out, tml);
        out += " ";
        print_training_tokens_parenthesize(out, tmr);
        out += ")";
    }else{ assert(tm.is_abs());
        out += "<v";
        print_training_type(out, tm.type());
        out += ".";
        print_training_tokens_parenthesize(out, tm.aterm());
        out += ">";
    }
}


static
uint funArity(Type ty)
{
    uint ret = 0;
    while (ty.is_tapp() && ty.tcon() == tcon_fun){
        ty = ty.targs()[1];
        ret++;
    }
    return ret;
}


static
void stripComb(Term tm, Term& out_hop, Vec<Term>& out_args)
{
    assert(out_args.size() == 0);
    while (tm.is_comb()){
        out_args.push(tm.arg());
        tm = tm.fun(); }
    reverse(out_args);
    out_hop = tm;
}


static
void print_training_tokens_vars(Out& out, Term tm, bool include_types)
{
    if (tm.is_cnst()){
        if (funArity(tm.type()) > 0)
            out += " part";
        out += " c",  tm.cnst();
        if (include_types)
            print_training_type(out, tm.type());

    }else if (tm.is_var()){
        Var x = tm.var();
        if (Term::isLambdaVar(x)){
            out += " b", Str(x).slice(1);   // -- to match Cezary's output more closely, remove the leading backtick
        }else{
            out += " f", x;
        }

    }else if (tm.is_comb()){
        static Cnst cnst_forall("!");
        if (tm.fun().is_cnst() && tm.fun().cnst() == cnst_forall && tm.arg().is_abs()){
            out += " !";
            if (include_types)
                print_training_type(out, tm.type());
            print_training_tokens_vars(out, tm.arg().aterm(), include_types);

        }else{
            Term      hop;
            Vec<Term> args;
            stripComb(tm, hop, args);

            if (hop.is_cnst() && funArity(hop.type()) == args.size()){
                out += " c", hop.cnst();
                if (include_types)
                    print_training_type(out, hop.type());
                for (auto&& arg : args)
                    print_training_tokens_vars(out, arg, include_types);
            }else{
                out += " *";
                print_training_tokens_vars(out, tm.fun(), include_types);
                print_training_tokens_vars(out, tm.arg(), include_types);
            }
        }

    }else{ assert(tm.is_abs());
        out += " /";    // -- why slash instead of backslash for lambda?
        if (include_types)
            print_training_type(out, tm.type());
        print_training_tokens_vars(out, tm.aterm(), include_types);
    }
}


String fmtHolStep(Term tm, bool include_types)
{
    String out;
    print_training_tokens_vars(out, stripForall(tm), include_types);
    trim(out);
    return out;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

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
#include "Parser.hh"
#include "zz/Generics/LineReader.hh"


namespace ZZ {
using namespace std;


namespace {
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Parser rules, readable form:


struct RulePattern {
    uchar tag;
    cchar* ret_kind;
    cchar* name;            // -- must match names from 'RuleKind' in 'ParserTypes.hh'
    cchar* arg_kinds;
};


RulePattern rule_patterns[] = {
    // tag, result, name, args ($ means string, not id)
    { 't', "type", "TVar"    , "$tvar" },
    { 'a', "type", "TAppl"   , "$tcon [type]" },    // -- type-inst.; must be preceded by a 'Y', except for builtin 'bool' and 'fun'

    { 'c', "term", "Cnst"    , "$cnst type" },      // -- const-inst.; must be preceded by a 'F' (or 'Y'), except for builtin "=" and "@"
    { 'v', "term", "Var"     , "$var type" },
    { 'f', "term", "App"     , "term term" },       // -- types must match: f x
    { 'l', "term", "Abs"     , "term term" },       // -- first term must be variable

    { 'R', "thm" , "REFL"    , "term" },
    { 'T', "thm" , "TRANS"   , "thm thm" },
    { 'C', "thm" , "MK_COMB" , "thm thm" },
    { 'L', "thm" , "ABS"     , "term thm" },        // -- var-term, eq-thm
    { 'B', "thm" , "BETA"    , "term" },
    { 'H', "thm" , "ASSUME"  , "term" },
    { 'E', "thm" , "EQ_MP"   , "thm thm" },
    { 'D', "thm" , "DEDUCT"  , "thm thm" },
    { 'S', "thm" , "INST"    , "[term term] thm" }, // -- list of pairs (var-term, term)
    { 'Q', "thm" , "INST_T"  , "[type type] thm" }, // -- list of pairs (var-type, type)

    { 'A', "thm" , "New_Ax"  , "$axiom term" },     // -- term is of type bool
    { 'F', "thm" , "New_Def" , "$cnst term" },      // -- introduces a polymorphic constant (think of it as a "constant constructor", with 'c' rule instantiating a specific subtype)
    { 'Y', "thm" , "New_TDef", "$tcon $cnst $cnst term term thm" },
                                                    // -- pushes a null theorem; the two terms can be extracted from the theorem ("existence proof")

    { '1', "thm" , "TDef_Ex1", "thm" },             // -- extract |- abs(rep a:P) = a
    { '2', "thm" , "TDef_Ex2", "thm" },             // -- extract |- P r = (rep(abs r) = r)
    { '+', "nil" , "TThm"    , "$tthm" },           // -- mark previous theorem as "top-level"
};

// NOTE! Unused characters in proof-logs are: "&():;[]`{|}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Rule-compiler:


struct PR_Arg {
    bool    as_string;
    ArgKind arg_kind;

    PR_Arg() : as_string(false), arg_kind(arg_NULL) {}
    PR_Arg(Str text);
};

struct PR_Unit {
    bool    list_of = false;
    PR_Arg  arg;
    PR_Arg  arg2;   // -- only used in conjunction with 'list_of'
};


struct PR {
    uchar        tag = 0;
    ArgKind      ret_kind = arg_NULL;
    RuleKind     rule_kind = rule_NULL;
    Vec<PR_Unit> units;
};


PR_Arg::PR_Arg(Str text) {
    if (text[0] == '$'){
        as_string = true;
        text = text.slice(1);
    }else
        as_string = false;

    if      (eq(text, "type" )) arg_kind = arg_TYPE;
    else if (eq(text, "term" )) arg_kind = arg_TERM;
    else if (eq(text, "thm"  )) arg_kind = arg_THM;
    else if (eq(text, "cnst" )) arg_kind = arg_CNST;
    else if (eq(text, "var"  )) arg_kind = arg_VAR;
    else if (eq(text, "tcon" )) arg_kind = arg_TCON;
    else if (eq(text, "tvar" )) arg_kind = arg_TVAR;
    else if (eq(text, "axiom")) arg_kind = arg_AXIOM;
    else if (eq(text, "tthm" )) arg_kind = arg_TTHM;
    else if (eq(text, "nil"  )) arg_kind = arg_NULL;
    else{ Dump(text); assert(false); }
}

PR parse_rule[256];


}
//=================================================================================================
// -- Pretty-printing for debugging:


template<> fts_macro void write_(Out& out, PR_Arg const& v) {
    FWrite(out) "%C%_", (v.as_string ? '$' : '\0'), v.arg_kind; }

template<> fts_macro void write_(Out& out, PR_Unit const& v) {
    if (v.list_of){
        if (v.arg2.arg_kind == arg_NULL)
            FWrite(out) "[%_]", v.arg;
        else
            FWrite(out) "[%_ %_]", v.arg, v.arg2;
    }else
        out += v.arg;
}

template<> fts_macro void write_(Out& out, PR const& v) ___unused;
template<> fts_macro void write_(Out& out, PR const& v) {
    FWrite(out) "PR{tag=%_; ret_kind=%_; rule_kind=%_; units=%_}",
        v.tag, v.ret_kind, v.rule_kind, v.units;
}


namespace {
//=================================================================================================
// -- Rule-compiler main function:


void genParseRules()
{
    if (parse_rule[0].tag == 255) return;     // -- allow for running 'genParseRules()' multiple times

    for (RulePattern const& pat : rule_patterns){
        PR& pr = parse_rule[pat.tag];
        pr.tag = pat.tag;

        PR_Arg pr_arg(slize(pat.ret_kind));
        assert(!pr_arg.as_string);
        assert(isComposite(pr_arg.arg_kind) || pr_arg.arg_kind == arg_NULL);
        pr.ret_kind = pr_arg.arg_kind;

        for (uint i = 0; i < RuleKind_size; i++)
            if (eq(RuleKind_name[i], pat.name)){
                pr.rule_kind = RuleKind(i);
                break; }
        assert(pr.rule_kind != rule_NULL);

        Vec<Str> units;
        splitArray(slize(pat.arg_kinds), " ", units);
        for (uint i = 0; i < units.size(); i++){
            pr.units.push();
            PR_Unit& u = pr.units.last();

            if (units[i][0] == '['){
                u.list_of = true;
                if (units[i][LAST] == ']'){
                    assert(i+1 == units.size());        // -- a list must be at the end
                    u.arg = PR_Arg(units[i].slice(1, units[i].size()-1));
                }else{
                    assert(i+1 < units.size());         // -- otherwise missing ']' at end of string
                    assert(i+3 >= units.size());        // -- a list of pair can have at most one element after itself
                    assert(units[i+1][LAST] == ']');    // -- otherwise not a list of pairs
                    u.arg  = PR_Arg(units[i].slice(1));
                    u.arg2 = PR_Arg(units[i+1].slice(0, units[i+1].size()-1));
                    i++;
                }
            }else
                u.arg = PR_Arg(units[i]);
        }
        //**/Dump(pr);
    }
    parse_rule[0].tag = 255;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Parser:


uint64 strToUInt64(Str text)
{
    uint64 result = 0;
    for (char c : text){
        if (c < '0' || c > '9') throw Excp_ParseNum(Excp_ParseNum::IllFormed);
        result = result * 10 + (c - '0');
    }
    return result;
}


Arg parseArg(Str text, PR_Arg a)
{
    if (a.as_string){
        assert(!has(text, '`'));    // -- backtick is a reserved character
        switch (a.arg_kind){
        case arg_CNST : return Cnst (text);
        case arg_VAR  : return Var  (text);
        case arg_TCON : return TCon (text);
        case arg_TVAR : return TVar (text);
        case arg_AXIOM: return Axiom(text);
        case arg_TTHM : return TThm (text);
        default: assert(false); }
    }else{
        if (text[0] == '-') text = text.slice(1);   // -- ignore minus sign, no GC.
        uint64 id = strToUInt64(text);
        switch (a.arg_kind){
        case arg_CNST : return Cnst (id);
        case arg_VAR  : return Var  (id);
        case arg_TCON : return TCon (id);
        case arg_TVAR : return TVar (id);
        case arg_AXIOM: return Axiom(id);
        case arg_TTHM : return TThm (id);
        case arg_TYPE : return Arg(arg_TYPE_IDX, id);
        case arg_TERM : return Arg(arg_TERM_IDX, id);
        case arg_THM  : return Arg(arg_THM_IDX , id);
        default: assert(false); }
    }
}


void parseLine(uint64 line_no, Vec<Str> const& fs, Vec<PR_Unit> const& units, /*out*/Vec<Arg>& args)
{
    uint i = 0;
    for (PR_Unit u : units){
        try{
            if (u.list_of){
                if (u.arg2.arg_kind == arg_NULL){
                    // List of singletons:
                    while (i < fs.size()){
                        args.push(parseArg(fs[i], u.arg)); i++;
                    }
                }else{
                    // List of pairs:
                    while (i+2 < fs.size()){
                        args.push(parseArg(fs[i], u.arg )); i++;
                        args.push(parseArg(fs[i], u.arg2)); i++;
                    }
                }
            }else{
                // Single argument:
                if (i >= fs.size()){
                    ShoutLn "ERROR! Missing arguments on line %_", line_no;
                    exit(1); }

                args.push(parseArg(fs[i], u.arg));
                i++;
            }
        }catch(Excp_ParseNum){
            ShoutLn "ERROR! Expected number on line %_ (arg%_='%_')", line_no, i, fs[i];
            exit(1);
        }
    }
    if (i != fs.size()){
        ShoutLn "ERROR! %_ extra arguments on line %_", fs.size() - i, line_no;
        exit(1); }
}


}
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Public funtions:


void parseProofLog(String const& filename, ProofCallback const& cb, bool with_args)
{
    genParseRules();

    LineReader in(filename);
    if (!in){
        ShoutLn "ERROR! Could not open: %_", filename;
        exit(1); }
    uint64 input_length = in.size();

    Vec<Str>  fs;
    uint64    line_no = 0;
    Vec<Arg>  args;

    while (!in.eof()){
        Str line = in.getLine();
        line_no++;
        if (line.size() == 0) continue;     // -- ignore empty lines

        // Parse and hash string symbols (Var, TCon etc.):
        uchar tag = line[0];
        splitArray(line.slice(1), ' ', fs);

        PR const& pr = parse_rule[tag];
        if (pr.rule_kind == rule_NULL){
            ShoutLn "ERROR! Unsupported rule tag: %_", tag;
            exit(1); }

        // Parse line:
        if (with_args || pr.rule_kind == rule_TThm){
            args.clear();
            parseLine(line_no, fs, pr.units, args); }

        // Call callback:
        double progress = 100.0*in.tell() / input_length;
        cb(pr.ret_kind, pr.rule_kind, args, progress);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

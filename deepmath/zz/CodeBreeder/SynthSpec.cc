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

#include ZZ_Prelude_hh
#include "SynthSpec.hh"

#include "Parser.hh"
#include "TypeInference.hh"
#include "Vm.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


cchar* typevar_suffix = "_tvar";    // -- types ending in this represent type-variables in the list of symbols
cchar* cost_tag = "cost_";          // -- symbols can be wrapped with 'cost_( <cost>, <symbol> )' to assign a non-unit cost
cchar* synth_subst_tag = "synth_subst";


Spec readSpec(String spec_file, bool spec_file_is_text, bool just_syms)
{
    static Atom a_cost(cost_tag);
    static Atom a_synth_subst(synth_subst_tag);

    // Parse specification file:
    Expr prog;
    try{
        String text;
        if (spec_file_is_text)
            spec_file.moveTo(text);
        else{
            // Allow ',' separated list of files:
            Vec<Str> files;
            splitArray(spec_file.slice(), ",", files);
            for (Str file : files)
                wrLn(text, "##include \"%_\";", file);
        }
        Vec<Expr> exprs;
        parseEvo(text.c_str(), exprs);
        prog = Expr::Block(exprs);

        inferTypes(prog);
        RunTime rt;
        rt.tryCompile(prog);
    }catch (Excp_ParseError err){
        wrLn("PARSE ERROR! %_", err.msg);
        exit(1);
    }

    Vec<Expr> new_prog(copy_, prog);

    // Put let/rec defs in hash:
    Map<Atom, Expr> defs;
    for (Expr const& e : prog)
        if ((e.kind == expr_LetDef || e.kind == expr_RecDef) && e[0].kind == expr_Sym)
            defs.set(e[0].name, e);

    // Extract symbol pool:
    Spec spec;
    Expr syms;  // -- the symbol pool is different; we need the actual names (at least until we can reverse lookup addresses)
    Expr res;
    Vec<Vec<Expr>> subst;
    if (defs.peek(Atom("syms"), syms)){
        while (syms[1].kind == expr_Sym){
            if (!defs.peek(syms[1].name, syms)) assert(false); }

        Vec<CExpr> pool_syms;
        for (Expr const& s : tupleSlice(syms[1])){
            if (s.isUnit()) continue;
            if (s.kind == expr_Appl && s[0].kind == expr_Sym && s[0].name == a_cost){
                assert(s[1].kind == expr_Tuple); assert(s[1].size() == 2);
                if (s[1][0].kind != expr_Lit){ wrLn("ERROR! Costs must be float literals."); exit(1); }
                pool_syms.push(CExpr(getFloat(s[1][0]), s[1][1]));
            }else if (s.kind == expr_Appl && s[0].kind == expr_Sym && s[0].name == a_synth_subst){
                assert(s[1].kind == expr_Tuple); assert(s[1].size() == 2);
                Expr const& lhs = s[1][0];
                subst.push();
                subst[LAST].push(lhs);
                for (auto&& rhs : tupleSlice(s[1][1])){
                    if (lhs.type != rhs.type){
                        wrLn("ERROR! Types must match in synthesis substitution specification (symbols '%_' and '%_')", lhs, rhs);
                        exit(1); }
                    subst[LAST].push(rhs);
                }

            }else
                pool_syms.push(CExpr(1.0, s));
        }

        double cost_Appl  = 1;
        double cost_Sel   = 1;
        double cost_Lamb  = 1;
        double cost_Tuple = 1;

        if (defs.peek(Atom("cost_Appl" ), res)) cost_Appl  = getFloat(res[1]);
        if (defs.peek(Atom("cost_Sel"  ), res)) cost_Sel   = getFloat(res[1]);
        if (defs.peek(Atom("cost_Lamb" ), res)) cost_Lamb  = getFloat(res[1]);
        if (defs.peek(Atom("cost_Tuple"), res)) cost_Tuple = getFloat(res[1]);

        spec.pool = Pool(pool_syms, cost_Appl, cost_Sel, cost_Lamb, cost_Tuple);
    }
    if (just_syms) return spec;     // -- EARLY EXIT

    // Extract remaining specification components:
    if (defs.peek(Atom("spec_name" ), res)) spec.name  = getAtom(res[1]);
    if (defs.peek(Atom("spec_descr"), res)) spec.descr = getAtom(res[1]);

    if (defs.peek(Atom("io_pairs"), res)) spec.io_pairs = res;
    else{ wrLn("ERROR! Specification file must contain 'io_pairs'."); exit(1); }

    if (defs.peek(Atom("runner_"    ), res)) spec.runner     = res;
    if (defs.peek(Atom("checker_"   ), res)) spec.checker    = res;
    if (defs.peek(Atom("wrapper_"   ), res)) spec.wrapper    = res;
    if (defs.peek(Atom("test_vec_"  ), res)) spec.test_vec   = res;
    if (defs.peek(Atom("test_hash_" ), res)) spec.test_hash  = res;
    if (defs.peek(Atom("init_state_"), res)) spec.init_state = res;

    // Set defaults:
    if (!spec.name && !spec_file_is_text){
        String tmp = baseName(spec_file);
        stripSuffix(tmp, ".evo");
        spec.name = Atom(tmp);
    }

    if (spec.io_pairs.type.name != a_Vec || spec.io_pairs.type[0].name != a_Tuple || spec.io_pairs.type[0].size() != 2){
        wrLn("ERROR! 'io_pairs' must be of type [(In, Out)]"); exit(1); }

    // Verify types of runner, checker and wrapper (or set defaults if not specified):
    {
        Type in  = spec.io_pairs.type[0][0];
        Type out = spec.io_pairs.type[0][1];

        Type checker_type = parseType(fmt("(%_, %_, %_) -> Bool", in, out, out).c_str());
        if (spec.checker){
            if (checker_type != spec.checker.type){
                wrLn("ERROR! 'checker_' must be of type: %_", checker_type); exit(1); }
        }else{
            Expr lhs = Expr::Sym(Atom("checker_"), {}, Type(checker_type));
            Expr rhs = Expr::Sym(Atom("default_checker_"), {Type(in), Type(out)}, Type(checker_type));
            spec.checker = Expr::LetDef(move(lhs), Type(checker_type), move(rhs));
            new_prog.push(spec.checker);
        }

        cchar* rlim_type_text = "(Int, Int, Int)";
        Type runner_type = parseType(fmt("([(%_, %_)], (%_)->(%_), %_, (%_, %_, %_)->Bool) -> [Int]", in, out, in, out, rlim_type_text, in, out, out).c_str());
            // -- ([(In, Out)], In->Out, RLim, (In, Out, Out)->Bool) -> [Int]
        if (spec.runner){
            if (runner_type != spec.runner.type){
                wrLn("ERROR! 'runner_' must be of type: %_", runner_type); exit(1); }
        }else{
            Expr lhs = Expr::Sym(Atom("runner_"), {}, Type(runner_type));
            Expr rhs = Expr::Sym(Atom("default_runner_"), {Type(in), Type(out)}, Type(runner_type));
            spec.runner = Expr::LetDef(move(lhs), Type(runner_type), move(rhs));
            new_prog.push(spec.runner);
        }

        //rec default_wrapper_<IN,OUT> : (IN->OUT) -> (IN->OUT) = id_<IN->OUT>;
        //fun wrapper_(target :Target) -> (From -> Into);
        if (spec.wrapper){
            Type wrapper_type = parseType(fmt("A -> %_ -> %_", in, out).c_str());
            if (!typeMatch(wrapper_type, spec.wrapper.type)){
                wrLn("ERROR! 'wrapper_' must match type: %_", wrapper_type); exit(1); }
        }else{
            Type wrapper_type = parseType(fmt("(%_ -> %_) -> (%_ -> %_)", in, out, in, out).c_str());
            Expr lhs = Expr::Sym(Atom("wrapper_"), {}, Type(wrapper_type));
            Expr rhs = Expr::Sym(Atom("default_wrapper_"), {Type(in), Type(out)}, Type(wrapper_type));
            spec.wrapper = Expr::LetDef(move(lhs), Type(wrapper_type), move(rhs));
            new_prog.push(spec.wrapper);
        }
    }

    spec.target = spec.wrapper.type[0];

    // Make sure rec-defs are instantiated:
    new_prog.push(spec.io_pairs[0]);
    new_prog.push(spec.runner  [0]);
    new_prog.push(spec.checker [0]);
    new_prog.push(spec.wrapper [0]);
    if (syms) new_prog.push(syms[0]);

    // Compute size of 'io_pairs':
    Expr e_size = Expr::Sym(a_size, {}, Type());
    new_prog.push(Expr::Appl(move(e_size), Expr(spec.io_pairs[0])));

    spec.prog = Expr::Block(new_prog);
  #if 0
    inferTypes(spec.prog);
  #else
    try{
        inferTypes(spec.prog);
    }catch (Excp_ParseError err) {
        wrLn("PARSE ERROR! %_", err.msg);
        //**/{ OutFile out("spec_prog.evo"); out += ppFmt(spec.prog); wrLn("Wrote 'spec_prog.evo'"); }
        exit(1);
    }
  #endif

    RunTime rt;
    addr_t ret;
    try{ ret = rt.run(spec.prog); assert(ret); }
    catch(Excp_ParseError err) { wrLn("PARSE ERROR! %_", err.msg); exit(1); }
    spec.n_io_pairs = rt.data(ret).val;

    // Extract pruning rules, if any:
    Expr prun;  // -- the symbol pool is different; we need the actual names (at least until we can reverse lookup addresses)
    if (defs.peek(Atom("pruning"), prun)){
        while (prun[1].kind == expr_Sym){
            if (!defs.peek(prun[1].name, prun)) assert(false); }
        prun = prun[1];
        if (prun.type != parseType("[Void]")){ wrLn("ERROR! 'pruning' must be of type '[Void]', not '%_'", prun.type); exit(1); }
        if (prun.kind != expr_MkVec){ wrLn("ERROR! 'pruning' must be a vector literal '[:Void ....]'"); exit(1); }
    }
    spec.pruning = Vec<Expr>(copy_, prun);

    return spec;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

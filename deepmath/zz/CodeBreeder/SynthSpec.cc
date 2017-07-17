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


Spec readSpec(String spec_file, bool spec_file_is_text, bool just_syms)
{
    static Atom a_cost(cost_tag);

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
    if (!defs.peek(Atom("syms"), syms)){
        wrLn("ERROR! Specification file must contain symbol pool definition 'syms'"); exit(1); }
    while (syms[1].kind == expr_Sym){
        if (!defs.peek(syms[1].name, syms)){
            wrLn("ERROR! 'syms' refer to non-let symbol. Add a dummy '()' if this is not an error."); exit(1); }
    }

    for (Expr const& s : tupleSlice(syms[1])){
        if (s.kind == expr_Appl && s[0].kind == expr_Sym && s[0].name == a_cost){
            assert(s[1].kind == expr_Tuple); assert(s[1].size() == 2);
            if (s[1][0].kind != expr_Lit){ wrLn("ERROR! Costs must be float literals."); exit(1); }
            spec.pool.syms.push(CExpr(getFloat(s[1][0]), s[1][1]));
        }else
            spec.pool.syms.push(CExpr(1.0, s));

        if (spec.pool.syms[LAST]->isUnit())
            spec.pool.syms.pop();    // -- exclude the empty tuple
    }

    if (just_syms) return spec;     // -- EARLY EXIT

    // Extract remaining specification components:
    Expr res;
    if (defs.peek(Atom("cost_Appl" ), res)) spec.pool.cost_Appl  = getFloat(res[1]);
    if (defs.peek(Atom("cost_Sel"  ), res)) spec.pool.cost_Sel   = getFloat(res[1]);
    if (defs.peek(Atom("cost_Lamb" ), res)) spec.pool.cost_Lamb  = getFloat(res[1]);
    if (defs.peek(Atom("cost_Tuple"), res)) spec.pool.cost_Tuple = getFloat(res[1]);

    if (defs.peek(Atom("spec_name" ), res)) spec.name  = getAtom(res[1]);
    if (defs.peek(Atom("spec_descr"), res)) spec.descr = getAtom(res[1]);

    auto getDef = [&](cchar* name) {
        Expr res;
        if (defs.peek(Atom(name), res))
            return res;
        wrLn("ERROR! Specification file must contain '%_'.", name);
        exit(1);
    };

    spec.io_pairs = getDef("io_pairs"  );
    spec.runner   = getDef("runner_"   );
    spec.checker  = getDef("checker_"  );
    if (defs.peek(Atom("in_wrap_"  ), res)) spec.in_wrap  = res;
    if (defs.peek(Atom("out_wrap_" ), res)) spec.out_wrap = res;

    // Set defaults:
    if (!spec.name && !spec_file_is_text){
        String tmp = baseName(spec_file);
        stripSuffix(tmp, ".evo");
        spec.name = Atom(tmp);
    }

    if (spec.io_pairs.type.name != a_Vec || spec.io_pairs.type[0].name != a_Tuple || spec.io_pairs.type[0].size() != 2){
        wrLn("ERROR! 'io_pairs' must be of type [(In, Out)]"); exit(1); }

    auto identFun = [](Atom name, Type t) {
        Expr x = Expr::Sym(Atom("x"), {}, Type(t));
        Type fun_type = Type(a_Fun, Type(t), Type(t));
        Expr lam = Expr::Lamb(Expr(x), Expr::Block({x}), Type(fun_type));
        return Expr::RecDef(Expr::Sym(name, {}, Type(fun_type)), Type(fun_type), move(lam));
    };

    spec.target = Type(a_Fun, spec.io_pairs.type[0][0], spec.io_pairs.type[0][1]);
    if (!spec.in_wrap){
        spec.in_wrap = identFun("in_wrap_", spec.target[0]);
        new_prog.push(spec.in_wrap);    // -- add definition to code
    }else{
        if (spec.in_wrap.type.name != a_Fun || spec.in_wrap.type[0] != spec.target[0]){
            wrLn("ERROR! 'in_wrap_' has to be of type: %_ -> ?", spec.target[0]); exit(1); }
        spec.target[0] = spec.in_wrap.type[1];
    }

    if (!spec.out_wrap){
        spec.out_wrap = identFun("out_wrap_", spec.target[1]);
        new_prog.push(spec.out_wrap);   // -- add definition to code
    }else{
        if (spec.out_wrap.type.name != a_Fun || spec.out_wrap.type[1] != spec.target[1]){
            wrLn("ERROR! 'out_wrap_' has to be of type: ? -> %_", spec.target[1]); exit(1); }
        spec.target[1] = spec.out_wrap.type[0];
    }

    // Verify types of runner and checker:
    {
        Type in  = spec.io_pairs.type[0][0];
        Type out = spec.io_pairs.type[0][1];

        Type checker_type = parseType(fmt("(%_, %_, %_) -> Bool", in, out, out).c_str());
        if (checker_type != spec.checker.type){
            wrLn("ERROR! 'checker_' must be of type: %_", checker_type); exit(1); }

        cchar* rlim_type_text = "(Int, Int, Int)";
        Type runner_type = parseType(fmt("([(%_, %_)], (%_)->(%_), %_, (%_, %_, %_)->Bool) -> [Int]", in, out, in, out, rlim_type_text, in, out, out).c_str());
            // -- ([(In, Out)], In->Out, RLim, (In, Out, Out)->Bool) -> [Int]
        if (runner_type != spec.runner.type){
            wrLn("ERROR! 'runner_' must be of type: %_", runner_type); exit(1); }
    }

    // Make sure rec-defs are instantiated:
    new_prog.push(spec.io_pairs[0]);
    new_prog.push(spec.runner[0]);
    new_prog.push(spec.checker[0]);
    new_prog.push(spec.in_wrap[0]);
    new_prog.push(spec.out_wrap[0]);
    new_prog.push(syms[0]);

    // Compute size of 'io_pairs':
    Expr e_size = Expr::Sym(a_size, {}, Type());
    new_prog.push(Expr::Appl(move(e_size), Expr(spec.io_pairs[0])));

    spec.prog = Expr::Block(new_prog);
    inferTypes(spec.prog);

    RunTime rt;
    addr_t ret = rt.run(spec.prog); assert(ret);
    spec.n_io_pairs = rt.data(ret).val;

    return spec;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

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
#include "RandomFuncs.hh"

#include "zz/Generics/IdHeap.hh"
#include "zz/Generics/Set.hh"

#include "SynthEnum.hh"
#include "SynthPrune.hh"
#include "DeriveGenealogy.hh"
#include "HeapSynth.hh"
#include "Vm.hh"
#include "Parser.hh"

namespace ZZ {
using namespace std;
using namespace ENUM;


/*
For now, we are considering the following types:
  Inputs: Int List<Int> (Int, Int) (Int, List<Int>) (List<Int>, List<Int>)
    - note: in progio, '(List<Int>, Int)' is also present; normalize?

  Outputs: Bool Int List<Int>
*/


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


String funName(int64 hash_val, Type const& type)
{
    hash_val = shuffleHash(defaultHash(tuple(hash_val, defaultHash(type))));

    char valid[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    constexpr uint base = (sizeof(valid) - 1);
    char out[14] = "f_xxxxxxxxxxx";
    uint64 h = hash_val;
    for (uint i = 0; i < 11; i++){
        out[i+2] = valid[h % base];
        h /= base;
    }
    assert(h == 0);
    return String(out);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


class RandomFuncs : public HeapSynth {
    Spec const&           spec;
    Params_RandFun const& P;

    Pruner      pruner;
    Set<uint64> seen;

    Vec<uint>     must_haves;       // -- pool IDs (positive index)
    Vec<uint>     cant_haves;       // -- pool IDs (positive index)
    Vec<state_id> found;

    // Internal helpers:
    uint lookupPoolSym(String const& sym_text);

    // Statistics:
    uint64 n_enqueues = 0;
    uint64 n_prunes   = 0;
    uint64 n_runs     = 0;
    uint64 n_unqiue   = 0;

public:
    // <<== should take Params_HeapSynth (or subset of them)
    RandomFuncs(Spec const& spec, Params_RandFun const& P);

    void expand(state_id s) override;
    void eval  (state_id s) override;
    void reportProgress(bool final_call) override;
};


//=================================================================================================
// TEMPORARY; MOVE THIS:


//=================================================================================================


void RandomFuncs::reportProgress(bool final_call)
{
    wr("\r>> \a/\a*#enq:\a* %_   \a*#prune:\a* %_   \a*#run:\a* %_   \a*#unique:\a* %_   \a*#found:\a* %_   [%t]\a/\f%c",
        n_enqueues, n_prunes, n_runs, n_unqiue, found.size(), cpuTime(), final_call?'\n':'\r');
}


RandomFuncs::RandomFuncs(Spec const& spec, Params_RandFun const& P) :
    HeapSynth(spec.prog, Params_HeapSynth()),
    spec(spec),
    P(P)
{
    if (spec.init_state){
        assert(spec.init_state.kind == expr_LetDef || spec.init_state.kind == expr_RecDef);
        Vec<State> hist = deriveGenealogy(spec.init_state, spec.pool);

        if (P.verbosity >= 1){
            wrLn("INIT STATE GENEALOGY:");
            for (State S : hist)
                wrLn("  %_", S.expr(spec.pool));
            wrLn("\nPRETTY-PRINTED:\t+\t+\n%_\t-\t-\n", ppFmt(hist[LAST].expr(spec.pool)));
        }

        enqueue(hist[LAST], spec.pool);
    }else
        enqueue(initialEnumState(spec.target), spec.pool);

    n_enqueues++;
    pruner.init(spec.pruning, spec.pool, P.verbosity >= 1);
    must_haves = map(P.must_haves, [&](String const& s){ return lookupPoolSym(s); });
    cant_haves = map(P.cant_haves, [&](String const& s){ return lookupPoolSym(s); });
}


uint RandomFuncs::lookupPoolSym(String const& sym_text)
{
    Vec<Expr> exprs;
    try{
        parseEvo(sym_text.c_str(), exprs);
    }catch (Excp_ParseError err){
        wrLn("PARSE ERROR! %_", err.msg);
        exit(1);
    }

    if (exprs.size() != 1){
        wrLn("ERROR! Invalid symbol: %_", sym_text); exit(1); }

    for (uint i = 0; i < spec.pool.size(); i++)
        if (exprs[0].untypedEqualTo(spec.pool.sym(i)))
            return i;

    wrLn("ERROR! No such symbol in symbol pool: %_", exprs[0]);
    exit(1);
}


void RandomFuncs::expand(state_id s)
{
    uint tgt_i;
    bool ok = state[s].getLast(ENUM::g_Obl, tgt_i); assert(ok);

    auto pruningEnqueue = [&](State S){
        if (!pruner.shouldPrune(S)){
            enqueue(S, s);
            n_enqueues++;
        }else
            n_prunes++;

    };

    expandOne(spec.pool, state[s], tgt_i, pruningEnqueue, P.P_enum, nullptr);
}


void RandomFuncs::eval(state_id s)
{
    assert(spec.test_hash.type[1] == Type(a_Int));

    Expr expr = state[s].expr(spec.pool);
    Expr e_run = mxAppl(spec.test_hash[0], expr);

    uint ret_code ___unused = 0;
    rt.push();
    try{
        String result;
        Params_RunTime P_rt;
        P_rt.verbose = false;
        P_rt.lim = P.lim;
        P_rt.out = &result;
        //**/wrLn("`` resource limits: %_ %_ %_", P_rt.lim.cpu, P_rt.lim.mem, P_rt.lim.rec);
        addr_t ret = rt.run(e_run, P_rt);
        n_runs++;
        if (ret == 0){
            ret_code = 1;
            if (P.verbosity >= 2){
                char maybe_nl = (result.size() == 0 || result[0] != '\n') ? '\n' : '\0';
                wrLn("RUNTIME ERROR:\n\t+\t+%_\t-\t-", ppFmt(expr));
                wrLn("OUTPUT:%C\t+\t+%_\t-\t-", maybe_nl, result);
            }
        }else{
            // Hash vector...
            RetVal val(rt, ret, spec.test_hash.type[1]);
            int64 hash_val = val.val();

            bool is_unique = (hash_val != -1 && !seen.add(hash_val));
            n_unqiue += (uint)is_unique;
            if (is_unique && usesToplevelFormals(state[s]) && (!P.print_only_recursive || hasRecursion(state[s]))){
                Vec<uint> musts(copy_, must_haves);
                bool has_cant = !state[s].forAllCond([&](GExpr const& elem){
                    if (elem.kind != g_Pool) return true;
                    uint sym_id = ~elem.ins[0];
                    if (has(cant_haves, sym_id))
                        return false;
                    uind idx = search(musts, sym_id);
                    if (idx != UIND_MAX){
                        musts[idx] = musts.last();
                        musts.pop(); }
                    return true;
                });

                if (!has_cant && musts.size() == 0){
                    // Get output vector:
                    rt.pop();
                    rt.push();
                    Expr e_run = mxAppl(spec.test_vec[0], expr);
                    addr_t ret = rt.run(e_run, P_rt); assert(ret != 0);
                    RetVal vec(rt, ret, spec.test_vec.type[1]);

                    // Present result:
                    wrLn("============================================================[%.2f s]", cpuTime());
                    wrLn("let %_ = %_;", funName(hash_val, spec.target), ppFmt(expr));
                    wrLn("\a*-->  %s\a*", vec);
                    newLn();

                    found.push(s);

                    // Time to terminate?
                    if (found.size() == P.n_funcs_to_generate)
                        Q.clear();
                // <<== extract positive and negative training examples here
                }

            }else if (P.verbosity >= 3){
                if (hash_val == -1)
                    wrLn("PARTIAL:\n%_", ppFmtI(expr, 4));
                else
                    wrLn("REDUNDANT:\n%_", ppFmtI(expr, 4));
            }

        }
    }catch (Excp_ParseError err){
        /**/wrLn("COMPILE ERROR:\n%_", ppFmt(expr));
        /**/wrLn("  - %_", err.msg);
        ret_code = 2;
    }
    rt.pop();
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


void generateRandomFunctions(String spec_filename, Params_RandFun const& P)
{
    Spec spec = readSpec(spec_filename, false); assert(spec.prog.kind == expr_Block);
    assert(spec.target.name == a_Fun);

    if (P.verbosity >= 1){
        wrLn("\a*TARGET:\a* %_", spec.target);
        wrLn("\a*POOL:\a*"); for (uint i = 0; i < spec.pool.size(); i++) wrLn("  $%>3%_:  @%_ = \a*%_\a*  \a/:%_\a/", spec.pool.cost(i), i, spec.pool.sym(i), spec.pool.sym(i).type);
        wrLn("\a*CONS COSTS:\a*");
        wrLn("  $%>3%_:  %_", spec.pool.costAppl (), "Appl");
        wrLn("  $%>3%_:  %_", spec.pool.costSel  (), "Sel");
        wrLn("  $%>3%_:  %_", spec.pool.costLamb (), "Lamb");
        wrLn("  $%>3%_:  %_", spec.pool.costTuple(), "Tuple");
    }

    RandomFuncs rf(spec, P);
    rf.run();
}


//
//    Vec<Expr> new_prog(copy_, prog);
//    spec.prog = Expr::Block(new_prog);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}


/*
Christian's functions:

to_range_inclusive_
sum_
prod_

fun list_exists_<A>(list :List<A>, pred_ :A->Bool) -> Bool {
    not_(list_for_all_<A>(list, compose_(not_, pred_))); };


data DivisionByZero = {()};

fun safe_div_(n :Int, m :Int) -> Int {
   if_(eq_(m, 0), `throw_<(), DivisionByZero>(DivisionByZero.0()));
   div_(n, m);
};

fun safe_mod_(n :Int, m :Int) -> Int {
   if_(eq_(m, 0), `throw_<(), DivisionByZero>(DivisionByZero.0()));
   cond_<Int>(negative_(n), `mod_(neg_(n),m), `mod_(n,m));
};
*/

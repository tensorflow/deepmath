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
#include "HeapSynth.hh"


namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


HeapSynth::HeapSynth(Expr const& prog, Params_HeapSynth const& P) :
    P(P),
    Q(cost),
    reportC(P.report_freq)
{
    try{
        rt.run(prog, P.P_rt);
    }catch (Excp_ParseError err){
        wrLn("PARSE ERROR! %_", err.msg);
        exit(1);
    }
}


void HeapSynth::enqueue(State const& S, Pool p, uind from)
{
    state_id id = state.size();
    state.push(S);
  #if 1
    cost.push(S.cost());
  #else
    // randomize a little
    /**/static uint64 seed = 42;
    double prio0 = (from == NO_PARENT) ? 0 : cost[from];
    double cost0 = (from == NO_PARENT) ? 0 : state[from].cost();
    double delta = S.cost() - cost0;
    cost.push(prio0 + delta * drand(seed));
  #endif
    pool.push(p);
    parent.push(from);
    Q.add(id);
}


void HeapSynth::getParents(state_id s, Vec<state_id>& out_parents)
{
    out_parents.clear();
    while (s != NO_PARENT){
        out_parents.push(s);
        s = parent[s]; }
    reverse(out_parents);
}


void HeapSynth::run()
{
    start();
    for(;;){
        if (Q.size() == 0) flush();
        if (Q.size() == 0) break;

        if (reportC != 0){
            reportC--;
            if (reportC == 0){
                reportProgress(false);
                reportC = P.report_freq;
            }
        }

        state_id s = Q.pop();
        uint dummy;
        if (state[s].getLast(ENUM::g_Obl, dummy))
            expand(s);
        else
            eval(s);
    }
    if (reportC != 0)
        reportProgress(true);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Default implementation:


// Returns 0 if successful, 1 on Evo runtime error, 2 on Evo compile time error.
static uint evalExpr(Expr const& expr, RunTime& rt, Spec const& spec, ResLims const& rlim,
         /*outputs:*/Vec<uint>& run_results, double& eval_time)
{
    // Type: fun run_all_<IN,OUT>(io_pairs :[(IN, OUT)], f_ :IN->OUT, rlim :RLim, checker_ :(IN, OUT, OUT)->Bool) -> [Int]
    Expr e_rlim = mxTuple({mxLit_Int(rlim.cpu), mxLit_Int(rlim.mem), mxLit_Int(rlim.rec)});
    Expr e_fun  = mxAppl(spec.wrapper[0], expr);
    Expr e_arg  = mxTuple({spec.io_pairs[0], e_fun, e_rlim, spec.checker[0]});
    Expr e_run  = mxAppl(spec.runner[0], e_arg);

    run_results.reset(res_SIZE+1, 0);
    uint ret_code = 0;

    rt.push();

    double T0_eval = cpuTime();
    try{
        Params_RunTime P_rt;
        P_rt.verbose = false;

        addr_t ret = rt.run(e_run, P_rt);
        if (ret == 0){
            ret_code = 1;
            //**/wrLn("RUNTIME ERROR:\n%_", ppFmt(expr));
            /**/wrLn("RUNTIME ERROR:\n%_", ppFmt(e_run));
        }else{
            // Extract result vector and summarize it:
            addr_t vec_head = rt.data(ret).val;
            addr_t vec_data = rt.data(vec_head).val;
            addr_t vec_size = rt.data(vec_head+1).val;
            Array<VM::Word const> results = slice(rt.data(vec_data), rt.data(vec_data + vec_size));

            for (VM::Word w : results){
                assert((uint64)w.val < res_SIZE);
                run_results[w.val]++;
                run_results[LAST]++;
            }
        }
    }catch (Excp_ParseError err){
        /**/wrLn("COMPILE ERROR:\n%_", ppFmt(expr));
        /**/wrLn("  - %_", err.msg);
        ret_code = 2;
    }
    double T1 = cpuTime();
    eval_time += T1 - T0_eval;

    rt.pop();
    return ret_code;
}


void SimpleSynth::expand(state_id s)
{
    /**/wrLn("expand %_  %_", s, state[s].expr(spec.pool));
    uint tgt_i;
    bool ok = state[s].getLast(ENUM::g_Obl, tgt_i); assert(ok);
    expandOne(spec.pool, state[s], tgt_i, [&](State S){ enqueue(S, s); }, P.P_enum, nullptr);
}


void SimpleSynth::eval(state_id s)
{
    Expr expr = state[s].expr(spec.pool);
    Vec<uint> result_counts;
    double cpu_time = 0;
    uint ret ___unused = evalExpr(expr, rt, spec, P.rlim, result_counts, cpu_time);

    /**/wrLn("eval %_: %_  %_", s, result_counts[res_RIGHT], expr);
    if (result_counts[res_RIGHT] == spec.n_io_pairs){
        wrLn("Found solution!");
        wrLn("PRETTY-PRINTED: [%_]\t+\t+\n%_\t-\t-\n", spec.name, ppFmt(expr));
        Q.clear();  // -- aborts search
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

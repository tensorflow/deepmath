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
#include "Synth.hh"

#include <memory>

#include "zz/Generics/IdHeap.hh"
#include "zz/Generics/PArr.hh"
#include "zz/Generics/IntSet.hh"
#include "zz/Generics/IntMap.hh"
#include "zz/Generics/Sort.hh"

#include "SynthSpec.hh"
#include "SynthEnum.hh"
#include "SynthPrune.hh"
#include "SynthHelpers.hh"
#include "Parser.hh"
#include "TypeInference.hh"
#include "Vm.hh"
#include "TrainingData.hh"


namespace ZZ {
using namespace std;
using namespace ENUM;


ZZ_PTimer_Add(synth_enum);
ZZ_PTimer_Add(synth_enum_heap_ops);
ZZ_PTimer_Add(synth_python_cost_fun);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Parameters:


ZZ_PTimer_Add(consistency_check);


void addParams_Synth(CLI& cli)
{
    Params_Synth P;
    cli.add("verb", "uint", fmt("%_", P.verbosity), "Verbosity: 0=final solution, 1=each improvement, 2=equal to best-so-far, 3=everything but trivial, >=4 everything");
    cli.add("dump-states", "bool", P.dump_states?"yes":"no", "For debugging, show dequeued states during search.");
    cli.add("dump-exprs", "bool", P.dump_exprs?"yes":"no", "For debugging, show dequeued states as partial expressions.");
    cli.add("thru-text", "bool", P.thru_text?"yes":"no", "For debugging, execute programs by printing them and then parsing them back.");
    cli.add("max-tries", "int", fmt("%_", (int64)P.max_tries), "Abort search after this many tried expressions. '-1' = never stop.");
    cli.add("max-queue", "int", fmt("%_", (int64)P.max_queue), "Abort serach after enqueuing this many partial expressions (essentially a memory limit). '-1' = no limit.");
    cli.add("max-cpu", "float", fmt("%_", P.max_cpu), "Abort search after this many seconds.");
    cli.add("max-cost", "float|{inf}", fmt("%_", P.max_cost), "Discard solutions above this threshold.");
    cli.add("try-cpu-lim", "uint", fmt("%_", P.try_cpu_lim), "Abort expression execution after this many steps.");
    cli.add("try-mem-lim", "uint", fmt("%_", P.try_mem_lim), "Abort expression execution if using more than this many 64-bit words of memory.");
    cli.add("try-rec-lim", "uint", fmt("%_", P.try_rec_lim), "Abort expression execution if recursion reaches this depth.");
    cli.add("big", "bool", "no", "Increase default try limits by 10x");
    cli.add("huge", "bool", "no", "Increase default try limits by 100x");
    cli.add("gen-training", "string", P.gen_training, "Write supervised training data to this file.");
    cli.add("use-formals", "bool", P.enum_.must_use_formals?"yes":"no", "If 'yes', all formal parameters to a function must be used.");
    cli.add("force-rec", "bool", P.enum_.force_recursion?"yes":"no", "If 'yes', synthesized function must contain a recursive call.");
    cli.add("ban-rec", "bool", P.enum_.ban_recursion?"yes":"no", "If 'yes', synthesized function must NOT contain a recursive call.");
    cli.add("inc-eval", "bool", P.inc_eval?"yes":"no", "Incremental evaluation (may turn off for debugging purposes).");
    cli.add("keep-going", "bool", P.keep_going?"yes":"no", "If 'yes', don't stop after first solution is found.");
    cli.add("reb", "ufloat", fmt("%_", P.base_rebate), "Use a value in ]0,1[ to prioritize search-space below (heuristic) basecase detection.");
    cli.add("soft-reb", "bool", P.soft_rebate?"yes":"no", "Apply rebate only to constructs added after discovery of potential basecase.");
    cli.add("batch-size", "uint", fmt("%_", P.batch_size), "Send states in batches of this size to Python callback.");
    cli.add("rnd-costs", "ufloat", fmt("%_", P.randomize_costs), "Add a random cost to each expanded state.");
    cli.add("enum-mode", "bool", P.enumeration_mode?"yes":"no", "If set, all complete programs are considered good.");
}


void setParams_Synth(const CLI& cli, Params_Synth& P)
{
    P.verbosity        = cli.get("verb").int_val;
    P.dump_states      = cli.get("dump-states").bool_val;
    P.dump_exprs       = cli.get("dump-exprs").bool_val;
    P.thru_text        = cli.get("thru-text").bool_val;
    P.max_tries        = cli.get("max-tries").int_val;
    P.max_queue        = cli.get("max-queue").int_val;
    P.max_cpu          = cli.get("max-cpu").float_val;
    P.max_cost         = (cli.get("max-cost").choice == 1) ? DBL_INF : cli.get("max-cost").float_val;
    P.try_cpu_lim      = cli.get("try-cpu-lim").int_val;
    P.try_mem_lim      = cli.get("try-mem-lim").int_val;
    P.try_rec_lim      = cli.get("try-rec-lim").int_val;
    P.gen_training     = cli.get("gen-training").string_val;
    P.inc_eval         = cli.get("inc-eval").bool_val;
    P.keep_going       = cli.get("keep-going").bool_val;
    P.base_rebate      = cli.get("reb").float_val;
    P.soft_rebate      = cli.get("soft-reb").bool_val;
    P.batch_size       = cli.get("batch-size").int_val;
    P.randomize_costs  = cli.get("rnd-costs").float_val;
    P.enumeration_mode = cli.get("enum-mode").bool_val;
    if (cli.get("big" ).bool_val){ P.try_cpu_lim *=  10; P.try_mem_lim *=  10; P.try_rec_lim *=  10; }
    if (cli.get("huge").bool_val){ P.try_cpu_lim *= 100; P.try_mem_lim *= 100; P.try_rec_lim *= 100; }
    P.enum_.must_use_formals = cli.get("use-formals").bool_val;
    P.enum_.force_recursion  = cli.get("force-rec").bool_val;
    P.enum_.ban_recursion    = cli.get("ban-rec").bool_val;

    if (P.enumeration_mode) P.keep_going = true;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Enumeration based synthesis:


class Synth {
    // Input:
    Spec         const& spec;
    Params_Synth const& P;
    CostFun             cost_fun;
    SolutionFun         sol_fun;

    // Compilation:
    Inferrer       infr;
    RunTime        rt;
    Params_RunTime P_rt;

    // Expression enumeration state:
    Vec<State>                  states;
    Vec<Pair<double,uint64>>    state_costs;
    Vec<uind>                   parent;
    Vec<float>                  rebate_mul;
    Vec<float>                  rebate_add;
    Vec<uint>                   part_score;
    IdHeap<Pair<double,uint64>> Q;
    Pruner                      pruner;

  #if defined(GOOGLE_CODE)
    // Batching:
    Vec<uint>  batch_parents;
    Vec<State> batch_states;
    std::vector<::CodeBreeder::StateProto> batch_protos;    // -- this vector is sent through CLIF to Pyhton.

    CodeBreeder::PoolProto pool_proto;
  #endif

    // Counters:
    uint64 n_attempts = 0;
    uint64 n_enqueues = 0;
    uint64 n_dequeues = 0;
    uint64 n_runs     = 0;
    uint64 n_halts    = 0;

    double T0 = 0;
    double eval_time = 0;
    double exec_time = 0;
    double latest_cost = 0;

    uint64 seed = 0;

    // Internal Methods:
    String evalExpr(Expr const& expr, String score_fun = "score_", String print_fun = "print_int_", Type score_type = Type(a_Int));
    bool   failsRules(State const& S);
    bool   hasUnreach(State const& S);
    void   enqueue(State S, uind from);
    void   reportProgress(bool keep_output);

  #if defined(GOOGLE_CODE)
    void   processBatch();
  #endif

public:
    Synth(Spec const& spec, Params_Synth const& P, CostFun cost_fun, SolutionFun sol_fun)
        : spec(spec), P(P), cost_fun(cost_fun), sol_fun(sol_fun), Q(state_costs)
    {
        if (P.use_prune_rules) pruner.init(spec.pruning, spec.pool);
    }

    int64 run();
};


// Returns 0 if successful, 1 on Evo runtime error, 2 on Evo compile time error.
static uint evalExpr_(Expr const& expr, RunTime& rt, Spec const& spec, ResLims const& rlim, /*out*/Vec<uint>& run_results, double& eval_time) ___unused;
static uint evalExpr_(Expr const& expr, RunTime& rt, Spec const& spec, ResLims const& rlim, /*out*/Vec<uint>& run_results, double& eval_time)
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
            /**/wrLn("RUNTIME ERROR:\n%_", ppFmt(e_run));       // <<== can get here if evaluating 'expr' runs out of default resources for the 'rt' object (this evaluation is not wrapped in a 'run()')
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
        //**/Dump(err.msg);
        ret_code = 2;
    }
    double T1 = cpuTime();
    eval_time += T1 - T0_eval;

    rt.pop();
    return ret_code;
}


// Preserves the old text-based interface. New 'HeapSynth.cc/hh' depricates this.
String Synth::evalExpr(Expr const& expr, String score_fun, String print_fun, Type score_type)
{
    // <<== running out of resources should perhaps be considered an error, at least for base cases
    Vec<uint> results;
    uint ret = evalExpr_(expr, rt, spec, ResLims(P.try_cpu_lim, P.try_mem_lim, P.try_rec_lim), results, eval_time);
    if (ret == 1) return "HALTED!";
    assert(ret != 2);       // -- compile error
    assert(spec.n_io_pairs == results[LAST]);
    int64 val = (results[LAST] == results[res_RIGHT]) ? INT64_MAX :
                                                        (int64)results[res_RIGHT] - 10000 * (int64)results[res_WRONG];
    return fmt("%_", val);
}


#if defined(GOOGLE_CODE)
void Synth::processBatch()
{
    std::vector<double> costs;
    { ZZ_PTimer_Scope(synth_python_cost_fun);
      costs = cost_fun(pool_proto, batch_protos); }
    assert(costs.size() == batch_protos.size());
    for (size_t i = 0; i < costs.size(); i++){
        uint from = batch_parents[i];
        double cost = costs[i] * rebate_mul[from] + rebate_add[from];
        if (cost <= P.max_cost){
            uind n = states.size();
            states.push(batch_states[i]);
            state_costs.push(tuple(cost, n_attempts + i - costs.size()));
            parent.push(from);
            double rm = rebate_mul[from]; rebate_mul.push(rm);
            double ra = rebate_add[from]; rebate_add.push(ra);
            { ZZ_PTimer_Scope(synth_enum_heap_ops); Q.add(n); }
        }
    }
    batch_parents.clear();
    batch_states.clear();
    batch_protos.clear();
}
#endif


// EXPERIMENTAL
bool Synth::hasUnreach(State const& S)
{
    // Check if state is testable:
    bool ok_to_test = true;
    uint cc = 0;
    S.forAllCond([&](GExpr const& g){
        if (g.kind == g_Obl && g.ins.psize() == 0){
            if (g.type().name == a_Fun)
                cc++;
            else{
                ok_to_test = false;
                return false; }
        }
        if (g.kind == g_Obl && g.ins.psize() == 1){
            ok_to_test = false;
            return false; }
        return true;
    });
    //**/if (!ok_to_test){ wrLn("CANNOT TEST:\n\t+\t+%_\t-\t-", ppFmt(S.expr(spec.pool))); }
    if (!ok_to_test || cc == 0)
        return false;

    // Run partial expression:
    // <<== note, we are testing redundantly here; we should mark reachable Obl and only test if there are unreachable ones (and maybe even only instrument those)
    uint n_cov_points = 0;
    Expr expr = S.expr(spec.pool, false, &n_cov_points);
    assert(cc == n_cov_points);

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Expr e_run_for_cover = Expr::Sym(Atom("run_for_cover_"), {spec.io_pairs.type[0][0], spec.io_pairs.type[0][1]}, Type(spec.runner[0].type));
    Expr e_rlim = mxTuple({mxLit_Int(P.try_cpu_lim), mxLit_Int(P.try_mem_lim), mxLit_Int(P.try_rec_lim)});
    Expr e_fun  = mxAppl(spec.wrapper[0], expr);
    Expr e_arg  = mxTuple({spec.io_pairs[0], e_fun, e_rlim, spec.checker[0]});
    Expr e_run  = mxAppl(e_run_for_cover, e_arg);

    //**/wrLn("Running:\n\t+\t+%_\t-\t-", ppFmt(e_run));
    rt.push();
    //double T0 = cpuTime();
    uint64 result = 0;
    try{
        Params_RunTime P_rt;
        P_rt.verbose = false;

        addr_t ret = rt.run(e_run, P_rt);
        if (ret == 0){
            wrLn("INTERNAL ERROR! Unreachability test failed at run-time.");
            exit(1);
        }else
            result = rt.data(ret).val;
    }catch (Excp_ParseError err){
        wrLn("INTERNAL ERROR! Unreachability test failed at compile-time.");
        exit(1);
    }
    //double T1 = cpuTime();
    //**/wrLn("  coverage test time: %t", T1 - T0);      // <<== collect statistics

    rt.pop();
    uint64 mask = (n_cov_points < 64) ? (1ull << n_cov_points) - 1u : ~uint64(0);
    //**/wrLn("  has unreach: %_   (coverage: %b)", result != mask, result);     // <<== collect statistics
    return result != mask;
}

//=================================================================================================


void Synth::enqueue(State S, uind from)
{
    /**/static bool test_unreach = getenv("UNREACH");
    /**/if (test_unreach && hasUnreach(S)) return;

  #if defined(GOOGLE_CODE)
     if (cost_fun && P.randomize_costs == 0.0){
        assert(!P.use_prune_rules || pruner.isEmpty());     // -- not implemented yet
        ::CodeBreeder::StateProto proto;
        S.toProto(states.size() + batch_states.size(), &proto);

        batch_protos .push_back(proto);
        batch_parents.push(from);
        batch_states .push(S);
        n_enqueues++;

        if (batch_protos.size() >= P.batch_size)
            processBatch();
    }else
  #endif
    {
        if (P.use_prune_rules && pruner.shouldPrune(S))
            return;

        double cost = S.cost() * rebate_mul[from] + rebate_add[from];
        if (cost <= P.max_cost){
            uind n = states.size();
            states.push(S);
            state_costs.push(tuple(cost, n_attempts));
            parent.push(from);
            double rm = rebate_mul[from]; rebate_mul.push(rm);     // -- push takes reference, which will be invalidated on a resize if using 'rebate[from]' directly instead of temporary 'r'.
            double ra = rebate_add[from]; rebate_add.push(ra);
            uint   ps = part_score[from]; part_score.push(ps);
            { ZZ_PTimer_Scope(synth_enum_heap_ops); Q.add(n); }
            n_enqueues++;
        }
    }
}


void Synth::reportProgress(bool keep)
{
    if (keep && P.verbosity == 0){ wr("\r"); return; }
    wr("\r>> \a/\a*cost:\a* $%.2f   \a*#enqueues:\a* %_ (%_)   \a*queue-size:\a* %_   \a*#runs:\a* %_    \a*#halts:\a* %_   \a*mem:\a* %DB   \a*eval-time:\a* %.0f%% (%.0f%%)   [%t]\a/\f%c",
        latest_cost, n_enqueues, n_attempts, Q.size(), n_runs, n_halts, memUsedNow(),
        100.0 * eval_time/(cpuTime() - T0), 100.0 * exec_time/(cpuTime() - T0),
        cpuTime() - T0,
        keep?'\n':'\r');
}


//=================================================================================================
// -- main loop:


int64 Synth::run()
{
    T0 = cpuTime();

    Pool const& pool = spec.pool;
  #if defined(GOOGLE_CODE)
    if (cost_fun)
        pool.toProto(&pool_proto);
  #endif

    // Incremental runtime:
    P_rt.lim.cpu = P.try_cpu_lim;
    P_rt.lim.mem = P.try_mem_lim;
    P_rt.lim.rec = P.try_rec_lim;
    P_rt.verbose = false;
    try{
        rt.run(spec.prog, P_rt);    // <<== should force 'score_' to be instantiated (just add 'score_;' as a statement)
    }catch (Excp_ParseError err){
        wrLn("PARSE ERROR! %_", err.msg);
        exit(1);
    }

    // Write task information to screen:
    if (P.verbosity > 0){
        if (spec.name) wrLn("\a*SPEC NAME:\a* %_", spec.name);
        if (spec.descr) wrLn("\a*SPEC DESCR:\a* %_", spec.descr);
        wrLn("\a*TARGET:\a* %_", spec.target);
        wrLn("\a*POOL:\a*"); for (uint i = 0; i < pool.size(); i++) wrLn("  $%>3%_:  @%_ = \a*%_\a*  \a/:%_\a/", pool.cost(i), i, pool.sym(i), pool.sym(i).type);
        wrLn("\a*CONS COSTS:\a*");
        wrLn("  $%>3%_:  %_", pool.costAppl (), "Appl");
        wrLn("  $%>3%_:  %_", pool.costSel  (), "Sel");
        wrLn("  $%>3%_:  %_", pool.costLamb (), "Lamb");
        wrLn("  $%>3%_:  %_", pool.costTuple(), "Tuple");
        if (spec.io_pairs) wrLn("\a*IO PAIRS:\a*\t+\t+\n%_\t-\t-", spec.io_pairs);      // <<== for now (just to see some representation of the problem)
        newLn();
    }

    // Expression enumeration state:
    states.push(initialEnumState(spec.target));
    state_costs.push(tuple(0.0, 0ull));
    parent.push(UIND_MAX);
    rebate_mul.push(1.0);
    rebate_add.push(0.0);
    part_score.push(0);
    Q.add(0);

    bool   trivial_results = (P.verbosity >= 4);
    int64  threshold = (P.verbosity >= 3) ? INT64_MIN/2 :
                       (P.verbosity == 2) ? 0 :
                       (P.verbosity == 1) ? 1 :
                       /*otherwise*/        INT64_MAX/2;
    int64  best = INT64_MIN/4;
    uint64 n_last_report = 0;

    if (P.verbosity > 0) wrLn("Expression enumeration starting...");
    ZZ_PTimer_Scope(synth_enum);
    for(;;){
      #if defined(GOOGLE_CODE)
        if (Q.size() == 0 && batch_protos.size() > 0)
            processBatch();
      #endif
        if (Q.size() == 0) break;   // -- loop condition

        uind s; { ZZ_PTimer_Scope(synth_enum_heap_ops); s = Q.pop(); }
        latest_cost = state_costs[s].fst;
        State S = states[s];
        n_dequeues++;

        if (n_attempts >= n_last_report + 250){
            reportProgress(false);
            n_last_report = n_attempts; }

        uint tgt_i;
        bool hilite = (P.dump_states || P.dump_exprs) && !S.getLast(g_Obl, tgt_i);

        if (hilite) wr("EVAL: ");
        if (P.dump_states)
            wrLn("(queue size: %_) dequeued: %_", Q.size(), S);
        if (P.dump_exprs)
            wrLn("(queue size: %_) dequeued: %_", Q.size(), S.expr(pool));

        if (S.getLast(g_Obl, tgt_i)){
            // Hardcode base heuristic for now:
            //if (P.base_rebate < 1.0 && rebate[s] == 1.0 && S[tgt_i].ins.psize() == 0){
            //if (S.size() > 1 && P.base_rebate < 1.0 && rebate_mul[s] >= P.base_rebate && S[tgt_i].ins.psize() == 0){        // <<== real heuristic should count if more io_pairs were solved!
            if (S.size() > 1 && P.base_rebate < 1.0                                   && S[tgt_i].ins.psize() == 0){        // <<== real heuristic should count if more io_pairs were solved!
                bool single_obl = true;
                S.enumAll([&](uind i, GExpr const& g) {
                    if (i != tgt_i && g.kind == g_Obl){
                        single_obl = false;
                        return false;
                    }else
                        return true;
                });

//                if (single_obl)
                {
                    Expr prog = S.expr(pool, true);
                    String result = evalExpr(prog);
                    if (strstr(result.c_str(), "HALTED!") == nullptr){  // -- if steps limit is reached while outputting result, "HALTED!" may not appear first
                        int64 val = 0;
                        try{ val = stringToInt64(result); }
                        catch (...){ /**/wrLn("`` strange result: %_", result); }

                        if (val > 0 && val > part_score[parent[s]]){
                        //if (rebate_mul[s] == 1.0 && val > 0){
                        //if ((rebate_mul[s] == 1.0 && val > 0) || val > 2){
                            if (P.verbosity >= 1) wrLn("\a*\a_Basecase?\a*\a_ (%_)  %_  [score: %_, cost: $%_]", rebate_mul[s], prog, val, latest_cost);

                            if (P.soft_rebate)
                                rebate_add[s] = latest_cost * rebate_mul[s] * (1 - P.base_rebate);    // -- experimental: will widen search
                            rebate_mul[s] *= P.base_rebate;
                            part_score[s] = val;
                        }
                    }
                }
            }   // -- base heuristic ends here

            //**/wrLn("  expanding: $%_  %_", states[s].cost(), states[s].expr(pool));
            expandOne(pool, S, tgt_i, [&](State S){ enqueue(S, s); }, P.enum_, &n_attempts);

        }else{
            // Run expression:
            Expr e = S.expr(pool);
            String result = evalExpr(e);
            n_runs++;

            if (strstr(result.c_str(), "HALTED!") == nullptr){  // -- if steps limit is reached while outputting result, "HALTED!" may not appear first
                int64 val = INT64_MAX;
                if (!P.enumeration_mode){
                    try{ val = stringToInt64(result); }
                    catch (...){ wrLn("Program must output a single integer, not:\n\t+\t+\t+\t+%_\n\t-\t-\t-\t-", result); exit(1); }
                }

                if (val == INT64_MAX){
                    if (P.verbosity > 0){
                        wrLn("\a*SOLVED!:\a* %_", e);
                        reportProgress(true);
                        if (!P.keep_going) wrLn("Found smallest solution!");
                        reportGenealogy(s, states, state_costs, parent, pool);
                        //**/{ String name = fmt("state%_.dot", n_attempts); exportDot(pool, S, name); wrLn("Wrote: \a*%_\a*", name); }
                    }
                    wrLn("PRETTY-PRINTED: [%_]\t+\t+\n%_\t-\t-\n", spec.name, ppFmt(e));
                    if (P.gen_training != "")
                        outputTrainingData(P.gen_training, s, states, parent, pool);
                    if (sol_fun){
                        ::CodeBreeder::TrainingProto tr_proto;
                        genTrainingData(s, states, parent, pool).toProto(&tr_proto);
                        tr_proto.set_evo_output(result.c_str());
                        sol_fun(tr_proto);
                    }
                    if (!P.keep_going)
                        return INT64_MAX;     // -- ABORT SEARCH!

                }else if (best != INT64_MAX && (result != "0" || trivial_results) && val >= best + threshold){
                    wrLn("\a/\a*score:\a* %_ [#%_, %t, $%_]:\a/ %_", result, n_attempts, cpuTime() - T0, latest_cost, e);
                }

                newMax(best, val);
            }else{
                // <<== collect statistics on reason for halt (cpu/mem/rec-depth/compile-err)
              #if 0
                /**/Dump(S.expr(pool));
                /**/Dump(result);
              #endif
                n_halts++;
                if (trivial_results) wrLn("HALTED: %_", e);
            }

            if (n_runs >= P.max_tries){
                reportProgress(true);
                wrLn("--reached limit on number of tries--");
                return best;     // -- ABORT SEARCH!
            }
        }

        if (P.max_cpu >= 0 && cpuTime() - T0 >= P.max_cpu){
            reportProgress(true);
            if (P.verbosity > 0) wrLn("--reached time limit--");
            return best;     // -- ABORT SEARCH!
        }

        if (P.max_queue != UINT_MAX && n_attempts >= P.max_queue) {
            reportProgress(true);
            if (P.verbosity > 0) wrLn("--reached max enqueue limit--");
            return best;     // -- ABORT SEARCH!
        }
    }

    reportProgress(true);
    if (P.verbosity > 0) wrLn("--exhaustive enumeration--");
    return best;
}


int64 synthesizeProgram(String spec_file, Params_Synth P, bool spec_file_is_text, CostFun cost_fun, SolutionFun sol_fun)
{
    if (P.verbosity > 0){
        wrLn("\a*==-----------------------------------------------------------------------------\a0");
        wrLn("\a*==\a* \a/%<76%_\a0", " CodeBreeder synthesis");
        wrLn("\a*==-----------------------------------------------------------------------------\a0");
        newLn();
    }

    Spec spec = readSpec(spec_file, spec_file_is_text); assert(spec.prog.kind == expr_Block);
    if (!spec.target){ wrLn("ERROR! Could not find 'score_' function."); exit(1); }
    if (!spec.pool){ wrLn("ERROR! Specification file must contain symbol pool definition 'syms'."); exit(1); }

    Synth synth(spec, P, cost_fun, sol_fun);
    return synth.run();
}




int64 pySynthesizeProgram(String prog_text, String params, CostFun cost_fun, SolutionFun sol_fun)
{
    CLI cli;
    addParams_Synth(cli);
    cli.parseCmdLine(params, "pySynthesizeProgram()");

    Params_Synth P;
    setParams_Synth(cli, P);

    return synthesizeProgram(prog_text, P, true, cost_fun, sol_fun);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}


/*
- type-closure computation

- bättre pretty-printing; håll reda på typad kontext
- extern/plug-in filtrering av uttryck?
    idempotens:      max_(x, max_(x, y)) = max(x, y)
    kommutativitet:  x+y = y+x
    associativitet:  (x+y)+z = x+(y+z)
    identitetselement: x+0 = x; x*1 = x

- optimering: om case splittar och rekursivt call sätter samman på exakt samma sätt, använd tupeln.

- om symboler inte är fullt typade från en target typ, PIs i scope är möjliga typer för överiga variabler
- för 'case' måste 'A' vara av sumtyp (specialfall)
- inte värt att instantiera mer än en variabel? eller gör instantieringar innan expr. enum? (förmodligen en god ide)
*/

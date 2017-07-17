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

#if defined(GOOGLE_CODE)    // -- for protobuf support:
  // Protobuf includes have to go before 'Prelude.hh'
  #include /*no-mangling*/ "third_party/deepmath/zz/CodeBreeder/synth.proto.h"
#else
  namespace CodeBreeder {
  struct NodeProto;
  struct StateProto;
  struct TypeProto;
  struct SymbolProto;
  struct PoolProto;
  struct TrainingProto;
  }
#endif

#include ZZ_Prelude_hh
#include "Synth.hh"

#include <memory>


#include "zz/Generics/IdHeap.hh"
#include "zz/Generics/PArr.hh"
#include "zz/Generics/IntSet.hh"
#include "zz/Generics/IntMap.hh"

#include "SynthSpec.hh"
#include "SynthEnum.hh"
#include "Parser.hh"
#include "TypeInference.hh"
#include "Vm.hh"


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
    cli.add("base-rebate", "ufloat", fmt("%_", P.base_rebate), "Use a value in ]0,1[ to prioritize search-space below (heuristic) basecase detection.");
    cli.add("batch-size", "uint", fmt("%_", P.batch_size), "Send states in batches of this size to Python callback.");
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
    P.base_rebate      = cli.get("base-rebate").float_val;
    P.batch_size       = cli.get("batch-size").int_val;
    if (cli.get("big" ).bool_val){ P.try_cpu_lim *=  10; P.try_mem_lim *=  10; P.try_rec_lim *=  10; }
    if (cli.get("huge").bool_val){ P.try_cpu_lim *= 100; P.try_mem_lim *= 100; P.try_rec_lim *= 100; }
    P.enum_.must_use_formals = cli.get("use-formals").bool_val;
    P.enum_.force_recursion  = cli.get("force-rec").bool_val;
    P.enum_.ban_recursion    = cli.get("ban-rec").bool_val;
}


#if defined(GOOGLE_CODE)
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Protobuf support:


void typeToProto(const Type& t, ::CodeBreeder::TypeProto* proto)
{
    proto->set_name(t.name.c_str());
    for (int i = 0; i < t.size(); i++)
        typeToProto(t[i], proto->mutable_arg()->Add());
}


void GExpr::toProto(::CodeBreeder::NodeProto* proto) const
{
    proto->set_kind(GExprKind_name[kind]);
    proto->set_internal(internal);
    proto->set_cost(cost);
    for (int i = 0; i < ins.psize(); i++)
        proto->add_input((int)ins[i]);
    if (type())
        typeToProto(type(), proto->mutable_type_proto());
}


void State::toProto(::CodeBreeder::StateProto* proto) const // <<== move to SynthEnum.cc
{
    for (int i = 0; i < size(); ++i)
        (*this)[i].toProto(proto->add_node());
}


void Pool::toProto(::CodeBreeder::PoolProto* proto) const
{
    for (CExpr const& s : syms){
        proto->add_name(s->name.c_str());
        typeToProto(s->type, proto->add_type_proto());
        proto->add_qualified_name(fmt("%_", *s).c_str());
    }
}


#else
void GExpr::toProto(::CodeBreeder::NodeProto * proto) const { assert(false); }
void State::toProto(::CodeBreeder::StateProto* proto) const { assert(false); }
void Pool ::toProto(::CodeBreeder::PoolProto * proto) const { assert(false); }


#endif
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


void reportGenealogy(uind s, Vec<State> const& states, Vec<Pair<double,uint64>> const& state_costs, Vec<uind> const& parent, Pool const& pool)
{
    Vec<uind> hist;
    while (s != UIND_MAX){
        hist.push(s);
        s = parent[s]; }
    reverse(hist);

    newLn();
    wrLn("GENEALOGY:");
    for (uind s : hist){
        wrLn("  $%.2f %_", state_costs[s].fst, states[s].expr(pool));
    }
    newLn();
}


#if defined(GOOGLE_CODE)
static Str slice(std::string const& str) {
    return slice(*str.begin(), *str.end()); }   // -- C++11 guarantees strings are contiguous


void outputTrainingData(String filename, uind s, Vec<State> const& states, Vec<uind> const& parent, Pool const& pool)
{
    ::CodeBreeder::TrainingProto tr_proto;
    pool.toProto(tr_proto.mutable_pool_proto());

    uint pos_count = 0;
    uint neg_count = 0;
    Vec<uind> avoid;

    auto writeExamples = [&](uind s, bool positive) {
        Vec<uind> hist;
        while (s != UIND_MAX){
            if (has(avoid, s)) break;
            hist.push(s);
            s = parent[s];
        }
        reverse(hist);

        if (hist.size() > 1){
            for (uind s : hist)
                states[s].toProto(positive ? tr_proto.add_positive() : tr_proto.add_negative());
            State().toProto(positive ? tr_proto.add_positive() : tr_proto.add_negative());
            if (positive) pos_count += hist.size();
            else          neg_count += hist.size();
        }
        append(avoid, hist);
    };

    writeExamples(s, true);
    uint64 seed = 0;
    while (neg_count < pos_count * 9)   // -- 10/90 ratio between positives and negatives
        writeExamples(irand(seed, states.size()), false);

    {
        OutFile out(filename);
        //wrLn(out, "%_", slice(tr_proto.DebugString()));
        out += slice(tr_proto.SerializeAsString());
    }
    wrLn("Wrote: \a*%_\a*", filename);
}

#else
void outputTrainingData(String filename, uind, Vec<State> const&, Vec<uind> const&, Pool const&) { assert(false); }
#endif


#if defined(GOOGLE_CODE)
::CodeBreeder::PoolProto getPool(String spec_file, String params, bool spec_file_is_text)
{
    Spec spec = readSpec(spec_file, spec_file_is_text, true);
    Pool& pool = spec.pool;
    ::CodeBreeder::PoolProto pool_proto;
    pool.toProto(&pool_proto);
    return pool_proto;
}
#endif


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Enumeration based synthesis:


class Synth {
    // Input:
    Spec         const& spec;
    Params_Synth const& P;
    CostFun             cost_fun;

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
    IdHeap<Pair<double,uint64>> Q;

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

    // Internal Methods:
    String evalExpr(Expr const& expr, String score_fun = "score_", String print_fun = "print_int_", Type score_type = Type(a_Int));
    void   enqueue(State S, uind from);
    void   reportProgress(bool keep_output);

  #if defined(GOOGLE_CODE)
    void   processBatch();
  #endif

public:
    Synth(Spec const& spec, Params_Synth const& P, CostFun cost_fun) : spec(spec), P(P), cost_fun(cost_fun), Q(state_costs) {}
    int64 run();
};


//-------------------------------------------------------------------------------------------------
enum RunResults {
    res_NULL   ,
    res_RIGHT  ,
    res_WRONG  ,
    res_ABSTAIN,
    res_CRASH  ,
    res_CPU_LIM,
    res_MEM_LIM,
    res_REC_LIM,
    res_SIZE   ,
};



struct ResLims {
    uint64 cpu;
    addr_t mem;
    uint   rec;
    ResLims(uint64 cpu = 0, addr_t mem = 0, uint rec = 0) : cpu(cpu), mem(mem), rec(rec) {}
};


//-------------------------------------------------------------------------------------------------


// Returns 0 if successful, 1 on Evo runtime error, 2 on Evo compile time error.
static uint evalExpr_(Expr const& expr, RunTime& rt, Spec const& spec, ResLims const& rlim, /*out*/Vec<uint>& run_results, double& eval_time) ___unused;
static uint evalExpr_(Expr const& expr, RunTime& rt, Spec const& spec, ResLims const& rlim, /*out*/Vec<uint>& run_results, double& eval_time)
{
    // Type: fun run_all_<IN,OUT>(io_pairs :[(IN, OUT)], f_ :IN->OUT, rlim :RLim, checker_ :(IN, OUT, OUT)->Bool) -> [Int]
    Expr e_rlim = mxTuple({mxLit_Int(rlim.cpu), mxLit_Int(rlim.mem), mxLit_Int(rlim.rec)});
    Expr e_arg  = mxTuple({spec.io_pairs[0], expr, e_rlim, spec.checker[0]});
    Expr e_run  = mxAppl(spec.runner[0], e_arg);

    run_results.reset(res_SIZE+1, 0);
    uint ret_code = 0;

    rt.push();

    double T0_eval = cpuTime();
    try{
        Params_RunTime P_rt;
        P_rt.verbose = false;

        addr_t ret = rt.run(e_run, P_rt);
        if (ret == 0)
            ret_code = 1;
        else{
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
        ret_code = 2;
    }
    double T1 = cpuTime();
    eval_time += T1 - T0_eval;

    rt.pop();
    return ret_code;
}


String Synth::evalExpr(Expr const& expr, String score_fun, String print_fun, Type score_type)
{
#if 0   /*DEBUG*/
    static Expr e_fun = Expr::Sym(Atom("run_f_"), {}, parseType("(Int, Int->Int) -> Int"));
    static Params_RunTime P_rt; P_rt.verbose = false;

    int64 score = 0;
    for (uint i = 0; i < spec.n_io_pairs; i++){
        Expr e_arg = mxTuple({mxLit_Int(i), Expr(expr)});

        rt.push();
        try{
            addr_t ret = rt.run(mxAppl(e_fun, e_arg), P_rt);
            if (ret == 0) return "HALTED!";
            int64 val = rt.data(ret).val;
            assert(val >= 0 && val < res_SIZE);

            if      (val == res_RIGHT) score++;
            else if (val == res_WRONG) score -= 10000;
        }catch (Excp_ParseError err){
            return "HALTED!";
        }
        rt.pop();
    }
    if (score == spec.n_io_pairs) score = INT64_MAX;
    return fmt("%_", score);
#endif  /*END DEBUG*/

#if 0   /*DEBUG*/
    Vec<uint> results;
    evalExpr_(expr, rt, spec, ResLims(P.try_cpu_lim, P.try_mem_lim, P.try_rec_lim), results, eval_time);
    assert(spec.n_io_pairs == results[LAST]);
    int64 val = (results[LAST] == results[res_RIGHT]) ? INT64_MAX :
                                                        (int64)results[res_RIGHT] - 10000 * (int64)results[res_WRONG];
    return fmt("%_", val);
#endif  /*END DEBUG*/

    String result;
    try{
        double T0_eval = cpuTime();
        if (P.inc_eval){
            // Incremental evaluation:
            Vec<Expr> code;
            Expr score_sym  = Expr::Sym(Atom(score_fun), {}, Type(a_Fun, Type(spec.target), score_type));
            Expr score_appl = Expr::Appl(move(score_sym), Expr(expr)).setType(Type(score_type));
            Expr print_sym  = Expr::Sym(Atom(print_fun), {}, Type(a_Fun, score_type, Type(a_Void)));
            Expr print_appl = Expr::Appl(move(print_sym), move(score_appl)).setType(Type(a_Void));
            code.push(move(print_appl));
            Expr new_prog = Expr::Block(code).setType(Type(code[LAST].type));

            rt.push();
            double T0_exec = cpuTime();

            // Enumerator will generate expressions of type "rec f : Int->Int = \{f}();" which are illformed.
            // Treat them as infinitely looping.
            result.clear();
            P_rt.out = &result;
            try{
                rt.run(new_prog, P_rt);
            }catch (Excp_ParseError err){
                result = "HALTED!";
            }
            double T1 = cpuTime();
            eval_time += T1 - T0_eval;
            exec_time += T1 - T0_exec;

            rt.pop();

        }else{
            // Fresh-runtime-system evaluation:
            Vec<Expr> code(copy_, spec.prog);    // -- note, shallow copy
            Expr new_prog;
            if (P.thru_text){
                parseEvo(fmt("%_(%_(%_));", print_fun, score_fun, expr).c_str(), code);
                new_prog = Expr::Block(code);
                inferTypes(new_prog);
            }else{
                Expr score_sym  = Expr::Sym(Atom(score_fun), {}, Type(a_Fun, Type(spec.target), score_type));
                Expr score_appl = Expr::Appl(move(score_sym), Expr(expr)).setType(Type(score_type));
                Expr print_sym  = Expr::Sym(Atom(print_fun), {}, Type(a_Fun, score_type, Type(a_Void)));
                Expr print_appl = Expr::Appl(move(print_sym), move(score_appl)).setType(Type(a_Void));
                code.push(move(print_appl));
                new_prog = Expr::Block(code).setType(Type(code[LAST].type));
            }

            Params_RunTime P_rt;
            P_rt.cpu_lim = P.try_cpu_lim;
            P_rt.mem_lim = P.try_mem_lim;
            P_rt.rec_lim = P.try_rec_lim;
            P_rt.verbose = false;
            P_rt.out = &result;
            RunTime rt;

            // Enumerator will generate expressions of type "rec f : Int->Int = \{f}();" which are illformed.
            // Treat them as infinitely looping.
            double T0_exec = cpuTime();
            try{
                rt.run(new_prog, P_rt);
            }catch (Excp_ParseError err){
                result = "HALTED!";
            }
            double T1 = cpuTime();
            eval_time += T1 - T0_eval;
            exec_time += T1 - T0_exec;
        }

    }catch (Excp_ParseError err){
        newLn();
        wrLn("PARSE ERROR! %_", err.msg);
        wrLn("EXPR: %_", expr);
        newLn();
        result = "HALTED!";
    }
    return result;
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


void Synth::enqueue(State S, uind from)
{
  #if defined(GOOGLE_CODE)
    if (cost_fun){
        ::CodeBreeder::StateProto proto;
        S.toProto(&proto);

        batch_protos .push_back(proto);
        batch_parents.push(from);
        batch_states .push(S);
        n_enqueues++;

        if (batch_protos.size() >= P.batch_size)
            processBatch();
    }else
  #endif
    {
        double cost = S.cost() * rebate_mul[from] + rebate_add[from];
        if (cost <= P.max_cost){
            uind n = states.size();
            states.push(S);
            state_costs.push(tuple(cost, n_attempts));
            parent.push(from);
            double rm = rebate_mul[from]; rebate_mul.push(rm);     // -- push takes reference, which will be invalidated on a resize if using 'rebate[from]' directly instead of temporary 'r'.
            double ra = rebate_add[from]; rebate_add.push(ra);
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
    P_rt.cpu_lim = P.try_cpu_lim;
    P_rt.mem_lim = P.try_mem_lim;
    P_rt.rec_lim = P.try_rec_lim;
    P_rt.verbose = false;
    try{
        rt.run(spec.prog, P_rt);    // <<== should force 'score_' to be instantiated (just add 'score_;' as a statement)
    }catch (Excp_ParseError err){
        wrLn("PARSE ERROR! %_", err.msg);
        exit(1);
    }

    // Write task information to screen:
    if (P.verbosity > 0){
        if (spec.name != "") wrLn("\a*SPEC NAME:\a* %_", spec.name);
        wrLn("\a*TARGET:\a* %_", spec.target);
        wrLn("\a*POOL:\a*"); for (uint i = 0; i < pool.syms.size(); i++) wrLn("  $%>3%_:  @%_ = \a*%_\a*  \a/:%_\a/", pool.syms[i].cost, i, *pool.syms[i], pool.syms[i]->type);
        wrLn("\a*CONS COSTS:\a*");
        wrLn("  $%>3%_:  %_", pool.cost_Appl , "Appl");
        wrLn("  $%>3%_:  %_", pool.cost_Sel  , "Sel");
        wrLn("  $%>3%_:  %_", pool.cost_Lamb , "Lamb");
        wrLn("  $%>3%_:  %_", pool.cost_Tuple, "Tuple");
        if (spec.io_pairs) wrLn("\a*IO PAIRS:\a*\t+\t+\n%_\t-\t-", spec.io_pairs);      // <<== for now (just to see some representation of the problem)
        newLn();
    }

    // Expression enumeration state:
    states.push(initialEnumState(spec.target));
    state_costs.push(tuple(0.0, 0ull));
    parent.push(UIND_MAX);
    rebate_mul.push(1.0);
    rebate_add.push(0.0);
    Q.add(0);

    bool   trivial_results = (P.verbosity >= 4);
    int64  threshold = (P.verbosity >= 3) ? INT64_MIN/2 :
                       (P.verbosity == 2) ? 0 :
                       (P.verbosity == 1) ? 1 :
                       /*otherwise*/        INT64_MAX/2;
    int64  best = INT64_MIN/4;
    uint64 n_last_report = 0;

    if (P.verbosity > 0) wrLn("Expression enumaration starting...");
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

        if (P.dump_states)
            wrLn("(queue size: %_) dequeued: %_", Q.size(), S);
        if (P.dump_exprs)
            wrLn("(queue size: %_) dequeued: %_", Q.size(), S.expr(pool));

        uint tgt_i;
        if (S.getLast(g_Obl, tgt_i)){
            // Hardcode base heuristic for now:
            //if (P.base_rebate < 1.0 && rebate[s] == 1.0 && S[tgt_i].ins.psize() == 0){
            if (S.size() > 1 && P.base_rebate < 1.0 && rebate_mul[s] >= P.base_rebate && S[tgt_i].ins.psize() == 0){        // <<== real heuristic should count if more io_pairs were solved!
                bool single_obl = true;
                S.enumAll([&](uind i, GExpr const& g) {
                    if (i != tgt_i && g.kind == g_Obl){
                        single_obl = false;
                        return false;
                    }else
                        return true;
                });

                if (single_obl){
                    Expr prog = S.expr(pool, true);
                    String result = evalExpr(prog);
                    if (strstr(result.c_str(), "HALTED!") == nullptr){  // -- if steps limit is reached while outputting result, "HALTED!" may not appear first
                        int64 val = 0;
                        try{ val = stringToInt64(result); }
                        catch (...){ /**/wrLn("`` strange result: %_", result); }

                        //if (val > 0){
                        if ((rebate_mul[s] == 1.0 && val > 0) || val > 2){    // <<== need to compare to score of guy that set the rebate
                            if (P.verbosity >= 1) wrLn("\a*\a_Basecase?\a*\a_ (%_)  %_  [score: %_]", rebate_mul[s], prog, val);

//                            rebate_add[s] = latest_cost * rebate_mul[s] * (1 - P.base_rebate);    // -- experimental: will widen search
                            rebate_mul[s] *= P.base_rebate;
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
                int64 val;
                try{ val = stringToInt64(result); }
                catch (...){ wrLn("Program must output a single integer, not:\n\t+\t+\t+\t+%_\n\t-\t-\t-\t-", result); exit(1); }

                if (val == INT64_MAX){
                    if (P.verbosity > 0){
                        wrLn("\a*SOLVED!:\a* %_", e);
                        reportProgress(true);
                        if (!P.keep_going) wrLn("Found smallest solution!");
                        reportGenealogy(s, states, state_costs, parent, pool);
                    }
                    wrLn("PRETTY-PRINTED: [%_]\t+\t+\n%_\t-\t-\n", spec.name, ppFmt(e));
                    if (P.gen_training != "")
                        outputTrainingData(P.gen_training, s, states, parent, pool);
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

    if (P.verbosity > 0) wrLn("--exhaustive enumeration--");
    return best;
}



int64 synthesizeProgram(String spec_file, Params_Synth P, bool spec_file_is_text, CostFun cost_fun)
{
    if (P.verbosity > 0){
        wrLn("\a*==-----------------------------------------------------------------------------\a0");
        wrLn("\a*==\a* \a/%<76%_\a0", " CodeBreeder synthesis");
        wrLn("\a*==-----------------------------------------------------------------------------\a0");
        newLn();
    }

    Spec spec = readSpec(spec_file, spec_file_is_text); assert(spec.prog.kind == expr_Block);
    if (!spec.target){ wrLn("ERROR! Could not find 'score_' function."); exit(1); }

    Synth synth(spec, P, cost_fun);
    return synth.run();
}




int64 pySynthesizeProgram(String prog_text, String params, CostFun cost_fun)
{
    CLI cli;
    addParams_Synth(cli);
    cli.parseCmdLine(params, "pySynthesizeProgram()");

    Params_Synth P;
    setParams_Synth(cli, P);

    return synthesizeProgram(prog_text, P, true, cost_fun);
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

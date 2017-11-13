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

#include "zz/CmdLine/CmdLine.hh"

#include "Parser.hh"
#include "TypeInference.hh"
#include "Vm.hh"
#include "Synth.hh"
#include "RandomFuncs.hh"
#include "HeapSynth.hh"
#include "ExtractPropTests.hh"
#include "CreateTestVectors.hh"

using namespace ZZ;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


void runEvoFile(String filename, bool catch_error, Params_RunTime const& P, bool interactive)
{
    Inferrer infr;
    RunTime rt;

    if (catch_error){
        try{
            Expr prog = parseEvoFile(filename);
            infr.inferTypes(prog);
            rt.run(prog, P);
        }catch (Excp_ParseError err){
            wrLn("PARSE ERROR! %_", err.msg);
        }
    }else{
        Expr prog = parseEvoFile(filename);
        infr.inferTypes(prog);
        rt.run(prog, P);
    }

    // Interactive mode (at this point, only for debugging):
    if (interactive){
        for(;;){
            printf("%% ");
            char buf[1024];
            char* ret = fgets(buf, sizeof(buf), stdin);
            if (ret == nullptr || buf[0] == '\n') break;

            try{
                Vec<Expr> exprs;
                parseEvo(buf, exprs);
                Expr prog = Expr::Block(exprs);
                infr.inferTypes(prog);   // <<== need incremental stuff too...
                rt.run(prog, P);
            }catch (Excp_ParseError err){
                wrLn("PARSE ERROR! %_", err.msg);
            }
        }
    }
}


void printTypeAt(String filename, uint line, uint col)
{
    Expr prog;
    try{
        prog = parseEvoFile(filename);
        inferTypes(prog);
    }catch (Excp_ParseError err){
        wrLn("PARSE ERROR! %_", err.msg);
        exit(1);
    }

    Loc ask_loc(line, col);
    Loc best_loc;
    Vec<Type> best_types;

    function<void(Expr const&)> recurse = [&](Expr const& e) {
        if (e.loc.file) return;     // -- only consider main file 'filename', not its includes
        for (Expr const& sub : e) recurse(sub);
        if (e.loc <= ask_loc){
            if (!best_loc || e.loc >= best_loc){
                if (e.loc > best_loc) best_types.clear();
                best_loc = e.loc;
                best_types.push(e.type);
            }
        }
    };

    recurse(prog);
    if (best_types.size() > 0)
        wrLn("%_: %_", best_loc, best_types[0]);
        //wrLn("%_", best_types[0]);
    else{
        wrLn("ERROR! No token under given position.");
        exit(1);
    }
}


void test()
{
    try{
        Expr prog = parseEvoFile("test.evo");
        Inferrer infr;
        infr.inferTypes(prog);

        wrLn("TYPE: %_", prog.type);
        //RunTime rt;
        //RetVal val = rt.runRet(prog);
        //wrLn("RETVAL: %_", val);

    }catch (Excp_ParseError err){
        wrLn("PARSE ERROR! %_", err.msg);
        exit(1);
    }
}


int main(int argc, char** argv)
{
    ZZ_Init;

#if 0  // DEBUG
    __uint128_t bignum = UINT64_MAX;
    bignum *= bignum;
    while (bignum != 0){
        wr("%_ ", uint(bignum % 10));
        bignum /= 10;
    }
    newLn();
    return 0;
#endif // END-DEBUG
    cli.add("input", "string", "", "Input .evo file", 0);
    addStandardSwitches(cli);
    cli_hidden.add("i", "bool", "no", "Run in interactive mode [unfinished]");

    CLI cli_run;
    cli_run.add("catch", "bool", "yes", "Catch parse errors and report them (turn off if running inside GDB).");
    cli_run.add("cpu", "uint", "0", "CPU limit in steps. 0=no limit.");
    cli_run.add("mem", "ufloat", "64", "Memory limit in MB.");
    cli_run.add("rec", "uint", "0", "Maximum recursion depth. 0=no limit");
    cli_run.add("verbose", "bool", "yes", "Set to 'no' to only show program output.");
    cli_run.add("out", "string", "", "Send output to this file.");
    cli.addCommand("run", "Run Evo program.", &cli_run);

    CLI cli_synth;
    addParams_Synth(cli_synth);
    cli.addCommand("synth", "Synthesize Evo program.", &cli_synth);
    cli.addCommand("heap-synth", "Heap synthesis (unfinshed).");

    CLI cli_type_at;
    cli_type_at.add("line", "uint", arg_REQUIRED, "Line number (startin from 1)");
    cli_type_at.add("col" , "uint", arg_REQUIRED, "Column number (starting from 0)");
    cli.addCommand("type-at", "Return the type for the expression at given location.", &cli_type_at);

    CLI cli_rand_fun;
    cli_rand_fun.add("n"          , "uint", "100", "Number of random functions to output.");
    cli_rand_fun.add("timeout"    , "ufloat", "0", "Run for at most this many seconds (cpu time).");
    cli_rand_fun.add("memout"     , "ufloat", "0", "Use at most this much memory (in MB).");
    cli_rand_fun.add("use-formals", "bool", "no", "If 'yes', all formal parameters to a function must be used.");
    cli_rand_fun.add("force-rec"  , "bool", "no", "If 'yes', synthesized function must contain a recursive call.");
    cli_rand_fun.add("ban-rec"    , "bool", "no", "If 'yes', synthesized function must NOT contain a recursive call.");
    cli_rand_fun.add("print-rec"  , "bool", "no", "If 'yes', only recursive functions are printed (but all functions are synthesized).");
    cli_rand_fun.add("must"       , "[string]", "[]", "List of symbols that must appear in output");
    cli_rand_fun.add("cant"       , "[string]", "[]", "List of symbols that can't appear in output");
    cli_rand_fun.add("cpu", "uint"  , "25000000", "Evo CPU limit in steps. 0=no limit.");
    cli_rand_fun.add("mem", "ufloat", "64"      , "Evo memory limit in MB.");
    cli_rand_fun.add("rec", "uint"  , "10000"   , "Evo maximum recursion depth. 0=no limit");
    cli_rand_fun.add("verb", "uint", "1", "Verbosity level.");
    cli_rand_fun.add("save-pfx", "string", "", "If provided, training data will be saved to files with this prefix.");
    cli_rand_fun.add("seen-in" , "string", "", "Read already seen hashes from this file.");
    cli_rand_fun.add("seen-out", "string", "", "Write seen hashes to this file.");
    cli.addCommand("rand-fun", "Experimental code for generating random functions.", &cli_rand_fun);

    CLI cli_prop_tests;
    cli_prop_tests.add("output", "string", "", "Output file for property test patterns.");
    cli.addCommand("prop-tests", "Collect property tests ('input' is interpreted as glob).", &cli_prop_tests);

    cli.addCommand("summarize-targets", "Count targets of each type in selected set of files.");

    cli.addCommand("test", "Placeholder for debugging.");

    CLI cli_testvecs;
    cli_testvecs.add("var", "string", "test_inputs", "Name of variable holding generated test patterns.");
    cli_testvecs.add("n", "uint", "20", "Number of test-vectors to generate for each type.");
    cli_testvecs.add("seed", "uint", "0", "Random seed to use.");
    cli_testvecs.add("cat", "int", "-1", "Restrict test vector to this category");
    cli_testvecs.add("types", "string", arg_REQUIRED, "File containint ';'-separated list of types.");
    cli_testvecs.add("closed", "bool", "yes", "Should sub-types of given types be added automatically?");
    cli.addCommand("testvecs", "Generate input test-vectors for a set of predefined types (see 'CreateTestVectors.cc').", &cli_testvecs);

    cli.parseCmdLine(argc, argv);
    processStandardSwitches(cli);

    // Run command:
    setupSignalHandlers();      // -- capture CTRL-C in a nicer way

    String in_filename = cli.get("input").string_val;
    if (cli.cmd == "run"){
        if (in_filename == ""){ wrLn("ERROR! Must provide input file."); exit(1); }
        bool catch_ = cli.get("catch").bool_val;
        bool interactive = cli.get("i").bool_val;
        Params_RunTime P;
        P.lim.cpu = cli.get("cpu").int_val; if (P.lim.cpu == 0) P.lim.cpu = UINT64_MAX;
        P.lim.mem = cli.get("mem").float_val * 1024 * 1024 / sizeof(VM::Word);
        P.lim.rec = cli.get("rec").int_val; if (P.lim.rec == 0) P.lim.rec = UINT_MAX;
        P.verbose = cli.get("verbose").bool_val;
        String out_filename = cli.get("out").string_val;
        if (out_filename != ""){
            OutFile out(out_filename);
            P.out = &out;
            runEvoFile(in_filename, catch_, P, interactive);
        }else
            runEvoFile(in_filename, catch_, P, interactive);

    }else if (cli.cmd == "synth"){
        if (in_filename == ""){ wrLn("ERROR! Must provide input file."); exit(1); }
        Params_Synth P;
        setParams_Synth(cli, P);
        int64 result = synthesizeProgram(in_filename, P);
        if (P.verbosity > 0){
            wrLn("CPU-time: %t", cpuTime ());
            wrLn("Realtime: %t", realTime());
        }
        return (result == INT64_MAX) ? 0 : 1;

    }else if (cli.cmd == "heap-synth"){
        if (in_filename == ""){ wrLn("ERROR! Must provide input file."); exit(1); }
        Spec spec = readSpec(in_filename, false); assert(spec.prog.kind == expr_Block);
        if (!spec.pool) { wrLn("ERROR! Specification file must contain symbol pool definition 'syms'."); exit(1); }
        SimpleSynth synth(spec);
        synth.run();

    }else if (cli.cmd == "type-at"){
        printTypeAt(in_filename, cli.get("line").int_val, cli.get("col").int_val);

    }else if (cli.cmd == "rand-fun"){
        Params_RandFun P;
        P.P_enum.must_use_formals = cli.get("use-formals").bool_val;
        P.P_enum.force_recursion  = cli.get("force-rec").bool_val;
        P.P_enum.ban_recursion    = cli.get("ban-rec").bool_val;
        P.n_funcs_to_generate     = cli.get("n").int_val;
        P.timeout                 = cli.get("timeout").float_val;
        P.memout                  = cli.get("memout").float_val * 1024 * 1024;
        P.print_only_recursive    = cli.get("print-rec").bool_val;
        P.lim.cpu = cli.get("cpu").int_val; if (P.lim.cpu == 0) P.lim.cpu = UINT64_MAX;
        P.lim.mem = cli.get("mem").float_val * 1024 * 1024 / sizeof(VM::Word);
        P.lim.rec = cli.get("rec").int_val; if (P.lim.rec == 0) P.lim.rec = UINT_MAX;
        P.verbosity = cli.get("verb").int_val;
        P.training_data_pfx = cli.get("save-pfx").string_val;
        P.seen_infile       = cli.get("seen-in").string_val;
        P.seen_outfile      = cli.get("seen-out").string_val;
        for (CLI_Val v : cli.get("must")) P.must_haves.push(v.string_val);
        for (CLI_Val v : cli.get("cant")) P.cant_haves.push(v.string_val);

        generateRandomFunctions(in_filename, P);

    }else if (cli.cmd == "prop-tests"){
        writePropTestFile(in_filename, cli.get("output").string_val);

    }else if (cli.cmd == "summarize-targets"){
        summarizeTargets(in_filename);

    }else if (cli.cmd == "testvecs"){
        String var_name   = cli.get("var").string_val;
        uint   n_patterns = cli.get("n").int_val;
        uint64 seed       = cli.get("seed").int_val;
        uint   cat        = cli.get("cat").int_val;
        String filename   = cli.get("types").string_val;
        bool   closed     = cli.get("closed").bool_val;
        genTestVectors(std_out, var_name, n_patterns, seed, cat, filename, closed);

    }else if (cli.cmd == "test"){
        test();
    }

    return 0;
}

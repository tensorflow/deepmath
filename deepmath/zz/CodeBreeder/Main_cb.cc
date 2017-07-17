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
#include "PropSynth.hh"

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
        Expr prog = parseEvoFile("bug.evo");
        Inferrer infr;
        infr.inferTypes(prog);

        RunTime rt;
        addr_t ret = rt.run(prog);
        /**/Dump(ret);
        addr_t head = rt.data(ret).val;
        addr_t vec_data = rt.data(head).val;
        addr_t vec_size = rt.data(head+1).val;
        /**/Dump(head, vec_data, vec_size);
        Array<VM::Word const> results = slice(rt.data(vec_data), rt.data(vec_data + vec_size));
        /**/Dump(results);

        prog = parseEvo("apa ()");
        infr.inferTypes(prog);
        ret = rt.run(prog);
        /**/Dump(ret);

    }catch (Excp_ParseError err){
        wrLn("PARSE ERROR! %_", err.msg);
        exit(1);
    }
}


int main(int argc, char** argv)
{
    ZZ_Init;

    cli.add("input", "string", "", "Input .evo file", 0);
    cli.add("env", "[string]", "", "Set environment variables.");
    cli.add("profile", "bool", "no", "Display developer's profile information at the end.");
    cli_hidden.add("i", "bool", "no", "Run in interactive mode [unfinished]");

    CLI cli_run;
    cli_run.add("catch", "bool", "yes", "Catch parse errors and report them (turn off if running inside GDB).");
    cli_run.add("cpu", "uint", "0", "CPU limit in steps. 0=no limit.");
    cli_run.add("mem", "ufloat", "8", "Memory limit in MB.");
    cli_run.add("rec", "uint", "0", "Maximum recursion depth. 0=no limit");
    cli_run.add("verbose", "bool", "yes", "Set to 'no' to only show program output.");
    cli_run.add("out", "string", "", "Send output to this file.");
    cli.addCommand("run", "Run Evo program.", &cli_run);

    CLI cli_synth;
    addParams_Synth(cli_synth);
    cli.addCommand("synth", "Run Evo program.", &cli_synth);

    CLI cli_prop_spec;
    cli_prop_spec.add("...", "string", "", "List of EVO files");
    cli.addCommand("prop-spec", "Create property specification file.", &cli_prop_spec);

    CLI cli_type_at;
    cli_type_at.add("line", "uint", arg_REQUIRED, "Line number (startin from 1)");
    cli_type_at.add("col" , "uint", arg_REQUIRED, "Column number (starting from 0)");
    cli.addCommand("type-at", "Return the type for the expression at given location.", &cli_type_at);

    cli.addCommand("test", "Placeholder for debugging.");

    cli.parseCmdLine(argc, argv);

    // Environment:
    Vec<CLI_Val>* env = cli.get("env").sub;
    if (env){
        for (auto&& elem : *env){
            putenv(elem.string_val.c_str());
        }
    }
    suppress_profile_output = !cli.get("profile").bool_val;

    // Run command:
    setupSignalHandlers();      // -- capture CTRL-C in a nicer way

    String in_filename = cli.get("input").string_val;
    if (cli.cmd == "run"){
        if (in_filename == ""){ wrLn("ERROR! Must provide input file."); exit(1); }
        bool catch_ = cli.get("catch").bool_val;
        bool interactive = cli.get("i").bool_val;
        Params_RunTime P;
        P.cpu_lim = cli.get("cpu").int_val; if (P.cpu_lim == 0) P.cpu_lim = UINT64_MAX;
        P.mem_lim = cli.get("mem").float_val * 1024 * 1024;
        P.rec_lim = cli.get("rec").int_val; if (P.rec_lim == 0) P.rec_lim = UINT_MAX;
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

    }else if (cli.cmd == "prop-spec"){
        Vec<String> files= {in_filename};
        for (auto&& arg : cli.tail)
            files.push(arg.string_val);
        createPropertySpecifications(files);

    }else if (cli.cmd == "type-at"){
        printTypeAt(in_filename, cli.get("line").int_val, cli.get("col").int_val);

    }else if (cli.cmd == "test"){
        test();
    }

    return 0;
}

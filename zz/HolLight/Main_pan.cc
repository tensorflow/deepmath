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
#include "CmdLine/CmdLine.hh"
#include "Printing.hh"
#include "Checker.hh"
#include "PremiseViewer.hh"
#include "WriteProof.hh"

using namespace ZZ;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


int main(int argc, char** argv)
{
    ZZ_Init;

    // Declare command line arguments:
    cli.add("input"   , "string", arg_REQUIRED, "Input proof-log. May be gzipped.", 0);
    cli.add("ansi"    , "bool", pp_use_ansi ? "yes" : "no", "Use ANSI codes in batch output.");

    CLI cli_check;
    cli_check.add("progress", "bool", "yes", "Show progress during proof-checking.");

    CLI cli_view;
    cli_view.add("thm", "string", "", "Go to detailed view for given theorem.");

    CLI cli_write;
    cli_write.add("output", "string", "", "Output proof-log file.");

    cli.addCommand("check", "Check proof (will throw assert if incorrect).", &cli_check);
    cli.addCommand("stats", "Quick proof statistics.");
    cli.addCommand("thms" , "List (human named) theorems.");
    cli.addCommand("appl" , "List proof-rule applications.");
    cli.addCommand("full" , "List proof-rule applications and results.");
    cli.addCommand("view" , "Interactive proof explorer.", &cli_view);
    cli.addCommand("write", "Rewrite the proof-log without redundancies.", &cli_write);

    // Parse command line arguments:
    cli.parseCmdLine(argc, argv);

    pp_use_ansi = cli.get("ansi").bool_val;

    String input_file = cli.get("input").string_val;
    String output_file = (cli.cmd != "write") ? String() : cli.get("output").string_val;
    if (output_file.size() == 0)
        output_file = setExtension(input_file, "new.pfl");

    // Run command:
    if      (cli.cmd == "check") checkProof(input_file, CheckMode::CHECK, cli.get("progress").bool_val);
    else if (cli.cmd == "stats") checkProof(input_file, CheckMode::STATS);
    else if (cli.cmd == "thms" ) checkProof(input_file, CheckMode::THMS);
    else if (cli.cmd == "appl" ) checkProof(input_file, CheckMode::APPL);
    else if (cli.cmd == "full" ) checkProof(input_file, CheckMode::FULL);
    else if (cli.cmd == "view" ) viewProof(input_file, cli.get("thm").string_val);
    else if (cli.cmd == "write") writeProof(input_file, output_file);
    else assert(false);

    if (cli.cmd != "thms"){     // -- we want this one to be completely clean from any other output
        NewLine;
        WriteLn "Mem used: %.1f MB", memUsed() / 1048576.0;
        WriteLn "CPU-time: %t", cpuTime();
        WriteLn "Realtime: %t", realTime();
    }

    return 0;
}

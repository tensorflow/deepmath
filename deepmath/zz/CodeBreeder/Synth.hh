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

#ifndef ZZ__CodeBreeder__Synth_hh
#define ZZ__CodeBreeder__Synth_hh

#include <vector>

#include "zz/CmdLine/CmdLine.hh"
#include "Types.hh"
#include "SynthSpec.hh"
#include "SynthEnum.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Synthesis:


struct Params_Synth {
    uint    verbosity    = 1;           // -- 0=show only solution, 1=show improvements, 2=show equal to best-so-far, 3=show everything but trivial, >=4 show everything
    bool    dump_states  = false;
    bool    dump_exprs   = false;
    bool    thru_text    = false;       // -- for debugging, run programs by printing and parsing back (can give better error messages)
    String  gen_training = "";          // -- file to write training protos to

    Params_SynthEnum enum_;
    bool    use_prune_rules  = true;

    bool    inc_eval         = true;
    bool    keep_going       = false;
    double  base_rebate      = 1.0;     // -- A factor of 1 means no rebate for base case (turn it off).
    bool    soft_rebate      = false;   // -- If true, rebate only apply to cost of constructs added AFTER discovering potential base-case

    // Overall search procedure limits:
    uint64  max_tries = UINT64_MAX;
    uint64  max_queue = UINT64_MAX;     // -- in enqueued states
    double  max_cpu   = -1;             // -- in seconds
    double  max_cost  = DBL_INF;

    // Individual evaluation limits:
    uint64  try_mem_lim = 8* 1048576;   // -- #words a single evaluation can take
    uint64  try_cpu_lim = 250000;       // -- number of execution steps a single evaluation can take
    uint    try_rec_lim = 1000;         // -- limit on recursion depth

    // Python interface:
    uint    batch_size = 64;
    double  randomize_costs = 0.0;      // -- add a random cost '[0,x[' to each actual cost computed (to shake up the heap a bit)
    bool    enumeration_mode = false;   // -- if TRUE, output of generated program is ignored and each candidate is considered a solution
};

void addParams_Synth(CLI& cli);
void setParams_Synth(const CLI& cli, Params_Synth& P);


// Forward declare protobuf for callback:
typedef function<std::vector<double>(::CodeBreeder::PoolProto const&, std::vector<::CodeBreeder::StateProto> const&)> CostFun;
typedef function<void(::CodeBreeder::TrainingProto const&)> SolutionFun;  // -- called with the genealogy of the solution

int64 synthesizeProgram(String spec_file, Params_Synth P = Params_Synth(), bool spec_file_is_text = false, CostFun cost_fun = CostFun(), SolutionFun sol_fun = SolutionFun());
int64 pySynthesizeProgram(String prog_text, String params, CostFun cost_fun = CostFun(), SolutionFun sol_fun = SolutionFun());
  // -- returns the best score achieved (which for the standard score function is the number of
  // correctly solved input/output pairs or INT64_MAX if all pairs are correct; INT64_MIN if
  // resource limits were such that not a single program was tried).
  //
  // EXAMPLE:
  //   int64 ret = pySynthesizeProgram(prog_text, "-max-cpu=10 -max-queue=1000000");


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

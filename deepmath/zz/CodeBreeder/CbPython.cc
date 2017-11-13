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

#if defined(GOOGLE_CODE)
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>

//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


#include "synth.proto.h"

#include ZZ_Prelude_hh
#include "Synth.hh"
#include "Parser.hh"
#include "TypeInference.hh"
#include "Expand.hh"
#include "Vm.hh"

void initialize() {
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    ZZ::zzInitialize ();
  }
}

bool does_compile(const std::string& progtext) {
  initialize();
  try {
    ZZ::Vec<ZZ::Expr> exprs;
    ZZ::parseEvo(progtext.c_str(), exprs);
    ZZ::Expr prog = ZZ::Expr::Block(exprs);
    ZZ::inferTypes(prog);
    ZZ::RunTime rt;
    rt.tryCompile(prog);
  } catch (ZZ::Excp_ParseError err) {
    ZZ::wrLn("PARSE ERROR! %_", err.msg);
    return false;
  } catch(...) {
    ZZ::wrLn("OTHER ERROR!");
    return false;
  }
  return true;
}

int64 synthesize_program(
  const std::string& progtext,
  const std::string& params,
  std::function<void(::CodeBreeder::TrainingProto const &)> sol_callback
) {
  initialize();
  ZZ::String prog_text(progtext.c_str());
  ZZ::String parameters(params.c_str());
  auto score = ZZ::pySynthesizeProgram(prog_text, parameters,
                                       ZZ::CostFun(), std::move(sol_callback));
  ZZ::wrLn("Score: %_", score);
  return score;
}

int64 synthesize_with_guidance(
    const std::string& progtext,
    const std::string& params,
    std::function<std::vector<double>(::CodeBreeder::PoolProto const &,
      std::vector<::CodeBreeder::StateProto> const &)> guide_callback,
    std::function<void(::CodeBreeder::TrainingProto const &)> sol_callback
) {
  initialize();
  ZZ::String prog_text(progtext.c_str());
  ZZ::String parameters(params.c_str());
  auto score =
      ZZ::pySynthesizeProgram(prog_text, parameters, std::move(guide_callback),
                              std::move(sol_callback));
  ZZ::wrLn("Score: %_", score);
  return score;
}

static ::CodeBreeder::PoolProto get_pool_helper(ZZ::String spec_file, ZZ::String params)
{
    ZZ::Spec spec = readSpec(spec_file, true, true);
    ZZ::Pool& pool = spec.pool;
    ::CodeBreeder::PoolProto pool_proto;
    pool.toProto(&pool_proto);
    return pool_proto;
}

::CodeBreeder::PoolProto get_pool(const std::string& progtext,
                                  const std::string& params) {
  initialize();
  ZZ::String prog_text(progtext.c_str());
  ZZ::String parameters(params.c_str());
  return get_pool_helper(prog_text, parameters);
}

struct EvoVm {
  ZZ::Inferrer inferrer;
  ZZ::RunTime  run_time;
};

//std::unique_ptr<EvoVm> evo_create_vm() { return std::unique_ptr<EvoVm>(new EvoVm); }
EvoVm* evo_create_vm() {
  initialize();
  return new EvoVm;
}

void evo_dispose_vm(EvoVm* vm) {
  delete vm;
}

void evo_push_state(EvoVm* vm) {
  vm->inferrer.push();
  vm->run_time.push();
}

void evo_pop_state(EvoVm* vm) {
  vm->inferrer.pop();
  vm->run_time.pop();
}

// Returns a string containing the output (unless 'to_string' is set to false)
// and a boolean that is TRUE if execution was successful, FALSE if an Evo
// excpetion was raised and uncaught (such as reaching resource limits or
// addressing a vector out of range) OR if the 'progtext' resulted in a compile
// error.
std::pair<std::string, bool> evo_run_rlim(EvoVm* vm, const std::string& progtext, bool to_string,
                                          uint64 cpu_lim, uint64 mem_lim, uint rec_lim) {
  ZZ::String text;
  ZZ::Params_RunTime P;
  P.lim.cpu = cpu_lim;
  P.lim.mem = mem_lim;
  P.lim.rec = rec_lim;
  P.verbose = false;
  P.out = to_string ? &text : &ZZ::std_out;

  try{
    ZZ::Expr prog = ZZ::parseEvo(progtext.c_str());
    vm->inferrer.inferTypes(prog);
    ZZ::addr_t ret = vm->run_time.run(prog, P);
    return make_pair(std::string(text.c_str()), ret != 0);
  }catch (ZZ::Excp_ParseError err){
    return make_pair(std::string(fmt("PARSE ERROR! %_", err.msg).c_str()), false);
  }
}

std::pair<std::string, bool> evo_run(EvoVm* vm, const std::string& progtext, bool to_string) {
  ZZ::Params_RunTime P;
  return evo_run_rlim(vm, progtext, to_string, P.lim.cpu, P.lim.mem, P.lim.rec);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
#endif

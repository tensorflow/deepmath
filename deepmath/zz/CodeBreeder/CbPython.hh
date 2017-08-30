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

#include <string>
#include "synth.proto.h"
#ifndef ZZ__CodeBreeder__CbPython_hh
#define ZZ__CodeBreeder__CbPython_hh


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


bool does_compile(const std::string& text);

::CodeBreeder::PoolProto get_pool(
  const std::string& progtext,
  const std::string& params
  );

int64 synthesize_program(
  const std::string& progtext,
  const std::string& params,
  std::function<void(::CodeBreeder::TrainingProto const &)> sol_callback
  );

int64 synthesize_with_guidance(
  const std::string& progtext,
  const std::string& params,
  std::function<std::vector<double>(CodeBreeder::PoolProto const &, std::vector<CodeBreeder::StateProto> const &)> guide_callback,
  std::function<void(::CodeBreeder::TrainingProto const &)> sol_callback
  );


// Exposing Evo compiler/runtime system:
struct EvoVm;

EvoVm* evo_create_vm();
void evo_dispose_vm(EvoVm* vm);
void evo_push_state(EvoVm* vm);
void evo_pop_state (EvoVm* vm);

std::pair<std::string, bool> evo_run     (EvoVm* vm, const std::string& progtext, bool to_string);
std::pair<std::string, bool> evo_run_rlim(EvoVm* vm, const std::string& progtext, bool to_string,
                                          uint64 cpu_lim, uint64 mem_lim, uint rec_lim);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
#endif
#endif

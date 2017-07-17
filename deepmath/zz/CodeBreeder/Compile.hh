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

#ifndef ZZ__CodeBreeder__Compile_hh
#define ZZ__CodeBreeder__Compile_hh

#include "Vm.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


// Only needed for optimization; basic compilation is oblivious of previously compiled code.
struct VmState {
    Vec<Instr>&     code;
    Vec<VM::Word>&  data;
    Vec<VM::WTag>&  tag;
    VmState(Vec<Instr> & code, Vec<VM::Word>& data, Vec<VM::WTag>& tag) : code(code), data(data), tag(tag) {}
};


// Compile whole expression. PRE-CONDITION: All literals have been added to 'tab0'.
addr_t compile(Vec<Instr>& code, Expr const& prog, SymTable<RelPos>& tab0, RelPos& result, addr_t glob_off, VmState const& vm_state);
    // -- NOTE! For now, illegal let-recs are detected in this (the compilation) phase.
    // Perhaps this should be moved to an earlier phase.
    // NOTE! The same now holds for 'write_' being passed a non-'Void->Void' function.


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

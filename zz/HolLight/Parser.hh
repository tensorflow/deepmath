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

#ifndef ZZ__HolLight__Parser_hh
#define ZZ__HolLight__Parser_hh

#include "ParserTypes.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


typedef function<void(ArgKind ret_kind, RuleKind rule_kind, Vec<Arg>& args, double progress_percent)> ProofCallback;
    // -- 'ret_kind' is one of: arg_TYPE, arg_TERM, arg_THM. For type, 'args' will use
    // proof-log indexes: arg_TYPE_IDX, arg_TERM_IDX, arg_THM_IDX.


void parseProofLog(String const& filename, ProofCallback const& cb, bool with_args = true);
    // -- if 'with_args' is FALSE, the 'args' vector in the callback will be empty EXCEPT fo
    // top-level theorems.


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

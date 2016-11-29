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

#ifndef ZZ__HolLight__DetailedViewer_hh
#define ZZ__HolLight__DetailedViewer_hh

#include "ProofStore.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


inline char objSymbol(ArgKind kind) {
    return (kind == arg_TYPE || kind == arg_TYPE_IDX) ? '^' :
           (kind == arg_TERM || kind == arg_TERM_IDX) ? '%' :
           (kind == arg_THM  || kind == arg_THM_IDX ) ? '#' : 0; }


bool detailedViewer(ProofStore& P, Vec<line_t> const& theorems, Vec<line_t> const& premises);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

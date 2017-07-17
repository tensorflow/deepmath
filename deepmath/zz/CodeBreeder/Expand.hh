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

#ifndef ZZ__CodeBreeder__Expand_hh
#define ZZ__CodeBreeder__Expand_hh

#include "Types.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


struct Expander_data;

struct Expander {
    Expander_data* data;

    Expander();
   ~Expander();

    void push();
    void pop();

    Expr expandTemplates(Expr const& prog);
        // -- Can be called multiple times; datatype definitions and let-recs will be remembered
        // from previous calls. Each call should only contain the NEW code in 'prog' (which must
        // be of type 'expr_Block), not all code up to this point. The return value will contain
        // a block with monomorphic functions and values only.
};


inline Expr expandTemplates(Expr const& prog) {
    Expander X;
    return X.expandTemplates(prog); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

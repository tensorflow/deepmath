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

#ifndef ZZ__HolLight__HolFormat_hh
#define ZZ__HolLight__HolFormat_hh

#include "zz/HolLight/HolOperators.hh"

#include ZZ_Prelude_hh
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


// Character Category
enum CharCat : uchar {
    cc_CNST,
    cc_EQUAL,       // -- constant '=' is marked separately
    cc_BINDER,
    cc_VAR,         // -- bound variable
    cc_ABSVAR,      // -- abstraction variable (variable at binding position)
    cc_FREEVAR,
    cc_OTHER,

    CharCat_size
};


// Decorated Character
struct DChar {
    char     chr;
    CharCat  cat;
    Term     term;
};


Vec<DChar> fmtTerm(Term tm);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

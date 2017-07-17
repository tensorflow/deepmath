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

#ifndef ZZ__CodeBreeder__Parser_hh
#define ZZ__CodeBreeder__Parser_hh

#include "Types.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


// NOTE! Parser will create pointers into 'text'; its life-span must outlive the use of the types
// and expressions returned.
Type parseType(cchar* text);
Expr parseEvoFile(String filename);         // -- returns a top-level expression (may contain type/data definitions)
void parseEvo(cchar* text, Vec<Expr>& out); // -- append expressions parsed from 'text' to 'out'

inline Expr parseEvo(cchar* text) {
    Vec<Expr> out;
    parseEvo(text, out);
    return Expr::Block(out);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

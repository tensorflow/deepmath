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

#ifndef ZZ__CodeBreeder__TrainingData_hh
#define ZZ__CodeBreeder__TrainingData_hh

#include "SynthSpec.hh"
#include "SynthEnum.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


struct TrainingData {
    typedef Pair<uind, ENUM::State> StatePair;  // -- (state_id, state)

    Pool             pool;
    Vec<StatePair>   positive;  // }- genealogies are separated by a null state
    Vec<StatePair>   negative;  // }
    Vec<bool>        property_features; // -- Christian's property vector
    String           evo_code;
    String           evo_output;
    String           type_string;

  #if 0
    Vec<schar>       prop_vec;
        // '-127'=property is false, '+127'=property holds, '0'=property does not apply.
        // In future: 1..126 may be used for partial functions where property holds on all valid inputs
  #endif

    void toProto(::CodeBreeder::TrainingProto* proto) const;
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

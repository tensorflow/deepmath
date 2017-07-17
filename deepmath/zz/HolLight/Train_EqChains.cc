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

#include ZZ_Prelude_hh
#include "Train_EqChains.hh"
#include "ProofStore.hh"
#include "ProofLogging.hh"
#include "Printing.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


void extractEqChains(String input_file)
{
    kernel_proof_logging = true;

    ProofStore P(input_file);
    IntMap<id_t, uint> dist(0);     // -- maps theorem IDs to length of equality chain
    IntMap<id_t, bool> sink(false); // -- last step of equality reasoning

    WriteLn "Evaluating proof...";
    for (line_t n = 0; n < P.size(); n++){
        if (!P.is_thm(n)) continue;

        th_t idx = P.line2idx[n];
        Thm  th  = P.evalThm(idx);
        List<IdBase> pr = th.proof();
        RuleKind rule = RuleKind(+pr[0]);

        // Exclude equivalences that include hypothises or are of type bool.
        if (rule == rule_TRANS && th.hyps().empty() && th.concl().arg().type() != type_bool){
            PArgKind kind0, kind1;
            id_t     id0, id1;
            l_tuple(kind0, id0) = decodeArg(pr[1]); assert(kind0 == parg_Thm);
            l_tuple(kind1, id1) = decodeArg(pr[2]); assert(kind1 == parg_Thm);
            dist(+th) = dist[id0] + dist[id1] + 1;
            sink(id0) = false;
            sink(id1) = false;
            sink(+th) = true;
        }
    }

    for (id_t i = 0; i < sink.size(); i++){
        if (sink[i] && dist[i] > 2){
            wrLn("%_: %_", dist[i], Thm(i));
        }
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

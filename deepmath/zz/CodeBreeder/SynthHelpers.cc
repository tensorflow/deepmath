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

#include ZZ_Prelude_hh
#include "SynthHelpers.hh"


namespace ZZ {
using namespace std;
using namespace ENUM;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


void reportGenealogy(uind s, Vec<State> const& states, Vec<Pair<double,uint64>> const& state_costs, Vec<uind> const& parent, Pool const& pool)
{
    Vec<uind> hist;
    while (s != UIND_MAX){
        hist.push(s);
        s = parent[s]; }
    reverse(hist);

    newLn();
    wrLn("GENEALOGY:");
    for (uind s : hist){
        wrLn("  $%.2f %_", state_costs[s].fst, states[s].expr(pool));
    }
    newLn();
}


TrainingData genTrainingData(uind s, Vec<State> const& states, Vec<uind> const& parent, Pool const& pool)
{
    TrainingData tr;
    tr.pool = pool;
    tr.evo_code = fmt("%_", states[s].expr(pool));
    tr.type_string = fmt("%_", states[s][0].type());

    uint pos_count = 0;
    uint neg_count = 0;
    Vec<uind> avoid;

    auto writeExamples = [&](uind s, bool positive) {
        Vec<Pair<uind,State>>& out = positive ? tr.positive : tr.negative;

        Vec<uind> hist;
        while (s != UIND_MAX){
            if (has(avoid, s)) break;
            hist.push(s);
            s = parent[s];
        }
        reverse(hist);

        if (hist.size() > 1){
            for (uind s : hist)
                out.push(tuple(s, states[s]));
            out.push(tuple((uind)0, State()));
            if (positive) pos_count += hist.size();
            else          neg_count += hist.size();
        }
        append(avoid, hist);
    };

    writeExamples(s, true);
    uint64 seed = 0;
    uint max_tries = 10000;
    while (neg_count < pos_count * 9 && max_tries > 0){   // -- 10/90 ratio between positives and negatives
        writeExamples(irand(seed, states.size()), false);
        max_tries--; }

    return tr;
}


void outputTrainingData(String filename, uind s, Vec<State> const& states, Vec<uind> const& parent, Pool const& pool)
{
    ::CodeBreeder::TrainingProto tr_proto;
    genTrainingData(s, states, parent, pool).toProto(&tr_proto);
    {
        OutFile out(filename);
        out += slice(tr_proto.SerializeAsString());
    }
    wrLn("Wrote: \a*%_\a*", filename);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

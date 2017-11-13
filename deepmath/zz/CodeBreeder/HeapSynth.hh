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

#ifndef ZZ__CodeBreeder__HeapSynth_hh
#define ZZ__CodeBreeder__HeapSynth_hh

#include "zz/Generics/IdHeap.hh"

#include "Types.hh"
#include "Vm.hh"
#include "SynthEnum.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


struct Params_HeapSynth {
    bool             hash_states   = false;     // <<== not implemented yet
    bool             pool_is_state = false;     // <<== not implemented yet
    uint64           report_freq   = 50;
    ResLims          rlim;
    Params_SynthEnum P_enum;
    Params_RunTime   P_rt;
};


struct HeapSynth {
    using State = ENUM::State;
    using state_id = uind;
    enum{ NO_PARENT = UIND_MAX };

  //________________________________________
  //  State:

    Params_HeapSynth     P;
    RunTime              rt;

    // Indexed by state ID:
    Vec<State>           state;
    Vec<Pool>            pool;
    Vec<double>          cost;      // -- ties are broken by state_id.
    Vec<state_id>        parent;

    StableIdHeap<double> Q;
    uint64               reportC;   // -- starts at 'report_freq' and counts down to zero, then 'reportProgress()' is called (and this counter reset)

    HeapSynth(Expr const& prog, Params_HeapSynth const& P);
    void run();

  //________________________________________
  //  Helpers for virtual functions:

    void enqueue(State const& S, Pool new_pool, uind from = NO_PARENT);
    void enqueue(State const& S, uind from) { enqueue(S, pool[from], from); }
    void getParents(state_id s, Vec<state_id>& out_parents); // -- return genealogy of current node (clearing and populating 'out_parents, 'out_parents[0]' is the initial state)

  //________________________________________
  //  Methods to override:

    virtual void expand(state_id s) = 0;
    virtual void eval  (state_id s) = 0;
    virtual void start () {}        // -- called at start of 'run()'
    virtual void flush () {}        // -- called when queue is empty to give sub-class a chance to populate it before aborting search

    virtual void reportProgress(bool final_call) {}     // -- called periodically to allow for some user output; if 'report_freq == 0', not called at all

    virtual ~HeapSynth() {}
};


struct SimpleSynth : HeapSynth {
    Spec const& spec;
    SimpleSynth(Spec const& spec) : HeapSynth(spec.prog, Params_HeapSynth()), spec(spec) {
        enqueue(initialEnumState(spec.target), spec.pool);
        P.rlim = ResLims(250000, 8388608, 1000); // <<== should be a parameter, but this code is unfinished
    }
    virtual void expand(state_id s) override;
    virtual void eval  (state_id s) override;
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

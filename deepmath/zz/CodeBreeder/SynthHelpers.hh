#ifndef ZZ__CodeBreeder__SynthHelpers_hh
#define ZZ__CodeBreeder__SynthHelpers_hh

#include "SynthEnum.hh"
#include "TrainingData.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


void reportGenealogy(
    uind s,
    Vec<ENUM::State> const& states,
    Vec<Pair<double,uint64>> const& state_costs,
    Vec<uind> const& parent,
    Pool const& pool);

TrainingData genTrainingData(
    uind s,
    Vec<ENUM::State> const& states,
    Vec<uind> const& parent,
    Pool const& pool);

void outputTrainingData(
    String filename,
    uind s,
    Vec<ENUM::State> const& states,
    Vec<uind> const& parent,
    Pool const& pool);

inline Str slice(std::string const& str) {
    return slice(*str.begin(), *str.end()); }   // -- C++11 guarantees strings are contiguous


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

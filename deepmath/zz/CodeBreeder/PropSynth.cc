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
#include "Parser.hh"
#include "Synth.hh"
#include "SynthSpec.hh"


namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


namespace {
struct Repo {
    Vec<Vec<Vec<Expr>>> id2specs;
    Vec<Type>           id2type;
    Map<Type, uint>     type2id;
};
}


void dumpSing(Repo& R, Type ty, Expr const& io_pairs, uint i0, uint i1)
{
}


void dumpPair(Repo& R, Type ty, Expr const& io_pairs, uint i0, uint i1, uint j0, uint j1)
{
    uint* id;
    if (!R.type2id.get(ty, id)){
        *id = R.id2specs.size();
        R.id2specs.push();
        R.id2type.push(ty);
    }
    Vec<Vec<Expr>>& specs = R.id2specs[*id];
    specs.push();

    assert(io_pairs.kind == expr_MkVec);
    for (Expr const& pair : io_pairs){
        assert(Type(a_Tuple, {tupleSlice(pair[i0])[i1].type, tupleSlice(pair[j0])[j1].type}) == ty);
        Expr new_pair = Expr::Tuple({tupleSlice(pair[i0])[i1], tupleSlice(pair[j0])[j1]}).setType(Type(ty));
        specs[LAST].push(new_pair);
    }
}


void createPropertySpecifications(Vec<String> const& spec_files)
{
    Type ty_ints = parseType("List<Int>");
    Repo R;

    // Collect input/output specifications of desired types:
    for (String const& file : spec_files){
        Spec spec = readSpec(file, false);
        assert(spec.target.name == a_Fun);

        Vec<Type> in_types (copy_, tupleSlice(spec.target[0]));
        Vec<Type> out_types(copy_, tupleSlice(spec.target[1]));

        for (uint i = 0; i < in_types.size(); i++)
            if (in_types[i] == ty_ints)
                dumpSing(R, in_types[i], spec.io_pairs, 0, i);

        for (uint j = 0; j < out_types.size(); j++)
            if (out_types[j] == ty_ints)
                dumpSing(R, out_types[j], spec.io_pairs, 1, j);

        for (uint i = 0; i < in_types.size(); i++){
            for (uint j = 0; j < out_types.size(); j++){
                if (in_types[i] == ty_ints || out_types[j] == ty_ints)
                    dumpPair(R, Type(a_Tuple, {in_types[i], out_types[j]}), spec.io_pairs, 0, i, 1, j);
            }
        }
    }

    // Write specification:
    uint threshold = 5;
    uint cc = 0;
    for (uint i = 0; i < R.id2type.size(); i++){
        if (R.id2specs[i].size() >= threshold){
            String filename = fmt("propspec%_.evo", cc++);
            OutFile out(filename);

            wrLn(out, "##include \"std.evo\";");
            wr(out, "let specs = [:[%_]\n        ", R.id2type[i]);
            for (Vec<Expr> const& spec : R.id2specs[i])
                wrLn(out, "    [:%_ %_],", R.id2type[i], join(",\n        ", spec));
            wrLn(out, "];");

            wrLn("Wrote: \a*%_\a*", filename);
        }
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

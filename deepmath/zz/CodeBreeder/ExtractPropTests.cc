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
#include "ExtractPropTests.hh"
#include "zz/Generics/Glob.hh"
#include "zz/Generics/Sort.hh"
#include "SynthEnum.hh"
#include "Vm.hh"

namespace ZZ {
using namespace std;


namespace{
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


struct IoElem {
    uint i0;        // -- 0 for input, 1 for output
    uint i1;        // -- tuple argument
    Type type;
};


String extract(RetVal const& val, IoElem e)
{
    String ret;
    wr(ret, "[:%_ ", e.type);
    for (uint i = 0; i < val.size(); i++)
        wr(ret, "%_, ", val[i](e.i0)(e.i1));
    wr(ret, "]");
    return ret;
}


String extract(RetVal const& val, IoElem e0, IoElem e1)
{
    String ret;
    wr(ret, "[:(%_, %_) ", e0.type, e1.type);
    for (uint i = 0; i < val.size(); i++)
        wr(ret, "(%_, %_), ", val[i](e0.i0)(e0.i1), val[i](e1.i0)(e1.i1));
    wr(ret, "]");
    return ret;

    return String();
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}


void summarizeTargets(String input_glob)
{
    Vec<String> evo_files = getMatchingFiles("", input_glob);
    if (evo_files.size() == 0){
        wrLn("ERROR! No files matching: %_", input_glob); exit(1); }

    Map<Type,uint> count;
    for (String const& file : evo_files){
        Spec spec = readSpec(file, false);

        uint* result;
        if (!count.get(spec.target, result))
            *result = 1;
        else
            *result += 1;
    }

    Vec<Pair<uint,Type>> pairs;
    For_Map(count){
        Type  key = Map_Key(count);
        uint  val = Map_Value(count);
        pairs.push(tuple(val, key));
    }
    sort_reverse(pairs);

    wrLn("COUNT              TYPE");
    wrLn("----------------   ------------------------------------------------------------");
    for (auto&& p : pairs)
        wrLn("%>16%,d   %_", p.fst, p.snd);
}


void writePropTestFile(String input_glob, String output_filename, uint min_patterns_per_type)
{
    Vec<String> evo_files = getMatchingFiles("", input_glob);
    if (evo_files.size() == 0){
        wrLn("ERROR! No files matching: %_", input_glob); exit(1); }

    Vec<Type>        types;
    Vec<Vec<String>> patterns;
    Map<Type,uint>   type2pat;

    auto getPat = [&](Type const& t) -> Vec<String>& {
        uint* idx;
        if (!type2pat.get(t, idx)){
            *idx = patterns.size();
            types.push(t);
            patterns.push(); }
        return patterns[*idx];
    };

    for (String const& file : evo_files){
        // Read and evaluate file:
        wrLn("Processing: \a*%_\a*", file);
        Spec spec = readSpec(file, false);
        assert(spec.target.name == a_Fun);

        RunTime rt;
        addr_t ret = rt.run(spec.prog); assert(ret);
        ret = rt.run(spec.io_pairs[0]); assert(ret);
        RetVal val(rt, ret, spec.io_pairs[0].type);     // -- 'val' is now the 'io_pairs' vector

        // Construct property input tuples:
        Array<Type const> ins  = tupleSlice(spec.target[0]);
        Array<Type const> outs = tupleSlice(spec.target[1]);

        Vec<IoElem> elems;
        for (uint i = 0; i < ins .size(); i++) elems.push(IoElem{0, i, ins [i]});
        for (uint i = 0; i < outs.size(); i++) elems.push(IoElem{1, i, outs[i]});

        for (uint i = 0; i < elems.size(); i++)     // -- one-input properties
            getPat(elems[i].type).push(extract(val, elems[i]));

        for (uint i = 0; i < elems.size()-1; i++){  // -- two-input properties
            for (uint j = i+1; j < elems.size(); j++){
                IoElem e0 = elems[i];
                IoElem e1 = elems[j];
                if (e0.type > e1.type) swp(e0, e1);
                getPat(Type(a_Tuple, {Type(e0.type), Type(e1.type)})).push(extract(val, e0, e1));
            }
        }
    }

    // Write patterns:
    newLn();
    wrLn("SUMMARY:");
    wrLn("    #patterns:   type:");
    for (uint i = 0; i < patterns.size(); i++){
        if (patterns[i].size() < min_patterns_per_type) continue;
        wrLn("%>14%,d   %_", patterns[i].size(), types[i]);
    }

    if (output_filename != ""){
        OutFile out(output_filename);
        if (!out){ wrLn("ERROR! Could not open file: %_", output_filename); exit(1); }

        // Build segments:
        Vec<String> data_seg;
        Vec<String> test_seg;
        for (uint i = 0; i < patterns.size(); i++){
            if (patterns[i].size() < min_patterns_per_type) continue;

            sortUnique(patterns[i]);    // <<== use size, then lex order...
            data_seg.push(fmt("    ##if (match<T, %_>, [:[%_]\n", types[i], types[i]));
            for (String const& pat : patterns[i])
                append(data_seg[LAST], fmt("        %_,\n", pat));
            append(data_seg[LAST], fmt("    ],\n"));

            cchar* norm_fmt = "        vec_map_append_(result, prop_tests<%_>, \\inputs{ for_all_(inputs, p_); });\n";
            cchar* swap_fmt = "        vec_map_append_(result, prop_tests<%_>, \\inputs{ for_all_(inputs, \\(x,y){ p_(y,x) }); });\n";

            assert(types[i].name != a_Tuple || types[i].size() == 2);

            test_seg.push(fmt("    ##if (match<IN, %_>, {\n", types[i]));
            if (types[i].name == a_Tuple && types[i][0] == types[i][1]){
                append(test_seg[LAST], fmt(norm_fmt, types[i]));
                append(test_seg[LAST], fmt(swap_fmt, types[i]));
            }else
                append(test_seg[LAST], fmt(norm_fmt, types[i]));
            append(test_seg[LAST], fmt("    },\n"));

            if (types[i].name == a_Tuple && types[i][0] != types[i][1]){
                test_seg.push(fmt("    ##if (match<IN, %_>, {\n", Type(a_Tuple, Type(types[i][1]), Type(types[i][0]))));
                append(test_seg[LAST], fmt(swap_fmt, types[i]));
                append(test_seg[LAST], fmt("    },\n"));
            }
        }

        // Write data:
        wrLn(out, "##include \"std.evo\";\n");

        wrLn(out, "rec prop_tests<T> : [[T]] =");
        wr(out, "% _", data_seg);
        wrLn(out, "        ##error \"No property test-patterns defined for type\"");
        wrLn(out, "    %_;\n", Vec<char>(data_seg.size(), ')'));


        wrLn(out, "// Run property 'p_' on extracted test patterns and return a vector of booleans.");
        wrLn(out, "fun test_prop_<IN>(p_ :IN->Bool) -> [Bool] {");
        wrLn(out, "    let result = [:Bool];");
        wr(out, "% _", test_seg);
        wrLn(out, "        ##error \"No property tester defined for type\"");
        wrLn(out, "    %_;", Vec<char>(test_seg.size(), ')'));
        wrLn(out, "    result;");
        wrLn(out, "};\n");

        // Write some boilerplate:
        wrLn(out, "fun summarize_prop_<IN>(p_ :IN->Bool) -> (Int, Int) {");
        wrLn(out, "    let c0 = &0;");
        wrLn(out, "    let c1 = &0;");
        wrLn(out, "    apply_all_(test_prop_(p_), \\res{");
        wrLn(out, "        if_else_(res, `c0 += 1, `c1 += 1); });");
        wrLn(out, "    (^c0, ^c1);");
        wrLn(out, "};");

        out.close();

        newLn();
        wrLn("Wrote: %\a*%_\a*", output_filename);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

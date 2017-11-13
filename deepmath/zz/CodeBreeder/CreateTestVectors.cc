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
#include "CreateTestVectors.hh"
#include "zz/Generics/Set.hh"
#include "zz/Generics/Map.hh"
#include "Parser.hh"

/*
Creates input test-vectors for the basic types:
  Bool
  Int
  Int -> Bool
  Int -> Int
  (Int, Int) -> Int
  List<Int> -> List<Int>

and composite types:
    List<A>
    (A, B)
    (A, B, C)

Each basic type is subdivided into categories (small ints, big ints, negative ints, short lists
etc.). When creating tuples, first categories are combined, then elements of those categories.
*/


namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


struct TestGen {
    virtual Array<double const> catWeights() const = 0;
        // -- higher weights means proportionally more samples from that category will be used.

    virtual String generate(uint64& seed, uint cat) const = 0;
        // -- generate a sample from category 'cat'

    virtual ~TestGen() {}
};

// From now on, a generator 'Gen' is a shared pointer to a constant TestGen...
namespace { typedef shared_ptr<TestGen const> Gen; }


function<String()> getMixer(uint64 seed, Gen gen, Vec<uint> cats)
{
    Vec<double> weights = map(cats, [&](uint i){ return gen->catWeights()[i]; });
    double scale = 1.0 / foldl(weights, -std::numeric_limits<double>::infinity(), max_<double>);
    for (double& w : weights) w *= scale;

    Arr<uint>   cs(cats);
    Arr<double> ws(weights);
    return [=]() mutable {
        // Inefficient, but will do.
        for(;;){
            uint i = irand(seed, cs.size());
            if (drand(seed) < ws[i]) return fmt("%_", gen->generate(seed, cs[i]));
        }
    };
}


inline function<String()> getMixer(uint64 seed, Gen gen) {
    return getMixer(seed, gen, enumTo<uint>(gen->catWeights().size())); }


//=================================================================================================
// -- Bool:


struct TestGen_Bool : TestGen {
    Array<double const> catWeights() const override {
        static double const weights[] = { 1.0 };
        return Array_new(weights, elemsof(weights)); }

    String generate(uint64& seed, uint cat) const override {
        assert(cat == 0);
        return (irand(seed, 2) == 0) ? "_0" : "_1"; }
};


//=================================================================================================
// -- Int:


struct TestGen_Int : TestGen {
    Array<double const> catWeights() const override {
        // -- small, big, negative, pathological
        static double const weights[] = { 4, 2, 2, 1 };
        return Array_new(weights, elemsof(weights)); }

    String generate(uint64& seed, uint cat) const override;
};


template<int64 k=5>
int64 logPick(uint64& seed, int64 lim)
{
  #if defined(GOOGLE_CODE)
    const double logk = log(k);
  #else
    constexpr double logk = log(k);
  #endif
    assert(lim > 0);
    assert(lim < INT64_MAX);
    double r = drand(seed);
    int64 ret = int64(exp(r * (log(lim+k) - logk) + logk) - k);
    assert(ret >= 0);
    assert(ret < lim);
    return ret;
}


String TestGen_Int::generate(uint64& seed, uint cat) const
{
    auto helper = [&]() -> int64 {
        switch (cat){
        case 0: return logPick(seed, 20);
        case 1: return logPick(seed, 1ll << 30);
        case 2: return ~logPick<1>(seed, 1ll << 30);
        case 3:{
            switch (irand(seed, 6)){
            case 0: case 1: return INT64_MAX - logPick(seed, 5);
            case 2: case 3: return INT64_MIN + logPick(seed, 5);
            case 4:         return (INT64_MAX >> 1) + logPick(seed, 11) - 5;
            case 5:         return (INT64_MIN >> 1) + logPick(seed, 11) - 5;
            default: assert(false); }
        }
        default: assert(false); }
    };
    return fmt("%_", helper());
}


//=================================================================================================
// -- List<A>:


// Half of the lists will be from a single category of 'a_gen', half will be mixed.
struct TestGen_List : TestGen {
    Gen         a_gen;
    Arr<double> weights;
    String      type_name;
    mutable function<String()> a_mix;       // -- initialized at first call to 'generate()'

    TestGen_List(Gen a, String const& type_name);

    Array<double const> catWeights() const override { return weights.slice(); }
    String generate(uint64& seed, uint cat) const override;
};


TestGen_List::TestGen_List(Gen a, String const& type_name) :
    a_gen(a),
    type_name(type_name)
{
    Vec<double> ws;
    double sum = 0;
    for (double w : a_gen->catWeights()){
        sum += w;
        ws.push(w);     // -- single category
    }
    ws.push(sum);       // -- fully mixed category

    weights = Arr<double>(ws);
}


String TestGen_List::generate(uint64& seed, uint cat) const
{
    if (!a_mix)
        a_mix = getMixer(splitSeed(seed), a_gen);

    uint len = logPick<1>(seed, 50);
    Vec<String> result;
    if (cat == a_gen->catWeights().size()){
        for (uint i = 0; i < len; i++)
            result.push(a_mix());
    }else{
        for (uint i = 0; i < len; i++)
            result.push(a_gen->generate(seed, cat));
    }
    return fmt("list_[:%_ %_]", type_name, join(", ", result));
}


//=================================================================================================
// -- (A, B):


struct TestGen_Pair : TestGen {
    Gen         a_gen;
    Gen         b_gen;
    Arr<double> weights;

    TestGen_Pair(Gen a, Gen b);

    Array<double const> catWeights() const override { return weights.slice(); }
    String generate(uint64& seed, uint cat) const override;
};


TestGen_Pair::TestGen_Pair(Gen a, Gen b) :
    a_gen(a),
    b_gen(b)
{
    Vec<double> ws;
    for (double wa : a_gen->catWeights())
    for (double wb : b_gen->catWeights())
        ws.push(wa * wb);
    weights = Arr<double>(ws);
}


String TestGen_Pair::generate(uint64& seed, uint cat) const
{
    uint nb = b_gen->catWeights().size();
    return fmt("(%_, %_)", a_gen->generate(seed, cat / nb), b_gen->generate(seed, cat % nb));
}


//=================================================================================================
// -- (A, B, C):


struct TestGen_Trip : TestGen {
    Gen         a_gen;
    Gen         b_gen;
    Gen         c_gen;
    Arr<double> weights;

    TestGen_Trip(Gen a, Gen b, Gen c);

    Array<double const> catWeights() const override { return weights.slice(); }
    String generate(uint64& seed, uint cat) const override;
};


TestGen_Trip::TestGen_Trip(Gen a, Gen b, Gen c) :
    a_gen(a),
    b_gen(b),
    c_gen(c)
{
    Vec<double> ws;
    for (double wa : a_gen->catWeights())
    for (double wb : b_gen->catWeights())
    for (double wc : c_gen->catWeights())
        ws.push(wa * wb * wc);
    weights = Arr<double>(ws);
}


String TestGen_Trip::generate(uint64& seed, uint cat) const
{
    uint c = cat % c_gen->catWeights().size(); cat /= c_gen->catWeights().size();
    uint b = cat % b_gen->catWeights().size(); cat /= b_gen->catWeights().size();
    uint a = cat; assert(a < a_gen->catWeights().size());
    return fmt("(%_, %_, %_)", a_gen->generate(seed, a), b_gen->generate(seed, b), c_gen->generate(seed, c));
}


//=================================================================================================
// -- Finite set:


struct TestGen_FiniteSet : TestGen {
    Vec<String> set;
    TestGen_FiniteSet(Vec<String> const& set0) : set(copy_, set0) {}
    template<size_t sz> TestGen_FiniteSet(cchar* (&set0) [sz]) : set(copy_, Array_new(set0, elemsof(set0))) {}

    Array<double const> catWeights() const override {
        static double const weights[] = { 1.0 };
        return Array_new(weights, elemsof(weights)); }

    String generate(uint64& seed, uint cat) const override {
        assert(cat == 0);
        return set[irand(seed, set.size())]; }
};


static cchar* funs_Int_to_Bool[] = {
    "is_zero_",
    "is_even_",
    "is_pow2_",
    "is_nonzero_",
};


static cchar* funs_Int_to_Int[] = {
    "rand_unop_",
    "dec_",
    "inc_",
    "sq_",
    "dbl_",
    "abs_",
    "sign_",
    "neg_",
    "inv_",
};


static cchar* funs_IntInt_to_Int[] = {
    "rand_binop_",
    "add_",
    "sub_",
    "mul_",
    "div_",
    "mod_",
    "lshift_",
    "rshift_",
    "urshift_",
    "bit_and_",
    "bit_or_",
    "bit_xor_",
};


static cchar* funs_Ints_to_Ints[] = {
    "list_reverse_<Int>",
    "list_rotate_<Int>",
    "tail_<Int>",
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Main:


class TestGenMgr {
    Map<Type, Gen> type2gen;
public:
    Gen getGen(Type const& type);
};


Gen TestGenMgr::getGen(Type const& type)
{
    Gen* val;
    if (!type2gen.getI(type, val)){
        if (type == Type(a_Bool))
            *val = make_shared<TestGen_Bool>();
        else if (type == Type(a_Int))
            *val = make_shared<TestGen_Int>();
        else if (type.name == a_List)
            *val = make_shared<TestGen_List>(getGen(type[0]), fmt("%_", type[0]));
        else if (type == Type(a_Fun, Type(a_Int), Type(a_Bool)))
            *val = make_shared<TestGen_FiniteSet>(funs_Int_to_Bool);
        else if (type == Type(a_Fun, Type(a_Int), Type(a_Int)))
            *val = make_shared<TestGen_FiniteSet>(funs_Int_to_Int);
        else if (type == Type(a_Fun, Type(a_Tuple, Type(a_Int), Type(a_Int)), Type(a_Int)))
            *val = make_shared<TestGen_FiniteSet>(funs_IntInt_to_Int);
        else if (type == Type(a_Fun, Type(a_List, Type(a_Int)), Type(a_List, Type(a_Int))))
            *val = make_shared<TestGen_FiniteSet>(funs_Ints_to_Ints);
        else if (type.name == a_Tuple){
            if (type.size() == 2)
                *val = make_shared<TestGen_Pair>(getGen(type[0]), getGen(type[1]));
            else if (type.size() == 3)
                *val = make_shared<TestGen_Trip>(getGen(type[0]), getGen(type[1]), getGen(type[2]));
            else{
                wrLn("ERROR! Cannot handle tuples of size > 3 in testpattern generation.");
                exit(1);
            }
        }else{
            wrLn("ERROR! No testpattern generator defined for: %_", type);
            exit(1);
        }
    }
    return *val;
}


static
Vec<String> genN(function<String()>& mix, uint n_patterns, uint64& seed)
{
    Set<String> result;
    uint n_tries = 0;
    while (result.size() < n_patterns && n_tries < 1000){
        if (result.add(mix()))
            n_tries++;      // -- test-pattern already exists
        else
            n_tries = 0;
    }

    Vec<String> vec;
    For_Set(result){ vec.push(Set_Key(result)); }
    shuffle(seed, vec);

    return vec;
}


static
void writeTestVectors(Out& out, Type const& type, function<String()>& mix, uint n_patterns, uint64& seed)
{
    wrLn("##if (match<T, %_>,", type);
    wrLn(out, "    [:%_", type);
    for (String const& text : genN(mix, n_patterns, seed))
        wrLn(out, "        %_,", text);
    wrLn(out, "    ],");
}


static
void addSubtypes(Vec<Type>& types, Type const& t)
{
    if (!has(types, t)) types.push(t);
    for (Type const& s : t)
        addSubtypes(types, s);
}


// 'cat' is just used for debugging test-pattern generation. 'closed_set' means add subtypes of 'types' to itself.
void genTestVectors(Out& out, String var_name, uint n_patterns, uint64 seed, uint cat, Vec<Type> const& types0, bool closed_set)
{
    Vec<Type> types(copy_, types0);
    if (closed_set)
        for (Type const& t : types0)
            addSubtypes(types, t);

    TestGenMgr mgr;

    wrLn(out, "##include \"std.evo\";\n");
    wrLn(out, "rec %_<T> : [T] =", var_name);
    for (Type const& type : types){
        auto mix = (cat == UINT_MAX) ? getMixer(seed, mgr.getGen(type)) : getMixer(seed, mgr.getGen(type), {cat});
        writeTestVectors(out, type, mix, n_patterns, seed);
    }
    wrLn(out, "##error :T \"No test pattern defined for type\"%)r;", types.size());

    // Instantiate templates for incremental compilation:
    newLn(out);
    for (Type const& type : types)
        wrLn(out, "%_<%_>;", var_name, type);
}


void genTestVectors(Out& out, String var_name, uint n_patterns, uint64 seed, uint cat, String types_filename, bool closed_set)
{
    Str text = readFile(types_filename);
    if (!text){ wrLn("ERROR! Could not read file: %_", types_filename); exit(1); }
    Vec<Str> fs;
    strictSplitArray(text, ";", fs);
    Vec<Type> types;
    for (Str type_text : fs){
        try{
            String text(type_text);
            trim(text);
            if (text.size() > 0)
                types.push(parseType(text.c_str()));
        }catch (Excp_ParseError err){
            wrLn("PARSE ERROR! %_", err.msg); exit(1);
        }
    }
    dispose(text);

    genTestVectors(out, var_name, n_patterns, seed, cat, types, closed_set);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

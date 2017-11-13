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
#include "Vm.hh"
#include "zz/Generics/Map.hh"
#include "zz/Generics/Sort.hh"
#include "Parser.hh"
#include "Expand.hh"
#include "Compile.hh"
#include "Instr.hh"

namespace ZZ {
using namespace std;
using namespace VM;


ZZ_PTimer_Add(vm_run);
ZZ_PTimer_Add(vm_gc);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


bool stress_gc = getenv("STRESS_GC");       // -- for debugging, stress test the garbage collection


Atom unquoteUnescape(Atom atom_)
{
    Str atom(atom_);
    assert(atom[0] == '\"' && atom[LAST] == '\"');

    Vec<char> ret;
    for (uint i = 1; i < atom.size() - 1; i++){
        if (atom[i] != '\\')
            ret.push(atom[i]);
        else{
            switch (atom[i+1]){
            case 'a': ret.push('\a'); break;
            case 'b': ret.push('\b'); break;
            case 't': ret.push('\t'); break;
            case 'n': ret.push('\n'); break;
            case 'v': ret.push('\v'); break;
            case 'f': ret.push('\f'); break;
            case 'r': ret.push('\r'); break;
            // <<== \nnn for octal, \xNN for hex  (if 2nd or 3rd 'n' is outside 0-7, stop, same for 'N' and hex)
            default: ret.push(atom[i+1]); }
            i++;
        }
    }
    ret.push(0);
    return Atom(ret.base());
}


namespace VM{
    Atom a_normal;
    Atom a_cpu_lim;
    Atom a_mem_lim;
    Atom a_rec_lim;
    Atom a_null_deref;
    Atom a_vec_op;
    Atom a_vop_get;
    Atom a_vop_set;
    Atom a_vop_growby;
    Atom a_vop_resize;

    Atom a_exit_code_type;
}

ZZ_Initializer(exit_codes, 10) {
    a_normal     = Atom("normal");
    a_cpu_lim    = Atom("cpu_lim");
    a_mem_lim    = Atom("mem_lim");
    a_rec_lim    = Atom("rec_lim");
    a_null_deref = Atom("null_deref");
    a_vec_op     = Atom("vec_op");
    a_vop_get    = Atom("get");
    a_vop_set    = Atom("set");
    a_vop_growby = Atom("growby");
    a_vop_resize = Atom("resize");

    a_exit_code_type = Atom(fmt("%_", parseType(exit_code_type)));   // -- parse and pretty-print to normalize
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Builtins:


// These are simple builtins. They cannot create references or modify the shape of the data segment.
typedef void (Builtin)(Word* args, Word* ret);


struct RegisterBin {
    cchar*   fun_name;
    cchar*   fun_type;
    Builtin* fun_ptr;
    RegisterBin* next;
    RegisterBin(cchar* fun_name, cchar* fun_type, Builtin* fun_ptr);
};

static RegisterBin* registered_bins = nullptr;
RegisterBin::RegisterBin(cchar* fun_name, cchar* fun_type, Builtin* fun_ptr) :
    fun_name(fun_name), fun_type(fun_type), fun_ptr(fun_ptr), next(registered_bins) { registered_bins = this; }


#define Bin_Define(fun_name, fun_type) \
    static void BINFUNC_##fun_name (Word*, Word*) ___aligned(4); \
    static void BINFUNC_##fun_name (Word* ___unused arg, Word* ___unused ret); \
    RegisterBin ___unused BINFUNC_##fun_name##_instance(#fun_name, fun_type, BINFUNC_##fun_name); \
    static void BINFUNC_##fun_name (Word* ___unused arg, Word* ___unused ret)


Bin_Define(neg_, "Int -> Int") { ret[0].val = -arg[0].val; }
Bin_Define(inv_, "Int -> Int") { ret[0].val = ~arg[0].val; }

Bin_Define(add_, "(Int, Int) -> Int") { ret[0].val = arg[0].val + arg[1].val; }
Bin_Define(sub_, "(Int, Int) -> Int") { ret[0].val = arg[0].val - arg[1].val; }
Bin_Define(mul_, "(Int, Int) -> Int") { ret[0].val = arg[0].val * arg[1].val; }
//Bin_Define(div_, "(Int, Int) -> Int") { ret[0].val = arg[0].val / arg[1].val; }
//Bin_Define(mod_, "(Int, Int) -> Int") { ret[0].val = arg[0].val % arg[1].val; }
Bin_Define(div_, "(Int, Int) -> Int") { ret[0].val = (arg[1].val == 0 || (arg[0].val == INT64_MIN && arg[1].val == -1)) ? 0 : arg[0].val / arg[1].val; } // <<== better to throw exception in Evo
Bin_Define(mod_, "(Int, Int) -> Int") { ret[0].val = (arg[1].val == 0 || (arg[0].val == INT64_MIN && arg[1].val == -1)) ? 0 : arg[0].val % arg[1].val; } // <<== better to throw exception in Evo
Bin_Define(lshift_  , "(Int, Int) -> Int") { ret[0].val = arg[0].val << (arg[1].val & 63); }
Bin_Define(rshift_  , "(Int, Int) -> Int") { ret[0].val = arg[0].val >> (arg[1].val & 63); }
Bin_Define(urshift_ , "(Int, Int) -> Int") { ret[0].val = uint64(arg[0].val) >> (arg[1].val & 63); }
Bin_Define(bit_and_ , "(Int, Int) -> Int") { ret[0].val = arg[0].val & arg[1].val; }
Bin_Define(bit_or_  , "(Int, Int) -> Int") { ret[0].val = arg[0].val | arg[1].val; }
Bin_Define(bit_xor_ , "(Int, Int) -> Int") { ret[0].val = arg[0].val ^ arg[1].val; }
Bin_Define(abs_     , "Int -> Int") { ret[0].val = abs(arg[0].val); }
Bin_Define(sign_    , "Int -> Int") { ret[0].val = (arg[0].val > 0) ? 1 : (arg[0].val < 0) ? -1 : 0; }
Bin_Define(hmix_    , "(Int, Int) -> Int") { ret[0].val = shuffleHash(uint64(arg[0].val) ^ (uint64(arg[1].val) * 602923773562679ull)); }

Bin_Define(f_neg_, "Float -> Float")          { ret[0].flt = -arg[0].flt; }
Bin_Define(f_add_, "(Float, Float) -> Float") { ret[0].flt = arg[0].flt + arg[1].flt; }
Bin_Define(f_sub_, "(Float, Float) -> Float") { ret[0].flt = arg[0].flt - arg[1].flt; }
Bin_Define(f_mul_, "(Float, Float) -> Float") { ret[0].flt = arg[0].flt * arg[1].flt; }
Bin_Define(f_div_, "(Float, Float) -> Float") { ret[0].flt = arg[0].flt / arg[1].flt; }
Bin_Define(f_mod_, "(Float, Float) -> Float") { ret[0].flt = fmod(arg[0].flt, arg[1].flt); }
Bin_Define(f_abs_, "Float -> Float") { ret[0].flt = fabs(arg[0].flt); }
Bin_Define(f_log_, "Float -> Float") { ret[0].flt = log(arg[0].flt); }

Bin_Define(ge_, "(Int, Int) -> Bool") { ret[0].val = arg[0].val >= arg[1].val; }
Bin_Define(gt_, "(Int, Int) -> Bool") { ret[0].val = arg[0].val >  arg[1].val; }
Bin_Define(le_, "(Int, Int) -> Bool") { ret[0].val = arg[0].val <= arg[1].val; }
Bin_Define(lt_, "(Int, Int) -> Bool") { ret[0].val = arg[0].val <  arg[1].val; }
Bin_Define(eq_, "(Int, Int) -> Bool") { ret[0].val = arg[0].val == arg[1].val; }
Bin_Define(ne_, "(Int, Int) -> Bool") { ret[0].val = arg[0].val != arg[1].val; }

Bin_Define(f_ge_, "(Float, Float) -> Bool") { ret[0].val = arg[0].flt >= arg[1].flt; }
Bin_Define(f_gt_, "(Float, Float) -> Bool") { ret[0].val = arg[0].flt >  arg[1].flt; }
Bin_Define(f_le_, "(Float, Float) -> Bool") { ret[0].val = arg[0].flt <= arg[1].flt; }
Bin_Define(f_lt_, "(Float, Float) -> Bool") { ret[0].val = arg[0].flt <  arg[1].flt; }
Bin_Define(f_eq_, "(Float, Float) -> Bool") { ret[0].val = arg[0].flt == arg[1].flt; }
Bin_Define(f_ne_, "(Float, Float) -> Bool") { ret[0].val = arg[0].flt != arg[1].flt; }
Bin_Define(f_hash_, "Float -> Int") { ret[0].val = arg[0].val; }
Bin_Define(f_bit_eq_, "(Float, Float) -> Bool") { ret[0].val = arg[0].val == arg[1].val; }     // -- must be used together with hash

Bin_Define(a_ge_, "(Atom, Atom) -> Bool") { ret[0].val = strcmp(Atom(arg[0].val).c_str(), Atom(arg[1].val).c_str()) >= 0; }
Bin_Define(a_gt_, "(Atom, Atom) -> Bool") { ret[0].val = strcmp(Atom(arg[0].val).c_str(), Atom(arg[1].val).c_str()) >  0; }
Bin_Define(a_le_, "(Atom, Atom) -> Bool") { ret[0].val = strcmp(Atom(arg[0].val).c_str(), Atom(arg[1].val).c_str()) <= 0; }
Bin_Define(a_lt_, "(Atom, Atom) -> Bool") { ret[0].val = strcmp(Atom(arg[0].val).c_str(), Atom(arg[1].val).c_str()) <  0; }
Bin_Define(a_eq_, "(Atom, Atom) -> Bool") { ret[0].val = arg[0].val == arg[1].val; }
Bin_Define(a_ne_, "(Atom, Atom) -> Bool") { ret[0].val = arg[0].val != arg[1].val; }
Bin_Define(a_hash_, "Atom -> Int") { ret[0].val = arg[0].val; }
Bin_Define(a_size_, "Atom -> Int") { ret[0].val = Str(Atom(arg[0].val)).size(); }
Bin_Define(a_get_ , "(Atom, Int) -> Int") { Str text = Str(Atom(arg[0].val)); uint64 idx = arg[1].val; ret[0].val = (idx < text.size()) ? text[idx] : 0; }

Bin_Define(int_to_flt_, "Int -> Float") { ret[0].flt = arg[0].val; }
Bin_Define(flt_to_int_, "Float -> Int") { ret[0].val = arg[0].flt; }

Bin_Define(irand_, "Int -> (Int, Int)") { uint64 seed = arg[0].val; ret[0].val = int64(irandl(seed) & 0x7FFFFFFFFFFFFFFFull); ret[1].val = seed; }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Virtual Machine:


struct VmLimits {
    uint64 cpu;
    addr_t mem;
    uint   rec;
    VmLimits(uint64 cpu = 0, addr_t mem = 0, uint rec = 0) : cpu(cpu), mem(mem), rec(rec) {}
};

template<> fts_macro void write_(Out& out, VmLimits const& v) {
    wr(out, "VmLimits{cpu=%_; mem=%_; rec=%_}", v.cpu, v.mem, v.rec); }


struct VmExcp {
    Atom     type_text = Atom();
    uint     frame     = 0;
    addr_t   closure   = 0;
    addr_t   ret       = 0;
    uint     ret_sz    = 0;
    addr_t   arg       = 0;
    uint     arg_sz    = 0;
    VmLimits lim;       // -- resource limits are handled as a special try/catch with 'type_text == "run_"'
};

template<> fts_macro void write_(Out& out, VmExcp const& v){
    wr(out, "VmExcp{type_text=%_; frame=%_; closure=%_; ret=%_; ret_sz=%_; arg=%_; arg_sz=%_; lim=%_}",
        v.type_text, v.frame, v.closure, v.ret, v.ret_sz, v.arg, v.arg_sz, v.lim); }


class Vm {
    Expander    expander;       // -- a bit unclean, but the virtual machine keep track of datadefs and rec-defs for compilation

    Vec<Instr>  code;           // -- "byte" code
    Vec<Word>   data;           // -- beginning is persistent data followed by heap, end is stack (growing downwards)
    Vec<WTag>   tag;            // -- same size as 'data';
    addr_t      pp = 0;         // -- persistent pointer (end of persistent data)
    addr_t      dp = 0;         // -- dynamic-data start pointer (start of heap data)
    addr_t      hp = 0;         // -- heap pointer (end of heap)
    addr_t      rel[5]{0};      // -- {zero, retval, args, local, closure} pointer
    Vec<addr_t> icall;          // -- instruction call stack (stores previous instruction pointer)
    Vec<addr_t> dcall;          // -- data call stack (stores previous {retval, args, local, closure} pointers)
    Vec<VmExcp> excp_stack;     // -- exception stack, added to by 'i_TRY' and consumed by 'i_THROW'
    RelPos      result;         // -- top-level result will be stored here (set by 'compile()' used by 'run()')
    addr_t      mem_used = 0;   // -- memory usage is measured during GC; this variable reflects the latest measurement
    int64       gc_count = 0;   // -- counts down for each step and each allocation; forces a GC when below zero

    SymTable<RelPos> tab0;      // -- base symbol table; will store built-in functions and program literals (maps literal text to address in 'data')

    Vec<Trip<uind,uind,uind>> undo;

    // Internal methods:
    addr_t  absPos(RelPos pos) { return rel[pos.base] + pos.offset; }
    void    pushLit(Word w);
    void    callClosure(addr_t* ip, uint n_words, addr_t clos_ptr, addr_t ret, addr_t arg);     // -- used by 'run()'
    void    dumpAddr(addr_t i, Out& out = std_out);

    void    gcMark(addr_t ref0);
    void    gc(uint failed_alloc_sz, bool persist, Vec<addr_t>* roots = nullptr);
                // -- if 'persist' is TRUE, then we ran out of persistent space, otherwise dynamic (heap/stack) space.
                // 'roots' is a list of external roots that should be marked and preserved; the vector will be updated to reflect new addresses.

    void    clear(addr_t addr, uint n);
    void    globAlloc(uint n);
    void    stackAlloc(uint n);
    void    heapAlloc(uint n, RelPos dst_);
    bool    vecResize(uint elem_sz, RelPos vec_, int64 new_sz, int64 mem_lim);
    bool    throwExcp(Atom type_text, Word const* src_data, WTag const* src_tag, uint n, addr_t& ip);

    void    storeLiteralsInData(Expr const& expr);

public:
    Vm();
    addr_t  compile(Expr prog, bool just_try);
    addr_t  run(addr_t start_addr, ResLims const& lim);

    void    push();
    void    pop ();

    Word const& readData(addr_t idx) { return data[idx]; }
    WTag const& readTag (addr_t idx) { return tag [idx]; }

    void    dumpState(Out& out = std_out);
    void    dumpData (Out& out = std_out);
    void    dumpCode (Out& out = std_out) { ::ZZ::dumpCode(code, out); }

    Out* out = &std_out;        // -- change this to send program output to another stream

    // Statistics for last call to 'run()'
    double gc_time = 0;
    uint64 steps = 0;
};


void Vm::push() {
    undo.push(tuple(code.size(), pp, tab0.size()));
    expander.push();
}


void Vm::pop ()
{
    code.shrinkTo(undo.last().fst);
    pp = undo.last().snd;
    while (tab0.size() > undo.last().trd) tab0.pop();
    undo.pop();
    expander.pop();
}


Vm::Vm()
{
    static_assert(sizeof(Instr) == 4, "Instr should be 4 bytes.");
    static_assert(sizeof(Word) == 8, "Word should be 8 bytes.");

    auto pushB = [&] (Word w) { data.push(w); tag.push(w_Val); };

    auto registerBin = [&](cchar* name, Builtin* func, cchar* type) {
        tab0.addQ(Atom(name), RelPos(b_ZERO, data.size()));
        pushB(Word((void*)func));   // }- closure
        pushB(Word());              // }
    };

    pushB(Word());   // -- reserve null address
    pushB(Word());   // -- reserve address 1 for 'vecResize()'

    for (RegisterBin* r = registered_bins; r; r = r->next)
        registerBin(r->fun_name, r->fun_ptr, r->fun_type);

    // Add compiler handled symbols:
    tab0.addQ(a_case  , RelPos(b_INTERNAL, 0));  // <<== BI
    tab0.addQ(a_ite   , RelPos(b_INTERNAL, 1));
    tab0.addQ(a_assign, RelPos(b_INTERNAL, 2));
    tab0.addQ(a_fail  , RelPos(b_INTERNAL, 3));
    tab0.addQ(a_write , RelPos(b_INTERNAL, 4));
    tab0.addQ(a_run   , RelPos(b_INTERNAL, 5));

    tab0.addQ(a_size  , RelPos(b_INTERNAL, 6));
    tab0.addQ(a_get   , RelPos(b_INTERNAL, 7));
    tab0.addQ(a_growby, RelPos(b_INTERNAL, 8));
    tab0.addQ(a_resize, RelPos(b_INTERNAL, 9));
    tab0.addQ(a_set   , RelPos(b_INTERNAL, 10));

    tab0.addQ(a_print_bool , RelPos(b_INTERNAL, 11));
    tab0.addQ(a_print_int  , RelPos(b_INTERNAL, 12));
    tab0.addQ(a_print_float, RelPos(b_INTERNAL, 13));
    tab0.addQ(a_print_atom , RelPos(b_INTERNAL, 14));

    tab0.addQ(a_try   , RelPos(b_INTERNAL, 15));
    tab0.addQ(a_throw , RelPos(b_INTERNAL, 16));
    tab0.addQ(a_ttry  , RelPos(b_INTERNAL, 17));
    tab0.addQ(a_tthrow, RelPos(b_INTERNAL, 18));
    tab0.addQ(a_block , RelPos(b_INTERNAL, 19));
    tab0.addQ(a_break , RelPos(b_INTERNAL, 20));

    tab0.addQ(a_line , RelPos(b_INTERNAL, 21));
    tab0.addQ(a_file , RelPos(b_INTERNAL, 22));

    // Set up data pointers:
    addr_t init_size = data.size() + (stress_gc ? 10 : 10000);

    memClear(rel);
    pp = data.size();
    dp = pp + init_size / 8;
    hp = dp;
    data.growTo(dp + init_size);
    tag .growTo(data.size(), w_Val);
    rel[b_LOCAL] = data.size();
}


// Push literals into persistent data area:
void Vm::pushLit(Word w)
{
    if (pp >= dp) gc(1, true);
    assert(pp < dp);
    data[pp] = w;
    tag [pp] = w_Val;
    pp++;
}


// Store literals in persistent area of 'data'.
void Vm::storeLiteralsInData(Expr const& expr)
{
    if (expr.kind == expr_Lit){
        Atom text = expr.name;
        if (tab0.has(text)) return;  // -- already stored this literal (note: we rely on the fact that differently typed literals cannot have the same text representation)
        tab0.addQ(text, RelPos(b_ZERO, pp));
        if      (expr.type.name == a_Bool ) pushLit(Word(int64(expr.name == a_false ? 0 : 1)));
        else if (expr.type.name == a_Int  ) pushLit(Word(stringToInt64(expr.name)));
        else if (expr.type.name == a_Float) pushLit(Word(stringToDouble(expr.name)));
        else if (expr.type.name == a_Atom ) pushLit(int64(+unquoteUnescape(expr.name)));
        else assert(false);
    }
    for (Expr const& e : expr)
        storeLiteralsInData(e);
}


addr_t Vm::compile(Expr prog, bool just_try)
{
    prog = expander.expandTemplates(prog);
    storeLiteralsInData(prog);
    if (just_try) push();
    addr_t ret = ::ZZ::compile(code, prog, tab0, result, pp, VmState(code, data, tag));
    if (just_try) pop();
    return ret;
}


//=================================================================================================
// Garbage collection:


void Vm::gcMark(addr_t ref0)
{
    Vec<addr_t> Q(1, ref0);;

    while (Q.size() > 0){
        addr_t ref = Q.popC();
        if (ref == 0 || tag[ref] & w_Mark) continue;

        // Find start of block:
        addr_t i = ref;
        for(;;){
            assert(i != 0);
            i--;
            if (tag[i] == w_Blk) break;
        }
        assert(i + data[i].val >= ref);      // -- 'ref' must be inside the block

        // Mark block:
        for (addr_t n = data[i].val + 1; n != 0; i++, n--){
            assert(!(tag[i] & w_Mark));
            if (tag[i] == w_Ref) Q.push(data[i].val);
            (uchar&)tag[i] |= w_Mark;
        }
    }
}


// If 'persist' is TRUE then run out of global space for persistent data.
void Vm::gc(uint failed_alloc_sz, bool persist, Vec<addr_t>* roots)
{
    ZZ_PTimer_Scope(vm_gc);
    double T0 = cpuTime();

    // Mark reachable data:
    addr_t sp = rel[b_LOCAL];
    for (addr_t i = 0; i < pp; i++){            // -- mark things pointed to by global vars
        if (tag[i] == w_Ref) gcMark(data[i].val);
        tag[i] |= w_Mark; }
    for (addr_t i = sp; i < data.size(); i++){  // -- mark things pointed to from the stack
        if (tag[i] == w_Ref) gcMark(data[i].val);
        tag[i] |= w_Mark; }
    for (uind i = 3; i < dcall.size(); i += 4){ // -- NOTE! '+4' must correspond to data pushed at call; '3' to location of closure data
        gcMark(dcall[i]); }
    for (VmExcp const& excp : excp_stack){
        gcMark(excp.closure);
        gcMark(excp.ret);
        gcMark(excp.arg);
    }
    if (roots)
        for (addr_t a : *roots) gcMark(a);

    // Count used area:
    addr_t heap_sz  = 0;
    addr_t stack_sz = 0;
    for (addr_t i = 0 ; i < hp         ; i++) if (tag[i] & w_Mark) heap_sz++;
    for (addr_t i = sp; i < data.size(); i++) if (tag[i] & w_Mark) stack_sz++;

    // Allocate new heap:
    mem_used = heap_sz + stack_sz + (persist ? 0 : failed_alloc_sz);    // -- for memory monitoring

    double grow_factor = stress_gc ? 1.0 : 2.5;     // -- increase the minimal amount for debugging
    addr_t new_mem_sz = max_(double(data.size() - dp), (heap_sz + stack_sz) * grow_factor + (persist ? 0 : failed_alloc_sz));
    addr_t new_per_sz = max_(double(dp), pp * grow_factor + (persist ? failed_alloc_sz : 0));
    newMax(new_mem_sz, new_per_sz / 4);             // -- neither part of data (persistent/dynamic) should be smaller than 1/4 of the other.
    newMax(new_per_sz, new_mem_sz / 4);

    addr_t new_sz = new_mem_sz + new_per_sz;
    Vec<Word> new_data(new_sz + 1); new_data.pop(); // -- reserve one extra word (see "Update references" below)
    Vec<WTag> new_tag (new_data.size(), w_Val);

    // Copy live data: (using old heap to store new address)
    addr_t j = 0;
    for (addr_t i = 0; i < hp; i++){
        if (i == dp) j = new_per_sz;     // -- leave room for more persistent data
        if (tag[i] & w_Mark){
            new_data[j] = data[i];
            new_tag [j] = tag[i] & ~w_Mark;
            data[i].val = j;
            j++;
        }else
            data[i].val = 0xAC1D0FF1CEC0FFEEll;     // -- to catch bugs
    }
    dp = new_per_sz;        // -- NEW data pointer set here
    hp = max_(dp, j);       // -- NEW heap pointer set here

    j = new_data.size() - 1;
    for (addr_t i = data.size(); i > sp;){ i--;
        if (tag[i] & w_Mark){
            new_data[j] = data[i];
            new_tag [j] = tag[i] & ~w_Mark;
            data[i].val = j;
            j--;
        }else
            data[i].val = 0xAC1D0FF1CEC0FFEEll;     // -- to catch bugs
    }

    // Update references:
    data.push(Word(int64(new_data.size())));        // -- the original stack-pointer is just outside of 'data'; add translation for it
    for (addr_t i = 0; i < new_data.size(); i++)
        if (new_tag[i] == w_Ref)
            new_data[i] = data[new_data[i].val];

    // Migrate to new data area:
    rel[b_LOCAL  ] = data[rel[b_LOCAL  ]].val;
    rel[b_RETVAL ] = data[rel[b_RETVAL ]].val;
    rel[b_ARGS   ] = data[rel[b_ARGS   ]].val;
    rel[b_CLOSURE] = data[rel[b_CLOSURE]].val;
    for (addr_t& a : dcall) a = data[a].val;
    for (VmExcp& excp : excp_stack){
        excp.closure = data[excp.closure].val;
        excp.ret     = data[excp.ret    ].val;
        excp.arg     = data[excp.arg    ].val;
    }
    if (roots)
        for (addr_t& a : *roots) a = data[a].val;

    new_data.moveTo(data);
    new_tag .moveTo(tag);

    gc_time += cpuTime() - T0;

    //**/wrLn("\a/-- GC completed -- [data size: %D, gc-time: %t]\a/", data.size(), gc_time);
}


//=================================================================================================
// -- Run function:


/*
There are two internal pointers, not accessible from the virtual machine-code:
   pp  - end of persistent data (global variables + literals + built-in function closures)
   dp  - beginning of heap-allocated data
   hp  - end of heap-allocated data

and ther are six pointers that can be part of a machine instruction, all stored in 'rel':
  rel: [zp, rp, ap, lp, cp]  (zero, retval, args, local, closure)

The layout of 'data[]' (type 'union Word { int64 val; double flt; void* bin; }') is:

    data[0]                     --heap grows-->       <--stack grows--                   data[LAST]

      zp________(gp)_____pp      dp_________hp        lp_______(lp')__ap__rp___(lp")________

        literals   global  free      heap       free     curr       prev local      rest of
        + simple   vars    global    data       stack    local      vars + curr     stack
        builtins           space                +heap    vars       args/retval
        + prev.
        globals

The "local" pointer is essentially the stack pointer. '_' denotes used memory.  The
closure pointer "cp" can live anywhere. "(gp)" denotes where global pointer used to be
in earlier versions of the VM.
*/


void Vm::clear(addr_t addr, uint n) {
    for (; n != 0; n--){
        data[addr] = Word();
        tag [addr] = w_Val;
        addr++;
    }
}


// Global allocation should be called once at the beginning of the program.
// Moves 'pp' forward to leave space for global variables in the persistent area.
void Vm::globAlloc(uint n) {
    if (pp + n > dp) gc(n, true);
    assert(pp + n <= dp);
    clear(pp, n);
    pp += n;
}


// Moves 'lp' backward to leave space for local variables
void Vm::stackAlloc(uint n) {
    if (hp + n > rel[b_LOCAL])
        gc(n, false);
    assert(hp + n <= rel[b_LOCAL]);

    rel[b_LOCAL] -= n;
    clear(rel[b_LOCAL], n);
    gc_count -= n;
}


// Reserve space on the heap by moving 'hp' forward. The address will be stored at 'dst'. Unless
// 'n == 0', the allocated area will be preceded by a size-tag, stating the size of the block (for
// garbage collecting purposes).
void Vm::heapAlloc(uint n, RelPos dst_)
{
    if (n == 0){
        addr_t dst = absPos(dst_);
        data[dst] = Word(int64(0));
        tag [dst] = w_Ref;
        return;
    }

    if (hp + n+1 > rel[b_LOCAL])
        gc(n+1, false);
    assert(hp + n+1 <= rel[b_LOCAL]);

    addr_t dst = absPos(dst_);
    clear(hp+1, n);
    data[hp]  = Word(int64(n));
    tag [hp]  = w_Blk;
    data[dst] = Word(int64(hp+1));
    tag [dst] = w_Ref;
    hp += n+1;
    gc_count -= n+1;
}


void Vm::callClosure(addr_t* ip, uint n_words, addr_t clos_ptr, addr_t ret, addr_t arg)
{
    Word fun = data[clos_ptr];
    if (fun.val & 1){
        // Evo function:
        icall.push(*ip);
        dcall.push(rel[b_RETVAL]);
        dcall.push(rel[b_ARGS]);
        dcall.push(rel[b_LOCAL]);
        dcall.push(rel[b_CLOSURE]);
            // -- if adding anything here, remember to update 'Vm::gc()' to get closure roots properly (also 'i_THROW')

        rel[b_RETVAL]  = ret;
        rel[b_CLOSURE] = data[clos_ptr+1].val;
        rel[b_ARGS]    = arg; // -- note: rel[b_LOCALS] is set up by function itself
        *ip = decodeCodePtr(fun.val);
    }else{
        // Builtin: (simple arithmetic functions)
        Builtin* bin = (Builtin*)fun.bin;
        bin(&data[arg], &data[ret]);
        for (uint i = 0; i < n_words; i++) tag[ret + i] = w_Val;    // -- tag results as values
    }
}


bool Vm::vecResize(uint elem_sz, RelPos vec_, int64 new_sz, int64 mem_lim)
{
    auto vecHead = [&](uint i) -> int64& { return data[data[absPos(vec_)].val + i].val; };
    auto vecTag  = [&](uint i) -> WTag&  { return tag [data[absPos(vec_)].val + i]; };

    int64 sz = vecHead(1);
    if (new_sz > sz){
        if (new_sz > mem_lim) return false; // -- will abort program
        int64 cap = vecHead(2);
        if (new_sz >= cap){
            int64 new_cap = max_(int64((cap + 4) * 1.5), new_sz);
            new_cap = (new_cap + 3) & ~int64(3);    // -- round up to nearest 4

            heapAlloc(new_cap * elem_sz, RelPos(b_ZERO, 1));

            addr_t old_ptr = vecHead(0);
            addr_t new_ptr = data[1].val;   // -- read reserved position
            data[1].val = 0;                // -- restore
            memcpy(&data[new_ptr], &data[old_ptr], elem_sz * sz * sizeof(Word));
            memcpy(&tag [new_ptr], &tag [old_ptr], elem_sz * sz * sizeof(WTag));
            clear(old_ptr, elem_sz * sz);
            vecHead(0) = new_ptr;
            vecTag (0) = w_Ref;
            vecHead(2) = new_cap;
        }
        vecHead(1) = new_sz;

    }else if (new_sz < sz){
        // Shrink:
        if (new_sz < 0) return false;       // -- will abort program
        clear(vecHead(0) + elem_sz * new_sz, elem_sz * (sz - new_sz));
        vecHead(1) = new_sz;
    }
    return true;
}


// Return TRUE if exception was caught.
bool Vm::throwExcp(Atom type_text, Word const* src_data, WTag const* src_tag, uint n, addr_t& ip)
{
    for(;;){
        if (excp_stack.size() == 1)     // -- exception stack has a resource limit sentinel element at index 0
            return false;

        VmExcp excp = excp_stack[LAST];
        if (excp.type_text != type_text)
            excp_stack.pop();
        else{
            assert(excp.arg_sz == n);
            while (icall.size() != excp.frame){
                if (excp_stack.size() > 1 && icall.size() == excp_stack[LAST].frame) excp_stack.pop();  // -- same code as for 'i_RETURN'
                icall.pop(); dcall.pop(); dcall.pop(); dcall.pop(); dcall.pop();
            }

            excp_stack.pop();
            rel[b_CLOSURE] = dcall.popC();      // -- same code as for 'i_RETURN'
            rel[b_LOCAL]   = dcall.popC();
            rel[b_ARGS]    = dcall.popC();
            rel[b_RETVAL]  = dcall.popC();
            ip             = icall.popC();

            memcpy(&data[excp.arg], src_data, n * sizeof(Word));
            memcpy(&tag [excp.arg], src_tag , n * sizeof(WTag));

            callClosure(&ip, excp.ret_sz, excp.closure, excp.ret, excp.arg);
            return true;
        }
    }
}


// NOTE! 'mem_lim0' is given in bytes, but will inside this function be converted to 'Word's.
addr_t Vm::run(addr_t start_addr, ResLims const& lim)
{
    ZZ_PTimer_Scope(vm_run);

    auto haltMsg = [&](String const& msg) {
        if (out == &std_out) wrLn(*out, "\n\a*\a/HALTED!\a* %_\a/", msg);
        else                 wrLn(*out, "\nHALTED! %_", msg);
    };

    auto throwExit = [&](ExitData exit_data, addr_t& ip) -> bool {
        return throwExcp(a_exit_code_type, &exit_data.exit_code, exit_tags, elemsof(exit_tags), ip);
    };

    // Resource limits:
    VmExcp excp;
    excp.type_text = a_run;
    excp.frame     = icall.size();
    excp.lim       = VmLimits{lim.cpu, lim.mem, lim.rec};
    excp_stack.push(excp);

    // Reset statistics:
    steps = 0;
    mem_used = 0;
    gc_count = excp_stack[LAST].lim.mem;

    auto getWord = [&](RelPos addr) -> Word { return data[absPos(addr)]; };

    bool trace = getenv("TRACE");
    addr_t ip = start_addr;
    for(;;){
        if (trace){ wr("`` EXECUTING: (ip=%>6%_  op=%>2%_)  ", ip, code[ip].op_code); dumpInstrAndData(&code[ip], getWord); std_out += FL; }

        steps++;

        switch (code[ip].op_code){
        case i_COPY:{
            uint n = code[ip].n_words;
            addr_t dst = absPos(code[ip+1].pos);
            addr_t src = absPos(code[ip+2].pos);
            memcpy(&data[dst], &data[src], n * sizeof(Word));
            memcpy(&tag [dst], &tag [src], n * sizeof(WTag));
            ip += 3;
            break; }

        case i_COPY_IS:{
            uint n = code[ip].n_words;
            addr_t dst = absPos(code[ip+1].pos);
            addr_t src = data[absPos(code[ip+2].pos)].val + code[ip+3].off;
            memcpy(&data[dst], &data[src], n * sizeof(Word));
            memcpy(&tag [dst], &tag [src], n * sizeof(WTag));
            ip += 4;
            break; }

        case i_COPY_ID:{
            uint n = code[ip].n_words;
            addr_t dst = data[absPos(code[ip+1].pos)].val + code[ip+3].off;
            addr_t src = absPos(code[ip+2].pos);
            memcpy(&data[dst], &data[src], n * sizeof(Word));
            memcpy(&tag [dst], &tag [src], n * sizeof(WTag));
            ip += 4;
            break; }

        case i_LEA:{
            int off = code[ip].n_words;
            addr_t dst = absPos(code[ip+1].pos);
            addr_t src = absPos(code[ip+2].pos);
            addr_t src_ptr = data[src].val; assert(tag[src] == w_Ref);
            data[dst] = Word(int64(src_ptr + off));
            tag [dst] = w_Ref;
            ip += 3;
            break; }

        case i_ITE:{
            uint n = code[ip].n_words;
            addr_t dst  = absPos(code[ip+1].pos);
            addr_t cond = absPos(code[ip+2].pos);
            addr_t src  = (data[cond].val != 0) ? cond + 1 : cond + 1 + n;
            memcpy(&data[dst], &data[src], n * sizeof(Word));
            memcpy(&tag [dst], &tag [src], n * sizeof(WTag));
            ip += 3;
            break; }

        case i_PUT:{
            addr_t dst = absPos(code[ip+1].pos);
            data[dst] = Word(int64(code[ip+2].off));
            tag [dst] = w_Val;
            ip += 3;
            break; }

        case i_PUT_CP:{
            addr_t dst = absPos(code[ip+1].pos);
            data[dst] = encodeCodePtr(code[ip+2].off);
            tag [dst] = w_Val;
            ip += 3;
            break; }

        case i_CALL:{   // -- note, 'code[ip+3]' is not used; its purpose is for code-optimization
            uint n = code[ip].n_words;
            ip += 5;
            callClosure(&ip, n, absPos(code[ip-5+2].pos), absPos(code[ip-5+1].pos), absPos(code[ip-5+4].pos));
            break; }

        case i_CASE:{   // [n] ret arg_sz arg       ('n' is 'ret_sz'; call 'k+1'th argument (a closure) of tuple 'arg' depending on the alt. 'k' of the 0th argument)
            // arg = sum_alt, sum_ref, code_ptr[0], clos_ptr[0], code_ptr[1], clos_ptr[1]....
            uint n = code[ip].n_words;
            addr_t ret = absPos(code[ip+1].pos);
            addr_t arg = absPos(code[ip+3].pos);
            int    alt = data[arg].val;
            addr_t clos_ptr = arg + 2*(alt + 1);
            ip += 4;
            callClosure(&ip, n, clos_ptr, ret, arg+1);
            break; }

        case i_RETURN:
            if (excp_stack.size() > 1 && icall.size() == excp_stack[LAST].frame) excp_stack.pop();
            rel[b_CLOSURE] = dcall.popC();
            rel[b_LOCAL]   = dcall.popC();
            rel[b_ARGS]    = dcall.popC();
            rel[b_RETVAL]  = dcall.popC();
            ip             = icall.popC();
            break;

        case i_LOCALS:{
            stackAlloc(code[ip].n_words);
            ip += 1;
            break; }

        case i_GLOBALS:{
            globAlloc(code[ip].n_words);
            ip += 1;
            break; }

        case i_ALLOC:{
            heapAlloc(code[ip].n_words, code[ip+1].pos);
            ip += 2;
            break; }

        case i_CHK_REF:{
            uint n = code[ip].n_words; assert(n != 0);  // -- must have an error message
            addr_t src = data[absPos(code[ip+1].pos)].val;
            if (src == 0){
                if (!throwExit(ExitData(a_null_deref, Atom(n)), ip)){
                    haltMsg(fmt("%_", Atom(n)));
                    result = RelPos(b_ZERO, 0);
                    goto Done; }
            }else
                ip += 2;
            break; }

        case i_HALT:{
            uint n = code[ip].n_words;
            if (n != 0)
                haltMsg(fmt("%_", Atom(n)));
            ip += 1;
            goto Done; }

        // VECTOR INSTRUCTIONS:  (header layout: 0=data_ptr, 1=size (in elems), 2=capacity (in elems))
        case i_VEC_SIZE:{
            addr_t dst = absPos(code[ip+1].pos);
            addr_t vec = data[absPos(code[ip+2].pos)].val;
            data[dst] = data[vec+1];
            tag [dst] = w_Val;
            ip += 3;
            break; }

        case i_VEC_GET:{
            uint n = code[ip].n_words;
            addr_t dst = absPos(code[ip+1].pos);
            addr_t vec = data[absPos(code[ip+2].pos)].val;
            int64  idx = data[absPos(code[ip+3].pos)].val;
            int64  sz  = data[vec+1].val;
            if (idx < 0) idx += sz;

            if (uint64(idx) >= uint64(sz)){
                if (!throwExit(ExitData(a_vec_op, a_vop_get, idx, sz), ip)){
                    haltMsg(fmt("Out-of-bounds read index '%_' into vector of size '%_'.", idx, sz));
                    result = RelPos(b_ZERO, 0);
                    goto Done; }
            }else{
                memcpy(&data[dst], &data[data[vec+0].val + n*idx], n * sizeof(Word));
                memcpy(&tag [dst], &tag [data[vec+0].val + n*idx], n * sizeof(WTag));
                ip += 4;
            }
            break; }

        case i_VEC_GROWBY:{
            RelPos vec    = code[ip+1].pos;
            int64  old_sz = data[data[absPos(vec)].val + 1].val;
            int64  delta  = data[absPos(code[ip+2].pos)].val;
            if (!vecResize(code[ip].n_words, vec, old_sz + delta, excp_stack[LAST].lim.mem)){
                if (!throwExit(ExitData(a_vec_op, a_vop_growby, old_sz + delta, old_sz), ip)){
                    if (delta < 0) haltMsg(fmt("Shrunk vector to negative size."));
                    else           haltMsg(fmt("Grow vector beyond memory limit."));
                    result = RelPos(b_ZERO, 0);
                    goto Done; }
            }else
                ip += 3;
            break; }

        case i_VEC_RESIZE:{
            RelPos vec    = code[ip+1].pos;
            int64  new_sz = data[absPos(code[ip+2].pos)].val;
            if (!vecResize(code[ip].n_words, vec, new_sz, excp_stack[LAST].lim.mem)){
                int64 old_sz = data[data[absPos(vec)].val + 1].val;
                if (!throwExit(ExitData(a_vec_op, a_vop_resize, new_sz, old_sz), ip)){
                    if (new_sz < 0) haltMsg(fmt("Resized vector to negative size."));
                    else            haltMsg(fmt("Resized vector beyond memory limit."));
                    result = RelPos(b_ZERO, 0);
                    goto Done; }
            }else
                ip += 3;
            break; }

        case i_VEC_SET:{
            uint n = code[ip].n_words;
            addr_t vec = data[absPos(code[ip+1].pos)].val;
            int64  idx = data[absPos(code[ip+2].pos)].val;
            addr_t src = absPos(code[ip+3].pos);
            int64  sz  = data[vec+1].val;
            if (idx < 0) idx += sz;

            if (uint64(idx) >= uint64(sz)){
                if (!throwExit(ExitData(a_vec_op, a_vop_set, idx, sz), ip)){
                    haltMsg(fmt("Out-of-bounds write index '%_' into vector of size '%_'.", idx, sz));
                    result = RelPos(b_ZERO, 0);
                    goto Done; }
            }else{
                memcpy(&data[data[vec+0].val + n*idx], &data[src], n * sizeof(Word));
                memcpy(&tag [data[vec+0].val + n*idx], &tag [src], n * sizeof(WTag));
                ip += 4;
            }
            break; }

        // PRINT INSTRUCTIONS:
        case i_PR_BOOL:{
            addr_t src = absPos(code[ip+1].pos);
            *out += data[src].val ? a_true : a_false;
            ip += 2;
            break; }

        case i_PR_INT:{
            addr_t src = absPos(code[ip+1].pos);
            *out += data[src].val;
            ip += 2;
            break; }

        case i_PR_FLOAT:{
            addr_t src = absPos(code[ip+1].pos);
            *out += data[src].flt;
            ip += 2;
            break; }

        case i_PR_ATOM:{
            addr_t src = absPos(code[ip+1].pos);
            *out += Atom(data[src].val);
            ip += 2;
            break; }

        case i_PR_TEXT:{
            uint idx = code[ip+1].off;
            *out += Atom(idx);
            ip += 2;
            break; }

        // EXCEPTION HANDLING:
        case i_TRY:{
            addr_t ret    = absPos(code[ip+1].pos);
            uint   ret_sz = code[ip].n_words;
            VmExcp excp;
            excp.type_text = Atom(code[ip+4].off);
            excp.frame     = icall.size() + 1;
            excp.closure   = absPos(code[ip+3].pos);
            excp.ret       = ret;
            excp.ret_sz    = ret_sz;
            excp.arg       = absPos(code[ip+5].pos);
            excp.arg_sz    = code[ip+6].off;
            excp.lim       = excp_stack[LAST].lim;
            excp_stack.push(excp);

            ip += 7;
            callClosure(&ip, ret_sz, absPos(code[ip-7+2].pos), ret, 0);
            break; }

        case i_THROW:{
            Atom   type_text = Atom(code[ip+1].off);
            addr_t src = absPos(code[ip+2].pos);        // -- addr of thrown value
            uint   n   = code[ip].n_words;              // -- size of thrown value
            if (!throwExcp(type_text, &data[src], &tag[src], n, ip)){
                haltMsg(fmt("Uncaught exception of type: %_", type_text));
                result = RelPos(b_ZERO, 0);
                goto Done; }
            break; }

        case i_RL_RUN:{
            uint   ret_sz = code[ip].n_words;
            addr_t ret  = absPos(code[ip+1].pos);
            addr_t clos = absPos(code[ip+2].pos);
            int64  cpu_lim = min_(data[absPos(code[ip+3].pos)].val                    , (int64)0x3FFFFFFFFFFFFFFFll);   // -- leave some room for arithmetic
            int64  mem_lim = min_(data[absPos(code[ip+4].pos)].val / (int)sizeof(Word), (int64)0x3FFFFFFFll);
            int64  rec_lim = min_(data[absPos(code[ip+5].pos)].val                    , (int64)0x3FFFFFFFll);
            VmLimits& curr = excp_stack[LAST].lim;

            // Values '<= 0' means "use current value". Positive values are relative to current resource usage.
            // Limits can only shrink.
            if (cpu_lim <= 0) cpu_lim = curr.cpu; else{ cpu_lim += steps;        if ((uint64)cpu_lim > curr.cpu) cpu_lim = curr.cpu; }
            if (mem_lim <= 0) mem_lim = curr.mem; else{ mem_lim += mem_used;     if ((uint64)mem_lim > curr.mem) mem_lim = curr.mem; }
            if (rec_lim <= 0) rec_lim = curr.rec; else{ rec_lim += icall.size(); if ((uint64)rec_lim > curr.rec) rec_lim = curr.rec; }

            VmExcp excp;
            excp.type_text = a_run;     // <<== if introducing blocks, maybe change this
            excp.frame     = icall.size();
            excp.lim       = VmLimits{uint64(cpu_lim), addr_t(mem_lim), uint(rec_lim)};
            excp_stack.push(excp);

            ip += 6;
            callClosure(&ip, ret_sz, clos, ret, 0);
            break; }

        default: assert(false); }

        // Resource limit:
        if (steps >= excp_stack[LAST].lim.cpu){
            uint64 lim = excp_stack[LAST].lim.cpu;
            if (!throwExit(ExitData(a_cpu_lim, Atom(), steps, lim), ip)){
                haltMsg(fmt("Exceeded CPU-limit of %_ steps.", lim));
                result = RelPos(b_ZERO, 0);
                goto Done; }
        }

        if (mem_used > excp_stack[LAST].lim.mem){
            uint64 lim = excp_stack[LAST].lim.mem;
            if (!throwExit(ExitData(a_mem_lim, Atom(), mem_used, lim), ip)){
                haltMsg(fmt("Exceeded memory limit of %^DB.", lim * sizeof(Word)));
                result = RelPos(b_ZERO, 0);
                goto Done; }
            gc(0, false);
        }

        if (icall.size() > excp_stack[LAST].lim.rec){
            uint64 lim = excp_stack[LAST].lim.rec;
            if (!throwExit(ExitData(a_rec_lim, Atom(), icall.size(), lim), ip)){
                haltMsg(fmt("Exceeded recursion depth of %_.", lim));
                result = RelPos(b_ZERO, 0);
                goto Done; }
        }

        // Force periodic GC to monitor memory:
        gc_count--;
        if (gc_count <= 0){
            gc(0, false);
            gc_count = excp_stack[LAST].lim.mem;
        }
    }
  Done:;

    // Cleanup:
    icall.clear();
    dcall.clear();
    excp_stack.clear();
    rel[b_RETVAL] = rel[b_ARGS] = rel[b_CLOSURE] = 0;
    rel[b_LOCAL] = data.size();

    addr_t ret = absPos(result);
    if (getenv("DUMP_STATE")){
        //**/Vec<addr_t> rvec(1, ret);
        //**/gc(0, false, &rvec);
        //**/ret = rvec[0];
        newLn();
        dumpState();
    }

    return ret;
}


// For debugging.
void Vm::dumpAddr(addr_t i, Out& out)
{
    wr(out, "(%>6%_)   ", i);
    char mark = (tag[i] & w_Mark) ? '*' : 0;
    switch (tag[i] & ~w_Mark){
    case w_Val: wrLn(out, "VAL%C %_ = $%:x = %%%:b = cp:%_", mark, data[i].val, data[i].val, data[i].val, data[i].val >> 1); break;
    case w_Ref: wrLn(out, "REF%C %_", mark, data[i].val); break;
    case w_Blk: wrLn(out, "BLK%C %_", mark, data[i].val); break;
    default: assert(false); }
}


// Should be called before 'pp' is restored to 'gp'.
void Vm::dumpState(Out& out)
{
    wrLn(out, "\a*DUMP STATE -- LITERALS/GLOBALS:\a*\t+\t+");
    for (addr_t i = 0; i < pp; i++)
        if (tag[i] != w_Val || data[i].val != 0) dumpAddr(i, out);
    wrLn("\t-\t-");

    wrLn(out, "\a*DUMP STATE -- HEAP:\a*\t+\t+");
    for (addr_t i = pp; i < hp; i++)
        if (tag[i] != w_Val || data[i].val != 0) dumpAddr(i, out);
    wrLn("\t-\t-");

    wrLn(out, "\a*DUMP STATE -- STACK:\a*\t+\t+");
    for (addr_t i = rel[b_LOCAL]; i < data.size(); i++) dumpAddr(i, out);
    wrLn("\t-\t-");
}


void Vm::dumpData(Out& out)
{
    for (Atom a : tab0.locals()){
        if (tab0.ref(a).base == b_ZERO)
            wrLn(out, "  (%>6%_)   %_", tab0.ref(a).offset, a);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Wrappers:


Vec<Pair<Atom,Type>> getBuiltinFunctions()
{
    Vec<Pair<Atom,Type>> ret;
    for (RegisterBin* r = registered_bins; r; r = r->next)
        ret.push(tuple(Atom(r->fun_name), parseType(r->fun_type)));
    return ret;
}


RunTime::RunTime() { vm = new Vm; }
RunTime::~RunTime() { delete vm; }


void RunTime::tryCompile(Expr prog) {
    RelPos result;
    vm->compile(prog, true); }


addr_t RunTime::run(Expr prog, Params_RunTime P)
{
    if (prog.kind != expr_Block){   // -- the compiler expects a Block expression
        Vec<Expr> sing(1, prog);
        prog = Expr::Block(sing).setType(Type(prog.type));
    }

    RelPos result;
    addr_t start = vm->compile(prog, false);

    if (getenv("DATA")){
        wrLn("\a*DATA:\a*\t+\t+");
        vm->dumpData();
        wrLn("\t-\t-");
    }

    if (getenv("CODE")){
        wrLn("\a*CODE:\a*\t+\t+");
        vm->dumpCode();
        wrLn("\t-\t-");
    }

    if (P.verbose) wrLn("\a*-- EXECUTION STARTING --\a*");

    double T0c = cpuTime();
    double T0r = realTime();
    vm->out = P.out;
    addr_t ret = vm->run(start, P.lim);
    vm->out->flush();

    if (P.verbose){
        wrLn("\a*-- EXECUTION FINISHED --\a*");
        wrLn("\a*-- STEPS: %_ --\a*", vm->steps);
        wrLn("\a*-- GC time: %t --\a*", vm->gc_time);
        newLn();
        wrLn("Mem used: %.1f MB", memUsed() / 1048576.0);
        wrLn("CPU-time: %t", cpuTime () - T0c);
        wrLn("Realtime: %t", realTime() - T0r);
    }

    return ret;
}


RetVal RunTime::runRet(Expr prog, Params_RunTime P)
{
    addr_t addr = run(prog, P);
    return (addr == 0) ? RetVal(*this, addr, Type()) : RetVal(*this, addr, prog.type);
}


void RunTime::push() { vm->push(); }
void RunTime::pop () { vm->pop (); }
Word const& RunTime::data(addr_t idx) const { return vm->readData(idx); }
WTag const& RunTime::tag (addr_t idx) const { return vm->readTag (idx); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// RetVal:


bool RetVal::isVoid() const {
    return type.name == a_Void; }


double RetVal::flt() const {
    assert(type.name == a_Float);
    return rt.data(addr).flt; }


int64 RetVal::val() const {
    assert(type.name == a_Bool || type.name == a_Int || type.name == a_Atom);
    return rt.data(addr).val; }


uint RetVal::alt() const {
    return rt.data(addr).val; }


VM::Word RetVal::word() const {
    return rt.data(addr); }


addr_t RetVal::clos() const {
    assert(type.name == a_Fun);
    return rt.data(addr+1).val; }


RetVal RetVal::operator()(uint idx) const
{
    Array<Type const> ts = tupleSlice(type);
    assert(idx < ts.size());
    addr_t a = addr;
    for (uint i = 0; i < idx; i++) a += reprSize(ts[i]);
    if (type.name != a_Tuple){
        assert(idx == 0);
        return RetVal(rt, a, type);
    }else
        return RetVal(rt, a, type[idx]);
}


RetVal RetVal::operator[](addr_t idx) const
{
    assert(type.name == a_Vec);
    addr_t head     = rt.data(addr).val;
    addr_t vec_data = rt.data(head).val;
    addr_t vec_size = rt.data(head+1).val;
    assert(idx < vec_size);
    return RetVal(rt, vec_data + idx * reprSize(type[0]), type[0]);
}


addr_t RetVal::size() const
{
    assert(type.name == a_Vec);
    addr_t head = rt.data(addr).val;
    return rt.data(head+1).val;
}


RetVal RetVal::operator*() const
{
    if (type.name == a_Ref)
        return RetVal(rt, rt.data(addr).val, type[0]);
    else if (type.name == a_OneOf)
        return RetVal(rt, rt.data(addr+1).val, type[rt.data(addr).val]);
    else
        assert(false);
}


RetVal RetVal::derefData(Type const& type_def) const {
    return RetVal(rt, rt.data(addr+1).val, type_def[rt.data(addr).val]); }


//=================================================================================================


// Print 'data' types defined in 'std.evo'.
static
void writeRetVal_stdEvo(Out& out, RetVal const& val, bool short_form)
{
    static Atom a_List("List");
    static Atom a_Maybe("Maybe");

    if (val.type.name == a_List){
        Type type_def = Type(a_OneOf, {Type(a_Void), Type(a_Tuple, {Type(val.type[0]), Type(val.type)})}); // -- this type must match code in 'std.evo'.
        if (short_form)
            out += '{';
        else
            out += "(list_[:", val.type[0], ' ';
        RetVal p = val;
        bool first = true;
        for(;;){
            p = p.derefData(type_def);
            if (p.isVoid()) break;

            if (first) first = false;
            else out += ", ";

            writeRetVal(out, p(0), short_form);
            p = p(1);
        }
        if (short_form)
            out += '}';
        else
            out += "])";

    }else if (val.type.name == a_Maybe){
        if (val.alt() == 0){
            if (short_form)
                out += '*';
            else
                out += "none<", val.type[0], '>';
        }else{
            Type type_def = Type(a_OneOf, {Type(a_Void), {Type(val.type[0])}}); // -- this type must match code in 'std.evo'.
            RetVal p = val.derefData(type_def);
            if (short_form)
                writeRetVal(out, p, short_form);
            else{
                out += "(some_ ";
                writeRetVal(out, p, short_form);
                out += ')';
            }
        }

    }else{
        wrLn("INTERNAL ERROR! Cannot print value of type '%_'", val.type);
        exit(1);
    }

}


void writeRetVal(Out& out, RetVal const& val, bool short_form)
{
    if (!val.type){
        out += "fail<?>";
    }else if (val.type.name == a_Void){
        out += "()";
    }else if (val.type.name == a_Bool){
        out += val ? "_1" : "_0";
    }else if (val.type.name == a_Int){
        out += val.val();
    }else if (val.type.name == a_Float){
        String tmp; tmp += val.flt();
        if (!has(tmp, '.') && !has(tmp, 'e') && !has(tmp, 'E')) tmp += ".0";    // -- make it parse as a Float
        out += tmp;
        // <<== +-nan/inf not handled here! (or in Parser.cc)
    }else if (val.type.name == a_Atom){
        out += '"', Atom(val), '"';
    }else if (val.type.name == a_OneOf){
        out += '.', val.type, '.', val.val(), '(',
        writeRetVal(out, *val, short_form);
        out += ')';
    }else if (val.type.name == a_Tuple){
        out += '(';
        for (uint i = 0; i < val.type.size(); i++){
            if (i != 0) out += ", ";
            writeRetVal(out, val(i), short_form);
        }
        out += ')';
    }else if (val.type.name == a_Ref){
        out += "&(",
        writeRetVal(out, *val, short_form);
        out += ')';
    }else if (val.type.name == a_Vec){
        if (short_form)
            out += '[';
        else
            out += "[:", val.type[0], ' ';
        for (addr_t i = 0; i < val.size(); i++){
            if (i != 0) out += ", ";
            writeRetVal(out, val[i], short_form);
        }
        out += ']';
    }else if (val.type.name == a_Fun){
        Word w = val.word();
        addr_t a = val.clos();
        if (isCodePtr(w))
            wr(out, "closure(fun=%_, clos=%_)", decodeCodePtr(w), a);
        else
            wr(out, "builtin(ptr=%p)", w.bin);
    }else
        writeRetVal_stdEvo(out, val, short_form);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

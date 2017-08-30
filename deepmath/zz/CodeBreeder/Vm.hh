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

#ifndef ZZ__CodeBreeder__Vm_hh
#define ZZ__CodeBreeder__Vm_hh

#include "Types.hh"
#include "Instr.hh"

namespace ZZ {
using namespace std;


namespace VM {
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Virtual Machine Types: [namespace 'VM']


// The 'data' vector stores elements of this type.
union Word {
    int64    val;       // -- Bool, Int, Atom, data pointer, lambda-code pointer (encoded with 'encodeCodePtr()' to distinguish from a 'bin' pointer)
    double   flt;       // -- Float
    void*    bin;       // -- pointer to built-in function (4 bytes aligned)

    Word()           : val(0)   {}
    Word(int64  val) : val(val) {}
    Word(double flt) : flt(flt) {}
    Word(void*  bin) : bin(bin) {}
};


// Each word is tagged with a char for garbage collection purposes.
typedef uchar WTag;
constexpr WTag w_Val = 0;
constexpr WTag w_Ref = 1;
constexpr WTag w_Blk = 2;
constexpr WTag w_Mark = 128;    // -- for garbage collection


/*
A CLOSURE consists of two words:
  - pointer to function (either a code-ptr ('addr_t') or a builtin-ptr ('Builtin*'))
  - pointer ('addr_t') to closure arguments
*/

inline bool   isCodePtr(Word fun)       { return fun.val & 1; }   // -- is 'fun' a pointer in the virtual machine (an 'addr_t'); otherwise C++ pointer to a 'Builtin' function.
inline Word   encodeCodePtr(addr_t addr){ return Word((int64(addr) << 1) | 1); }
inline addr_t decodeCodePtr(Word fun)   { assert(isCodePtr(fun)); return fun.val >> 1; }


// EXIT CODES: (extra info: atom, index, size)
//
// Never throws:
//    "normal"
// Throws under 'run_()':
//    "cpu_lim"        -- index=#actual-steps, size=#cpu-limit
//    "mem_lim"        -- index=#actual-words, size=#mem-limit
//    "rec_lim"        -- index=#actual-recs , size=#rec-limit
// Always throws:
//    "null_deref"     -- atom=message
//    "vec_op"         -- atom={get,set,shrunk,resize}, index=get-or-set-index/new-size, size=old-size
//
struct ExitData {
    Word exit_code;     // -- always present, must be first (and remaining elements of the struct of type 'Word')
    Word atom;          // }
    Word index;         // }- depends on exit code
    Word size;          // }

    ExitData(Atom exit_code, Atom atom = Atom(), int64 index = 0, int64 size = 0) :
        exit_code(int64(+exit_code)), atom(int64(+atom)), index(index), size(size) {}
};

static constexpr cchar* exit_code_type =  "(Atom, Atom, Int, Int)";

static const WTag exit_tags[] = {w_Val, w_Val, w_Val, w_Val};

extern Atom a_normal;
extern Atom a_cpu_lim;
extern Atom a_mem_lim;
extern Atom a_rec_lim;
extern Atom a_null_deref;
extern Atom a_vec_op;

extern Atom a_vop_get;
extern Atom a_vop_set;
extern Atom a_vop_growby;
extern Atom a_vop_resize;

extern Atom a_exit_code_type;


}
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Wrapper functions:


Vec<Pair<Atom,Type>> getBuiltinFunctions();
void runEvo(Expr const& expr, uint64 mem_lim = 8*1024*1024, uint64 cpu_lim = UINT64_MAX);


template<> fts_macro void write_(Out& out, VM::Word const& v) {   // -- assume its the 'val' field we want
    out += v.val; }


struct ResLims {
    uint64 cpu;
    addr_t mem;     // -- in words ('sizeof(Word)')
    uint   rec;
    ResLims(uint64 cpu = UINT64_MAX, addr_t mem = 128 * 1024 * 1024, uint rec = 1000000) : cpu(cpu), mem(mem), rec(rec) {}
};


struct Params_RunTime {
    ResLims lim;
    bool   verbose = false;
    Out*   out     = &std_out;
};


// Wrapper for internal 'Vm'.
class Vm;
class RetVal;
class RunTime {
    Vm* vm;
public:
    RunTime();
   ~RunTime();
    void   tryCompile(Expr prog);   // -- try to compile, then clean up (to catch semantical error such as unresolvable "rec" definitions)
    addr_t run   (Expr prog, Params_RunTime P = Params_RunTime());  // -- returns address of result, or '0' if program was halted
    RetVal runRet(Expr prog, Params_RunTime P = Params_RunTime());  // -- returns a value of type 'Type()' if program was halted

    void push();    // -- push current state
    void pop();     // -- restore state to last 'push()' operation

    VM::Word const& data(addr_t idx) const;
    VM::WTag const& tag (addr_t idx) const;
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// RetVal -- helper class for extracting Evo return values:


class RetVal {
    RunTime const&  rt;
    addr_t          addr;
public:
    Type            type;       // -- use 'type.name' to determine which access function/operator to use

    RetVal(RunTime const& rt, addr_t addr, Type const& type) : rt(rt), addr(addr), type(type) {}
        // -- 'addr' is return value from 'Vm::run()'

    RetVal(RetVal const& s) : rt(s.rt), addr(s.addr), type(s.type) {}
    RetVal& operator=(RetVal const& s) { this->~RetVal(); new (this) RetVal(s); return *this; }

    bool     isVoid() const;                        // -- is value 'Void'?
    double   flt   () const;                        // -- value access for Float
    int64    val   () const;                        // -- value access (Int, Bool, Atom)
    uint     alt   () const;                        // -- alternative# for 'OneOf's
    operator int64 () const { return val(); }       // -- convenience operator

    VM::Word word  () const;                        // -- value access (any type)
    addr_t   clos  () const;                        // -- closure reference (Fun)

    RetVal operator()(uint idx) const;              // -- tuple access; a single value can be treated as a unit tuple (note: a little expensive; can be improved if bottleneck)
    RetVal operator[](addr_t idx) const;            // -- vector access
    addr_t size() const;                            // -- vector size
    RetVal operator*() const;                       // -- follow '&' or 'OneOf' reference
    RetVal derefData(Type const& type_def) const;   // -- dereference 'data' types; pass in value from 'Inferrer::lookupType(type)'
};


void writeRetVal(Out& out, RetVal const& val, bool short_form);     // -- the short form is more readable but cannot be parsed back as an Evo expression
template<> fts_macro void write_(Out& out, RetVal const& val) { writeRetVal(out, val, false); }
template<> fts_macro void write_(Out& out, RetVal const& val, Str /*flags*/) { writeRetVal(out, val, true); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

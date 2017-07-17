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

#ifndef ZZ__CodeBreeder__Instr_hh
#define ZZ__CodeBreeder__Instr_hh
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Auxiliary types:


typedef uint addr_t;    // -- zero-based location in 'code' or 'data'; for implicit documentation

constexpr uint closure_sz = 2;  // (code_addr, ref_to_captures_tuple)
constexpr uint vechead_sz = 3;  // (ref_to_data, size, capacity)


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// VM instruction:


enum Base { b_ZERO, b_RETVAL, b_ARGS, b_LOCAL, b_CLOSURE, /*special:*/b_INTERNAL };


struct RelPos {
    Base   base   : 3;
    int    offset : 29;

    RelPos(Base base = b_ZERO, int offset = 0) : base(base), offset(offset) {}
    explicit operator bool() const { return base != b_ZERO || offset != 0; }

    RelPos operator+(int delta) const { return RelPos(base, offset + delta); }
    bool operator==(RelPos const& h) const { return memEq(*this, h); }
};

template<> fts_macro void write_(Out& out, const RelPos& v) {
    wr(out, "%C%C[%_]", "\0ralcg"[v.base], "\0ppppp"[v.base], v.offset); }


// NOTE! Anything ending in '#' is stored in the 'off' field of 'Instr'.
enum OpCode : uint {
    // Core instructions:
    i_COPY,     // [n] dst src              (data[dst] = data[src])
    i_COPY_IS,  // [n] dst src_base off#    (data[dst] = data[data[src_base] + off])
    i_COPY_ID,  // [n] dst_base src off#    (data[data[dst_base] + off] = data[src])
    i_LEA,      // [n] dst base             (data[dst] = data[base] + n; so 'n' act as offset, 'data[base]' must store a reference)
    i_ITE,      // [n] dst src              (data[dst] = data[src] ? data[src+1] : data[src+1+n])
    i_PUT,      // [1] dst val#             (copy immediate value (small integer, using 'off' in Instr)' into location 'dst')
    i_PUT_CP,   // [1] dst code_ptr#        (copy immediate address 'code_ptr' (using 'off' in 'Instr') into location 'dst' with proper translation for closures)
    i_CALL,     // [n] ret clos arg_sz# arg ('n' is 'ret_sz'; call 'clos' on 'arg' and store result at 'ret')
    i_CASE,     // [n] ret arg_sz# arg      ('n' is 'ret_sz'; call 'k+1'th argument (a closure) of tuple 'arg' depending on the alt. 'k' of the 0th argument)
    i_RETURN,   // [0]                      (return from function)
    i_LOCALS,   // [n]                      (reserve n words on stack; will be zeroed)
    i_GLOBALS,  // [n]                      (reserve n words on global heap; must be issued before any 'ALLOC'; persistent throughout execution; will be zeroed)
    i_ALLOC,    // [n] dst                  (reserve n words on heap, store address at 'dst'; will be zeroed)
    i_CHK_REF,  // [n] src                  (assert that 'data[src] != 0'; n = Atom index for error message)
    i_HALT,     // [n]                      (stop execution; n = Atom index, 0 = no error)

    // Vector instructions: (^ indicates vector ref, 'n' is always the size of an element, here 'delta/new_sz' are *positions*)
    i_VEC_SIZE,     // [n] dst src^
    i_VEC_GET,      // [n] dst src^ idx     (read "src[idx]"; moral equivalent of COPY_IS, viewing the vector as a reference)
    i_VEC_GROWBY,   // [n] vec^ delta       (relative resizing, 'delta' can be negative)
    i_VEC_RESIZE,   // [n] vec^ new_sz      (absolute resizing)
    i_VEC_SET,      // [n] dst^ idx src     ("dst[idx] = src")

    // Print instructions:
    i_PR_BOOL,  // [1] src
    i_PR_INT,   // [1] src
    i_PR_FLOAT, // [1] src
    i_PR_ATOM,  // [1] src
    i_PR_TEXT,  // [0] atom#                (print a specific, compile-time atom; used by compiler for 'write_')

    // Exception handling:
    i_TRY,      // [n] 1:ret 2:clos_try 3:clos_catch 4:type_catch# 5:carg 6:carg_sz#
                //                          (calls 'clos_try' taking Void and returning 'n' words; if 'i_THROW' is issued
                //                           on type 'type_catch'then 'clos_catch' is called on args copied to 'cargs' of size 'carg_sz')
    i_THROW,    // [n] type_throw# src      (throw exception of type 'type_throw' at 'src' of size 'n')

    // Resource limits:
    i_RL_RUN,   // [n] ret clos cpu mem rec (call closure 'clos' (taking Void argument) and execute it under resource limit)

    // Info:
    i_SUBR,     // [n]                      (mark the beginning of a subroutine; 'n' is size of the code, including 'LOCALS' and 'RETURN' instructions but not the 'SUBR' instruction itself)

    // Pseudo-instructions used only during code-generation (removed before execution):
    pi_LABEL,   // [n]
    pi_PUT_CP,  // [n] dst

    OpCode_size
};
/*
Proposals:
    JMP    cp
    BRANCH [n=negate?] src cp         -- if n=0, jump to 'cp' if '*src != 0' (if n=1, jump if '*src == 0')
    JMPTAB [n] src cp_1 ... cp_n-1    -- if '*src == 0' go to next instruction, otherwise use the n-1 provided code-pointers.
    SWAPCLOS                          -- swap closure pointer with value at address
*/
/*
Operands
  p = pos
  h = pos on heap (zero-based and beyond pp)
  n = offset, size, index, or delta
  c = code ptr (address)
  t = type (as Atom)
*/

union Instr {
    struct {
        uint op_code : 5;
        uint n_words : 27;
    };
    RelPos pos;        // -- source, destination, offset (relative addresses, can be negative)
    int    off;

    Instr() : pos() {}

    Instr(uint op_code, uint n_words) : op_code(op_code), n_words(n_words) {}
    Instr(RelPos pos)                 : pos(pos) {}
    Instr(int off)                    : off(off) {}
};


static const uchar instr_size[OpCode_size] = {
    3,  //  i_COPY
    4,  //  i_COPY_IS
    4,  //  i_COPY_ID
    3,  //  i_LEA
    3,  //  i_ITE
    3,  //  i_PUT
    3,  //  i_PUT_CP
    5,  //  i_CALL
    4,  //  i_CASE
    1,  //  i_RETURN
    1,  //  i_LOCALS
    1,  //  i_GLOBALS
    2,  //  i_ALLOC
    2,  //  i_CHK_REF
    1,  //  i_HALT

    3,  //  i_VEC_SIZE
    4,  //  i_VEC_GET
    3,  //  i_VEC_GROWBY
    3,  //  i_VEC_RESIZE
    4,  //  i_VEC_SET

    2,  //  i_PR_BOOL
    2,  //  i_PR_INT
    2,  //  i_PR_FLOAT
    2,  //  i_PR_ATOM
    2,  //  i_PR_TEXT

    7,  //  i_TRY
    3,  //  i_THROW

    6,  //  i_RL_RUN

    1,  //  i_SUBR

    1,  //  pi_LABEL
    2,  //  pi_PUT_CP
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Debug:


namespace VM { union Word; }

void dumpInstrCode(Instr const* p, Out& out = std_out);
void dumpInstrData(Instr const* p, function<VM::Word(RelPos)> getData, Out& out = std_out);

inline void dumpInstr       (Instr const* p, Out& out = std_out) { dumpInstrCode(p, out); out += NL; }
inline void dumpInstrAndData(Instr const* p, function<VM::Word(RelPos)> getData, Out& out = std_out) { dumpInstrCode(p, out); out += "; "; dumpInstrData(p, getData, out); out += NL; }

void dumpCode(Array<Instr const> code, Out& out = std_out);
inline void dumpCode(Vec<Instr> const& code, Out& out = std_out) { dumpCode(code.slice(), out); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

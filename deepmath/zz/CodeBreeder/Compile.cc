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
#include "Compile.hh"
#include "zz/Generics/IntSet.hh"
#include "zz/Generics/Set.hh"

namespace ZZ {
using namespace std;


ZZ_PTimer_Add(evo_core_compilation);


namespace {
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Internal types:


enum Unit : uchar { UNIT };     // -- dummy type to use SymTable as a SymSet.


// Allocator for temporaries (local variables) for functions.
class TmpAlloc {
    Vec<uint> hist;
    uint      curr;
    uint      max;

public:
    TmpAlloc() : curr(0), max(0) {}

    void    push() { hist.push(curr); }
    void    pop () { curr = hist.popC(); }
    uint    alloc(uint n) { uint ret = curr; curr += n; newMax(max, curr); return ret; }

    uint    getMaxAlloc() const { return max; }
};


// Environment for code emission. Keeps track of code pieces and linearizes and backpatches them once
// whole program has been compiled. Also handles stack space allocation for local variables and
// symbol tables.
class Env {
    typedef SymTable<RelPos> SymTab;

    Vec<uint>             active;     // -- which element in 'code[]' are we working on?
    Vec<Vec<Instr>>       codes;
    Vec<TmpAlloc>         tmps;
    Vec<SymTab*>          tabs;
    Set<SymTab*>          tab_allocs;
    uint                  block_counter = 0;
    Vec<Pair<Atom, Type>> blocks;

    Vec<Instr>& code() { return codes[active.last()]; }
    TmpAlloc&   tmp () { return tmps [active.last()]; }
    SymTab*&    tab () { return tabs [active.last()]; }

    Pair<uint,uint> currILoc() const { return tuple(active.last(), codes[active.last()].size()); }

    SymTab const* global_tab;
    addr_t        glob_off;

public:
    typedef Pair<uint,uint> ILoc;       // -- instruction location ('codes[fst][snd]'); can be used to later backpatch emitted instructions

    Env(SymTable<RelPos> const& tab0, addr_t glob_off) : global_tab(&tab0), glob_off(glob_off) { startFun(allocFun()); } // -- ready to receive top-level code
   ~Env() { For_Set(tab_allocs){ delete Set_Key(tab_allocs); } }

  //________________________________________
  //  Scope and code-block management:

    // Returns a 'fun_id' that can be used in 'startFun' and 'emit_PUT_CP. Will create a new symbol
    // table on top of the global one.
    uint allocFun() {
        int ret = codes.size();
        codes.push();
        tmps.push();
        tabs.push(new SymTab(global_tab));
        tab_allocs.add(tabs.last());
        return ret;
    }

    void startFun(uint fun_id) {
        active.push(fun_id);
        startScope();
        emit(pi_LABEL, fun_id);
        emit((active.size() == 1) ? i_GLOBALS : i_LOCALS, 0);          // -- will be backpatched in 'endFun()'
    }

    void endFun() {
        emit((active.size() == 1) ? i_HALT : i_RETURN, 0);
        code()[1].n_words = tmp().getMaxAlloc();    // -- backpatch
        endScope();
        active.pop();
    }

    void startScope() {
        tmp().push();
        tab() = new SymTab(*tab());
        tab_allocs.add(tab());
    }

    void endScope() {
        tmp().pop();
        SymTab* parent = const_cast<SymTab*>(tab()->getParent());
        if (global_tab == tab()) global_tab = parent;
        delete tab();
        if (!tab_allocs.exclude(tab())) assert(false);
        tab() = parent;
    }

    void setGlobalSymTab() { global_tab = tab(); }     // -- mark where global scope begins
    SymTab const& getGlobalSymTab() { assert(global_tab); return *global_tab; }

    addr_t finalize(Vec<Instr>& code, VmState const& vm_state);

  //________________________________________
  //  Code emission:

    RelPos reserve(uint n) {  // -- reserve space on stack (or start of heap for global scope)
        return active.size() == 1 ? RelPos(b_ZERO , tmp().alloc(n) + glob_off)
                                  : RelPos(b_LOCAL, tmp().alloc(n));
    }

    template<class... Args>
    ILoc emit(OpCode op, uint n, Args&&... args) {
        ILoc iloc = currILoc();
        code().push(Instr(op, n));
        emitArgs(forward<Args>(args)...);
        return iloc;
    }

    void emitArgs() {}

    template<class T, class... Args>
    void emitArgs(T t, Args... args) {
        code().push(Instr(t));
        emitArgs(args...); }

    ILoc emit_PUT_CP(RelPos dst, uint fun_id) {     // -- will be backpatched
        ILoc iloc = currILoc();
        code().push(Instr(pi_PUT_CP, fun_id));
        code().push(Instr(dst));
        return iloc;
    }

    Instr* backpatch(ILoc instr_loc){ return &codes[instr_loc.fst][instr_loc.snd]; }

    void addBlock(Type const& ty){
        Atom tag = fmt("\"__block#_%___\"", block_counter++);   // -- block names have to be unique
        blocks.push(tuple(tag, ty)); }
    Atom getBlockTag (uint level = 1) { return (level > blocks.size()) ? Atom() : blocks[END - level].fst; }
    Type getBlockType(uint level = 1) { return (level > blocks.size()) ? Type() : blocks[END - level].snd; }
        // -- 1=last block, 2=second last block etc.

    // Symbol table:
    SymTab& symTab() { return *tab(); }
    void    addQ(Atom name, RelPos pos) { symTab().addQ(name, pos); }       // }- for convenience
    RelPos  ref (Atom name)             { return symTab().ref(name); }      // }
    RelPos  operator[](Atom name) const { return const_cast<Env*>(this)->symTab().operator[](name); }  // -- deep lookup
};


addr_t Env::finalize(Vec<Instr>& code, VmState const& vm_state)
{
    endFun();

//**/wrLn("`` Experimental code optimization BEGINS");
//**/optimize(codes, glob_off, vm_state);
//**/wrLn("`` Experimental code optimization ENDS");

    addr_t ret = code.size();
    Map<uint, int> label2addr;
    Vec<Pair<uind, uint>> backpatch;    // (index_into_code, label)

    auto isEmptyCopy = [](Instr const& ins) {
        return ins.n_words == 0 && (ins.op_code == i_COPY || ins.op_code == i_COPY_IS || ins.op_code == i_COPY_ID); };

    bool dump_subroutines = getenv("SUBS");
    for (uint i = 0; i < codes.size(); i++){
        if (dump_subroutines){
            wrLn("==== codes[%_] ====", i);
            dumpCode(codes[i]); newLn(); }

        addr_t subr_loc = code.size();
        code.push(Instr(i_SUBR, 0));
        for (uint j = 0; j < codes[i].size();){
            // <<== this might be a good place to apply some simple copy-elimination optimization
            Instr* ins = &codes[i][j];
            uint sz = instr_size[ins->op_code];

            if (ins->op_code == pi_LABEL){
                assert(int64(code.size()) < int64(INT_MAX));
                label2addr.set(ins->n_words, code.size());

            }else if (ins->op_code == pi_PUT_CP){
                uint label = ins->n_words;
                code.push(Instr(i_PUT_CP, 1));
                code.push(ins[1]);
                backpatch.push(tuple(code.size(), label));
                code.push(0);

            }else if (!isEmptyCopy(*ins)){
                for (uint n = 0; n < sz; n++)
                    code.push(ins[n]);
            }

            j += sz;
        }
        code[subr_loc].n_words = code.size() - subr_loc - 1;
    }

    for (auto&& p : backpatch){
        int addr;
        if (!label2addr.peek(p.snd, addr)) assert(false);
        code[p.fst] = addr;
    }

    return ret + 1;     // -- '+ 1' due to the 'i_SUBR' pseudo instruction
}


}
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


static IntMap<Atom, uint> repr_size(2); // -- default is 'OneOf' size == 'data' type size.
ZZ_Initializer(repr_size, 10){
    repr_size(a_Void ) = 0;             // -- no space needed for voids
    repr_size(a_Bool ) = 1;             //
    repr_size(a_Int  ) = 1;             //
    repr_size(a_Float) = 1;             // -- double (word is 64-bit, so one word)
    repr_size(a_Atom ) = 1;             // -- integer
    repr_size(a_Tuple) = UINT_MAX;      // -- (illegal to take size of)
    repr_size(a_Ref  ) = 1;             // -- address (integer)
    repr_size(a_OneOf) = 2;             // -- (alt#, ref)
    repr_size(a_Fun  ) = closure_sz;    // -- closure
    repr_size(a_Vec  ) = 1;             // -- reference
    repr_size(a_Any  ) = UINT_MAX;      // -- (illegal to take size of)
    repr_size(Atom() ) = UINT_MAX;      // -- (illegal to take size of)
}


// Returns the number of 'uint's needed to represent an object of type 'type'.
uint reprSize(Type const& type)
{
    if (type.name == a_Tuple){
        uint sz = 0;
        for (Type const& t : type) sz += reprSize(t);
        return sz;
    }else{
        assert(repr_size[type.name] != UINT_MAX);
        return repr_size[type.name];
    }
}

inline uint reprSize(Expr const& expr) { return reprSize(expr.type); }


// Bind elements of tuple stored at 'pos' into symbols of pattern 'pat'. NOTE! 'pos' will be
// modified.
static void addPatternToSymTable(Expr const& pat, RelPos& pos, SymTable<RelPos>& tab)
{
    if (pat.kind == expr_Sym){
        assert(!pat.targs);
        tab.addQ(pat.name, pos);
        pos = pos + reprSize(pat);

    }else{ assert(pat.kind == expr_Tuple);
        for (Expr const & e : pat)
            addPatternToSymTable(e, pos, tab);
    }
}


static void addPatternToSymSet(Expr const& pat, SymTable<Unit>& tab)
{
    if (pat.kind == expr_Sym){
        assert(!pat.targs);
        tab.addQ(pat.name, UNIT);

    }else{ assert(pat.kind == expr_Tuple);
        for (Expr const & e : pat)
            addPatternToSymSet(e, tab);
    }
}


// Compute variables captured by expression, excluding local table 'ltab' (symbols introduced below
// point of capture) and global symbols of 'tab' (which are always available and don't need to be
// captured by closure). 'ltab' is the local part of the symbol table, belonging to the lambda we
// are doing capture analysis on. It is only used as a set, so the positions stored are unused
// (hence "dummy" type 'Unit'). The output result is stored in 'captures', mapping symbols to
// (location, size) representing from where they should be copied when creating the closure.
//
static void capturedVars(Expr const& expr, SymTable<RelPos> const& tab, SymTable<RelPos> const& tab_global, SymTable<Unit> const& ltab, SymTable<Pair<RelPos,uint>>& captures)
{
    switch (expr.kind){
    case expr_Sym:
        if (expr.name != a_underscore && !ltab.has(expr.name) && !captures.has(expr.name) && (!tab_global.has(expr.name) || tab_global[expr.name] != tab[expr]))
            // -- not underscore, not masked by 'ltab', not captured already, not in global scope (unless it is a different variable with the same name from the one in 'tab')...
            captures.addQ(expr.name, tuple(tab[expr], reprSize(expr)));
        break;

    case expr_Lamb:{
        SymTable<Unit> stab(ltab);
        addPatternToSymSet(expr[0], stab);
        capturedVars(expr[1], tab, tab_global, stab, captures);
        break; }

    case expr_Block:{
        SymTable<Unit> stab(ltab);
        uint j = 0;
        for (uint i = 0; i < expr.size(); i++){
            Expr const& e = expr[i];
            if (i >= j && e.kind == expr_RecDef){   // -- trigger on first rec-def only
                for (j = i; j < expr.size() && expr[j].kind == expr_RecDef; j++)
                    addPatternToSymSet(expr[j][0], stab);
            }
            if (e.kind == expr_LetDef)
                addPatternToSymSet(e[0], stab);
            capturedVars(e, tab, tab_global, stab, captures);
        }
        break; }

    case expr_Lit:
    case expr_MkVec:
    case expr_Tuple:
    case expr_MkRef:
    case expr_Deref:
    case expr_Appl:
    case expr_Cons:
    case expr_Sel:
    case expr_LetDef:
    case expr_RecDef:
        for (Expr const& e : expr)
            capturedVars(e, tab, tab_global, ltab, captures);
        break;

    case expr_TypeDef:
    case expr_DataDef:
        // Nothing to do:
        break; //

    default: assert(false); }
}


// Wrapper function. NOTE: 'tab_global' is an ancestor of 'tab', marking where the global scope begins
static SymTable<Pair<RelPos,uint>> capturedVars(Expr const& expr, SymTable<RelPos> const& tab, SymTable<RelPos> const& tab_global)
{
    SymTable<Pair<RelPos,uint>> captures;
    SymTable<Unit> ltab;
    capturedVars(expr, tab, tab_global, ltab, captures);
    return captures;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Code generation:


static void emitCode(Env& E, Expr const& expr, RelPos result, SymTable<RelPos>* global_block = nullptr);


static uint sizeOfCaptures(SymTable<Pair<RelPos,uint>> const& captures)
{
    uint sz = 0;
    for (Atom name : captures.locals())
        sz += captures.ref(name).snd;
    return sz;
}


static void emitLambda(Env& E, uint fun_id, SymTable<Pair<RelPos,uint>> const& captures, RelPos closure_pos, Expr const& lamb_expr)
{
    int off = 0;
    for (Atom name : captures.locals()){ RelPos pos = captures[name].fst; uint sz = captures[name].snd;
        E.emit(i_COPY_ID, sz, closure_pos+1, pos, off);
        off += sz; }

    // Create symbol table for captures and emit code for lambda function:
    E.startFun(fun_id);
    off = 0;
    for (Atom name : captures.locals()){ uint sz = captures[name].snd;
        E.addQ(name, RelPos(b_CLOSURE, off));
        off += sz; }
    RelPos arg(b_ARGS, 0);
    addPatternToSymTable(lamb_expr[0], arg, E.symTab());  // -- arguments
    emitCode(E, lamb_expr[1], RelPos(b_RETVAL, 0));
    E.endFun();
}


//=================================================================================================


// There are four reference types, with the following internal representations (after the ~>):
//
//     Ref<T>     = &T;     ~> ptr
//     Vec<T>     = [T];    ~> ptr
//     Fun<A,B>   = A -> B  ~> (addr, ptr)
//     OneOf<A,B> = {A,B}   ~> (alt#, ptr)
//
// For all but the last one, the pointer points to a fixed-sized area, and can thus be allocated
// before it is initialized. This semi-initialized variables are sufficient for "reference use" (as
// opposed to "value use") and can be copied, for instance when a lambda closure is created. This
// allows cyclic dependencies in let-rec constructs to be broken.
//
// A full anaylsis, however, seems overly complicated, so we will only allow cycles for 'rec'
// definitions of type 'Fun' (not even pairs '(Int->Int, Int-Int)' are allowed). Any free variable
// in a lambda is allowed to refer to a symbol initialized after the current definition, but all
// other symbols must be initialized fully before.
//
// SUMMARY: Evo's "let rec" allows simple functions to be mutually recursive; nothing else (no "rec
// (f, g) = (\.., \..)" or references "rec xs = List.1(42, xs)".
//
static void emitRecs(Env& E, Array<Expr const> block)
{
    SymTable<Expr const*> defs;
    SymTable<uint>        fun_ids;
    SymTable<Env::ILoc>   heap_allocs;
    SymTable<RelPos>      sympos;   // -- to catch cyclic definitions, we must delay adding symbols to 'E'; this table is used for that purpose

    auto isFunDef = [](Expr const& e) { assert(e.kind == expr_RecDef); return e[1].kind == expr_Lamb; };

    for (Expr const& e : block){
        if (e.kind == expr_RecDef){
            assert(!e[0].targs);
            Atom sym = e[0].name;
            defs.addQ(sym, &e);     // -- for evaluation order computation

            RelPos pos = E.reserve(reprSize(e[0].type));
            sympos.addQ(sym, pos);  // -- storage location

            if (isFunDef(e)){
                uint fun_id = E.allocFun();
                fun_ids.addQ(sym, fun_id);
                E.emit_PUT_CP(pos, fun_id);
                heap_allocs.addQ(sym, E.emit(i_ALLOC, 0, pos+1));   // -- the '0' will be backpatched
                E.addQ(sym, sympos.ref(sym));   // -- these symbols are added immediately
            }
        }
    }

    IntSeen<Atom> in_process;   // -- cycle detection
    IntSeen<Atom> done;

    function<void(Expr const&)> scanExpr;

    // Emit a single 'rec' definition after first emitting all 'rec' definitions it "value" depends on.
    function<void(Expr const&)> emitRec = [&](Expr const& e) {
        assert(e.kind == expr_RecDef);
        Atom sym = e[0].name;
        if (done.has(sym)) return;
        if (in_process.add(sym)) throw Excp_ParseError(fmt("%_: Cyclic 'rec' definition detected for '%_'.", e.loc, sym));

        scanExpr(e[1]);
        if (isFunDef(e)){
            SymTable<Pair<RelPos,uint>> captures = capturedVars(e[1], E.symTab(), E.getGlobalSymTab());
            uint fun_id = fun_ids.ref(sym);
            E.backpatch(heap_allocs.ref(sym))[0].n_words = sizeOfCaptures(captures);
            emitLambda(E, fun_id, captures, E.ref(sym), e[1]);
        }else{
            emitCode(E, e[1], sympos.ref(sym));
            E.addQ(sym, sympos.ref(sym));
        }
        done.add(sym);
    };

    // Emit all "value" dependencies of 'e'.
    scanExpr = [&](Expr const& e) {
        switch (e.kind){
        case expr_Sym:
            if (defs.has(e.name)){ emitRec(*defs.ref(e.name)); }
            break;
        case expr_Block:
        case expr_Lamb:{
            SymTable<Pair<RelPos,uint>> captures = capturedVars(e, E.symTab(), E.getGlobalSymTab());
            for (Atom name : captures.locals())
                if (defs.has(name) && (e.kind == expr_Block || !isFunDef(*defs.ref(name)))){
                    emitRec(*defs.ref(name)); }
            break; }
        case expr_LetDef: case expr_RecDef: case expr_TypeDef: case expr_DataDef:
            break;
        default:
            for (Expr const& s : e) scanExpr(s);
        }
    };

    // Emit all 'rec' definitions:
    for (Expr const& e : block)
        if (e.kind == expr_RecDef){
            emitRec(e); }
}


// Curry a built in instruction (create a lambda closure that runs the instruction once the argument is given).
// 'result' is the location to put the lambda closure in.
template<class F>
inline void curry(Env& E, RelPos result, F f, uint alloc_sz = 0) {
    uint fun_id = E.allocFun();
    E.emit_PUT_CP(result, fun_id);
    E.emit(i_ALLOC, alloc_sz, result+1);
    E.startFun(fun_id);
    f();
    E.endFun();
}


void emitWrite(Env& E, Type const& type, RelPos arg, Loc err_loc, bool top_tuple = false)
{
    if      (type.name == a_Void ) E.emit(i_PR_TEXT , 1, +Atom("()"));
    else if (type.name == a_Bool ) E.emit(i_PR_BOOL , 1, arg);
    else if (type.name == a_Int  ) E.emit(i_PR_INT  , 1, arg);
    else if (type.name == a_Float) E.emit(i_PR_FLOAT, 1, arg);
    else if (type.name == a_Atom ) E.emit(i_PR_ATOM , 1, arg);

    else if (type.name == a_Tuple){
        if (!top_tuple) E.emit(i_PR_TEXT, 1, +Atom("("));      // <<== utom top-tuple...
        bool first = true;
        for (Type const& t : type){
            if (first) first = false;
            else if (!top_tuple) E.emit(i_PR_TEXT, 1, +Atom(", "));
            emitWrite(E, t, arg, err_loc);
            arg = arg + reprSize(t);
        }
        if (!top_tuple) E.emit(i_PR_TEXT, 1, +Atom(")"));

    }else if (type.name == a_Fun){
        if (type[0].name != a_Void || type[1].name != a_Void)
            throw Excp_ParseError(fmt("%_: '%_' can only take Void->Void functions, not '%_'.", err_loc, a_write, type));
        E.emit(i_CALL, 0, RelPos(), arg, 0, RelPos());

    }else if (type.name == a_Ref || type.name == a_Vec){    // -- for debugging
        E.emit(i_PR_TEXT, 1, +Atom("@"));
        E.emit(i_PR_INT , 1, arg);

    }else{                                                  // -- for debugging
        E.emit(i_PR_TEXT, 1, +type.name);
        E.emit(i_PR_TEXT, 1, +Atom("("));
        E.emit(i_PR_INT , 1, arg);
        E.emit(i_PR_TEXT, 1, +Atom(", @"));
        E.emit(i_PR_INT , 1, arg+1);
        E.emit(i_PR_TEXT, 1, +Atom(")"));
    }
}


static void emitCode(Env& E, Expr const& expr, RelPos result, SymTable<RelPos>* global_block)
{
    assert(result);
    switch (expr.kind){
    case expr_Sym:
    case expr_Lit:{
        RelPos pos = E[expr.name];
        if (pos.base != b_INTERNAL)
            E.emit(i_COPY, reprSize(expr), result, pos);
        else{
            // <<== BI for now; later we are going to generalize compile-time generated functions' interface
            if (expr.name == a_case){
                curry(E, result, [&](){     // -- for T = {T0, T1, T2, ...} in 'case<&T,R>', type is: (T, &T0->R, &T1->R, ...) -> R
                    E.emit(i_CASE, reprSize(expr.type[1]), RelPos(b_RETVAL, 0), reprSize(expr.type[0]), RelPos(b_ARGS, 0)); });

            }else if (expr.name == a_ite){
                curry(E, result, [&](){     // -- type is: (Bool, A, A) -> A
                    E.emit(i_ITE, reprSize(expr.type[1]), RelPos(b_RETVAL, 0), RelPos(b_ARGS, 0)); });

            }else if (expr.name == a_assign){
                curry(E, result, [&](){     // -- type is: (&A, A) -> Void
                    E.emit(i_COPY_ID, reprSize(expr.type[0][1]), RelPos(b_ARGS, 0), RelPos(b_ARGS, 1), 0); });

            }else if (expr.name == a_fail){
                E.emit(i_THROW, 0, +a_Void, RelPos(b_ZERO, 0));
                //E.emit(i_HALT, +Atom(fmt("%_: %_<%_> executed.", expr.loc, a_fail, expr.type)));

            }else if (expr.name == a_try){
                Atom a_type(fmt("%_", expr.type[0][1][0]));
                int tt_sz = reprSize(expr.type[0][1][0]);
                curry(E, result, [&](){     // -- type is: (Void->A, TT->A) -> A
                    E.emit(i_TRY, reprSize(expr.type[1]), RelPos(b_RETVAL, 0), RelPos(b_ARGS, 0), RelPos(b_ARGS, closure_sz), +a_type, RelPos(b_CLOSURE, 0), tt_sz); }
                    , tt_sz);

            }else if (expr.name == a_throw){
                Atom a_type(fmt("%_", expr.type[0]));
                curry(E, result, [&](){     // -- type is: TT -> A
                    E.emit(i_THROW, reprSize(expr.type[0]), +a_type, RelPos(b_ARGS, 0)); });

            }else if (expr.name == a_run){
                curry(E, result, [&](){     // -- type is: (Void->A, Int, Int, Int) -> A
                    E.emit(i_RL_RUN, reprSize(expr.type[1]), RelPos(b_RETVAL, 0), RelPos(b_ARGS, 0), RelPos(b_ARGS, closure_sz), RelPos(b_ARGS, closure_sz+1), RelPos(b_ARGS, closure_sz+2)); });

            // VECTOR INSTRUCTIONS:   (types of form: ([T], ...) -> R; with empty '...' part for 'size' and 'dup')
            }else if (expr.name == a_size  ){ curry(E, result, [&](){ E.emit(i_VEC_SIZE  , reprSize(expr.type[0][0]   ), RelPos(b_RETVAL, 0), RelPos(b_ARGS, 0)); });
            }else if (expr.name == a_get   ){ curry(E, result, [&](){ E.emit(i_VEC_GET   , reprSize(expr.type[0][0][0]), RelPos(b_RETVAL, 0), RelPos(b_ARGS, 0), RelPos(b_ARGS, 1)); });
            }else if (expr.name == a_growby){ curry(E, result, [&](){ E.emit(i_VEC_GROWBY, reprSize(expr.type[0][0][0]), RelPos(b_ARGS, 0), RelPos(b_ARGS, 1)); });
            }else if (expr.name == a_resize){ curry(E, result, [&](){ E.emit(i_VEC_RESIZE, reprSize(expr.type[0][0][0]), RelPos(b_ARGS, 0), RelPos(b_ARGS, 1)); });
            }else if (expr.name == a_set   ){ curry(E, result, [&](){ E.emit(i_VEC_SET   , reprSize(expr.type[0][0][0]), RelPos(b_ARGS, 0), RelPos(b_ARGS, 1), RelPos(b_ARGS, 2)); });

            // PRINT INSTRUCTIONS:
            }else if (expr.name == a_print_bool ){ curry(E, result, [&](){ E.emit(i_PR_BOOL , 1, RelPos(b_ARGS, 0)); });
            }else if (expr.name == a_print_int  ){ curry(E, result, [&](){ E.emit(i_PR_INT  , 1, RelPos(b_ARGS, 0)); });
            }else if (expr.name == a_print_float){ curry(E, result, [&](){ E.emit(i_PR_FLOAT, 1, RelPos(b_ARGS, 0)); });
            }else if (expr.name == a_print_atom ){ curry(E, result, [&](){ E.emit(i_PR_ATOM , 1, RelPos(b_ARGS, 0)); });

            // PRE-PROCESSOR LIKE SYMBOLS:
            }else if (expr.name == a_line){ E.emit(i_PUT, 1, result, expr.loc.line);
            }else if (expr.name == a_file){ E.emit(i_PUT, 1, result, +expr.loc.file);

            }else assert(false);
        }
        break; }

    case expr_MkVec:{
        // Alloc:
        uint elem_sz = reprSize(expr.type[0]);
        E.emit(i_ALLOC, vechead_sz, result);        // -- alloc will clear memory which is the proper representation for an empty vector
        RelPos vsz = E.reserve(1);
        E.emit(i_PUT, 1, vsz, (int)expr.size());    // -- currently no instruction for resizing to an absolute size, so copy size to stack variable 'vsz'
        E.emit(i_VEC_RESIZE, elem_sz, result, vsz);
        // Populate:
        RelPos ptr = E.reserve(1);
        E.emit(i_COPY_IS, 1, ptr, result, 0);       // -- layout of vec-head is (data, sz, cap), so copy data pointer (now after resize) to stack variable 'ptr'
        RelPos tmp = E.reserve(elem_sz);
        int off = 0;
        for (Expr const& e: expr){
            emitCode(E, e, tmp);
            E.emit(i_COPY_ID, elem_sz, ptr, tmp, off);
            off += elem_sz;
        }
        break; }

    case expr_Tuple:{
        for (Expr const& e : expr){
            emitCode(E, e, result);
            result.offset += reprSize(e); }
        break; }

    case expr_MkRef:{
        E.emit(i_ALLOC, reprSize(expr.type[0]), result);
        uint src_sz = reprSize(expr[0]);
        RelPos src = E.reserve(src_sz);
        emitCode(E, expr[0], src);
        E.emit(i_COPY_ID, src_sz, result, src, 0);
        break; }

    case expr_Deref:{
        uint src_sz = reprSize(expr);
        if (src_sz > 0){    // -- no point following '&Void's
            RelPos src = E.reserve(src_sz);
            emitCode(E, expr[0], src);
            Atom msg(fmt("%_: Dereferencing null of type '%_'.", expr.loc, expr[0].type[0]));
            E.emit(i_CHK_REF, +msg, src);
            E.emit(i_COPY_IS, src_sz, result, src, 0);
        }
        break; }

    case expr_Lamb:{
        SymTable<Pair<RelPos,uint>> captures = capturedVars(expr, E.symTab(), E.getGlobalSymTab());
        uint fun_id = E.allocFun();
        E.emit_PUT_CP(result, fun_id);
        E.emit(i_ALLOC, sizeOfCaptures(captures), result+1);
        emitLambda(E, fun_id, captures, result, expr);
        break; }

    case expr_Appl:{
        // Special builtins 'write_', 'ttry_', 'tthrow_' must take arguments:
        if (expr[0].kind == expr_Sym && expr[0].name == a_write){
            // Special builtin 'write_' must take argument:
            RelPos arg = E.reserve(reprSize(expr[1]));
            emitCode(E, expr[1], arg);
            emitWrite(E, expr[1].type, arg, expr[1].loc, /*top_tuple*/true);
            break;

        }else if (expr[0].kind == expr_Sym && expr[0].name == a_ttry){
            Expr const& arg = expr[1];
            if (arg.kind != expr_Tuple || arg[0].kind != expr_Lit)
                throw Excp_ParseError(fmt("%_: Tagged try must have constant Atom as first argument.", expr.loc));
            Atom a_tagged_type(fmt("%_:%_", arg[0].name, arg.type[2][0]));

            int tt_sz = reprSize(arg.type[2][0]);
            RelPos try_fun   = E.reserve(closure_sz);
            RelPos catch_fun = E.reserve(closure_sz);
            RelPos excp_tmp  = E.reserve(1);
            emitCode(E, expr[1][1], try_fun);
            emitCode(E, expr[1][2], catch_fun);
            E.emit(i_ALLOC, tt_sz, excp_tmp);
            E.emit(i_TRY, reprSize(expr.type), result, try_fun, catch_fun, +a_tagged_type, excp_tmp, tt_sz);
            break;

        }else if (expr[0].kind == expr_Sym && expr[0].name == a_tthrow){
            Expr const& arg = expr[1];
            if (arg.kind != expr_Tuple || arg[0].kind != expr_Lit)
                throw Excp_ParseError(fmt("%_: Tagged throw must have constant Atom as first argument.", expr.loc));
            Atom a_tagged_type(fmt("%_:%_", arg[0].name, expr[0].type[0][1]));

            uint thr_sz = reprSize(arg[1]);     // -- thrown object
            RelPos thr = E.reserve(thr_sz);
            emitCode(E, arg[1], thr);
            E.emit(i_THROW, thr_sz, +a_tagged_type, thr);
            break;

        }else if (expr[0].kind == expr_Sym && expr[0].name == a_block){
            // Emit 'ttry_<T,T>(tag, \{ code }, \ret{ret})':
            Type ty_T = expr[1].type;
            E.addBlock(ty_T);
            Type ty_ttry = Type(a_Fun, {Type(a_Tuple, {Type(a_Atom), Type(a_Fun, {Type(a_Void), ty_T}), Type(a_Fun, {ty_T, ty_T})}), ty_T});    // -- "(Atom, Void->T, T->T) -> T"
            Expr e_thunk = mxLamb(mxTuple({}), expr[1]);
            Expr e_ident = mxLamb(mxSym(a_x, ty_T), mxSym(a_x, ty_T));
            Expr e = mxAppl(mxSym(a_ttry, ty_ttry), mxTuple({mxLit_Atom(E.getBlockTag()), e_thunk, e_ident}));
            emitCode(E, e, result);
            break;

        }else if (expr[0].kind == expr_Sym && expr[0].name == a_break){
            Expr const& arg = expr[1];
            uint        lev;
            Expr const* val;
            if (arg.type.name == a_Tuple){
                if (arg.kind != expr_Tuple || arg.size() != 2 || arg[0].type.name != a_Int || arg[0].kind != expr_Lit)
                    throw Excp_ParseError(fmt("%_: Keyword 'break' takes a direct argument of type 'T' or '(Int, T)' where the integer must be a literal 1, 2, 3...", arg.loc));
                lev = stringToInt64(arg[0].name);
                val = &arg[1];
            }else{
                lev = 1;
                val = &arg;
            }
            Atom tag = E.getBlockTag(lev);
            if (!tag) throw Excp_ParseError(fmt("%_: Breaking out of more blocks than declared.", arg.loc));
            Type ty_T = E.getBlockType(lev);
            if (val->type != ty_T) throw Excp_ParseError(fmt("%_: Breaking with wrong type '%_'. Should be '%_'.", arg.loc, val->type, ty_T));
            // Emit 'tthrow_<(),T>(tag, val)'
            Type ty_tthrow = Type(a_Fun, {Type(a_Tuple, {Type(a_Atom), ty_T}), Type(a_Void)});
            Expr e_arg = mxTuple({mxLit_Atom(tag), *val});
            Expr e = mxAppl(mxSym(a_tthrow, ty_tthrow), e_arg);
            emitCode(E, e, result);
            break;
        }

        // Normal function application:
        uint arg_sz = reprSize(expr[1]);
        uint ret_sz = reprSize(expr);
        RelPos fun = E.reserve(closure_sz);
        RelPos arg = E.reserve(arg_sz);
        emitCode(E, expr[1], arg);  // -- for side-effects such as exception, we choose to evaluate the argument before the function
        emitCode(E, expr[0], fun);
        Atom msg(fmt("%_: Calling null of type '%_'.", expr[0].loc, expr[0].type));
        E.emit(i_CHK_REF, +msg, fun);
        E.emit(i_CALL, ret_sz, result, fun, arg_sz, arg);
        break; }

    case expr_Cons:{    // -- 'expr.targs[0]' = type to construct, 'expr.name' is sum-type alternative number
        uint fun_id = E.allocFun();
        E.emit_PUT_CP(result, fun_id);
        E.emit(i_ALLOC, 0, result+1);

        E.startFun(fun_id);
        Type t0 = expr.targs[0];
        uint idx = stringToInt64(expr.name); assert(idx < t0.size());
        Type t = t0[idx];
        uint sz = reprSize(t);
        E.emit(i_PUT, 1, RelPos(b_RETVAL, 0), idx);
        E.emit(i_ALLOC, sz, RelPos(b_RETVAL, 1));
        E.emit(i_COPY_ID, sz, RelPos(b_RETVAL, 1), RelPos(b_ARGS, 0), 0);
        E.endFun();
        break; }

    case expr_Sel:{
        RelPos src = E.reserve(reprSize(expr[0]));
        emitCode(E, expr[0], src);
        uint sel_idx = stringToInt64(expr.name);
        uint sz = 0;
        bool is_ref = (expr[0].type.name == a_Ref);
        Type type = is_ref ? expr[0].type[0] : expr[0].type;
        uint off = 0; assert(type.size() > 0);
        for (Type const& t : type){
            sz = reprSize(t);
            if (sel_idx == 0) break;
            off += sz;
            sel_idx--;
        }
        if (is_ref){
            Atom msg(fmt("%_: Select on null of type '%_'.", expr.loc, expr.type[0]));
            E.emit(i_CHK_REF, +msg, src);
            E.emit(i_LEA, off, result, src);
        }else
            E.emit(i_COPY, sz, result, src + off);
        break; }

    case expr_Block:{
        // LetDef: 'expr[0]' is variable pattern, 'expr[1]' values, 'expr.type' is the type of the pattern
        // RecDef: 'expr[0]' is constant symbol , 'expr[1]' value, 'expr.type' is the type of the new symbol
        if (expr.size() == 0) break;

        E.startScope();
        if (global_block)
            E.setGlobalSymTab();

        uint j = 0;
        for (uint i = 0; i < expr.size(); i++){
            Expr const& e = expr[i];
            if (i >= j && e.kind == expr_RecDef){   // -- trigger on first rec-def only
                for (j = i+1; j < expr.size() && expr[j].kind == expr_RecDef; j++);
                emitRecs(E, slice(expr[i], expr[j]));
            }
            if (e.kind != expr_RecDef){
                emitCode(E, e, (i+1 == expr.size()) ? result : E.reserve(reprSize(e))); }
        }

        if (global_block){
            // Migrate symbols to global symbol table ('tab0' of VM):
            for (Atom name : E.symTab().locals())
                global_block->addQ(name, E.ref(name));
        }
        E.endScope();
        break; }

    case expr_LetDef:{
        if (expr.size() == 1){  // -- special let: reference to default value
            assert(expr[0].kind == expr_Sym);
            if (expr.type.name == a_Ref)
                E.emit(i_ALLOC, reprSize(expr.type[0]), result);
            else assert(expr.type.name == a_Vec),
                E.emit(i_ALLOC, vechead_sz, result);
        }else
            emitCode(E, expr[1], result);
        addPatternToSymTable(expr[0], result, E.symTab());
        break; }

    case expr_RecDef:
        // (nothing to do, handled by parent 'expr_Block')
        break;

    case expr_TypeDef:
    case expr_DataDef:
        // (nothing to do)
        break;

    default: assert(false); }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Wrapper:


// Adds code to 'code' and increases persitent data pointer 'pp'; start address is returned.
addr_t compile(Vec<Instr>& code, Expr const& prog, SymTable<RelPos>& tab0, RelPos& result, addr_t glob_off, VmState const& vm_state)
{
    ZZ_PTimer_Scope(evo_core_compilation);

    Env E(tab0, glob_off);

    result = E.reserve(reprSize(prog));     // -- we may want to place this on the heap so that it is not kept forever
    emitCode(E, prog, result, &tab0);
    return E.finalize(code, vm_state);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

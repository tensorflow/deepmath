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

#ifndef ZZ__CodeBreeder__Types_hh
#define ZZ__CodeBreeder__Types_hh

#include "zz/Generics/Arr.hh"
#include "zz/Generics/Atom.hh"
#include "zz/Generics/Map.hh"
#include <limits>
#include "ProtoBuf.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Symbol naming conventions:


bool isType(Str str);
bool isTVar(Str str);
bool isVar (Str str);

extern Atom a_x;
extern Atom a_y;
extern Atom a_z;

extern Atom a_case;
extern Atom a_ite;
extern Atom a_assign;
extern Atom a_fail;
extern Atom a_write;
extern Atom a_run;
extern Atom a_match;

extern Atom a_size;
extern Atom a_get;
extern Atom a_growby;
extern Atom a_resize;
extern Atom a_set;
extern Atom a_dup;

extern Atom a_print_bool;
extern Atom a_print_int;
extern Atom a_print_float;
extern Atom a_print_atom;

extern Atom a_try;
extern Atom a_throw;
extern Atom a_ttry;
extern Atom a_tthrow;
extern Atom a_block;
extern Atom a_break;

extern Atom a_underscore;
extern Atom a_false;
extern Atom a_true;
extern Atom a_appl_op;      // -- pseudo-operator, outside the operator alphabet

extern Atom a_line;
extern Atom a_file;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Program locations:


struct Loc {
    uint line;
    uint col;
    Atom file;
    Loc(uint line = 0, uint col = 0, Atom file = Atom()) : line(line), col(col), file(file) {}
    explicit operator bool() const { return line != 0; }

    bool operator==(Loc q) const { return line == q.line && col == q.col && file == q.file; }
    bool operator< (Loc q) const { return tuple(file, line, col) < tuple(q.file, q.line, q.col); }
};

template<> fts_macro void write_(Out& out, const Loc& v) {
    if (!v.file) wr(out, "[%_,%_]", v.line, v.col);
    else         wr(out, "[%_: %_,%_]", v.file, v.line, v.col);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Types:


extern Atom a_Void;
extern Atom a_Bool;
extern Atom a_Int;
extern Atom a_Float;
extern Atom a_Atom;
extern Atom a_Tuple;
extern Atom a_Ref;      // }
extern Atom a_OneOf;    // }- reference semantics
extern Atom a_Fun;      // }
extern Atom a_Vec;      // }

extern Atom a_Any;      // -- special type used for 'write_' in type-inference
extern Atom a_Internal; // -- special type used to mark built-in symbols in expansion phase
extern Atom a_A;        // }
extern Atom a_B;        // }- type variables

extern Atom a_List;     // -- standard types defined in 'std.evo'
extern Atom a_Maybe;
extern Atom a_Tree;
extern Atom a_Hide;

extern Vec<Atom> bit_primitive;     // -- nullary types from the list above
extern Vec<Atom> bit_composite;     // -- the remaining types (parser will guarantee arity constraints are met; type name is outside syntactic scope of types)


struct Type {
    Atom      name;
    Arr<Type> args;
    Loc       loc;

    explicit Type(Atom name = Atom())    : name(name) {}
    Type(Atom name, Type t0)             : name(name), args({t0}) {}
    Type(Atom name, Type t0, Type t1)    : name(name), args({t0, t1}) {}
    Type(Atom name, Vec<Type> const& ts) : name(name), args(ts) {}

    Type& setLoc(Loc loc_) { loc = loc_; return *this; }

    explicit operator bool() const { return name; }

    uint        size()             const { return args ? args.size() : 0; }
    Type&       operator[](uint i)       { return args[i]; }
    Type const& operator[](uint i) const { return args[i]; }
    Std_Array_Funcs(Type);

    bool operator==(Type const& t) const { return name == t.name && vecEqual(+args, +t.args); }
        // -- here we make no distinction between null 'args' and empty 'args'.
    bool operator< (Type const& t) const {
        if (+name < +t.name) return true;
        if (+name > +t.name) return false;
        return vecCompare(+args, +t.args);
    }

    bool isTVar() const { return !args && ::ZZ::isTVar(name); }
};


void writeType(Out& out, Type const& t, bool is_arg = false);
template<> fts_macro void write_(Out& out, Type const& t)            { writeType(out, t); }
template<> fts_macro void write_(Out& out, Type const& t, Str flags) { assert(eq(flags, "a")); writeType(out, t, true); }

// Hashing (ignores 'loc' field).
template<> struct Hash_default<Type> {
    uint64 hash (Type const& t) const { return defaultHash(tuple(defaultHash(t.name), defaultHash(+t.args))); }
    bool   equal(Type const& t1, Type const& t2) const { return t1 == t2; }
};


inline Array<Type const> tupleSlice(Type const& type) {    // -- unify treatment of tuples and expressions (considered here as tuple of size 1)
    return (type.name == a_Tuple || type.name == a_Void) ? type.slice() : Array<Type const>(&type, 1); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Expressions:


// NOTE! We treat statements as expressions; the distinction is mostly for the BNF grammar
// to make sure variable declarations are only allowed in reasonable places where there is no
// ambiguity about the scope.


enum ExprKind : uchar {
    expr_NULL,

    expr_Sym,
    expr_Op,
    expr_Lit,
    expr_MkVec,         // -- pseudo-literal; mostly for convenience when testing programs as a human
    expr_Tuple,
    expr_MkRef,
    expr_Deref,
    expr_Lamb,
    expr_Appl,
    expr_Cons,
    expr_Sel,
    expr_Block,

    expr_LetDef,
    expr_RecDef,

    expr_TypeDef,       // }- pseudo-expression representing the introduction of a type (only at
    expr_DataDef,       // }  global scope)

    expr_MetaIf,        // -- meta-programming if
    expr_MetaErr,       // -- if expanded, it is a compile error

    ExprKind_size
};

extern cchar* ExprKind_name[ExprKind_size];
template<> fts_macro void write_(Out& out, ExprKind const& v) { out += ExprKind_name[(uint)v]; }


struct Expr {
    Expr(ExprKind kind, Atom name = Atom(), Arr<Expr>&& exprs = Arr<Expr>(), Type&& type = Type(), Arr<Type>&& targs = Arr<Type>(), Loc loc = Loc()) :
        type(move(type)), loc(loc), kind(kind), name(name), targs(move(targs)), exprs(move(exprs)) {}

    // <<== mark type as given ("tc") or bottom-up inferred: may be important when doing mutations
    Type type;
    Loc  loc;           // -- for parsed expressions, reference into the text from which it was genereated

    ExprKind  kind;
    Atom      name;     // -- most often symbol name, but also stores the text of literals and selector indices
    Arr<Type> targs;    // -- type arguments (only for 'Sym' and 'Cons')
    Arr<Expr> exprs;    // -- sub-expressions (0-2, except for Tuple)

    Expr(): kind(expr_NULL) {}

    Expr(Expr const& e) : type(e.type), loc(e.loc), kind(e.kind), name(e.name), targs(e.targs), exprs(e.exprs) {}
    Expr(Expr&& e) { type = move(e.type); loc = e.loc; kind = e.kind; name = e.name; targs = move(e.targs); exprs = move(e.exprs); e.kind = expr_NULL; }

    Expr& setLoc(Loc loc_) { loc = loc_; return *this; }
    Expr& setType(Type&& t) { type = move(t); return *this; }

    Expr& operator=(Expr const& e) { type = e.type; loc = e.loc; kind = e.kind; name = e.name; targs = e.targs; exprs = e.exprs; return *this; }
    Expr& operator=(Expr&& e)      { type = move(e.type); loc = e.loc; kind = e.kind; name = e.name; targs = move(e.targs); exprs = move(e.exprs); e.kind = expr_NULL; return *this; }

    explicit operator bool() const { return kind != expr_NULL; }

    bool isUnit() const { return kind == expr_Tuple && exprs.size() == 0; }

    // Array access to 'exprs':
    uint        size()             const { return exprs ? exprs.size() : 0; }
    Expr&       operator[](uint i)       { return exprs[i]; }
    Expr const& operator[](uint i) const { return exprs[i]; }
    Std_Array_Funcs(Expr);

    // Named constructors:
    typedef Vec<Expr> const& Exprs;
    typedef Vec<Type> const& Types;

    static Expr Sym  (Atom name, Types targs, Type&& t){ Expr ret(expr_Sym, name, {}, move(t)); if (targs.size() > 0) ret.targs = Arr<Type>(targs); return ret; }
    static Expr Op   (Atom name)                       { Expr ret(expr_Op , name); return ret; }
    static Expr Lit  (Atom name, Type&& t)             { return Expr(expr_Lit  , name, {}, move(t)); }
    static Expr MkVec(Exprs es, Type&& t)              { return Expr(expr_MkVec, Atom()   , es, move(t)); }
    static Expr Tuple(Exprs es)                        { return (es.size() == 1) ? es[0] : Expr(expr_Tuple, Atom(), es); }
    static Expr Lamb (Expr&& hd, Expr&& bd, Type&& t)  { return Expr(expr_Lamb , Atom()   , {move(hd), move(bd)}, move(t)); }    // -- type only given for typed lambda "\:"
    static Expr MkRef(Expr&& e)                        { return Expr(expr_MkRef, Atom()   , {move(e)}); }
    static Expr Deref(Expr&& e)                        { return Expr(expr_Deref, Atom()   , {move(e)}); }
    static Expr Sel  (Expr&& e, Str idx)               { return Expr(expr_Sel  , Atom(idx), {move(e)}); }
    static Expr Appl (Expr&& fun, Expr&& arg)          { return Expr(expr_Appl , Atom()   , {move(fun), move(arg)}); }
    static Expr Cons (Type&& constr, Str idx)          { return Expr(expr_Cons,  Atom(idx), {}, Type(), {move(constr)}); }   // -- stores constructor type in 'targs[0]', which may match 'name'+'targs' of 'DataDef'.
    static Expr Block(Exprs&& es)                      { return Expr(expr_Block, Atom()   , es); }

    static Expr LetDef(Expr&& var , Type&& t)              { return Expr(expr_LetDef, Atom(), {move(var)             }, move(t)); }     // -- only references can be uninitialized
    static Expr LetDef(Expr&& vars, Type&& t, Expr&& init) { return Expr(expr_LetDef, Atom(), {move(vars), move(init)}, move(t)); }
    static Expr RecDef(Expr&& sym , Type&& t, Expr&& init) { return Expr(expr_RecDef, Atom(), {move(sym ), move(init)}, move(t)); }

    static Expr NewType(bool synon, Str name, Types targs, Type tdef) { return Expr(synon ? expr_TypeDef : expr_DataDef, Atom(name), {}, move(tdef), targs); }
        // -- 'tdef' is stored in 'type'

    static Expr MetaIf(Expr&& cond, Expr&& tt, Expr&& ff) { return Expr(expr_MetaIf, Atom(), {move(cond), move(tt), move(ff)}); }
    static Expr MetaErr(Atom msg = Atom(), Type&& t = Type()) { return Expr(expr_MetaErr, msg, {}, Type(), t ? Arr<Type>{move(t)} : Arr<Type>{}); }

    bool untypedEqualTo(Expr const& e) const {
        return kind == e.kind
            && name == e.name
            && vecEqual(+targs, +e.targs)
            && vecEqual(+exprs, +e.exprs);
    }

    // Comparison: (we make no distinction between null and empty 'Arr's, but *type* is part of expression)
    bool operator==(Expr const& e) const {
        return type == e.type && untypedEqualTo(e); }

    bool operator<(Expr const& e) const {
        if (type < e.type) return true;
        if (type > e.type) return false;
        if (kind < e.kind) return true;
        if (kind > e.kind) return false;
        if (+name < +e.name) return true;
        if (+name > +e.name) return false;
        if (vecCompare(+targs, +e.targs)) return true;
        if (vecCompare(+e.targs, +targs)) return false;
        return vecCompare(+exprs, +e.exprs);
    }
};


void writeExpr(Out& out, Expr const& e, bool is_arg = false);
template<> fts_macro void write_(Out& out, Expr const& e) { writeExpr(out, e); }
template<> fts_macro void write_(Out& out, Expr const& e, Str flags) { assert(flags[0] == 'a'); writeExpr(out, e, true); }

String ppFmtI(Expr const& expr, uint indent_level);     // -- pretty-print format
inline String ppFmt(Expr const& expr) { return ppFmtI(expr, 0); }


// Hashing (ignores 'loc' field).
template<> struct Hash_default<Expr> {
    uint64 hash (Expr const& e) const { return defaultHash(tuple(tuple(defaultHash(e.type), (uint)e.kind, +e.name), tuple(defaultHash(+e.targs), defaultHash(+e.exprs)))); }
    bool   equal(Expr const& e1, Expr const& e2) const { return e1 == e2; }
};

inline Array<Expr const> tupleSlice(Expr const& expr) {    // -- unify treatment of tuples and expressions (considered here as tuple of size 1)
    return (expr.kind == expr_Tuple) ? expr.slice() : Array<Expr const>(&expr, 1); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Bottom-up typed expression construction: [incomplete, add functions as necessary]


// mx = make expression

// Make sure 'e' is of type 'expr_Block' so that it fits the body of a lambda.
inline Expr wrapBlock(Expr const& e) {
    if (e.kind == expr_Block) return e;
    Vec<Expr> es(1, e);
    return Expr::Block(es).setType(Type(e.type));
}


inline Expr mxSym(Atom name, Type type) { return Expr::Sym(name, {}, move(type)); }

inline Expr mxLit_Bool(bool   v) { return Expr::Lit(v ? a_true : a_false, Type(a_Bool)); }
inline Expr mxLit_Int (int64  v) { return Expr::Lit(Atom(fmt("%_", v)), Type(a_Int)); }
inline Expr mxLit_Atom(Atom   v) { return Expr::Lit(v, Type(a_Atom)); }
    // -- ignore 'mkLit(Float)' for now; parse and print may not be perfect duals.

inline Expr mxTuple(Vec<Expr> const& es) {
    Vec<Type> ts(reserve_, es.size());
    for (Expr const& e : es) ts.push(Type(e.type));
    return Expr::Tuple(es).setType(ts.size() == 0 ? Type(a_Void) : Type(a_Tuple, ts));
}


inline Expr mxAppl(Expr fun, Expr arg) {
    assert(fun.type.name == a_Fun);
    assert(arg.type == fun.type[0]);
    Type type = fun.type[1];
    return Expr::Appl(move(fun), move(arg)).setType(Type(type));
}


inline Expr mxBlock(Vec<Expr> const& es) {
    Type t = (es.size() == 0 || es[LAST].kind == expr_RecDef) ? Type(a_Void) : Type(es[LAST].type);
    return Expr::Block(es).setType(move(t));
}


inline bool isPattern(Expr const& expr) {
    return expr.kind == expr_Sym
        || (expr.kind == expr_Tuple && trueForAll(expr, [](Expr const& e){ return isPattern(e); }));
}


inline Expr mxLamb(Expr head, Expr body) {
    assert(isPattern(head));
    Type t = Type(a_Fun, {Type(head.type), Type(body.type)});
    return Expr::Lamb(move(head), wrapBlock(body), move(t));
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Symbol table:


// A symbol here is just the name without the '<T1, T2...>' part.
// NOTE! If using 'add()', then template type 'T' must have a field 'Loc loc'.
template<class T>
class SymTable {
    Map<Atom, T>    sym2val;
    Vec<Atom>       syms;       // -- list of symbols (keys) in map 'sym2val' in the order they were added.
    SymTable const* parent;

    T const& lookup(Atom sym, Loc err_loc) const {
        T* ret;
        if (sym2val.peek(sym, ret)) return *ret;
        if (!parent){ throw Excp_ParseError(fmt("%_: Symbol '%_' undefined.", err_loc, sym)); }
        return parent->lookup(sym, err_loc); }

public:
    // Constructors:
    SymTable()                       : parent(nullptr) {}
    SymTable(SymTable const* parent) : parent(parent)  {}
    SymTable(SymTable const& parent) : parent(&parent) {}

    // Move semantics:
    SymTable(SymTable&& src) : sym2val(move(src.sym2val)), syms(move(src.syms)), parent(src.parent) {}
    SymTable& operator=(SymTable&& other) { other.sym2val.moveTo(sym2val); other.syms.moveTo(syms); parent = other.parent; return *this; }

    // Hierarchy:
    SymTable const* getParent() const { return parent; }
    void            setParent(SymTable const* p) { parent = p; }

    // Local scope:
    void add(Atom sym, T const& val) {
        if (sym == a_underscore) return;
        T* t;
        if (sym2val.get(sym, t)) throw Excp_ParseError(fmt("%_: Symbol '%_' already defined at %_.", val.loc, sym, t->loc));
        new (t) T(val); // -- copy construct 'val' into location returned by 'get()'
        syms.push(sym);
    }

    void addQ(Atom sym, T const& val) {     // -- quick add, when we know symbol cannot already exist
        if (sym == a_underscore) return;
        T* t;
        if (sym2val.get(sym, t)){
            wrLn("INTERNAL ERROR! 'SymTable::addQ(\"%_\")' on existing symbol.", sym);
            assert(false); }
        new (t) T(val);
        syms.push(sym);
    }

    // Shallow lookup, use this version only when the symbol is known to exist:
    T&       ref(Atom sym)       { T* ret; if (sym2val.peek(sym, ret)) return *ret; assert(false); }
    T const& ref(Atom sym) const { return const_cast<SymTable*>(this)->ref(sym); }

    Vec<Atom> const& locals() const { return syms; }

    // Deep lookups:
    T const& operator[](Type const& t) const { return lookup(t.name, t.loc); }
    T const& operator[](Expr const& e) const { assert(e.name); return lookup(e.name, e.loc); }
    T const& operator[](Atom name)     const { return lookup(name, Loc()); }

    bool has(Atom sym) const { return sym2val.has(sym) || (parent && parent->has(sym)); }
    bool has(Expr const &e) const { assert(e.name); return has(e.name); }
    bool has(Type const &t) const { return has(t.name); }

    // Debug:
    void dump(bool skip_global = false, function<void(T const&)> valPrinter = 0) const {
        uint scope = 0;
        for (SymTable const* p = this; p; p = p->parent, scope++);
        for (SymTable const* p = this; p; p = p->parent){
            scope--;
            if (scope == 0 && skip_global) return;
            wrLn("=== scope %_ ========================================", scope);
            if (!valPrinter)
                for (Atom a : p->locals()) wrLn("- %_", a);
            else
                for (Atom a : p->locals()){ wr("- %_ =", a); valPrinter(p->ref(a)); newLn(); }
        }
    }
    void dumpVal(bool skip_global = false) const { dump(skip_global, [](T const& t){ std_out += t; }); }

    // Undo:
    uind size() const { return syms.size(); }
    void pop() { sym2val.exclude(syms.last()); syms.pop(); }
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Functions:


// Represents type-substitution "a := ty".
struct TSubst {
    Type a;
    Type ty;
    TSubst(Type a = Type(), Type ty = Type()) : a(a), ty(ty) {}
};

template<> fts_macro void write_(Out& out, TSubst const& v) {
    wr(out, "%_ := %_", v.a, v.ty); }


Type typeSubst(Type const& type, Arr<Type> const& from, Arr<Type> const& into);
Type typeSubst(Type const& type, Vec<TSubst> const& subs);
Expr typeSubst(Expr expr, Arr<Type> const& from, Arr<Type> const& into);
Expr typeSubst(Expr expr, Vec<TSubst> const& subs);


bool        typeMatch(Type general, Type specific, /*out*/Vec<TSubst>& subs);
inline bool typeMatch(Type general, Type specific) { Vec<TSubst> subs; return typeMatch(general, specific, subs); }

double getFloat(Expr const& expr);
Atom   getAtom (Expr const& expr);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Cost types:


constexpr double DBL_INF = std::numeric_limits<double>::infinity();


// Pair of cost and 'T', ordered primarily on cost.
template<class T>
struct Cost {
    double cost;
    T      data;

    Cost() : cost(DBL_INF) {}
    Cost(double cost, T data = T()) : cost(cost), data(data) {}
    explicit operator bool() const { return bool(data); }

    // Access 'T':
    T const& operator* () const { return data; }
    T const* operator->() const { return &data; }

    bool operator==(Cost const& s) const { return cost == s.cost && data == s.data; }
    bool operator< (Cost const& s) const { return cost < s.cost || (cost == s.cost && data < s.data); }
};


typedef Cost<Type> CType;
typedef Cost<Expr> CExpr;

template<> struct Hash_default<CExpr> {
    uint64 hash (CExpr const& e) const { return defaultHash(tuple(*e, e.cost)); }
    bool   equal(CExpr const& e1, CExpr const& e2) const { return defaultEqual(*e1, *e2) && defaultEqual(e1.cost, e2.cost); }
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

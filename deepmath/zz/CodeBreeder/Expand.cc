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
#include "Expand.hh"
#include <memory>
#include "zz/Generics/Set.hh"

// Expands polymorphic functions and values. This must happen before compilation. The non-rec
// statements of the global scope defines the "sources". Only rec-expressions required to evaluate
// the sources are instantiated. The compiler may execute these rec-expressions of value type in
// any (feasible) order. For this reason, non-trivial side-effects in rec-expressions are
// discouraged.

namespace ZZ {
using namespace std;


ZZ_PTimer_Add(evo_template_expansion);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Rewrite helpers:


// This is pretty horrible, but it is not worth having a better rewrite interface unless it
// becomes a more prevalent method for compiling.
static Expr expandValueLambda(Expr const& e, SymTable<Expr> const& datadefs)
{
    // INPUT:  case_<A,B> : (A, T0->B, T1->B, ..., TN->B)->B
    // where:  type A = {T0, T1, ..., TN}
    //    or:  data A = {T0, T1, ..., TN}
    //
    // OUTPUT: \(a, f0, f1, ..., fN) { case_<&A, B>(a, \ref{f0 *ref}, \ref{f1 *ref}, ..., \ref{fN *ref}) }

    auto pNam = [](uint i) {
        static Vec<String> param_names = {"_a"};
        while (i >= param_names.size()) param_names.push(fmt("_f%_", param_names.size()-1));
        return param_names[i].slice();
    };

    // Build types:
    auto normType = [&datadefs](Type const t) {
        return datadefs.has(t.name) ? typeSubst(datadefs[t].type, datadefs[t].targs, t.args) : t;
    };
    assert(e.type.name == a_Internal);
    Type e_type = e.type[0];
    Type ty_A = normType(e_type[0][0]); assert(ty_A.name == a_OneOf);
    Type ty_B = e_type[1];

    Vec<Type> arg_ts(1, e_type[0][0]);  // -- we intentionally keep the unnormalized type here (for clarity of produced code)
    for (Type const& t : ty_A) arg_ts.push(Type(a_Fun, t, ty_B));
    Type ty_arg(a_Tuple, arg_ts); assert_debug(Type(a_Fun, ty_arg, ty_B) == e_type);

    // Build lambda-header and ref-case args:
    Vec<Expr> arg_es(1, Expr::Sym(pNam(0), {}, Type(arg_ts[0])));
    for (uint i = 1; i <= ty_A.size(); i++)
        arg_es.push(Expr::Sym(pNam(i), {}, Type(arg_ts[i])));
    Expr ex_arg = Expr::Tuple(arg_es);
    ex_arg.type = ty_arg;

    Vec<Expr> rcall_es(1, Expr::Sym(pNam(0), {}, Type(arg_ts[0])));
    Vec<Type> rcall_ts(1, rcall_es.last().type);
    for (uint i = 1; i <= ty_A.size(); i++){            // -- each argument is: \ref{fN *ref}
        Expr ref   = Expr::Sym(slize("_ref"), {}, Type(a_Ref, Type(ty_A[i-1])));
        Expr deref = Expr::Deref(Expr(ref)); deref.type = ty_A[i-1];
        Expr appl  = Expr::Appl(Expr(arg_es[i]), move(deref)); appl.type = ty_B;
        Expr body  = Expr::Block({move(appl)}); body.type = ty_B;
        Expr lamb  = Expr::Lamb(move(ref), move(body), Type(a_Fun, Type(a_Ref, Type(ty_A[i-1])), Type(ty_B)));
        rcall_es.push(move(lamb));
        rcall_ts.push(rcall_es.last().type);
    }
    Expr rcall_ex = Expr::Tuple(rcall_es);
    rcall_ex.type = Type(a_Tuple, rcall_ts);

    // Piece it together
    Expr case_ = Expr::Sym(a_case, {Type(a_Ref, ty_A), ty_B}, Type(a_Internal, Type(a_Fun, rcall_ex.type, ty_B)));  // -- must tag 'case_' with "Internal" to avoid name mangling
    Expr appl  = Expr::Appl(move(case_), move(rcall_ex)); appl.type = ty_B;
    Expr body  = Expr::Block({move(appl)}); body.type = ty_B;
    Expr lamb  = Expr::Lamb(move(ex_arg), move(body), Type(a_Fun, ty_arg, ty_B));

    return lamb;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Demand driven template expansion:


namespace {
struct RDef {   // -- 'rec' definition
    Expr                       expr;     // -- pointer to the rec-def expression
    Vec<Expr>*                 out;      // -- the output vector for the scope in which the definition appears
    shared_ptr<Set<Arr<Type>>> emitted;  // -- for which types has this symbol been emitted?
    RDef() : out(nullptr) {}
    RDef(Expr const& expr, Vec<Expr>& out) : expr(expr), out(&out), emitted(make_shared<Set<Arr<Type>>>()) {}
};

typedef Vec<Pair<shared_ptr<Set<Arr<Type>>>, Arr<Type>>> Emits;   // -- to store undo information for 'pop()'
}


static Expr expr_hidden;
static Vec<Expr> out_hidden;    // -- serves as marker for hidden symbols


// Remove symbols in pattern 'pat' from 'defs'
static void hideSyms(Expr const& pat, SymTable<RDef>& defs) {
    if (pat.kind == expr_Sym)
        defs.addQ(pat.name, RDef(expr_hidden, out_hidden));
    else
        for (Expr const& e : pat) hideSyms(e, defs);
}

inline bool isHidden(RDef const& rdef) { return rdef.out == &out_hidden; }


static Expr expand(Expr const& expr, SymTable<RDef>& defs, Emits& emits)
    // -- note that 'defs' is almost const, except for hiding of symbols after 'let's which introduces a different scope below it.
{
    if (expr.kind == expr_Sym){
        if (defs.has(expr) && !isHidden(defs[expr])){
            if (!defs[expr].expr)  // -- 'defs[expr].expr' is false for internal symbols like 'case_'
                return Expr(expr_Sym, expr.name, {}, Type(a_Internal, expr.type), Arr<Type>(expr.targs), expr.loc);
            else if (!defs[expr].emitted->add(expr.targs)){
                emits.push(tuple(defs[expr].emitted, expr.targs));
                Expr def = defs[expr].expr;
                Expr new_expr = def[0].targs ? typeSubst(def, def[0].targs, expr.targs) : def;
                new_expr = Expr::RecDef(Expr(new_expr[0]), Type(new_expr.type), expand(new_expr[1], defs, emits)).setLoc(def.loc);
                defs[expr].out->push(new_expr);
            }
        }
        return expr;

    }else if (expr.kind == expr_Block){
        SymTable<RDef> sub_defs(defs);
        Vec<Expr>      sub_out;

        uint j = 0;
        for (uint i = 0; i < expr.size(); i++){
            Expr const& e = expr[i];
            if (i >= j && e.kind == expr_RecDef){   // -- trigger on first rec-def only
                for (j = i; j < expr.size() && expr[j].kind == expr_RecDef; j++){
                    sub_defs.addQ(expr[j][0].name, RDef(expr[j], sub_out)); }
            }
            if (e.kind != expr_RecDef)
                sub_out.push(expand(e, sub_defs, emits));
        }
        return Expr::Block(sub_out).setType(Type(expr.type)).setLoc(expr.loc);

    }else if (expr.kind == expr_MetaIf){
        if (expr[0].targs[0] == expr[0].targs[1])   // -- fast test
            return expand(expr[1], defs, emits);
        else{
            Vec<TSubst> subs;
            if (typeMatch(expr[0].targs[1], expr[0].targs[0], subs))
                return expand(typeSubst(expr[1], subs), defs, emits);
            else
                return expand(expr[2], defs, emits);
        }

    }else if (expr.kind == expr_MetaErr){
        if (expr.name){
            Str msg(expr.name);
            if (expr.targs.psize() > 0)
                throw Excp_ParseError(fmt("%_: `%_`. %_", expr.loc, expr.targs[0], msg.slice(1, msg.size()-1)));
            else
                throw Excp_ParseError(fmt("%_: %_", expr.loc, msg.slice(1, msg.size()-1)));
        }else
            throw Excp_ParseError(fmt("%_: Meta-if reached undefined branch.", expr.loc));

    }else{ assert(expr.kind != expr_RecDef);
        // Variable hiding -- needed for things like:
        //     fun f<A>(a: A) -> A { a };
        //     { let f = 42; }
        SymTable<RDef>* defs_p = &defs;
        SymTable<RDef> sub_defs(defs);
        if (expr.kind == expr_LetDef || expr.kind == expr_Lamb){
            if (expr.kind == expr_Lamb) defs_p = &sub_defs;
            hideSyms(expr[0], *defs_p);
        }

        // Recurse:
        Arr<Expr> es;
        if (expr.exprs) es = map(expr.exprs, [&](Expr const& e){ return expand(e, *defs_p, emits); });
        return Expr(expr.kind, expr.name, move(es), Type(expr.type), Arr<Type>(expr.targs), expr.loc);
    }
}


static Expr mangleL(Expr const& e, SymTable<Expr>& datadefs);


static Expr mangle(Expr const& e, SymTable<Expr>& datadefs)
{
    Arr<Expr> es;   // -- recursively mangle
    if (e.exprs) es = map(e.exprs, [&](Expr const& ex){ return mangleL(ex, datadefs); });

    bool builtin = (e.kind == expr_Sym && e.type.name == a_Internal);

    Atom name = (e.kind != expr_Sym || !e.targs || builtin) ? e.name : Atom(fmt("'%_'", e));
    Arr<Type> ts = (e.kind == expr_Sym) ? Arr<Type>() : Arr<Type>(e.targs);

    return Expr(e.kind, name, move(es), Type(builtin ? e.type[0] : e.type), move(ts), e.loc);
        // -- for reference: Expr(ExprKind kind, Atom name, Arr<Expr>&& exprs, Type&& type, Arr<Type>&& targs, Loc loc)
}


// Mangle and expand value-lambdas.
static Expr mangleL(Expr const& e, SymTable<Expr>& datadefs)
{
    Expr e2;
    if (e.kind == expr_Cons){
        Type t = e.targs[0];
        if (t.name != a_OneOf){
            Expr const& dd = datadefs.ref(t.name); assert(dd.type.name == a_OneOf);   // -- data definition
            Type new_t = dd.targs ? typeSubst(dd.type, dd.targs, t.args) : dd.type;
            e2 = Expr::Cons(Type(new_t), e.name).setLoc(e.loc);   // 'e.name' is an integer index (represented as an atom)
        }else
            e2 = e;
    }else if (e.kind == expr_Sym && e.type.name == a_Internal && e.name == a_case && e.targs[0].name != a_Ref)  // <<== BI
        e2 = expandValueLambda(e, datadefs);

    return mangle(e2 ? e2 : e, datadefs);
}


struct Expander_data {
    SymTable<RDef> bi_defs;     // -- build-in pseudo-definitions (base for 'defs' symbol table)
    SymTable<Expr> datadefs;    // -- types declared by 'data'
    SymTable<RDef> defs;        // -- polymorphic value introduced by 'rec'
    Emits          emits;
    Vec<Expr>      out;
    Vec<Trip<uind,uind,uind>> undo;
    Expander_data() : defs(bi_defs) {}
};


Expander::Expander()
{
    data = new Expander_data;

    // Builtin symbols (these will be presented to compiler without name mangling):
    SymTable<RDef>& bi_defs = data->bi_defs;
    bi_defs.addQ(a_case  , RDef());  // <<== BI
    bi_defs.addQ(a_ite   , RDef());
    bi_defs.addQ(a_assign, RDef());
    bi_defs.addQ(a_fail  , RDef());
    bi_defs.addQ(a_write , RDef());
    bi_defs.addQ(a_run   , RDef());

    bi_defs.addQ(a_size  , RDef());
    bi_defs.addQ(a_get   , RDef());
    bi_defs.addQ(a_growby, RDef());
    bi_defs.addQ(a_resize, RDef());
    bi_defs.addQ(a_set   , RDef());

    bi_defs.addQ(a_print_bool , RDef());
    bi_defs.addQ(a_print_int  , RDef());
    bi_defs.addQ(a_print_float, RDef());
    bi_defs.addQ(a_print_atom , RDef());

    bi_defs.addQ(a_try   , RDef());
    bi_defs.addQ(a_throw , RDef());
    bi_defs.addQ(a_ttry  , RDef());
    bi_defs.addQ(a_tthrow, RDef());
    bi_defs.addQ(a_block , RDef());
    bi_defs.addQ(a_break , RDef());

    bi_defs.addQ(a_line, RDef());
    bi_defs.addQ(a_file, RDef());
}


Expander::~Expander() {
    delete data; }


void Expander::push()  {
    data->undo.push(tuple(data->datadefs.size(), data->defs.size(), data->emits.size())); }


void Expander::pop() {
    while (data->datadefs.size() > data->undo.last().fst) data->datadefs.pop();
    while (data->defs    .size() > data->undo.last().snd) data->defs    .pop();
    while (data->emits   .size() > data->undo.last().trd){
        data->emits.last().fst->exclude(data->emits.last().snd);
        data->emits.pop(); }
    data->undo.pop();
}


// Public function.
Expr Expander::expandTemplates(Expr const& prog)
{
    assert(prog.kind == expr_Block);
    ZZ_PTimer_Scope(evo_template_expansion);

    // Collect rec-defs and datadefs and expand the required ones with concrete types:
    SymTable<Expr>& datadefs = data->datadefs;
    SymTable<RDef>& defs     = data->defs;
    Vec<Expr>     & out      = data->out;
    out.clear();

    // Process main block:
    for (Expr const& e : prog)
        if (e.kind == expr_DataDef)
            datadefs.addQ(e.name, e);

    uint j = 0;
    for (uint i = 0; i < prog.size(); i++){
        Expr const& e = prog[i];
        if (i >= j && e.kind == expr_RecDef){   // -- trigger on first rec-def only
            for (j = i; j < prog.size() && prog[j].kind == expr_RecDef; j++){
                defs.addQ(prog[j][0].name, RDef(prog[j], out)); }
        }
        if (e.kind != expr_RecDef && e.kind != expr_TypeDef && e.kind != expr_DataDef)
            out.push(expand(e, defs, data->emits));
    }

    // Mangle names with types:
    Expr new_prog = mangleL(Expr::Block(out).setType(Type(prog.type)).setLoc(prog.loc), datadefs);

    if (getenv("EXPANDED")){
        wrLn("\a*EXPANDED:\a0\t+\t+");
        wrLn("%_;", join(";\n\n", new_prog));
        wrLn("\t-\t-");
    }

    return new_prog;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

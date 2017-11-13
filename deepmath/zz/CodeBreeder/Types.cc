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

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Symbol naming conventions:


// NOTE! 'str' is assumed to contain only characters from 'cc_Ident' as defined in 'Parser.cc'.

bool isType(Str str) {  // -- start with capital, contain at least one lower-case or end in underscore
    if (!str || str.size() == 0 || str[0] < 'A' || str[0] > 'Z') return false;
    for (char c : str) if (c >= 'a' && c <= 'z') return true;
    if (str.last() == '_') return true;
    return false; }

bool isTVar(Str str) {  // -- only capitals and not ending in underscore
    if (!str || str.size() == 0 || str[0] < 'A' || str[0] > 'Z') return false;
    for (char c : str) if (c >= 'a' && c <= 'z') return false;
    if (str.last() == '_') return false;
    return true; }

bool isVar(Str str) {   // -- start with lower-case letter or underscore
    return str && str.size() > 0 && ((str[0] >= 'a' && str[0] <= 'z') || str[0] == '_'); }


Atom a_x;
Atom a_y;
Atom a_z;

Atom a_case;
Atom a_ite;
Atom a_assign;
Atom a_fail;
Atom a_write;
Atom a_run;
Atom a_match;

Atom a_size;
Atom a_get;
Atom a_growby;
Atom a_resize;
Atom a_set;

Atom a_print_bool;
Atom a_print_int;
Atom a_print_float;
Atom a_print_atom;

Atom a_try;
Atom a_throw;
Atom a_ttry;
Atom a_tthrow;
Atom a_block;
Atom a_break;

Atom a_underscore;
Atom a_false;
Atom a_true;
Atom a_appl_op;

Atom a_line;
Atom a_file;

ZZ_Initializer(a_special_symbols, 0) {
    a_x           = Atom("x");
    a_y           = Atom("y");
    a_z           = Atom("z");

    a_case        = Atom("case_");
    a_ite         = Atom("ite_");
    a_assign      = Atom("assign_");
    a_fail        = Atom("fail");    // -- expression, not a function
    a_write       = Atom("write_");
    a_run         = Atom("run_");
    a_match       = Atom("match");

    a_size        = Atom("size_");
    a_get         = Atom("get_");
    a_growby      = Atom("growby_");
    a_resize      = Atom("resize_");
    a_set         = Atom("set_");

    a_print_bool  = Atom("print_bool_");
    a_print_int   = Atom("print_int_");
    a_print_float = Atom("print_float_");
    a_print_atom  = Atom("print_atom_");

    a_try         = Atom("try_");
    a_throw       = Atom("throw_");
    a_ttry        = Atom("ttry_");
    a_tthrow      = Atom("tthrow_");
    a_block       = Atom("block");  // }- keywords, so no trailing '_' (but parsed as functions)
    a_break       = Atom("break");  // }

    a_underscore  = Atom("_");
    a_false       = Atom("_0");
    a_true        = Atom("_1");
    a_appl_op     = Atom("o");

    a_line        = Atom("__LINE__");
    a_file        = Atom("__FILE__");
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Type atoms:


Atom a_Void;
Atom a_Bool;
Atom a_Int;
Atom a_Float;
Atom a_Atom;
Atom a_Ref;
Atom a_OneOf;
Atom a_Fun;
Atom a_Tuple;
Atom a_Vec;
Atom a_Any;
Atom a_Internal;
Atom a_A;
Atom a_B;
Atom a_T;
Atom a_List;
Atom a_Maybe;
Atom a_Tree;
Atom a_Hide;

Vec<Atom> bit_primitive;
Vec<Atom> bit_composite;

ZZ_Initializer(a_atoms, 0) {
    a_Void     = Atom("Void");
    a_Bool     = Atom("Bool");
    a_Int      = Atom("Int");
    a_Float    = Atom("Float");
    a_Atom     = Atom("Atom");
    a_Ref      = Atom("_Ref");     // -- underscore to make sure there the atom is outside the namespace of user defined types
    a_OneOf    = Atom("_OneOf");
    a_Fun      = Atom("_Fun");
    a_Tuple    = Atom("_Tuple");
    a_Vec      = Atom("_Vec");
    a_Any      = Atom("*");
    a_Internal = Atom("_Internal");
    a_A        = Atom("A");
    a_B        = Atom("B");
    a_List     = Atom("List");
    a_Maybe    = Atom("Maybe");
    a_Tree     = Atom("Tree");
    a_Hide     = Atom("Hide");

    bit_primitive += a_Void, a_Bool, a_Int, a_Float, a_Atom;
    bit_composite += a_Ref, a_OneOf, a_Fun, a_Tuple, a_Vec;
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Printing:


cchar* ExprKind_name[ExprKind_size] = {
    "<null>",
    "Sym",
    "Op",
    "Lit",
    "MkVec",
    "Tuple",
    "MkRef",
    "Deref",
    "Lamb",
    "Appl",
    "Cons",
    "Sel",
    "Block",
    "LetDef",
    "RecDef",
    "TypeDef",
    "DataDef",
    "MetaIf",
    "MetaErr",
};


void writeType(Out& out, Type const& t, bool is_arg)
{
    if      (!t)                out += "<null-type>";
    else if (t.name == a_Ref  ) wr(out, "&%a", t[0]);
    else if (t.name == a_Fun  ) { if (is_arg) wr(out, "(%a->%_)", t[0], t[1]); else wr(out, "%a->%_", t[0], t[1]); }
    else if (t.name == a_Vec  ) wr(out, "[%_]", t[0]);
    else if (t.name == a_Tuple) wr(out, "(%_)", join(", ", t.args));
    else if (t.name == a_OneOf) wr(out, "{%_}", join(", ", t.args));
    else if (!t.args)           wr(out, "%_", t.name);
    else                        wr(out, "%_<%_>", t.name, join(", ", t.args));
}


struct EvoFunHead { Expr head; };
template<> fts_macro void write_(Out& out, EvoFunHead const& v) {
    bool first = true;
    for (Expr const& e : tupleSlice(v.head)){
        if (first) first = false;
        else out += ", ";
        wr(out, "%_ :%_", e, e.type);
    }
}


void writeExpr(Out& out, Expr const& e, bool is_arg)
{
    switch (e.kind){
    case expr_NULL : out += "<null>"; break;
    case expr_Sym  : out += e.name; if (e.targs) wr(out, "<%_>", join(", ", e.targs)); break;
    case expr_Lit  : out += e.name; break;
    case expr_MkVec: wr(out, "[:%_ %_]", e.type[0], join(", ", e)); break;
    case expr_Tuple:
        if (is_arg){
            out += '(';
            for (uint i = 0; i < e.size(); i++){
                if (i != 0) out +=", ";
                wr(out, "%a", e[i]); }
            out += ')';
        }else
            wr(out, "(%_)", join(", ", e));
        break;
    case expr_Lamb :
        if (is_arg || !e.type){ if (e[0].isUnit()) wr(out, "\\%_", e[1]); else wr(out, "\\%_ %_", e[0], e[1]); }
        else       { if (e[0].isUnit()) wr(out, "\\:%_ %_", e.type, e[1]); else wr(out, "\\:%_ %_ %_", e.type, e[0], e[1]); }
        break;
    case expr_MkRef: out += "& ", e[0]; break;  // <<== don't need space if next character is non-op char
    case expr_Deref: out += "^ ", e[0]; break;  // <<== switch to function 'deref_'
    case expr_Sel  : wr(out, "#%_ (%_)", e.name, e[0]); break;  // <<== may not always need parenthesis
    case expr_Appl : if (is_arg) wr(out, "(%_ %a)", e[0], e[1]); else wr(out, "%_ %a", e[0], e[1]); break;
    case expr_Cons : wr(out, ".%_.%_", e.targs[0], e.name); break;      // <<== eliminate initial '.' if 'e.targs[0]' starts with uppercase letter in string representation
    case expr_Block: wr(out, "{%_}", join("; ", e)); break;

    case expr_LetDef: wr(out, "let %_" , e[0]); if (e.type) wr(out, " : %_", e.type); if (e.size() == 2) wr(out, " = %_", e[1]); break;
    case expr_RecDef:
        if (e.type && e.type.name == a_Fun && e[1].kind == expr_Lamb){
            wr(out, "fun %_(%_) ", e[0], EvoFunHead{e[1][0]});
            if (e.type[1].name != a_Void) wr(out, "-> %_ ", e.type[1]);
            wr(out, "%_", e[1][1]);
        }else{
            wr(out, "rec %_", e[0]); if (e.type) wr(out, " : %_", e.type); wr(out, " = %_", e[1]);
        }break;

    case expr_TypeDef:
    case expr_DataDef:
        wr(out, "%_ %_", (e.kind == expr_TypeDef) ? "type" : "data", e.name);
        if (e.targs.psize() > 0) wr(out, "<%_>", join(", ", e.targs));
        out += " = ", e.type; break;

    case expr_MetaIf:  wr(out, "##if(%_, %_, %_)", e[0], e[1], e[2]); break;
    case expr_MetaErr: wr(out, "##error"); if (e.name) wr(" \"%_\"", e.name); break;

    case expr_Op  : out += e.name; break;   // -- FOR DEBUGGING ONLY (parser will expand operators to function symbols)
    default: assert(false); }

    //**/if (e.type) wr(out, " \a/:%_\a/", e.type);
}


String ppFmtI(Expr const& expr, uint indent_level)
{
    String tmp; tmp += expr;
    String out;
    Vec<char> indent(indent_level, ' ');
    out += indent;
    for (uind i = 0; i < tmp.size(); i++){
        char c = tmp[i];
        char d = tmp[i+1];
        if      (c == '{' && d == '}'){ out += "{}"; i++; }
        else if (c == '{')            { indent.push(' '); indent.push(' '); out += '{', NL, indent; }
        else if (c == '}')            { indent.pop(), indent.pop(); out += NL, indent, '}'; }
        else if (c == ';' && d == ' '){ out += ';', NL, indent; i++; }
        else if (c == '\n')           { out += NL, indent; }
        else out += c;
    }
    return out;
}


void gdbPrintType(Type const& type)
{
    printf("%s\n", fmt("%_", type).c_str());
    fflush(stdout);
}


void gdbPrintExpr(Expr const& expr)
{
    printf("TYPE: ");
    gdbPrintType(expr.type);
    printf("%s\n", ppFmt(expr).c_str());
    fflush(stdout);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


// PRE-CONDITION 'from' must consist only of type-variables and have the same length as 'into'.
// Substitution will be applied once (in case 'into' types contains type-variables from 'from').
Type typeSubst(Type const& type, Arr<Type> const& from, Arr<Type> const& into)
{
    if (type.isTVar()){
        assert(from.size() == into.size());
        uind n = search(from, type);
        return (n == UIND_MAX) ? type : into[n];
    }else if (type.args){
        Vec<Type> ts;
        for (Type const& t : type)
            ts.push(typeSubst(t, from, into));
        return vecEqual(ts, type) ? type : Type(type.name, ts);
    }else
        return type;
}


Type typeSubst(Type const& type, Vec<TSubst> const& subs)
{
    Vec<Type> from(reserve_, subs.size());
    Vec<Type> into(reserve_, subs.size());
    for (TSubst s : subs){
        from.push(s.a);
        into.push(s.ty); }
    return typeSubst(type, from, into);
}


Expr typeSubst(Expr e, Arr<Type> const& from, Arr<Type> const& into)
{
    Arr<Expr> es;   // -- type substituted 'e.exprs'
    Arr<Type> ts;   // -- type substituted 'e.targs'
    if (e.exprs) es = map(e.exprs, [&](Expr const& ex){ return typeSubst(ex, from, into); });
    if (e.targs) ts = map(e.targs, [&](Type const& ty){ return typeSubst(ty, from, into); });

    return Expr(e.kind, e.name, move(es), typeSubst(e.type, from, into), move(ts), e.loc);
        // -- for reference: Expr(ExprKind kind, Atom name, Arr<Expr>&& exprs, Type&& type, Arr<Type>&& targs, Loc loc)
}


Expr typeSubst(Expr expr, Vec<TSubst> const& subs)
{
    Vec<Type> from(reserve_, subs.size());
    Vec<Type> into(reserve_, subs.size());
    for (TSubst s : subs){
        from.push(s.a);
        into.push(s.ty); }
    return typeSubst(expr, from, into);
}


// Is 'sub_ty' a specialization of 'base_ty'? Populates substitution map as a side-effect.
bool typeMatch(Type general, Type specific, /*out*/Vec<TSubst>& subs)
{
    if (general.isTVar()){
        for (TSubst& s : subs)
            if (general == s.a)
                return specific == s.ty;      // -- substitutions must be consistent
        subs.push(TSubst(general, specific));
        return true;

    }else{
        if (specific.isTVar() || general.name != specific.name || general.size() != specific.size())
            return false;

        for (uint i = 0; i < general.size(); i++)
            if (!typeMatch(general[i], specific[i], subs))
                return false;
        return true;
    }
}


// 'expr' must be int or float literal
double getFloat(Expr const& expr) {
    if (expr.kind == expr_Lit && expr.type.name == a_Int)   return stringToInt64(expr.name);
    if (expr.kind == expr_Lit && expr.type.name == a_Float) return stringToDouble(expr.name);
    wrLn("ERROR! %_: Expected a number.", expr.loc);
    exit(1);
}


// 'expr' must be atom
Atom getAtom(Expr const& expr) {
    if (expr.kind == expr_Lit && expr.type.name == a_Atom) return expr.name;
    wrLn("ERROR! %_: Expected an Atom.", expr.loc);
    exit(1);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

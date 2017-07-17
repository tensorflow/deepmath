/* Copyright 2016 Google Inc. All Rights Reserved.

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
#include "HolFormat.hh"

#include <memory>           // -- for 'unique_ptr'

#include "HolOperators.hh"
#include "Printing.hh"
    // -- 'Printing.hh' for printing types in 'simpleFmtTerm()'.

#define TRANSLATE_EQ_TO_EQUIV

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


// From 'printer.ml':
//
// static cchar* hol_spaces = " \t\n\r";
// static cchar* hol_separators = ",;";
// static cchar* hol_brackets = "()[]{}";
// static cchar* hol_symbs = "\\!@#$%^&*-+|\\<=>/?~.:";
// static cchar* hol_alphas = "'abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ";
// static cchar* hol_nums = "0123456789";


inline bool is_identChar(char c) {
    return c == '\''
        || c == '_'
        || (c >= 'a' && c <= 'z')
        || (c >= 'A' && c <= 'Z');
}


// Returns TRUE if term 'tm' is on one of the following forms:
//
//   1. (~ x)     = comb(~, x)          = "~x
//   2. ((+ x) y) = comb(comb(+, x), y) = "x + y"
//
// In the first case, 'x' will be bound to null-vaule 'Term()'.
//
static bool getOpExpr(Term tm, /*outs:*/Term& op, Term& x, Term& y, HolOp& h)
{
    if (!tm.is_comb())
        return false;

    Term f = tm.fun();
    Term g = tm.arg();
    if (f.is_cnst() && (h = hol_ops[f.cnst()])){
        if (h.kind == op_PREFIX || (h.kind == op_BINDER && g.is_abs())){
            op = f;
            x  = Term();
            y  = g;
            return true;
        }
    }

    if (f.is_comb() && f.fun().is_cnst() && (h = hol_ops[f.fun().cnst()])){
        if (h.kind == op_INFIXL || h.kind == op_INFIXR){
            op = f.fun();
            x  = f.arg();
            y  = g;
          #if defined(TRANSLATE_EQ_TO_EQUIV)
            if (op.cnst() == cnst_eq && x.type() == type_bool && y.type() == type_bool){
                op = tmCnst(cnst_equiv, op.type());
                h  = hol_ops[cnst_equiv]; }
          #endif
            return true;
        }
    }

    return false;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Bound variable naming:


/*
bool: u, v, w
num: x, y, z
A->bool: P, Q, R, S
A->B: f, g, h,
prod<A,B>: p, q
other: a, b, c, d
*/


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Decorated strings:


struct DStr;
typedef std::unique_ptr<DStr> DStrP;


struct DStr {
    bool is_atom;
    union {
        Vec<DChar> atom;
        Vec<DStrP> comp;        // -- owns data
    };

    DStr(Vec<DChar>&& atom) : is_atom(true) , atom(move(atom)) {}
    DStr(Vec<DStrP>&& comp) : is_atom(false), comp(move(comp)) {}
   ~DStr() { if (is_atom) destruct(atom); else destruct(comp); }

    void   flatten();   // -- turn 'DStr' into atom
    String str() const; // -- only for atoms
};


void DStr::flatten() {    // -- turn 'DStr' into atom
    if (is_atom) return;

    Vec<DChar> out;
    function<void(DStr*)> rec = [&](DStr* ds) {
        if (ds->is_atom){
            for (DChar c : ds->atom)
                out.push(c);
        }else{
            for (DStrP const& s : ds->comp)
                rec(s.get());
        }
    };
    rec(this);

    this->~DStr();
    is_atom = true;
    atom = move(out);
}


String DStr::str() const {
    assert(is_atom);
    String out;
    for (DChar d : atom) out.push(d.chr);
    return out;
}


// FACTORY FUNCTIONS:

// USAGE:
//   - 'dsAtom(text, char_category, term, add_quotes)'
//   - 'dsComp(str1, str2, ..., strN)', where all strings are temporary 'DStrP's. 'nullptr':s
//     are allowed and are discarded/ignored.

inline DStrP dsAtom(Str text, CharCat cat, Term tm, bool quote = false) {
    Vec<DChar> atom;
    if (quote) atom.push(DChar{'`', cat, tm});        // -- HOL Light uses '('
    for (char c : text)
        atom.push(DChar{c, cat, tm});
    if (quote) atom.push(DChar{'`', cat, tm});        // -- HOL Light uses ')'
    return DStrP(new DStr(move(atom)));
}


inline void dsComp(Vec<DStrP>& comp) {}

template<typename... Args>
inline void dsComp(Vec<DStrP>& comp, DStrP x, Args&&... args) {
    if (x) comp.push(move(x));
    dsComp(comp, forward<Args>(args)...); }

template<typename... Args>
inline DStrP dsComp(Args&&... args) {
    Vec<DStrP> comp;
    dsComp(comp, forward<Args>(args)...);
    return DStrP(new DStr(move(comp)));
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Formatted subexpression, return type:


struct FmtRet {
    HolOp h;        // -- top-level operator of expression; 'op.kind' is used only for infix operator
    Term  term;

    // Symbolic representation of the text of the expression; turn to decorated string through 'realize()':
    bool  paren;    // If term is non-nil, expression will be wrapped in parenthesis (associated with that term)
    Cnst  binder;
    DStrP vars;
    DStrP body;

    DStrP realize() {    // -- will reset 'vars' and 'body'
        static const Str s_lparen("(");
        static const Str s_rparen(")");
        static const Str s_space(" ");
        static const Str s_dot(". ");

        DStrP ds_lparen = !paren ? nullptr : dsAtom(s_lparen, cc_OTHER, term);
        DStrP ds_rparen = !paren ? nullptr : dsAtom(s_rparen, cc_OTHER, term);

        assert((bool)binder == (bool)vars);
        DStrP ds_abs = nullptr;
        if (binder){
            DStrP ds_binder = dsAtom(Str(binder), cc_BINDER, term);
            DStrP ds_space  = is_identChar(Str(binder).last()) ? dsAtom(s_space, cc_OTHER, term) : (DStrP)nullptr;
            DStrP ds_dot    = dsAtom(s_dot, cc_OTHER, term);
            ds_abs = dsComp(move(ds_binder), move(ds_space), move(vars), move(ds_dot));
        }

        DStrP ret = dsComp(move(ds_lparen), move(ds_abs), move(body), move(ds_rparen));
        return ret;
    }
};


// TODO: Take the following input parameters and format accordingly:
//   - operator syntax on/off
//   - abbreviations (list of terms)
//   - depth cut-off
//
FmtRet fmtTerm(Term tm, uint depth)
{
    static const Str s_space(" ");

    auto fmt = [&](Term tm) { return fmtTerm(tm, depth + 1); };

    HolOp h;        // -- 'h.kind' is used only for infix operator
    Cnst  binder;
    DStrP vars = nullptr;
    DStrP body = nullptr;

    auto mkBinder = [&](Cnst sym, Term tm) {
        h.prec = prec_BINDER;
        binder = sym;
        vars = dsAtom(Str(tm.avar().var()), cc_ABSVAR, tm.avar());
#if 1  // DEBUG
        String ttext;
        wr(ttext, " :%_", tm.avar().type());
        DStrP type = dsAtom(ttext.slice(), cc_TYPE, tm.avar());
        vars = dsComp(move(vars), move(type));
#endif // END-DEBUG

        FmtRet sub = fmt(tm.aterm());
        if (sub.binder == sym){
            DStrP space = dsAtom(s_space, cc_OTHER, tm);
            vars = dsComp(move(vars), move(space), move(sub.vars));
            body = move(sub.body);
        }else
            body = sub.realize();
    };

    // Main:
    if (tm.is_var()){
        body = dsAtom(Str(tm.var()), Term::isLambdaVar(tm.var()) ? cc_VAR : cc_FREEVAR, tm);
        h.prec = prec_ATOMIC;

    }else if (tm.is_cnst()){
        Cnst tm_cnst = tm.cnst();
      #if defined(TRANSLATE_EQ_TO_EQUIV)
        if (tm_cnst == cnst_eq && tm.type() == type_booleq)
            tm_cnst = cnst_equiv;
      #endif

        body = dsAtom(Str(tm_cnst), (tm_cnst == cnst_eq) ? cc_EQUAL : cc_CNST, tm, /*quote?*/(bool)hol_ops[tm_cnst]);
        h.prec = prec_ATOMIC;

    }else if (tm.is_abs()){
        mkBinder(cnst_lam, tm);

    }else{ assert(tm.is_comb());
        Term op, x, y;
        if (getOpExpr(tm, op, x, y, h)){
            // Operators/binders:
            DStrP op_str = dsAtom(Str(op.cnst()), (op.cnst() == cnst_eq) ? cc_EQUAL : cc_CNST, op);
            bool space_after_op = is_identChar(Str(op.cnst()).last());    // -- if binder/prefix-op ends in [a-zA-Z0-9_], add space after it

            if (x){     // -- infix operator
                FmtRet subx = fmt(x);
                FmtRet suby = fmt(y);
                if (subx.h.prec < h.prec || (subx.h.prec == h.prec && (h.kind != op_INFIXL || subx.h.kind != op_INFIXL))) subx.paren = true;
                if (suby.h.prec < h.prec || (suby.h.prec == h.prec && (h.kind != op_INFIXR || suby.h.kind != op_INFIXR))) suby.paren = true;

                DStrP space1 = dsAtom(s_space, cc_OTHER, tm);
                DStrP space2 = dsAtom(s_space, cc_OTHER, tm);
                body = dsComp(subx.realize(), move(space1), move(op_str), move(space2), suby.realize());

            }else if (h.kind == op_PREFIX){
                assert(h.prec == prec_PREFIX);
                FmtRet sub = fmt(y);
                if (sub.h.prec < prec_PREFIX) sub.paren = true;
                else if (sub.h.prec == prec_PREFIX) space_after_op = true;

                DStrP  space = space_after_op ? dsAtom(s_space, cc_OTHER, tm) : (DStrP)nullptr;
                body = dsComp(move(op_str), move(space), sub.realize());

            }else{ assert(h.kind == op_BINDER);
                mkBinder(op.cnst(), y);     // -- NOTE: 'getOpExpr()' guarantees 'y' is an abs-term
            }

        }else{
            // Standard function application: (f g)
            h.prec = prec_COMB;
            FmtRet f = fmt(tm.fun());
            FmtRet g = fmt(tm.arg());
            if (f.h.prec < prec_COMB)   f.paren = true;
            if (g.h.prec < prec_ATOMIC) g.paren = true;

            DStrP space = dsAtom(s_space, cc_OTHER, tm);
            body = dsComp(f.realize(), move(space), g.realize());
        }
    }

    return FmtRet{h, tm, false, binder, move(vars), move(body)};
}


Vec<DChar> fmtTerm(Term tm)
{
    DStrP d = fmtTerm(tm, 0).realize();
    d->flatten();
    return move(d->atom);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

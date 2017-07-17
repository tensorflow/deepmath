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

#ifndef ZZ__HolLight__Kernel_hh
#define ZZ__HolLight__Kernel_hh

#include "ProofLogging.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Definitional information:


struct Ax {
    Axiom   name;
    Thm     th;
    explicit operator bool() const { return bool(name); }
};


struct Def {
    Cnst    name;
    Term    def;    // -- polymorphically typed term of the constant
    Thm     th;     // -- theorem introducting constant
    explicit operator bool() const { return bool(name); }
};


struct TDef {
    uint arity;     // -- derived from 'witness': 'setOf_tvars(witness.concl().fun()).size()'
    TCon name;
    Cnst abs_name;
    Cnst rep_name;
    Thm  witness;
    Thm  th;        // -- theorem introducting type

    TDef(uint arity, TCon name, Cnst abs_name, Cnst rep_name, Thm witness, Thm th) :
        arity(arity), name(name), abs_name(abs_name), rep_name(rep_name), witness(witness), th(th) {}

    TDef() : arity(UINT_MAX) {}
    explicit operator bool() const { return arity != UINT_MAX; }
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


Ax   getAxiom(Axiom name);
Def  getDef  (Cnst  name);
TDef getTDef (TCon  name);

Vec<Ax  > kernelAxioms();
Vec<Def > kernelConsts();
Vec<TDef> kernelTypecons();


Thm kernel_REFL   (Term tm);
Thm kernel_BETA   (Term tm);
Thm kernel_ASSUME (Term tm);
Thm kernel_ABS    (Thm thm, Term x);
Thm kernel_TRANS  (Thm th1, Thm th2);
Thm kernel_MK_COMB(Thm funs_eq, Thm args_eq);
Thm kernel_EQ_MP  (Thm props_eq, Thm prop);
Thm kernel_DEDUCT (Thm th1, Thm th2);
Thm kernel_INST   (Thm th, Vec<Subst>& subs);
Thm kernel_INST_T (Thm th, Vec<TSubst>& subs);

Thm kernel_New_Ax  (Axiom ax, Term tm);
Thm kernel_New_Def (Cnst c, Term tm);
Thm kernel_New_TDef(TCon tc_name, Cnst abs_name, Cnst rep_name, Thm th);
Thm kernel_Extract1(Thm th);    // }- these go together with 'kernel_New_TDef'; no other use
Thm kernel_Extract2(Thm th);    // }
    // -- NOTE! the 'New_' functions affect the global state of the kernel (list of axioms, constant-constant and type-constructors)

Term kernel_Inst_Cnst(Cnst c , Vec<TSubst>& subs);
Type kernel_Inst_Type(TCon tc, List<Type> ts);


// Safe helpers:
Type typeOf(Term tm);
Type mkTApp(TCon c, List<Type> ts); // -- just another name for 'kernel_Inst_Type'
Type mkTVar(TVar v);
Term mkCnst(Cnst c, Type ty);       // -- matches 'ty' against 'c' and calls 'kernel_Inst_Cnst' with the right substitutions.
Term mkVar (Var  x, Type ty);
Term mkComb(Term f, Term tm);
Term mkAbs (Term x, Term tm);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

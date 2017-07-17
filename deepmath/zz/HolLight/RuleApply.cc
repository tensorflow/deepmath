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

#define OPAQUE_HOL_TYPES    // -- makes sure this file does not break the HOL-light kernel contract
#include "RuleApply.hh"
#include "Kernel.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


static Thm helper_INST(Array<Arg const> args)
{
    // NOTE! In the proof-logs, variables `x1`..`xn` from the substitution list are not
    // necessarily distinct, and may even  have inconsistent assignments. In that case,
    // the first assignment to `xi` should be used.

    Vec<Subst> subs;
    for (uind i = 0; i+1 < args.size(); i += 2){
        Term x  = args[i].term();
        Term tm = args[i+1].term();     // -- could add duplicate check here if prevalent
        subs.push(Subst(x, tm));
    }

    return kernel_INST(args.last().thm(), subs);
}


static Thm helper_INST_T(Array<Arg const> args)
{
    Vec<TSubst> subs;
    for (uind i = 0; i+1 < args.size(); i += 2){
        Type a  = args[i].type();
        Type ty = args[i+1].type();     // -- could add duplicate check here if prevalent
        subs.push(TSubst(a, ty));
    }

    return kernel_INST_T(args.last().thm(), subs);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


Type produceType(RuleKind rule, Array<Arg const> args)
{
    auto getType = [](Arg arg){ return arg.type(); };

    switch (rule){
    case rule_TVar : return mkTVar(args[0].tvar());
    case rule_TAppl: return mkTApp(args[0].tcon(), mkList(args.slice(1), getType));
    default: assert(false); }
}


Term produceTerm(RuleKind rule, Array<Arg const> args)
{
    switch (rule){
    case rule_Cnst: return mkCnst(args[0].cnst(), args[1].type());
    case rule_Var:  return mkVar (args[0].var (), args[1].type());
    case rule_App:  return mkComb(args[0].term(), args[1].term());
    case rule_Abs:  return mkAbs (args[0].term(), args[1].term());
    default: assert(false); }
}


// NOTE! For 'rule_New_TDef', a fake theorem with
Thm produceThm(RuleKind rule, Array<Arg const> args)
{
    switch (rule){
    case rule_REFL:     return kernel_REFL   (args[0].term());
    case rule_TRANS:    return kernel_TRANS  (args[0].thm(), args[1].thm());
    case rule_MK_COMB:  return kernel_MK_COMB(args[0].thm(), args[1].thm());
    case rule_ABS:      return kernel_ABS    (args[1].thm(), args[0].term());
    case rule_BETA:     return kernel_BETA   (args[0].term());
    case rule_ASSUME:   return kernel_ASSUME (args[0].term());
    case rule_EQ_MP:    return kernel_EQ_MP  (args[0].thm(), args[1].thm());
    case rule_DEDUCT:   return kernel_DEDUCT (args[0].thm(), args[1].thm());
    case rule_INST:     return helper_INST   (args);
    case rule_INST_T:   return helper_INST_T (args);
    case rule_New_Ax:   return kernel_New_Ax (args[0].axiom(), args[1].term());
    case rule_New_Def:  return kernel_New_Def(args[0].cnst(), args[1].term());
    case rule_New_TDef: return kernel_New_TDef(args[0].tcon(), args[1].cnst(), args[2].cnst(), args[5].thm());  //  -- "$tcon $cnst $cnst thm thm thm"
    case rule_TDef_Ex1: return kernel_Extract1(args[0].thm());
    case rule_TDef_Ex2: return kernel_Extract2(args[0].thm());
    default: assert(false); }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

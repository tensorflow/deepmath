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

#ifndef ZZ__HolLight__Printing_hh
#define ZZ__HolLight__Printing_hh

#include "Types.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


extern bool pp_full_internal;
extern bool pp_use_ansi;
extern bool pp_show_types;
extern bool pp_show_bound_types;
extern bool pp_readable_eqs;
extern bool pp_readable_nums;


#define PP_TYPE "\x1B[31m"
#define PP_CNST "\x1B[32m"
#define PP_VAR  "\x1B[33m"
#define PP_NUM  "\x1B[35m"
#define PP_END  "\x1B[0m"


// More readable: ($=var, #=const, `=type)
template<> fts_macro void write_(Out& out, Type const& ty, Str flags)
{
    assert(flags.size() <= 1);
    if (pp_full_internal){
        if (ty.is_tvar()) FWrite(out) "tvar(%_)", ty.tvar();
        else              FWrite(out) "tapp(%_, %_)", ty.tcon(), ty.targs();

    }else{
        if (ty.is_tvar())
            FWrite(out) "%_", ty.tvar();
        else{
            if (ty.tcon() == tcon_bool){
                assert(ty.targs().empty());
                FWrite(out) "bool";
            }else if (ty.tcon() == tcon_fun){
                List<Type> it = ty.targs();
                Type from = *it++;
                Type into = *it++;
                assert(!it);
                if (flags.size() == 1 && flags[0] == 'p') out += '(';
                FWrite(out) "%p->%_", from, into;
                if (flags.size() == 1 && flags[0] == 'p') out += ')';
            }else{
                if (ty.targs().empty()) FWrite(out) "%_", ty.tcon();
                else                    FWrite(out) "%_<%n>", ty.tcon(), ty.targs();    // -- this is not HOL-Light syntax
            }
        }
    }
}

template<> fts_macro void write_(Out& out, Type const& ty) { write_(out, ty, Str(empty_)); }


template<> fts_macro void write_(Out& out, Term const& tm)
{
    if (pp_full_internal){
        FWrite(out) "[%.8X/%_]", tm->var_mask, tm->lambda_c;
        if      (tm.is_var ()) FWrite(out) "var(%_)", tm.var();
        else if (tm.is_cnst()) FWrite(out) "cnst(%_)", tm.cnst();
        else if (tm.is_comb()) FWrite(out) "comb(%_, %_)", tm.fun(), tm.arg();
        else                   FWrite(out) "abs(%_, %_)", tm.avar(), tm.aterm();
        FWrite(out) ":%_", tm.type();

    }else{
        auto isEqTerm = [](Term tm) {
            return tm.is_comb()
                && tm.fun().is_comb()
                && tm.fun().fun().is_cnst()
                && tm.fun().fun().cnst() == cnst_eq;
        };

        auto isNum = [](Term tm) {
            return tm.is_comb()
                && tm.fun().is_cnst()
                && tm.fun().cnst() == cnst_NUMERAL;
        };

        function<bool(Term, uint64&)> getNum = [&getNum](Term tm, uint64& result) -> bool {
            if (tm.is_cnst()){
                if (tm.cnst() != cnst__0) return false;
                result = 0;
                return true;
            }else{
                if (!tm.is_comb() || !tm.fun().is_cnst()) return false;
                if (tm.fun().cnst() == cnst_NUMERAL)
                    return getNum(tm.arg(), result);
                else if (tm.fun().cnst() == cnst_BIT0){
                    if (!getNum(tm.arg(), result)) return false;
                    if (2 * result < result) return false;
                    result *= 2;
                    return true;
                }else if (tm.fun().cnst() == cnst_BIT1){
                    if (!getNum(tm.arg(), result)) return false;
                    if (2 * result + 1 < result) return false;
                    result = 2 * result + 1;
                    return true;
                }else
                    return false;
            }
        };

        if (tm.is_var()){
            if (pp_use_ansi) FWrite(out) PP_VAR  "%_" PP_END, tm.var();
            else             FWrite(out) "%_", tm.var();
        }else if (tm.is_cnst()){
            if (pp_use_ansi) FWrite(out) PP_CNST "%_" PP_END, tm.cnst();
            else             FWrite(out) "%_", tm.cnst();
        }else if (tm.is_comb()){
            if (pp_readable_eqs && isEqTerm(tm))
                FWrite(out) "[%_ = %_]", tm.fun().arg(), tm.arg();
            else if (pp_readable_nums && isNum(tm)){
                uint64 val;
                if (getNum(tm, val)){
                    if (pp_use_ansi) FWrite(out) PP_NUM "%_" PP_END, val;
                    else             FWrite(out)        "%_"       , val;
                }
                else FWrite(out) "(%_ %_)", tm.fun(), tm.arg();
            }else
                FWrite(out) "(%_ %_)", tm.fun(), tm.arg();
        }else{
            if (!pp_show_types && pp_show_bound_types){
                if (pp_use_ansi) FWrite(out) "(\\%_ " PP_TYPE ":%_" PP_END ". %_)", tm.avar(), tm.type(), tm.aterm();
                else             FWrite(out) "(\\%_ :%_. %_)"                     , tm.avar(), tm.type(), tm.aterm();
            }else
                FWrite(out) "(\\%_. %_)", tm.avar(), tm.aterm();
        }
    }

    if (pp_show_types){
        if (pp_use_ansi) FWrite(out) PP_TYPE ":%_" PP_END, tm.type();
        else             FWrite(out) ":%_", tm.type();
    }
}


template<> fts_macro void write_(Out& out, Thm const& th)
{
    FWrite(out) "%_ |- %_", th.hyps(), th.concl();
}


template<> fts_macro void write_(Out& out, const Subst&  v) { FWrite(out) "{%_ := %_}", v.x, v.tm; }
template<> fts_macro void write_(Out& out, const TSubst& v) { FWrite(out) "[%_ := %_]", v.a, v.ty; }


#undef PP_TYPE
#undef PP_CNST
#undef PP_VAR
#undef PP_END


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

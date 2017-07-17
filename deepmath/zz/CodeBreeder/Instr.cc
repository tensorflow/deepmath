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
#include "Instr.hh"
#include "Vm.hh"
#include "zz/Generics/Atom.hh"

namespace ZZ {
using namespace std;
using namespace VM;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


namespace{ struct ErrMsg { Instr p0; }; }

template<> fts_macro void write_(Out& out, const ErrMsg& v) {
    if (v.p0.n_words != 0)
        wr(out, " -- \"%_\"", Atom(v.p0.n_words));
}


void dumpInstrCode(Instr const* p, Out& out)
{
    switch (p[0].op_code){
    case i_COPY   : wr(out, "%_ = %_ :%_", p[1].pos, p[2].pos, p[0].n_words); break;
    case i_COPY_IS: wr(out, "%_ = [%_ + %_] :%_", p[1].pos, p[2].pos, p[3].off, p[0].n_words); break;
    case i_COPY_ID: wr(out, "[%_ + %_] = %_ :%_", p[1].pos, p[3].off, p[2].pos, p[0].n_words); break;
    case i_LEA    : wr(out, "%_ = LEA %_ + %_ :1", p[1].pos, p[2].pos, p[0].n_words); break;
    case i_ITE    : wr(out, "%_ = ITE %_ :%_", p[1].pos, p[2].pos, p[0].n_words); break;
    case i_PUT_CP : wr(out, "%_ = CODE_PTR %_", p[1].pos, p[2].off); break;
    case i_PUT    : wr(out, "%_ = %_", p[1].pos, p[2].off); break;
    case i_CALL   : wr(out, "%_ = CALL %_ (%_ :%_) :%_", p[1].pos, p[2].pos, p[4].pos, p[3].off, p[0].n_words); break;
    case i_CASE   : wr(out, "%_ = CASE (%_ :2*(1+%_)) :%_", p[1].pos, p[3].pos, p[2].off/2-1, p[0].n_words); break;
    case i_RETURN : wr(out, "RETURN"); break;
    case i_LOCALS : wr(out, "LOCALS %_", p[0].n_words); break;
    case i_GLOBALS: wr(out, "GLOBALS %_", p[0].n_words); break;
    case i_ALLOC  : wr(out, "%_ = ALLOC %_", p[1].pos, p[0].n_words); break;
    case i_CHK_REF: wr(out, "CHK_REF %_%_", p[1].pos, ErrMsg{p[0]}); break;
    case i_HALT   : wr(out, "HALT%_", ErrMsg{p[0]}); break;

    case i_VEC_SIZE  : wr(out, "%_ = ^%_.size :%_"   , p[1].pos, p[2].pos,           p[0].n_words); break;
    case i_VEC_GET   : wr(out, "%_ = ^%_.get(%_) :%_", p[1].pos, p[2].pos, p[3].pos, p[0].n_words); break;
    case i_VEC_GROWBY: wr(out, "^%_.growby(%_) :%_"  , p[1].pos, p[2].pos,           p[0].n_words); break;
    case i_VEC_RESIZE: wr(out, "^%_.resize(%_) :%_"  , p[1].pos, p[2].pos,           p[0].n_words); break;
    case i_VEC_SET   : wr(out, "^%_.set(%_, %_) :%_" , p[1].pos, p[2].pos, p[3].pos, p[0].n_words); break;

    case i_PR_BOOL : wr(out, "PRINT_BOOL %_", p[1].pos); break;
    case i_PR_INT  : wr(out, "PRINT_INT %_", p[1].pos); break;
    case i_PR_FLOAT: wr(out, "PRINT_FLOAT %_", p[1].pos); break;
    case i_PR_ATOM : wr(out, "PRINT_ATOM %_", p[1].pos); break;
    case i_PR_TEXT : wr(out, "PRINT_TEXT \"%_\"", Atom(p[1].off)); break;

    case i_TRY  : wr(out, "%_ = TRY %_ CATCH<%_> %_(%_ :%_) :%_", p[1].pos, p[2].pos, Atom(p[4].off), p[3].pos, p[5].pos, p[6].off, p[0].n_words); break;
    case i_THROW: wr(out, "THROW<%_> %_ :%_", Atom(p[1].off), p[2].pos, p[0].n_words); break;

    case i_RL_RUN: wr(out, "%_ = RL_RUN %_() :%_  {mem=%_, cpu=%_, rec=%_}", p[1].pos, p[2].pos, p[0].n_words, p[3].pos, p[4].pos, p[5].pos); break;

    case i_SUBR:  wr(out, "__SUBROUTINE__ :%_", p[0].n_words); break;

    case pi_PUT_CP: wr(out, "%_ = @__%_", p[1].pos, p[0].n_words); break;
    case pi_LABEL : wr(out, "@__%_", p[0].n_words); break;
    default: shoutLn("Invalid op-code: %_", p[0].op_code); assert(false); }
}


// NOTE: Only dumps the first word of the data (enough for most debugging).
void dumpInstrData(Instr const* p, function<VM::Word(RelPos)> getData, Out& out)
{
    std_out += "\a/";
    switch (p[0].op_code){
    case i_COPY   : wr(out, "*%_=%_", p[2].pos, getData(p[2].pos).val); break;
    case i_COPY_IS: wr(out, "*[%_ + %_]=%_", p[2].pos, p[3].off, getData(p[2].pos + p[3].off).val); break;
    case i_COPY_ID: wr(out, "*%_=%_", p[2].pos, getData(p[2].pos).val); break;
    case i_LEA    : wr(out, "*%_=%_", p[2].pos, getData(p[2].pos).val); break;
    case i_ITE    : wr(out, "*%_= %_? %_: %_", p[2].pos, getData(p[2].pos).val, getData(p[2].pos + 1).val, getData(p[2].pos + p[0].n_words + 1).val); break;
    default: break; }
    std_out += "\a/";
}


void dumpCode(Array<Instr const> code, Out& out)
{
    for (uint i = 0; i < code.size();){
        wr(out, "(%>6%_)   ", i);
        dumpInstr(&code[i], out);
        i += instr_size[code[i].op_code];
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

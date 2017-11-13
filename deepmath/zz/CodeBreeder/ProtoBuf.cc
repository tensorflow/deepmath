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
#include "ProtoBuf.hh"

#include "SynthSpec.hh"
#include "SynthEnum.hh"
#include "TrainingData.hh"

namespace ZZ {
using namespace std;
using namespace ENUM;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


void typeToProto(const Type& t, ::CodeBreeder::TypeProto* proto)
{
    proto->set_name(t.name.c_str());
    for (uint i = 0; i < t.size(); i++)
        typeToProto(t[i], proto->mutable_arg()->Add());
}


void GExpr::toProto(::CodeBreeder::NodeProto* proto) const
{
    proto->set_kind(GExprKind_name[kind]);
    proto->set_internal(internal);
    proto->set_cost(cost);
    for (uint i = 0; i < ins.psize(); i++)
        proto->add_input((int)ins[i]);
    if (type())
        typeToProto(type(), proto->mutable_type_proto());
}


void State::toProto(uint64 id, ::CodeBreeder::StateProto* proto) const
{
    proto->set_state_id(id);
    for (uint i = 0; i < size(); ++i)
        (*this)[i].toProto(proto->add_node());
}


void Pool::toProto(::CodeBreeder::PoolProto* proto) const
{
    for (uint i = 0; i < size(); i++){
        CExpr s = (*this)[i];
        proto->add_name(s->name.c_str());
        typeToProto(s->type, proto->add_type_proto());
        proto->add_qualified_name(fmt("%_", *s).c_str());
    }
}


void TrainingData::toProto(::CodeBreeder::TrainingProto* proto) const
{
    pool.toProto(proto->mutable_pool_proto());
    for (StatePair const& p : positive) p.snd.toProto(p.fst, proto->add_positive());
    for (StatePair const& p : negative) p.snd.toProto(p.fst, proto->add_negative());
    proto->set_evo_code   (evo_code   .c_str());
    proto->set_evo_output (evo_output .c_str());
    proto->set_type_string(type_string.c_str());
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

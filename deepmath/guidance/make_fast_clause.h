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

#ifndef RESEARCH_MATH_CLAUSE_SEARCH_MAKE_FAST_CLAUSE_H_
#define RESEARCH_MATH_CLAUSE_SEARCH_MAKE_FAST_CLAUSE_H_

#include "deepmath/eprover/clause.pb.h"
#include "deepmath/eprover/fol_formula.pb.h"
#include "deepmath/eprover/prover_clause_examples.pb.h"
#include "deepmath/eprover/fast_clause.pb.h"

namespace deepmath {

template <class Vocab>
void MakeFastTerm(const deepmath::FirstOrderLogicTerm& term,
                  const Vocab& vocab, deepmath::FastClause::Term* fast_term) {
  switch (term.term_type_case()) {
    case deepmath::FirstOrderLogicTerm::kVariable:
      fast_term->set_id(vocab(term.variable().name()));
      break;
    case deepmath::FirstOrderLogicTerm::kNumber:
      fast_term->set_id(vocab(term.number().value()));
      break;
    case deepmath::FirstOrderLogicTerm::kFunction: {
      fast_term->set_id(vocab(term.function().name()));
      for (const auto& arg : term.function().args()) {
        MakeFastTerm(arg, vocab, fast_term->add_args());
      }
      break;
    }
    default:
      LOG(FATAL) << "Invalid term type " << term.term_type_case();
  }
}

template <class Vocab>
void MakeFastClause(const ProverClause& clause, const Vocab& vocab,
                    deepmath::FastClause* fast_clause) {
  fast_clause->Clear();
  for (const auto& equation : clause.clause().equations()) {
    auto* fast_equation = fast_clause->add_equations();
    fast_equation->set_negated(equation.negated());
    MakeFastTerm(equation.left(), vocab, fast_equation->mutable_left());
    if (equation.has_right()) {
      MakeFastTerm(equation.right(), vocab, fast_equation->mutable_right());
    }
  }
}

}  // namespace deepmath

#endif  // RESEARCH_MATH_CLAUSE_SEARCH_MAKE_FAST_CLAUSE_H_

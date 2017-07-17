//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : ExprParser.cc
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : Parses expressions with infix, postfix and prefix operators and parenthesis.
//|
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//|
//| Operators are applied in order of falling precedence (priority), starting with the operator of
//| the highest priority. If two operators have the same priority, right-associative operators goes
//| first (counting prefix operators as left-associative and post-fix operators as right-
//| associative), thereafter operators of the same priority are applied from left to right.
//|________________________________________________________________________________________________

#include ZZ_Prelude_hh
#include "ExprParser.hh"

namespace ZZ {
using namespace std;

typedef XP_OpType OpType;
typedef XP_TokenStream TokenStream;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Run function with formatted error:


void* TokenStream::parse(String& err_msg)
{
    try{
        return parse();
    }catch (Excp_XP err){
        err_msg.clear();
        switch (err.type){
        case Excp_XP::MISSING_EXPRESSION:
            FWrite(err_msg) "[%_] Expected expression.", namePos(err.pos);
            break;
        case Excp_XP::PREFIX_OPERATOR_OUT_OF_PLACE:
            FWrite(err_msg) "[%_] Prefix operator '%_' found in postfix position.", namePos(err.pos), nameOp(err.tag);
            break;
        case Excp_XP::MISMATCHED_PARENTHESIS:
            FWrite(err_msg) "[%_] Mismatched parenthesis '%_'.", namePos(err.pos), nameRParen(err.tag);
            break;
        case Excp_XP::MISSING_CLOSING_PARENTHESIS:
            FWrite(err_msg) "[%_] Missing closing parenthesis for '%_'.", namePos(err.pos), nameLParen(err.tag);
            break;
        default: assert(false); }

        return NULL;
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Reduce expression:


// Called initially and whenever a prefix or postfix operator is reduced.
void TokenStream::activate(uint index)
{
    ParseElem& prev = xs[xs[index].prev];
    ParseElem& next = xs[xs[index].next];

    switch (xs[index].type){
    case xop_PREFIX:
        if (isExpr(next)) ns.add(OpNode(-xs[index].prio, index));
        break;
    case xop_POSTFIX:
        if (isExpr(prev)) ns.add(OpNode(-xs[index].prio, -index));
        break;
    case xop_INFIXL:
        if (isExpr(next) && isExpr(prev)) ns.add(OpNode(-xs[index].prio, index));
        break;
    case xop_INFIXR:
        if (isExpr(next) && isExpr(prev)) ns.add(OpNode(-xs[index].prio, -index));
        break;
    case xop_NULL:
    case xop_SENTINEL:
        /*nothing*/
        break;
    default: assert(false); }
}


void TokenStream::unlink(uint index)
{
    uint next = xs[index].next;
    uint prev = xs[index].prev;
    xs[next].prev = prev;
    xs[prev].next = next;
}


void TokenStream::reduce(uint from)
{
    assert_debug(xs[from].prio == INT_MIN+1);

    // Establish links:
    for (uint i = from; i < xs.size(); i++){
        xs[i].prev = i-1;
        xs[i].next = i+1;
    }
    xs[from].prev = xs.size()-1;
    xs[LAST].next = from;

    // Queue operators:
    ns.clear();
    for (uint i = from+1; i < xs.size(); i++)
        activate(i);

    // Reduce to single expression:
    while (ns.size() > 0){
        uint i = abs(ns.pop().index);
        uint next = xs[i].next;
        uint prev = xs[i].prev;
        if (xs[i].type == xop_PREFIX){
            assert_debug(isExpr(xs[next]));
            xs[next].expr = applyPrefix(xs[i].op_tag, xs[next].expr, xs[i].pos);
            unlink(i);
            activate(xs[next].prev);

        }else if (xs[i].type == xop_POSTFIX){
            assert_debug(isExpr(xs[prev]));
            xs[prev].expr = applyPostfix(xs[i].op_tag, xs[prev].expr, xs[i].pos);
            unlink(i);
            activate(xs[prev].next);

        }else{ assert_debug(xs[i].type == xop_INFIXL || xs[i].type == xop_INFIXR);
            assert_debug(isExpr(xs[next]));
            assert_debug(isExpr(xs[prev]));
            xs[prev].expr = applyInfix(xs[i].op_tag, xs[prev].expr, xs[next].expr, xs[i].pos);
            unlink(i);
            unlink(next);
        }
    }

    // Finish up:
    uint r = xs[from].next;
    assert_debug(xs[r].next == from);
    swp(xs[from], xs[r]);
    xs.shrinkTo(from + 1);
}


void TokenStream::disposeAllExprs()
{
    for (uint i = 0; i < xs.size(); i++)
        if (xs[i].type == xop_NULL && xs[i].expr != NULL)
            disposeExpr(xs[i].expr);
    xs.clear();
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Parse tokens:


void* TokenStream::parseExpr()
{
    ps.clear();
    xs.clear();

    xs.push();
    xs[LAST].type = xop_SENTINEL;      // -- sentinel
    xs[LAST].prio = INT_MIN+1;

    for(;;){
        // Must parse an expression:
        xs.push();
        ParseElem& x = xs[LAST];
        ParenElem p;

        if (parseOp(x.op_tag, x.pos, x.type, x.prio)){
            assert(x.type != xop_NULL);
            if (x.type != xop_PREFIX){
                disposeAllExprs();
                throw Excp_XP(Excp_XP::MISSING_EXPRESSION, x.pos); }

        }else if (parseLParen(p.paren_tag, p.pos)){
            p.index = xs.size() - 1;
            ps.push(p);
            xs[LAST].type = xop_SENTINEL;      // -- sentinel
            xs[LAST].prio = INT_MIN+1;

        }else if (parseAtom(x.expr, x.pos)){
            // May continue the expression:
            for(;;){
                xs.push();
                ParseElem& x = xs[LAST];
                ParenElem p;

                if (parseOp(x.op_tag, x.pos, x.type, x.prio)){
                    assert(x.type != xop_NULL);
                    if (x.type == xop_PREFIX){
                        disposeAllExprs();
                        throw Excp_XP(Excp_XP::PREFIX_OPERATOR_OUT_OF_PLACE, x.pos, x.op_tag);
                    }else if (x.type != xop_POSTFIX){
                        assert_debug(x.type == xop_INFIXL || x.type == xop_INFIXR);
                        break; }            // -- exit point: go back to "must parse expression"

                }else{
                    xs.pop();

                    if (ps.size() == 0){
                        goto Done;          // -- exit point: end of expression

                    }else if (parseRParen(p.paren_tag, p.pos)){
                        if (p.paren_tag != ps[LAST].paren_tag){
                            disposeAllExprs();
                            throw Excp_XP(Excp_XP::MISMATCHED_PARENTHESIS, p.pos, p.paren_tag); }

                        reduce(ps[LAST].index); assert_debug(xs.size() == ps[LAST].index+1); assert_debug(xs[LAST].op_tag == xop_NULL);
                        ps.pop();

                    }else{
                        disposeAllExprs();
                        throw Excp_XP(Excp_XP::MISSING_CLOSING_PARENTHESIS, ps[LAST].pos, ps[LAST].paren_tag); }
                }
            }

        }else{
            if (xs.size() == 2)
                return NULL;
            else
                throw Excp_XP(Excp_XP::MISSING_EXPRESSION, xs[LAST].pos);
        }
    }
  Done:;

    reduce(0); assert_debug(xs.size() == 1); assert_debug(xs[0].op_tag == xop_NULL);
    return xs[0].expr;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

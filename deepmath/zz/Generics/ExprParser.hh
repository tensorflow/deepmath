//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : ExprParser.hh
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

#ifndef ZZ__Generics__ExprParser_hh
#define ZZ__Generics__ExprParser_hh

#include "Heap.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helper types:


struct Excp_XP {
    enum Type {
        MISSING_EXPRESSION,             // tag unused
        PREFIX_OPERATOR_OUT_OF_PLACE,   // tag = op_tag
        MISMATCHED_PARENTHESIS,         // tag = paren_tag
        MISSING_CLOSING_PARENTHESIS,    // tag = paren_tag
    };

    Type type;
    uint pos;
    uint tag;
    Excp_XP(Type type_, uint pos_, uint tag_ = 0) : type(type_), pos(pos_), tag(tag_) {}
};


enum XP_OpType {
    xop_NULL,
    xop_PREFIX,
    xop_POSTFIX,
    xop_INFIXL,
    xop_INFIXR,
    xop_SENTINEL,
    XP_OpType_size
};


static const char* XP_OpType_name[XP_OpType_size] = { "<null>", "pre", "post", "inl", "inr", "sentinel" };

template<> fts_macro void write_(Out& out, const XP_OpType& v) {
    out += XP_OpType_name[v]; }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Expression parser (XP):


struct XP_TokenStream {
  //________________________________________
  //  Interface -- implement this:

    virtual ~XP_TokenStream() {}

    // Parsing: ("try parse", if cannot, return FALSE)
    virtual bool parseOp(uint& op_tag, uint& pos, XP_OpType& type, int& prio) = 0;
        // -- Everything defaults to zero/false; only 'type' has to be set.
        // The references should not be touched if FALSE is returned.
        // 'prio == INT_MIN' is reserved for internal use.
    virtual bool parseLParen(uint& paren_tag, uint& pos) = 0;
    virtual bool parseRParen(uint& paren_tag, uint& pos) = 0;
        // -- Defaults to zero (which is fine if only one type of parenthesis is used).
        // Tags for open and closing parenthesis must match.
    virtual bool parseAtom(void*& atom_expr, uint& pos) = 0;
        // -- Should return a newly allocated expression representing the atom.

    // Building expression:
    virtual void* applyPrefix (uint op_tag, void* expr)              { assert(false); }
    virtual void* applyPostfix(uint op_tag, void* expr)              { assert(false); }
    virtual void* applyInfix  (uint op_tag, void* expr0, void* expr1){ assert(false); }
        // -- If you don't care about operator position, override these, otherwise the three methods below.

    virtual void* applyPrefix (uint op_tag, void* expr, uint op_pos)               { return applyPrefix(op_tag, expr); }
    virtual void* applyPostfix(uint op_tag, void* expr, uint op_pos)               { return applyPostfix(op_tag, expr); }
    virtual void* applyInfix  (uint op_tag, void* expr0, void* expr1, uint op_pos) { return applyInfix(op_tag, expr0, expr1); }

    // Cleanup on parse error:
    virtual void disposeExpr(void* expr) = 0;

    // Optional methods for error messages:
    virtual String nameLParen(uint paren_tag) { assert(false); }
    virtual String nameRParen(uint paren_tag) { assert(false); }
    virtual String nameOp    (uint op_tag)    { assert(false); }
    virtual String namePos   (uint pos)       { assert(false); }

  //________________________________________
  //  Run functions -- call to parse:

    void* parse() { return parseExpr(); }
        // -- Returns an expression or throws 'Excp_XP' upon parse error. It is not
        // considered a parse error if the next token does not the start an expression;
        // instead NULL is returned so that this function can be used for probing.

    void* parse(String& err_msg);
        // -- returns NULL on parse error and a formatted error string through 'err_msg'
        // (requires optional methods to be defined). NOTE! 'err_msg' can also be the
        // empty string if the first token does not start an expression.

private:
  //________________________________________
  //  Internal stuff -- don't look:

    // Types:
    struct ParseElem {
        void*     expr;
        uint      prev;
        uint      next;

        uint      op_tag;
        uint      pos;
        XP_OpType type;
        int       prio;

        ParseElem() : expr(NULL), prev(0), next(0), op_tag(0), pos(0), type(xop_NULL), prio(0) {}
    };

    bool isOp  (const ParseElem& x) const { return x.type != xop_NULL; }
    bool isExpr(const ParseElem& x) const { return !isOp(x); }

    struct ParenElem {
        uint    paren_tag;
        uint    pos;
        uint    index;          // -- Index into 'ps' where sentinel for subexpression is
        ParenElem() : paren_tag(0), pos(0), index(0) {}
    };

    struct OpNode {
        int  prio;
        int  index;             // -- we use '-orig_index' for right-associative operators
        OpNode() {}
        OpNode(int prio_, int index_) : prio(prio_), index(index_) {}
        bool operator<(const OpNode& other) const { return prio < other.prio || (prio == other.prio && index < other.index); }
    };

    // Temporaries
    Vec<ParenElem>  ps;
    Vec<ParseElem>  xs;
    KeyHeap<OpNode> ns;

    // Methods:
    void  activate(uint index);
    void  unlink(uint index);
    void  reduce(uint from);
    void  disposeAllExprs();
    void* parseExpr();
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

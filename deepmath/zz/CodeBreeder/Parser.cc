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
#include "zz/Generics/IntSet.hh"
#include "zz/Generics/ExprParser.hh"
#include "TypeInference.hh"
#include "Vm.hh"

#define OPERATOR_SUPPORT

// Although meant for automatic generation, it is convenient to be able to test hand-written
// Evo-programs, especially while experimenting with language features.

namespace ZZ {
using namespace std;


ZZ_PTimer_Add(evo_parsing);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


static void updateLoc(cchar* start, cchar* end, Loc& loc)
{
    for (cchar* p = start; p != end; p++){
        if (*p == '\n')
            loc.col = 0, loc.line++;
        else
            loc.col++;          // -- screw tabs, you shouldn't use them anyway!
    }
}


namespace{
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Operators:   [internal namespace]


typedef XP_OpType OpType;  //  xop_NULL, xop_PREFIX, xop_POSTFIX, xop_INFIXL, xop_INFIXR


// Evo syntax: `##op(++, post, 5, inc_)` or `##op(<-->, pre, 10, list_reverse_<Int>)`
struct Op {
    Atom    name;
    OpType  type;
    int     prio;   // -- between -1000 and 1000, 0=priority of function application
    Expr    sym;    // -- operator will be expaned into this symbol (i.e. "2 + 2" will become "add_(2, 2)").

    Op() : type(xop_NULL), prio(0) {}
    Op(Atom name, OpType type, int prio, Expr sym) : name(name), type(type), prio(prio), sym(sym) {}

    explicit operator bool() const { return type != xop_NULL; }
};


static const Op Op_NULL;
constexpr int op_min_prio = -1000;
constexpr int op_max_prio = +1000;


class OpDecls {
    Vec<Op>         id2op;
    Map<Atom, uint> op2id;

public:
    OpDecls() {
        id2op.push(Op(a_appl_op, xop_INFIXL, 0, Expr()));     // -- synthetic function application operator
        op2id.set(a_appl_op, 0u); }

    void add(Atom name, OpType type, int prio, Expr const& sym) {
        op2id.set(name, id2op.size());
        id2op.push(Op(name, type, prio, sym)); }

    uint      has   (Atom name) const { return op2id.has(name); }
    uint      getId (Atom name) const { uint id; if (!op2id.peek(name, id)) assert(false); return id; }
    Op const& getOp (uint id)   const { return id2op[id]; }
    Op const& getOp (Atom name) const { return id2op[getId(name)]; }
};


// Token stream specialization for 'ZZ/Generics/ExprParser':
struct OpStream : XP_TokenStream {
    OpDecls   const& op_decls;
    Vec<Expr> const& exprs;
    uint             idx;     // -- index into 'exprs'

    OpStream(OpDecls const& op_decls, Vec<Expr> const& exprs) : op_decls(op_decls), exprs(exprs), idx(0) {}

    virtual bool parseLParen(uint& paren_tag, uint& pos){ return false; }
    virtual bool parseRParen(uint& paren_tag, uint& pos){ return false; }

    virtual bool parseOp(uint& op_tag, uint& pos, XP_OpType& type, int& prio){
        if (idx < exprs.size() && exprs[idx].kind == expr_Op){
            op_tag = op_decls.getId(exprs[idx].name);
            pos = idx++;
            type = op_decls.getOp(op_tag).type;
            prio = op_decls.getOp(op_tag).prio;
            return true;
        }else
            return false;
    }

    virtual bool parseAtom(void*& atom_expr, uint& pos){
        if (idx < exprs.size() && exprs[idx].kind != expr_Op){
            atom_expr = (void*)new Expr(exprs[idx]);
            pos = idx++;
            return true;
        }else
            return false;
    }

    virtual void* applyPrefix (uint op_tag, void* expr, uint op_pos) {
        Expr* ret = new Expr(Expr::Appl(Expr(op_decls.getOp(op_tag).sym), Expr(*(Expr*)expr)));
        ret->setLoc(exprs[op_pos].loc);
        disposeExpr(expr);
        return (void*)ret;
    }

    virtual void* applyPostfix(uint op_tag, void* expr, uint op_pos) {
        return applyPrefix(op_tag, expr, op_pos); }

    virtual void* applyInfix  (uint op_tag, void* expr0,  void* expr1, uint op_pos) {
        Expr* ret;
        if (op_tag == 0){   // -- tag 0 reserved for function application
            ret = new Expr(Expr::Appl(Expr(*(Expr*)expr0), Expr(*(Expr*)expr1)));
            ret->setLoc(((Expr*)expr0)->loc);
        }else{
            Loc loc = exprs[op_pos].loc;
            Expr arg = Expr::Tuple({Expr(*(Expr*)expr0), Expr(*(Expr*)expr1)}).setLoc(loc);
            ret = new Expr(Expr::Appl(Expr(op_decls.getOp(op_tag).sym), move(arg)));
            ret->setLoc(loc);
        }
        disposeExpr(expr0);
        disposeExpr(expr1);
        return (void*)ret;
    }

    virtual void disposeExpr(void* expr) { delete (Expr*)expr; }
};


}
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Tokenizer:


// Classes above 'cc_Space' form strings; below are parsed as single characters.
enum CharClass {
    cc_Null,        // '\0'
    cc_Op,          // !@#%&/=\\?+-^~*:|$<>   (with special hack for '>>...>')
    cc_Ident,       // A-Z a-z 0-9 _
    cc_Space,       // space, tab, newline, cr
    cc_Quote,       // `'"
    cc_Paren,       // ()[]{}<>
    cc_Seq,         // ,;.
    cc_Error,       // (everything else)
};

uchar char_class[256];


ZZ_Initializer(CharClass, 0) {
    for (uint c = 0; c < 256; c++){
        if (c == 0)
            char_class[c] = cc_Null;
        else if (strchr("!@#%&/=\\?+-^~*:|$<>", c))
            char_class[c] = cc_Op;
        else if (strchr("`'\"", c))
            char_class[c] = cc_Quote;
        else if (strchr("()[]{}", c))
            char_class[c] = cc_Paren;
        else if (strchr(",;.", c))
            char_class[c] = cc_Seq;
        else if (isIdentChar(c))
            char_class[c] = cc_Ident;
        else if (strchr(" \t\r\n", c))
            char_class[c] = cc_Space;
        else
            char_class[c] = cc_Error;
    }
};


class Tokenizer {
    cchar* text;
    Str    curr;
    Loc    loc_from;
    Loc    loc_upto;
    String cwd;         // -- '##include' statements are relative to this
    Atom   filename;    // -- hold on to filename for '__FILE__' symbol

    uchar cc(char c) const { return char_class[(uchar)c]; }
    void advance();

public:
    // Methods:
    Tokenizer(cchar* text, String cwd, OpDecls& op_decls, Atom filename = Atom(), bool suppress_filename = false) :
        // -- note that constructor may throw exception (fine as long as we have no destructor; otherwise function try-block should be used on constructor)
        text(text), cwd(cwd), filename(filename), op_decls(op_decls)
    {
        loc_upto = Loc(1, 0, suppress_filename ? Atom() : filename);
        curr = Str(text, 0u);
        advance();
    }

    Str  peek() const { return curr; }  // -- can be used to check for EOF (returns 'Str_NULL', which is false)
    Str  read() { Str ret = curr; advance(); return ret; }

    bool peekWs() const { return curr && isWS(curr.end_()); }   // -- is the current token (returned by 'peek()' followed by a whitespace?)

    Loc  loc() const { return loc_from; }
    String const& currDir() const { return cwd; }
    Atom currFile() const { return filename; }

    char operator[](uint i) const { return curr ? curr[i] : '\0';}
    bool operator==(CharClass cl) const { return cc((*this)[0]) == cl; }
    bool operator!=(CharClass cl) const { return !((*this) == cl); }

    bool is(cchar* text) const {      // -- compares next token to a string
        return curr ? eq(curr, text) : text[0] == '\0'; }
    bool match(cchar* text) {   // -- consuming version of 'is()'.
        if (is(text)){ read(); return true; } else return false; }
    void expect(cchar* text) {  // -- forced match (consume or fail with error)
        if (!match(text)) error(fmt("Expected '%_' not '%_'.", text, curr).c_str()); }

    void error(String const& msg) ___noreturn {
        throw Excp_ParseError(fmt("%_: %_", loc_from, msg)); }

    // Public state:
    OpDecls& op_decls;
};


void Tokenizer::advance()
{
    if (!curr) return;

    cchar* p = &curr.end_();
    cchar* p0 = p;
    for(;;){
        // Skip white-space:
        while (cc(*p) == cc_Space) p++;
        if (*p == '/' && p[1] == '/'){
            // Skip line comment:
            p += 2;
            while (*p != '\0' && *p++ != '\n');
        }else if (*p == '/' && p[1] == '*'){
            // Skip multi-line comment:
            p += 2;
            uint nesting = 1;
            for(;;){
                if (*p == '*' && p[1] == '/'){
                    p += 2;
                    nesting--;
                    if (nesting == 0) break;
                }else
                    p++;
            }
        }else
            break;
    }
    updateLoc(p0, p, loc_upto);
    loc_from = loc_upto;
    p0 = p;

    if (*p == '\0'){
        // End-of-file:
        curr = Str();

    }else if (*p == '"'){
        // String
        cchar* q = p+1;
        while (*q != '"' && *q != 0){
            if (*q == '\\') q++;
            q++;
        }
        curr = Str(p, q-p+1);
        updateLoc(p0, q+1, loc_upto);
        if (*q == 0) error("Unterminated string.");

    }else{
        cchar* q = p + 1;   // -- next character
        if (isDigit(*p) || ((p[0] == '-' || p[0] == '+') && isDigit(p[1]))){
            // Float or Int number:
            while (isIdentChar(*q) || *q == '.' || ((*q == '+' || *q == '-') && (q[-1] == 'e' || q[-1] == 'E')))
                q++;
        }else{
            // Normal token:
            if (cc(*p) < cc_Space
            && (*p != '>' || *q != '>')    // -- ugly hack to split some pairs ('>>' parsed as '> >', '<&' as '< &', ':&' ad ': &' and so on)
            && (*p != '<' || *q != '&')
            && (*p != ':' || *q != '&')
            && (*p != '>' || *q != '-')
            ){
                if (*p == '\\' && *q == ':')   // -- prefix '\:' is always split of as a token
                    q++;
                else
                    while (cc(*q) == cc(*p)) q++;
            }
        }
        curr = Str(p, q-p);
        updateLoc(p0, q, loc_upto);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Parser:


inline bool isType(Tokenizer const& tok) { return tok == cc_Ident && isType(tok.peek()); }
inline bool isTVar(Tokenizer const& tok) { return tok == cc_Ident && isTVar(tok.peek()); }
inline bool isVar (Tokenizer const& tok) { return tok == cc_Ident && isVar (tok.peek()); }
inline bool isOp  (Tokenizer const& tok) { return tok == cc_Op; }

inline bool isStr(Tokenizer const& tok) {
    return tok == cc_Quote && tok[0] == '"'; }

inline bool isInt(Tokenizer const& tok) {
    Str text = tok.peek();
    if (!text) return false;
    if (text[0] == '-' || text[0] == '+') text = text.slice(1);
    if (text.size() == 0 || char_class[(uchar)text[0]] != cc_Ident) return false;
    for (char c : text) if (c < '0' || c > '9') return false;
    return true; }

inline bool isFloat(Tokenizer const& tok) {
    Str text = tok.peek();
    if (!text) return false;
    if (text[0] == '-' || text[0] == '+') text = text.slice(1);
    if (text.size() == 0 || char_class[(uchar)text[0]] != cc_Ident) return false;
    bool may_be_float = false;
    for (char c : text)
        if (c == '.' || c == 'e' || c == 'E'){ may_be_float = true; break; }
    if (!may_be_float) return false;
    try{ stringToDouble(tok.peek()); return true; }
    catch (...){ return false; }
}


inline Str expectType(Tokenizer& tok){ if (!isType(tok)) tok.error("Expected type name.");     return tok.read(); }
inline Str expectTVar(Tokenizer& tok){ if (!isTVar(tok)) tok.error("Expected type variable."); return tok.read(); }
inline Str expectVar (Tokenizer& tok){ if (!isVar (tok)) tok.error("Expected variable name."); return tok.read(); }
inline Str expectInt (Tokenizer& tok){ if (!isInt (tok)) tok.error("Expected integer.");       return tok.read(); }
inline Str expectStr (Tokenizer& tok){ if (!isStr (tok)) tok.error("Expected string.");        return tok.read(); }
inline Str expectOp  (Tokenizer& tok){ if (!isOp  (tok)) tok.error("Expected operator.");      return tok.read(); }


inline bool isEndOfExpr(Tokenizer const& tok) {
    if (tok == cc_Seq || tok == cc_Null) return true;
    if (tok == cc_Paren && (tok[0] == ')' || tok[0] == '}' || tok[0] == ']')) return true;
    return false; }


// 'open_brace' can be empty if already consumed; an empty 'close_brace' means "end-of-file".
template<class T>
static Vec<T> parseList(Tokenizer& tok, T (*parseElem)(Tokenizer& tok), cchar* open_brace, cchar* close_brace, cchar* separator = ",")
{
    if (open_brace[0] != '\0')
        tok.expect(open_brace);

    Vec<T> ts;
    for(;;){
        if (tok.match(close_brace)) return ts;
        ts.push(parseElem(tok));
        if (tok.match(close_brace)) return ts;
        tok.expect(separator);
    }
}


//=================================================================================================
// -- Type parser:


static Type parseType(Tokenizer& tok);


static Type parseType_nonFun(Tokenizer& tok)
{
    Loc loc = tok.loc();
    Type ret;

    if (isType(tok)){
        Atom kind = Atom(tok.read());
        ret = tok.match("<") ? Type(kind, parseList(tok, parseType, "", ">")) : Type(kind);

    }else if (isTVar(tok)){
        ret = Type(tok.read());

    }else if (tok.match("&")){
        ret = Type(a_Ref, parseType_nonFun(tok));

    }else if (tok.match("(")){
        if (tok.match(")")) return Type(a_Void).setLoc(loc);
        Vec<Type> ts = parseList(tok, parseType, "", ")");
        ret = (ts.size() > 1) ? Type(a_Tuple, ts) : ts[0];

    }else if (tok.match("{")){
        ret = Type(a_OneOf, parseList(tok, parseType, "", "}"));

    }else if (tok.match("[")){
        Type t = parseType(tok);
        tok.expect("]");
        ret = Type(a_Vec, t);

    }else
        tok.error(fmt("Expected type not '%_'.", tok.peek()));

    return ret.setLoc(loc);
}


static Type parseType(Tokenizer& tok)
{
    Loc loc = tok.loc();

    Type t = parseType_nonFun(tok);
    return (tok.match("->")) ? Type(a_Fun, t, parseType(tok)).setLoc(loc) : t;
}


//=================================================================================================
// -- Expression parser:


static Expr parseExpr   (Tokenizer& tok);
static Expr parseExpr_tc(Tokenizer& tok);
static Expr parseStmt   (Tokenizer& tok);


// Patterns are represented as tuples.
static Expr parsePattern(Tokenizer& tok)
{
    Loc loc = tok.loc();

    if (tok.is("{"))   // -- epsilon pattern; next token must be '{'
        return Expr::Tuple({}).setLoc(loc);
    else if (isVar(tok))
        return Expr::Sym(tok.read(), {}, Type()).setLoc(loc);
    else
        return Expr::Tuple(parseList(tok, parsePattern, "(", ")")).setLoc(loc);    // -- 'Tuple' constructor handles singleton case (will not create tuple)
}


static Expr parseTypedPattern(Tokenizer& tok)
{
    Loc loc = tok.loc();

    if (isVar(tok)){
        Str var = tok.read();
        tok.expect(":");
        return Expr::Sym(var, {}, parseType(tok)).setLoc(loc);
    }else
        return Expr::Tuple(parseList(tok, parseTypedPattern, "(", ")")).setLoc(loc);
}


static Type parseTypeVar(Tokenizer& tok) {
    Loc loc = tok.loc();
    return Type(expectTVar(tok)).setLoc(loc); }


// A symbol is a variable name + optional type list in angle braces ('<' '>').
static Expr parseSymbol_(Tokenizer& tok, Type (*parse_typearg)(Tokenizer&))
{
    Loc loc = tok.loc();
    bool has_ws = tok.peekWs();
    Atom name = expectVar(tok);
    Vec<Type> targ = (!has_ws && tok.match("<")) ? parseList(tok, parse_typearg, "", ">") : Vec<Type>();
    return Expr::Sym(name, targ, Type()).setLoc(loc);
}
static Expr parseSymbol   (Tokenizer& tok){ return parseSymbol_(tok, parseType); }
static Expr parseDefSymbol(Tokenizer& tok){ return parseSymbol_(tok, parseTypeVar); }


static Expr parseConstr(Tokenizer& tok)
{
    Loc loc = tok.loc();
    Type t   = parseType(tok);
    tok.expect(".");
    Str  idx = expectInt(tok);
    return Expr::Cons(move(t), idx).setLoc(loc);
}


static Expr parseBlock(Tokenizer& tok, bool skip_open_brace = false) {
    Loc loc = tok.loc();
    return Expr::Block(parseList(tok, parseStmt, skip_open_brace?"":"{", "}", ";")).setLoc(loc); }



// No function application "f g" or tuple element selection "p'0".
static Expr parseExpr_simple(Tokenizer& tok)
{
    Loc loc = tok.loc();
    Expr ret;

    // Literals:
    if      (tok.is(";"))    ret = Expr::Tuple({});
    else if (tok.is("_0"))   ret = Expr::Lit(tok.read(), Type(a_Bool));
    else if (tok.is("_1"))   ret = Expr::Lit(tok.read(), Type(a_Bool));
    else if (isInt(tok))     ret = Expr::Lit(tok.read(), Type(a_Int));
    else if (isFloat(tok))   ret = Expr::Lit(tok.read(), Type(a_Float));
    else if (isVar(tok))   { ret = parseSymbol(tok); if (ret.name == a_file) loc.file = tok.currFile(); }   // -- make sure '__FILE__' has full location
    else if (isType(tok) || tok.match(".")) ret = parseConstr(tok);

    // Identifiers:
    else if (tok == cc_Ident) tok.error(fmt("Unexpected symbol '%_'.", tok.peek()));
    else if (isStr(tok))     ret = Expr::Lit(tok.read(), Type(a_Atom));

    // Special, non-operator characters:
    else if (tok.match("{")) ret = parseBlock(tok, true);
    else if (tok.match("(")) ret = Expr::Tuple(parseList(tok, parseExpr, "", ")"));
    else if (tok.match("&")) ret = Expr::MkRef(parseExpr_simple(tok));  // <<== normal op
    else if (tok.match("^")) ret = Expr::Deref(parseExpr_simple(tok));  // <<== normal op
    else if (tok.match("#")){ Atom idx = expectInt(tok); ret = Expr::Sel(parseExpr_simple(tok), idx); }
    else if (tok.match("`")){ Expr block = Expr::Block({parseExpr(tok)}); ret = Expr::Lamb(move(Expr::Tuple({}).setLoc(loc)), move(block.setLoc(loc)), Type()); }
    else if (tok.match("[")){
        tok.expect(":");
        Type t = parseType(tok);
        ret = Expr::MkVec(parseList(tok, parseExpr, "", "]"), Type(a_Vec, t)); }
    else if (tok.match("\\:")){
        Type type = parseType(tok);
        Expr head = parsePattern(tok);
        Expr body = parseBlock(tok);
        ret = Expr::Lamb(move(head), move(body), move(type)); }
    else if (tok.is("\\"))
        tok.error("Use typed lambda '\\:' in untyped contexts.");
    else if (tok.match("##")){
        if (tok.match("if")){
            tok.expect("(");
            Expr tup = Expr::Tuple(parseList(tok, parseExpr, "", ")"));
            if (tup.kind != expr_Tuple || tup.size() != 3){ throw Excp_ParseError(fmt("%_: Static-if '##if' expects three arguments.", loc)); }
            ret = Expr::MetaIf(move(tup[0]), move(tup[1]), move(tup[2]));
        }else if (tok.match("error")){
            Type t = tok.match(":") ? parseType(tok) : Type();
            Atom msg = isStr(tok) ? Atom(tok.read()) : Atom();
            ret = Expr::MetaErr(msg, move(t));
        }else
            throw Excp_ParseError(fmt("%_: Invalid compile-time directive.", loc));
    }
  #if defined(OPERATOR_SUPPORT)
    else if (isOp(tok)){
        ret = Expr::Op(tok.read());
        if (!tok.op_decls.has(ret.name))
            throw Excp_ParseError(fmt("%_: Undefined operator: %_", loc, ret.name)); }
  #endif
    else tok.error(fmt("Invalid expression: %_", tok.peek()).c_str());

    return ret.setLoc(loc);
}


// Function argument (always in a typed context)
static Expr parseExpr_arg(Tokenizer& tok)
{
    Loc loc = tok.loc();

    if (tok.match("("))
        return Expr::Tuple(parseList(tok, parseExpr_tc, "", ")")).setLoc(loc);
    else if (tok.match("\\")){
        Expr head = parsePattern(tok);
        Expr body = parseBlock(tok);
        return Expr::Lamb(move(head), move(body), Type()).setLoc(loc);
    }else
        return parseExpr_simple(tok);
}


// Expression, including function application.
static Expr parseExpr_(Tokenizer& tok, bool tc)
{
  #if defined(OPERATOR_SUPPORT)
    auto getOp = [&](Expr const& expr) -> Op const& {
        if (expr.kind != expr_Op) return Op_NULL;
        return tok.op_decls.getOp(expr.name);
    };

    Vec<Expr> es;
    es.push(tc ? parseExpr_arg(tok) : parseExpr_simple(tok));
    while (!isEndOfExpr(tok)){
        Expr e = parseExpr_arg(tok);
        Op const& op0 = getOp(es[LAST]);
        Op const& op1 = getOp(e);
        if ((!op0 || (op0.type == xop_POSTFIX && op0.prio > 0)) && (!op1 || (op1.type == xop_PREFIX  && op1.prio > 0)))
            es.push(Expr::Op(a_appl_op));
        es.push(e);
    }

    OpStream op_stream(tok.op_decls, es);
    try{
        void* ptr = op_stream.parse(); assert(ptr);
        Expr ret = *(Expr*)ptr;
        op_stream.disposeExpr(ptr);
        return ret;
    }catch(Excp_XP excp){
        if (excp.type == Excp_XP::MISSING_EXPRESSION)
            throw Excp_ParseError(fmt("%_: Operator missing operand.", op_stream.exprs[excp.pos].loc));
        else if (excp.type == Excp_XP::PREFIX_OPERATOR_OUT_OF_PLACE)
            throw Excp_ParseError(fmt("%_: Prefix operator '%_' out-of-place.", op_stream.exprs[excp.pos].loc, tok.op_decls.getOp(excp.tag).name));
        else
            assert(false);
    }

  #else
    Loc loc = tok.loc();
    Expr e = tc ? parseExpr_arg(tok) : parseExpr_simple(tok);
    while (!isEndOfExpr(tok)){
        Expr arg = parseExpr_arg(tok);
        e = Expr::Appl(move(e), move(arg)).setLoc(loc); }
    return e;
  #endif
}

static Expr parseExpr   (Tokenizer& tok) { return parseExpr_(tok, false); }
static Expr parseExpr_tc(Tokenizer& tok) { return parseExpr_(tok, true ); }


//=================================================================================================
// -- Statement parser:


static Type argType(Expr args)
{
    if (args.kind == expr_Sym)
        return args.type;
    else{ assert(args.kind == expr_Tuple);
        if (args.size() == 0)
            return Type(a_Void);
        Vec<Type> ts;
        for (Expr const& e : args) ts.push(argType(e));
        return Type(a_Tuple, ts);
    }
}


static Expr parseStmt(Tokenizer& tok)
{
    Loc loc = tok.loc();

    if (tok.match("let")){
        Expr vars = parsePattern(tok);
        Type type = tok.match(":") ? parseType(tok) : Type();
        Expr init = tok.match("=") ? (type ? parseExpr_tc(tok) : parseExpr(tok)) : Expr();
        if (!type && !init) tok.error("Variables must either have a type or be initialized to an expression with a bottom-up derivable type.");
        if (!init && type.name != a_Ref && type.name != a_Vec) tok.error("Only reference or vector variables may omit initialization in 'let'.");
        return (init ? Expr::LetDef(move(vars), move(type), move(init)) : Expr::LetDef(move(vars), move(type))).setLoc(loc);

    }else if (tok.match("rec")){
        Expr sym = parseDefSymbol(tok);
        tok.expect(":");
        Type type = parseType(tok);
        tok.expect("=");
        Expr init = type ? parseExpr_tc(tok) : parseExpr(tok);
        return Expr::RecDef(move(sym), move(type), move(init)).setLoc(loc);

    }else if (tok.match("fun")){
        auto metaErr = [](Loc loc) { return Expr::Block({Expr::MetaErr().setLoc(loc)}).setLoc(loc); };
        Expr sym = parseDefSymbol(tok);
        Expr args = parseTypedPattern(tok);
        Type ret_type = tok.match("->") ? parseType(tok) : Type(a_Void).setLoc(tok.loc());    // -- may leave out Void return type
        Expr body = (tok.is(";")) ? metaErr(tok.loc()) : parseBlock(tok);
        Type arg_type = argType(args);
        Expr lamb = Expr::Lamb(move(args), move(body), Type()).setLoc(args.loc);
        return Expr::RecDef(move(sym), Type(a_Fun, move(arg_type), move(ret_type)), move(lamb)).setLoc(loc);

    }else
        return parseExpr(tok);
}


//=================================================================================================
//  -- Typedefs and programs:


static Expr parseTypeDef(Tokenizer& tok, bool synon)
{
    Loc loc = tok.loc();

    Str name = expectType(tok);
    Vec<Type> targ = tok.match("<") ? parseList(tok, parseTypeVar, "", ">") : Vec<Type>();
    tok.expect("=");
    if (!synon && !tok.is("{")) tok.error("Data types must be sum-types \"data T = {T0, T1, ...}\".");
    Type tdef = parseType(tok);
    return Expr::NewType(synon, name, targ, move(tdef)).setLoc(loc);
}


static void parseOpDecl(Tokenizer& tok)
{
    tok.expect("(");
    Atom op_name = Atom(expectOp(tok));
    if (tok.op_decls.has(op_name))
        tok.error(fmt("Operator already introduced: %_", op_name));

    tok.expect(",");
    OpType op_type = xop_NULL;
    if (isVar(tok)){
        Str type_text = tok.peek();
        op_type = eq(type_text, "pre" ) ? xop_PREFIX :
                  eq(type_text, "post") ? xop_POSTFIX :
                  eq(type_text, "inl" ) ? xop_INFIXL :
                  eq(type_text, "inr" ) ? xop_INFIXR :
                  /*otherwise*/           xop_NULL;
    }
    if (op_type == xop_NULL)
        tok.error("Operator type must be one of: pre, post, inl, inr");
    tok.read();

    tok.expect(",");
    Str prio_text = expectInt(tok);
    int64 op_prio = INT_MIN;
    try{
        op_prio = stringToInt64(prio_text);
        if (op_prio < op_min_prio || op_prio > op_max_prio) op_prio = INT_MIN;
    }catch (...){}
    if (op_prio == INT_MIN)
        tok.error(fmt("Operator priorities must be in the range [%_, %_]", op_min_prio, op_max_prio));

    tok.expect(",");
    Expr op_sym = parseSymbol(tok);
    tok.expect(")");

    // Store it:
    tok.op_decls.add(op_name, op_type, op_prio, op_sym);
}


static void parseProg(Atom filename, String cwd, bool suppress_filename, IntSeen<Atom>& includes, Vec<Expr>& out, OpDecls& op_decls);


static void parseProg(Tokenizer& tok, IntSeen<Atom>& includes, Vec<Expr>& out)
{
    for(;;){
        if (tok.match("type"))
            out.push(parseTypeDef(tok, true));
        else if (tok.match("data"))
            out.push(parseTypeDef(tok, false));
        else if (tok.match("##")){
            if (tok.match("op")){
                parseOpDecl(tok);
            }else{
                tok.expect("include");
                Atom filename = expectStr(tok).slice(1, END-1);     // <<== maybe here is the place where the filename should be expanded
                if (!includes.add(filename))
                    parseProg(filename, tok.currDir(), false, includes, out, tok.op_decls);
            }
        }else
            out.push(parseStmt(tok));

        if (tok.match("")) return;  // -- matches end-of-file
        tok.expect(";");
        if (tok.match("")) return;
    }
}


static void parseProg(Atom filename, String cwd, bool suppress_filename, IntSeen<Atom>& includes, Vec<Expr>& out, OpDecls& op_decls)
{
    if (filename.str()[0] != '/'){
        char* abs_filename = realpath(fmt("%_/%_", cwd, filename).c_str(), nullptr);    // -- filename is relative to 'tok.cwd'
        if (!abs_filename) throw Excp_ParseError(fmt("Missing include file: %_  (resolved relative to: %_)", filename, cwd));
        filename = Atom(abs_filename);
        xfree(abs_filename);
    }

    Vec<char> text;
    if (!readFile(Str(filename), text, true))
        throw Excp_ParseError(fmt("Could not read file '%_'.", filename));

    Tokenizer tok(text.base(), dirName(filename.str()), op_decls, filename, suppress_filename);
    parseProg(tok, includes, out);
}


static Expr parseProg(Atom filename)
{
    Vec<Expr> out;
    IntSeen<Atom> includes;
    OpDecls op_decls;
    parseProg(filename, currDir(), true, includes, out, op_decls);
    return Expr::Block(out);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Main:


Expr parseEvoFile(String filename)
{
    ZZ_PTimer_Scope(evo_parsing);

    Expr prog = parseProg(filename);
    if (getenv("PARSED")){
        wrLn("\a*PARSED:\a0\t+\t+");
        wrLn("%\n_;", join(";\n\n", map(prog, ppFmt)));
        wrLn("\t-\t-");
        newLn();
    }
    return prog;
}


void parseEvo(cchar* text, Vec<Expr>& out)
{
    ZZ_PTimer_Scope(evo_parsing);

    OpDecls op_decls;
    Tokenizer tok(text, currDir(), op_decls);
    IntSeen<Atom> includes;
    parseProg(tok, includes, out);
}


Type parseType(cchar* text)
{
    OpDecls op_decls;
    Tokenizer tok(text, currDir(), op_decls);
    return parseType(tok);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

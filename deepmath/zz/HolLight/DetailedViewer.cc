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
#include "DetailedViewer.hh"
#include "zz/Console/ConsoleStd.hh"
#include "PremiseViewer.hh"
#include "HolFormat.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


static void pad(Vec<AChar>& text, uint width, Attr imbue)
{
    assert(width > 0);
    while (text.size() < width)
        text.push(' ' + imbue);
    for (AChar& c : text)
        c.bg = imbue.bg;
    if (text.size() > width){
        text.shrinkTo(width - 1);
        text.push('~' + darkenFg(imbue));
    }
}


inline bool isEqTerm(Term tm) {
    return tm.is_comb()
        && tm.fun().is_comb()
        && tm.fun().fun().is_cnst()
        && tm.fun().fun().cnst() == cnst_eq; }


template<class T>
inline T dist(T x, T y) { return (x > y) ? x - y : y - x; }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Colors:


BgAttr bg_left  = bgGray(3);
BgAttr bg_right = bgGray(0);
BgAttr bg_info  = bgRgb(0, 0, 1);

FgAttr fg_rule = fgRgb(4, 4, 2, sty_BOLD);
FgAttr fg_str  = fgGray(20);
FgAttr fg_idx  = fgGray(23);
FgAttr fg_glue = fgGray(23, sty_BOLD);
FgAttr fg_turn = fgRgb(5,2,5, sty_BOLD);

FgAttr fg_type_glue = fgRgb(4, 4, 5, sty_ITAL | sty_BOLD);
FgAttr fg_type_cnst = fgRgb(3, 4, 5, sty_ITAL);
FgAttr fg_type_var  = fgRgb(2, 3, 4, sty_ITAL);
FgAttr fg_type_sep  = fgRgb(1, 2, 3);

FgAttr fg_term[CharCat_size] = {
    fgGray(20),              // CNST
    fgRgb(5,3,0, sty_BOLD),  // EQUAL
    fgRgb(1,5,1, sty_BOLD),  // BINDER
    fgRgb(0,4,0),            // VAR
    fgRgb(1,5,1, sty_BOLD),  // ABSVAR
    fgRgb(5,1,1),            // FREEVAR
    fgRgb(3,4,5, sty_ITAL),  // TYPE
    fgGray(23, sty_BOLD),    // OTHER
};


inline FgAttr fg_obj(ArgKind kind) {
    if (kind == arg_TYPE_IDX) return fgRgb(3,4,5);
    if (kind == arg_TERM_IDX) return fgRgb(5,1,1);
//  if (kind == arg_THM_IDX ) return fgRgb(5,5,5);
    if (kind == arg_THM_IDX ) return fgRgb(1,5,1);
    assert(false);
}


/*
I infobox: fulla teoremnamnet (om något)
Hypoteser utspellade
*/


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// DetailedViewer:


class DetailedViewer {
    ProofStore& P;

    void addFaninConeToView(line_t n);
    void filterView();

    Vec<AChar> formatArg(Arg arg);
    Vec<AChar> formatRuleApplication(line_t n);

    Vec<AChar> formatTerm(Term tm);
    Vec<AChar> formatType(Type ty);
    void outputRuleResult(Vec<AChar>& out, line_t n);

    void redraw();
    void redrawInfo();

  //________________________________________
  //  Data selection ("what to view"):

    IntZet<line_t> prems;
    IntZet<line_t> sinks;

    IntZet<line_t> full_view;   // -- includes types and terms
    IntZet<line_t> view;
    IntZet<line_t> abbr;        // -- these objects are abbreviated on the RHS

    uind cursor = UIND_MAX;     // -- index into 'view'
    uind top = 0;               // -- index into 'view'
    bool show_only_thms = true;

  //________________________________________
  //  Formatting ("how to view it"):

    uint rule_width = 35;
    uint info_rows = 8;

    //bool hide_term_building;

public:

    DetailedViewer(ProofStore& P) : P(P) {}
    void init(Vec<line_t> const& theorems, Vec<line_t> const& premises);
    bool run();
};


//=================================================================================================


void DetailedViewer::addFaninConeToView(line_t n)
{
    if (view.add(n) || prems.has(n)) return;
    for (Arg arg : P.args(n)){
        if      (arg.kind == arg_TYPE_IDX) addFaninConeToView(P.ty_idx2line[arg.id]);
        else if (arg.kind == arg_TERM_IDX) addFaninConeToView(P.tm_idx2line[arg.id]);
        else if (arg.kind == arg_THM_IDX ) addFaninConeToView(P.th_idx2line[arg.id]);
    }
}


void DetailedViewer::filterView() {
    if (!show_only_thms)
        full_view.copyTo(view);
    else{
        view.clear();
        for (line_t n : full_view.list())
            if (P.ret(n) == arg_THM) view.add(n);
    }
}


void DetailedViewer::init(Vec<line_t> const& theorems, Vec<line_t> const& premises)
{
    cursor = UIND_MAX;
    top = 0;
    show_only_thms = true;

    prems.clear(); for (line_t n : premises) prems.add(n);
    sinks.clear(); for (line_t n : theorems) sinks.add(n);

    view.clear();
    for (line_t n : theorems)
        addFaninConeToView(n);
    sort(view.list());

    view.copyTo(full_view);
    filterView();
}


Vec<AChar> DetailedViewer::formatArg(Arg arg) {
    return arg.isAtomic() ? attrFmt({fg_str}, "\"%_\"", arg.str()) :
                            attrFmt({fg_obj(arg.kind)}, "%_%_", objSymbol(arg.kind), arg.id);      // <<== localize ID
}


Vec<AChar> DetailedViewer::formatRuleApplication(line_t n)
{
    static const Vec<Attr> attrs{fg_glue, fg_rule};

    String args_text;
    bool first = true;
    for (Arg arg : P.args(n)){
        if (first) first = false;
        else args_text += ", ";
        args_text += formatArg(arg);
    }

    Arg ret{makeIndex(P.ret(n)), P.line2idx[n]};
    return attrFmt(attrs, "%_ := \a1{%_\a}(%_)", formatArg(ret), P.rule(n), args_text);
}


Vec<AChar> DetailedViewer::formatTerm(Term tm)
{
    Vec<AChar> out;
    for (DChar d : fmtTerm(tm)){
        //**/if (d.chr == '`' && d.cat != cc_CNST) continue;
        out.push(AChar(d.chr, fg_term[d.cat])); }
    return out;
}


// <<== do this properly when have time
Vec<AChar> DetailedViewer::formatType(Type ty0)
{
    Vec<AChar> out;
    function<void(Type)> rec = [&](Type ty) {
        if (ty.is_tvar())
            attrWrite(out, {fg_type_var}, "%_", ty.tvar());
        else{
            if (ty.tcon() == tcon_fun){
                Type from = ty.targs()[0];
                Type into = ty.targs()[1];
                out += '(' + fg_type_glue;
                rec(from);
                out += '-' + fg_type_glue, '>' + fg_type_glue;
                rec(into);
                out += ')' + fg_type_glue;
            }else{
                attrWrite(out, {fg_type_cnst}, "%_", ty.tcon());
                if (!ty.targs().empty()){
                    out += '<' + fg_type_glue;
                    bool first = true;
                    for (List<Type> it = ty.targs(); it; ++it){
                        if (first) first = false;
                        else out += ',' + fg_type_glue, ' ' + fg_type_glue;
                        rec(*it);
                    }
                    out += '>' + fg_type_glue;
                }
            }
        }
    };

    rec(ty0);
    return out;
}


void DetailedViewer::outputRuleResult(Vec<AChar>& out, line_t n)
{
    if (P.ret(n) == arg_TERM){
        Term tm = P.evalTerm(P.line2idx[n]);
        append(out, formatTerm(tm));
        append(out, attrFmt({fg_type_sep}, " : "));
        append(out, formatType(tm.type()));

    }else if (P.ret(n) == arg_THM){
        Thm th = P.evalThm(P.line2idx[n]);
        if (!th.hyps().empty()){
            bool first = true;
            for (List<Term> it = th.hyps(); it; ++it){
                if (first) first = false;
                else append(out, attrFmt({fg_turn * bg_right}, ", "));
                append(out, formatTerm(*it));
            }
            out += bg_right;
        }
        append(out, attrFmt({fg_turn * bg_right}, "|- "));
        append(out, formatTerm(th.concl()));

        if (isEqTerm(th.concl())){
            append(out, attrFmt({fg_type_sep}, "  ["));
            append(out, formatType(th.concl().arg().type()));
            append(out, attrFmt({fg_type_sep}, "]"));
        }else{
            append(out, attrFmt({fg_type_sep}, " : "));
            append(out, formatType(th.concl().type()));
        }

    }else if (P.ret(n) == arg_TYPE){
        append(out, formatType(P.evalType(P.line2idx[n])));
    }
}


void DetailedViewer::redraw()
{
    fillScreen();
    uint y = 0;
    for (uind i = top; i < view.list().size(); i++){
        line_t n = view[i];

        Vec<AChar> out;

        char marker = sinks.has(n) ? 0xAB : prems.has(n) ? 0xBB : ' ';
        out.push(marker + fgGray(23)*bg_left);

        append(out, formatRuleApplication(n));
        pad(out, rule_width, bg_left);

        out += bg_right;
        outputRuleResult(out, n);

        while (out.size() < con_cols()) out += bg_right;
        if (i == cursor){ for (AChar& c : out) c = hiliteBg(c); }
        printAt(y, 0, out);

        y++;
        if (y + (cursor != UIND_MAX ? info_rows : 0) >= con_rows()) break;
    }
    redrawInfo();
}


/*
Print arguments in full (with types), including names for human theorems.
Print types for all variables (free and bound) as well as constants
*/
void DetailedViewer::redrawInfo()
{
    if (cursor == UIND_MAX) return;

    fill(~info_rows, 0, ~0, ~0, bg_info);

    uint y = 0;
    Vec<AChar> out;

    auto flushOut = [&]() {
        pad(out, out.size(), bg_info);
        for (uind i = 0; i < out.size(); i += con_cols()){
            printAt(~info_rows + y, 0, out.slice(i));
            y++;
            if (y >= info_rows) break;
        }
        out.clear();
    };

#if 1   /*DEBUG*/
    attrWrite(out, {fg_glue, fg_rule}, "\a1{%_\a}(", P.rule(view[cursor]));
    bool first = true;
    for (Arg arg : P.args(view[cursor])){
        if (first) first = false;
        else out += ','+fg_glue, ' '+fg_glue;

        Term tm; if (arg.kind == arg_TERM_IDX) tm = P.evalTerm(arg.id);
        if (arg.isAtomic())
            attrWrite(out, {fg_str}, "\"%_\"", arg.str());
        else if (arg.kind == arg_TYPE_IDX)
            append(out, formatType(P.idx2type[arg.id]));
        else if (arg.kind == arg_TERM_IDX && !tm.is_composite())
            append(out, formatTerm(tm));
        else
            attrWrite(out, {fg_obj(arg.kind)}, "%_%_", objSymbol(arg.kind), arg.id);      // <<== localize ID
    }

    attrWrite(out, {fg_glue}, ")");
    flushOut();

    for (Arg arg : P.args(view[cursor])){
        Term tm; if (arg.kind == arg_TERM_IDX) tm = P.evalTerm(arg.id);
        if (arg.kind == arg_THM_IDX){
            out += ' '+fg_glue, ' '+fg_glue;
            attrWrite(out, {fg_obj(arg.kind), fg_glue}, "%_%_\a1{:\a} ", objSymbol(arg.kind), arg.id);      // <<== localize ID
            outputRuleResult(out, P.th_idx2line[arg.id]);
            flushOut();
        }else if (arg.kind == arg_TERM_IDX && tm.is_composite()){
            out += ' '+fg_glue, ' '+fg_glue;
            attrWrite(out, {fg_obj(arg.kind), fg_glue}, "%_%_\a1{:\a} ", objSymbol(arg.kind), arg.id);      // <<== localize ID
            append(out, formatTerm(tm));
            flushOut();
        }
    }

    attrWrite(out, {fg_rule}, "Result:");
    flushOut();

#endif  /*END DEBUG*/

    out += ' '+fg_glue, ' '+fg_glue;
    outputRuleResult(out, view[cursor]);
    flushOut();
}


bool DetailedViewer::run()
{
    for(;;){
        redraw();
        ConEvent ev = con_getEvent();

        uint page_rows = con_rows() - (cursor != UIND_MAX ? info_rows : 0);
        bool moved_cursor = false;

        if (ev.type == ev_KEY){
            if (ev.key == 'q')
                return false;
            if (ev.key == (chr_META | 0x4F50) || ev.key == (chr_META | 0x4F51) || ev.key == (chr_CSI | 0x32357E))
                return true;

            if (ev.key == 'c'){
                line_t n = (cursor == UIND_MAX) ? 0 : view[cursor];
                show_only_thms ^= 1;
                filterView();
                if (cursor != UIND_MAX){
                    // Find nearest row to where I was:
                    cursor = 0;
                    for (uind i = 0; i < view.list().size(); i++)
                        if (dist(view.list()[i], n) < dist(view.list()[cursor], n))
                            cursor = i;
                    moved_cursor = true;
                }
            }

            if (ev.key == '<'){ if (rule_width > 10) rule_width--; }
            if (ev.key == '>'){ if (rule_width < con_cols()) rule_width++; }
            if (ev.key == ','){ if (info_rows > 3) info_rows--; }
            if (ev.key == '.'){ if (info_rows < con_rows()) info_rows++; }
        }

        navigate(ev, view.size(), page_rows, top, cursor, moved_cursor, info_rows, 0);
    }
}


// If 'theorems' is empty, go back to last view. Returns TRUE if should continue, FALSE if should quit program.
bool detailedViewer(ProofStore& P, Vec<line_t> const& theorems, Vec<line_t> const& premises)
{
    static DetailedViewer D(P);
    if (theorems.size() > 0)
        D.init(theorems, premises);
    return D.run();
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

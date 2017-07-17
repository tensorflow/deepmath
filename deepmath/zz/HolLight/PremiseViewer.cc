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
#include "PremiseViewer.hh"
#include "zz/Console/ConsoleStd.hh"
#include "HolFormat.hh"
#include "ProofStore.hh"
#include "DetailedViewer.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


// Return a string of length 5 with the suffix (if any) protruding
static String compactNum(uint64 num)
{
    String out;
    if (num < 10000) FWrite(out) "%>4%_ ", num;
    else             FWrite(out) "%>5%'D", num;
    return out;
}


static Vec<AChar> labelPad(Vec<AChar> text, uint width)
{
    AChar c = text.last();
    c.chr = ' ';

    reverse(text); text.push(c); reverse(text);
    while (text.size() < width - 1)
        text.push(c);
    c.alt &= ~sty_UNDER;
    text.push(c);

    for (uint i = 1; i < text.size()-1; i++)
        text[i].alt |= sty_UNDER;

    return text;
}


template<class T>
struct ColumnLT {
    typedef line_t Key;
    Vec<T> const& prio;
    ColumnLT(Vec<T> const& prio) : prio(prio) {}
    bool operator()(line_t m, line_t n) const { return prio[m] < prio[n] || (prio[m] == prio[n] && m < n); }
};

template<class T>
ColumnLT<T> columnLT(Vec<T> const& prio) { return ColumnLT<T>(prio); }


// <<== This is hackish; will clean this up
void navigate(ConEvent ev, uind view_size, uint& page_rows, uind& top, uind& cursor, bool& moved_cursor, uint no_cursor_extra_rows, uint top_margin)
{
    auto sub = [](uind& num, uint amount) { num = (amount > num) ? 0 : num - amount; };

    if (ev.type == ev_KEY){
        if (ev.key == 27){
            if (cursor != UIND_MAX){
                cursor = UIND_MAX;
                page_rows += no_cursor_extra_rows;    // -- screen just got a little bit bigger; adjustments below need this info
                moved_cursor = true;
            }
        }

        if (ev.key == 10){
            if (cursor == UIND_MAX) cursor = top, moved_cursor = true;
        }

       if (ev.key == (chr_CSI|0x42)){      // arrow down
            if (cursor == UIND_MAX) top++;
            else cursor = min_(view_size, cursor + 1), moved_cursor = true; }
        if (ev.key == (chr_CSI|0x41)){      // arrow up
            if (cursor == UIND_MAX) sub(top, 1);
            else sub(cursor, 1), moved_cursor = true; }
        if (ev.key == (chr_CSI|0x3242) || ev.key == (chr_CSI|0x313B3242))   // -- shift + arrow down
            top++;
        if (ev.key == (chr_CSI|0x3241) || ev.key == (chr_CSI|0x313B3241))   // -- shift + arrow up
            sub(top, 1);

        if (ev.key == (chr_CSI|0x367E)){    // page-down
            if (cursor == UIND_MAX) top += page_rows;
            else top += page_rows, cursor = min_(view_size, cursor + page_rows), moved_cursor = true; }
        if (ev.key == (chr_CSI|0x357E)){    // page-up
            if (cursor == UIND_MAX) sub(top, page_rows);
            else sub(top, page_rows), sub(cursor, page_rows), moved_cursor = true; }

        if (ev.key == (chr_CSI|0x363B357E)){
            top = view_size - page_rows;
            if (cursor != UIND_MAX) cursor = view_size-1; }
        if (ev.key == (chr_CSI|0x353B357E)){
            top = 0;
            if (cursor != UIND_MAX) cursor = 0; }

    }else if (ev.type == ev_MOUSE){
        if (ev.key == 'u') sub(top, 5);
        if (ev.key == 'd') top += 5;

        if (ev.key == 'l' && ev.row >= top_margin && ev.row - top_margin < page_rows){
            cursor = top + ev.row - top_margin;
            moved_cursor = true;            // -- toggle info pane
        }
    }

    // Adjust top-row:
    if (page_rows >= view_size)
        top = 0;
    else if (top + page_rows > view_size)
        top = view_size - page_rows;

    // Adjust cursor, if moved it:
    if (moved_cursor && cursor != UIND_MAX){
        if (cursor > view_size - 1) cursor = view_size - 1;
        if (cursor < top) top = cursor;
        if (cursor - top > page_rows-1) top = cursor - page_rows + 1;
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Colors:


FgAttr fg_labels     = fgRgb(4, 4, 2, sty_BOLD);

BgAttr bg_left_pane  = bgRgb(1, 0, 0);
FgAttr fg_left_thm   = fgGray(23, sty_BOLD);
FgAttr fg_left_def   = fgRgb(3,4,5, sty_BOLD | sty_ITAL);

BgAttr bg_right_pane = bgGray(0);
FgAttr fg_right_thm  = fgGray(23);
FgAttr fg_right_def  = fgRgb(3,4,5, sty_ITAL);

BgAttr bg_mid_pane   = bgGray(3);
FgAttr fg_mid_text   = fgRgb(2,5,2);

BgAttr bg_info_pane  = bgRgb(0, 0, 1);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Column interface:


struct Column {
    virtual uint       width() = 0;
    virtual Vec<AChar> label() = 0;
    virtual Vec<AChar> print(line_t) = 0;
    virtual ~Column() {}
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Columns declarations:


struct PremiseViewer;


struct Column_Theorem : Column {
    PremiseViewer const& V;
    uint name_width = UINT_MAX; // -- must be set before 'print()'

    Column_Theorem(PremiseViewer const& V) : V(V) {}

    // Interface:
    uint width();
    Vec<AChar> label();
    Vec<AChar> print(line_t);
};


struct Column_Premises : Column {
    PremiseViewer const& V;

    uint width_ = UINT_MAX;      // -- must be set before 'print()'
    uint xoffset = 0;
    bool show_defs = true;
    bool show_thms = true;
    bool compact   = false;      // -- strip '_DEF' and '_THM'

    Column_Premises(PremiseViewer const& V) : V(V) {}

    // Interface:
    uint width();
    Vec<AChar> label();
    Vec<AChar> print(line_t);
};


struct Column_ProofSize : Column {
    PremiseViewer const& V;
    Column_ProofSize(PremiseViewer const& V) : V(V) {}

    // Interface:
    uint width();
    Vec<AChar> label();
    Vec<AChar> print(line_t);
};


struct Column_Uses : Column {
    PremiseViewer const& V;
    Column_Uses(PremiseViewer const& V) : V(V) {}

    // Interface:
    uint width();
    Vec<AChar> label();
    Vec<AChar> print(line_t);
};


struct Column_Depth : Column {
    PremiseViewer const& V;
    Column_Depth(PremiseViewer const& V) : V(V) {}

    // Interface:
    uint width();
    Vec<AChar> label();
    Vec<AChar> print(line_t);
};


struct Column_WDepth : Column {
    PremiseViewer const& V;
    Column_WDepth(PremiseViewer const& V) : V(V) {}

    // Interface:
    uint width();
    Vec<AChar> label();
    Vec<AChar> print(line_t);
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// PremiseViewer:


struct PremiseViewer {
    ProofStore& P;

    String thmName(line_t n) const { return P.thmName(n) ? String(P.thmName(n)) : objName(n); }

    bool matchGranularity(line_t n) const;
    void getPremises(line_t n, Vec<line_t>& prems, IntZet<line_t>& seen) const;

    void viewAll();     // -- view selection; more stuff here later
    void computeColumns();

    void redraw();
    void redrawInfo();

    void eventLoop();

    // Small helpers:
    String objName(line_t n) const { return (FMT "%C%_", objSymbol(P.ret(n)), P.line2idx[n]); }

  //________________________________________
  //  Data for column printing:

    IntZet<line_t>  seen;
    Vec<line_t>     prems;

    Vec<uint>       proof_sz;
    Vec<uint>       n_uses;
    Vec<uint>       depth;
    Vec<uint64>     wdepth;

  //________________________________________
  //  Data selection ("what to view"):

    uint            thm_granul = 0;     // -- theorem granularity: 0=human named, 1='!'-marked, 2=multi-fanout
    IntZet<line_t>  view;               // -- view restriction: only list theorems in this set

    uind            cursor = UIND_MAX;  // -- index into 'view'
    IntZet<line_t>  cursor_inputs;
    IntZet<line_t>  cursor_outputs;

    line_t          info_thm_line = 0;

    IntZet<line_t>  consider;           // -- detailed view consideration (may include elements outside view restriction)
   /*consider_expr*/

    uind top = 0;   // -- index into 'view.list()'

    // Columns:
    Column_Theorem   col_theorem;
    Column_Premises  col_premises;
    Column_ProofSize col_proof_size;
    Column_Uses      col_uses;
    Column_Depth     col_depth;
    Column_WDepth    col_wdepth;

  //________________________________________
  //  Formatting ("how to view it"):

    uint horz_step_size = 10;   // -- for cursor left/right
    uint info_pane_rows = 5;

  //________________________________________
  //  Creation and launch:

    PremiseViewer(ProofStore& P) :
        P(P),
        col_theorem   (*this),
        col_premises  (*this),
        col_proof_size(*this),
        col_uses      (*this),
        col_depth     (*this),
        col_wdepth    (*this)
    {}

    void oneOff(String theorem);
    void run();
};


bool PremiseViewer::matchGranularity(line_t n) const
{
    if (!P.is_thm(n)) return false;
    return (thm_granul == 0 && P.is_humanThm(n))
        || (thm_granul == 1 && P.is_markedThm(n))
        || (thm_granul == 2 && P.is_fanoutThm(n));
}


void PremiseViewer::getPremises(line_t n, Vec<line_t>& prems, IntZet<line_t>& seen) const
{
    assert(P.is_thm(n));
    for (line_t m : P.thms(n)){
        if (seen.add(m)) continue;

        if (matchGranularity(m))
            prems.push(m);
        else
            getPremises(m, prems, seen);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Columns implementations:


//=================================================================================================
// -- Column "Theorem":


uint Column_Theorem::width() {
    return name_width + 2; }


Vec<AChar> Column_Theorem::label() {
    return attrFmt({bg_left_pane*fg_labels}, "Theorem:"); }


Vec<AChar> Column_Theorem::print(line_t n)
{
    Vec<AChar> out;
    String name = V.thmName(n);
    bool is_def = (V.P.rule(n) == rule_New_Def);
    Attr attr = bg_left_pane * (is_def ? fg_left_def : fg_left_thm);

    //if (in selection) add '>>'; else
    out += ' ' + attr;

    if (name.size() > name_width){
        attrWrite(out, {attr}, "%_", name.slice(0, name_width));
        out += '~' + darkenFg(attr);
    }else{
        attrWrite(out, {attr}, "%_", name);
        for (uint i = name.size(); i < name_width+1; i++)
            out += ' ' + attr;
    }
    return out;
}


//=================================================================================================
// -- Column "Premises":


uint       Column_Premises::width() { assert(width_ != UIND_MAX); return width_; }
Vec<AChar> Column_Premises::label() { return attrFmt({bg_right_pane*fg_labels}, "Premises (inputs):"); }

Vec<AChar> Column_Premises::print(line_t n)
{
    Vec<AChar> out;

    for (line_t m : V.prems){
        bool is_def = (V.P.rule(m) == rule_New_Def);
        if (is_def ? !show_defs : !show_thms) continue;

        String name = V.thmName(m);
        if (compact && (suf(name, "_DEF") || suf(name, "_THM")))
            name.shrinkTo(name.size() - 4);

        Attr attr = (is_def ? fg_right_def : fg_right_thm) * bg_right_pane;
        attrWrite(out, {attr}, "%C %_", (compact?0:' '), name);
    }

    // Trim:        // <<== be smarter about this; premise list can be very long so wasteful to print and erase so much
    reverse(out);
    for (uint i = 0; i < xoffset && out.size() > 0; i++) out.pop();
    reverse(out);

    while (out.size() > width_) out.pop();
    while (out.size() < width_) out.push(bg_right_pane);

    return out;
}


//=================================================================================================
// -- Column "ProofSize":


uint       Column_ProofSize::width() { return 7; }
Vec<AChar> Column_ProofSize::label() { return attrFmt({fg_labels * bg_mid_pane}, "PfSz:"); }

Vec<AChar> Column_ProofSize::print(line_t n) {
    return attrFmt({fg_mid_text * bg_mid_pane}, " %_ ", compactNum(V.seen.size())); }


//=================================================================================================
// -- Column "Uses":


uint       Column_Uses::width() { return 7; }
Vec<AChar> Column_Uses::label() { return attrFmt({fg_labels * bg_mid_pane}, "Uses:"); }

Vec<AChar> Column_Uses::print(line_t n) {
    return attrFmt({fg_mid_text * bg_mid_pane}, " %_ ", compactNum(V.n_uses[n])); }


//=================================================================================================
// -- Column "Depth":


uint       Column_Depth::width() { return 7; }
Vec<AChar> Column_Depth::label() { return attrFmt({fg_labels * bg_mid_pane}, "Dep.:"); }

Vec<AChar> Column_Depth::print(line_t n) {
    return attrFmt({fg_mid_text * bg_mid_pane}, " %_ ", compactNum(V.depth[n])); }


//=================================================================================================
// -- Column "WDepth":


uint       Column_WDepth::width() { return 7; }
Vec<AChar> Column_WDepth::label() { return attrFmt({fg_labels * bg_mid_pane}, "WDep:"); }

Vec<AChar> Column_WDepth::print(line_t n) {
    return attrFmt({fg_mid_text * bg_mid_pane}, " %_ ", compactNum(V.wdepth[n])); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Rendering:


void PremiseViewer::redraw()
{
    fillScreen();

    col_premises.width_ = con_cols() - col_theorem.width() - col_proof_size.width();

    AChar sep = (char)0xA6 + darkenFg(bg_mid_pane * fg_mid_text);
    Vec<AChar> out;
    append(out, labelPad(col_theorem   .label(), col_theorem   .width()));
    append(out, labelPad(col_proof_size.label(), col_proof_size.width())); out += sep;
    append(out, labelPad(col_uses      .label(), col_uses      .width())); out += sep;
    append(out, labelPad(col_depth     .label(), col_depth     .width())); out += sep;
    append(out, labelPad(col_wdepth    .label(), col_wdepth    .width()));
    append(out, labelPad(col_premises  .label(), col_premises  .width()));
    printAt(0, 0, out);

    uint y = 1;
    for (uind i = top; i < view.list().size(); i++){
        line_t n = view[i];

        // Compute premises for theorem 'n':
        prems.clear();
        seen.clear();
        getPremises(n, prems, seen);
        sort(prems);

        out.clear();
        append(out, col_theorem   .print(n));
        append(out, col_proof_size.print(n)); out += sep;
        append(out, col_uses      .print(n)); out += sep;
        append(out, col_depth     .print(n)); out += sep;
        append(out, col_wdepth    .print(n));
        append(out, col_premises  .print(n));

        if (i == cursor){ for (AChar& c : out) c = hiliteBg(c); }
        printAt(y, 0, out);

        y++;
        if (y >= con_rows()) break;     // <<== can stop a little earlier if info pane is visible
    }

    if (info_thm_line) redrawInfo();
}


/*
Ta bort "rule" rad, istället: fullständigt namn följt av typ:  (Thm, Def, Axiom)
Visa hyps()
Radbryt, skrolla om inte ryms (mus-hjul), kanske TAB för att byta panel?
*/
void PremiseViewer::redrawInfo()
{
    fill(~info_pane_rows, 0, ~0, ~0, bg_info_pane);

    {
        String out;
        FWrite(out) "%_ = %_(", objName(info_thm_line), P.rule(info_thm_line);
        for (Arg arg : P.args(info_thm_line)){
            if (arg.isAtomic()) FWrite(out) " %_(\"%_\")", arg.kind, arg.str();
            else                FWrite(out) " %_%d", objSymbol(arg.kind), arg.id;
        }
        FWrite(out) " )";
        printAt(~4, 0, out, fgRgb(3,5,5) * bg_info_pane);
    }

    Thm thm = P.evalThm(P.line2idx[info_thm_line]);
    Vec<DChar> text = fmtTerm(thm.concl());

    uint y = ~2;
    uint x = 0;
    for (DChar d : text){
        if (x >= con_cols()) break;
//            if (d.chr == '`' && d.cat != cc_CNST) continue;
        Attr attr = (d.cat == cc_CNST   ) ? fgGray(20)              * bg_info_pane :
                    (d.cat == cc_EQUAL  ) ? fgRgb(5,3,0, sty_BOLD)  * bg_info_pane :
                    (d.cat == cc_BINDER ) ? fgRgb(1,5,1, sty_BOLD)  * bg_info_pane :
                    (d.cat == cc_FREEVAR) ? fgRgb(5,1,1)            * bg_info_pane :
                    (d.cat == cc_VAR    ) ? fgRgb(0,4,0)            * bg_info_pane :
                    (d.cat == cc_ABSVAR ) ? fgRgb(0,4,0, sty_UNDER) * bg_info_pane :
                    (d.cat == cc_TYPE   ) ? fgRgb(3,4,5, sty_ITAL)  * bg_info_pane :
                    (d.cat == cc_OTHER  ) ? fgGray(23, sty_BOLD)    * bg_info_pane :
                    /*otherwise*/          (assert(false), Attr());
        putAt(y, x++, d.chr + attr);

        // <<== also, name lambda variables differently (<4: x,y,z,w, else a,b,c,d... skip used constants, use all letters)
        // <<== operator "," seems to be missing (builtin?)
        // <<== välj lambda-var symbol efter typ. Om T->bool, använd predikatsymbol (kapilär). Om S->T använd f,g,h för funktioner, om S->T->Q, infix op?
    }

    if (thm.hyps().size() > 0)
        printAt(~1, 0, (FMT "#hyps: %_", thm.hyps().size()), FgAttr(color_white)*bg_info_pane);
}


// Set up the view restriction to include all theorems of the current granularity.
void PremiseViewer::viewAll()
{
    view.clear();
    for (uind n = 0; n < P.size(); n++)
        if (matchGranularity(n))
            view.push(n);
}


void PremiseViewer::computeColumns()       // <<== analyze
{
    proof_sz.reset(P.size(), 0);
    n_uses  .reset(P.size(), 0);
    depth   .reset(P.size(), 0);
    wdepth  .reset(P.size(), 0);

    IntZet<line_t> seen;
    Vec<line_t>    prems;
    for (line_t n = 0; n < P.size(); n++){
        if (matchGranularity(n)){
            prems.clear();
            seen.clear();
            getPremises(n, prems, seen);

            uint   d = 0;
            uint64 w = 0;
            for (line_t m : prems){
                n_uses[m]++;
                newMax(d, depth[m]);
                newMax(w, wdepth[m]);
            }
            proof_sz[n] = seen.size();
            depth [n]   = d + 1;
            wdepth[n]   = w + seen.size();
        }
    }
}


static String prompt(String query)
{
    Vec<AChar> bup;
    for (uint x = 0; x < con_cols(); x++) bup.push(getAt(~1, x));

    String text;
    for(;;){
        fillRow(~1, 0, ~0, bgRgb(0, 2, 0));
        printAt(~1, 0, query, bgRgb(0, 2, 0) * fgRgb(5, 5, 5));
        printAt(~1, query.size(), text, bgRgb(0, 2, 0) * fgRgb(5, 5, 1));
        con_showCursor(~1, query.size() + text.size());

        ConEvent ev = con_getEvent();
        if      (ev.key == 0x7F){ if (text.size() > 0) text.pop(); }
        else if (ev.key == '\n') break;
        else if (ev.key == 27){ text.clear(); break; }
        else if (ev.key >= 32 && ev.key < 127) text.push(ev.key);
    }
    con_hideCursor();

    for (uint x = 0; x < bup.size(); x++) putAt(~1, x, bup[x]);

    return text;
}


void PremiseViewer::eventLoop()
{
    line_t eval_n = 1;
    for(;;){
        // Get event (and evaluate proof-lines in background):
        ConEvent ev;
        if (eval_n < P.size()){
            ev = con_getEvent(0.0);
            if (ev.type == ev_NULL){
                double T0 = realTime();
                while (eval_n < P.size() && realTime() < T0 + 0.01)
                    P.evalLine(eval_n++);
                redraw();
                printAt(con_rows()-1, con_cols()-10, (FMT " %>5%.1f %%  ", 100.0*eval_n / P.size()), fgRgb(5,5,2, sty_ITAL) * color_gray(6));
            }
        }else{
            redraw();
            ev = con_getEvent(0.0);
            if (ev.type == ev_NULL)
                ev = con_getEvent(0.04);
        }

        // React to event:
        auto sub = [](uint& num, uint amount) { num = (amount > num) ? 0 : num - amount; };
        uint page_rows = con_rows() - 1 - (info_thm_line ? info_pane_rows : 0);
        bool moved_cursor = false;

        navigate(ev, view.size(), page_rows, top, cursor, moved_cursor, info_pane_rows, 1);

        if (ev.type == ev_RESIZE){
            redraw();

        }else if (ev.type == ev_KEY){
            if (ev.key == 'q') break;

            if (ev.key == 'g'){
                thm_granul = (thm_granul + 1) % 3;
                viewAll();              // <<== if view expression, reapply it
                computeColumns();       // -- must be updated     <<== only do this if columnt is visible (expensive)
            }

            if (ev.key == 's' && cursor != UIND_MAX){
                Vec<line_t> theorems{view[cursor]};
                prems.clear();
                seen.clear();
                getPremises(view[cursor], prems, seen);
                if (!detailedViewer(P, theorems, prems))
                    break;
            }

            if (ev.key == (chr_META | 0x4F50) || ev.key == (chr_META | 0x4F51) || ev.key == (chr_CSI | 0x32357E)){
                if (!detailedViewer(P, {}, {}))
                    break;
            }

            if (ev.key == 'c') col_premises.compact ^= 1;
            if (ev.key == 'd') col_premises.show_defs ^= 1;
            if (ev.key == 't') col_premises.show_thms ^= 1;

            if (ev.key == (chr_CSI|0x43))       // arrow right
                col_premises.xoffset += horz_step_size;
            if (ev.key == (chr_CSI|0x44))       // arrow left
                sub(col_premises.xoffset, horz_step_size);

            if (ev.key == 'a'){
                viewAll();
                cursor = UIND_MAX;
                moved_cursor = true;
            }

            if (ev.key == 'f' || ev.key == '/'){
                auto match = [](String const& name, String const& pattern) {
                    return strstr(name.c_str(), pattern.c_str());   // <<== for now
                };

                String pattern = prompt("Search for theorem: ");
                if (pattern != ""){
                    IntZet<line_t> new_view;
                    for (uind n = 0; n < P.size(); n++)
                        if (matchGranularity(n) && match(thmName(n), pattern))
                            new_view.push(n);
                    if (new_view.size() > 0){
                        new_view.moveTo(view);
                        cursor = 0;
                        moved_cursor = true;
                    }
                }
            }


#if 1   /*DEBUG*/
            if (ev.key == 'z'){
                view.clear();
                for (uind n = 0; n < P.size(); n++)
                    if (P.rule(n) == rule_New_TDef)
                        view.push(n);
                moved_cursor = true;
                cursor = UIND_MAX;
            }
#endif  /*END DEBUG*/

        }else if (ev.type == ev_MOUSE){
            if (ev.type == ev_MOUSE && ev.key == 'l' && ev.row == 0){
                Vec<line_t> old_order(copy_, view.list());
                bool did_sort = false;
                uint x = ev.col;

                uint w0 = 0; uint w1 = col_theorem.width();
                if (x >= w0 && x < w1) sort(view.list()), did_sort = true;

                w0 = w1; w1 += col_proof_size.width() + 1;  // <<== +1 here is temporary
                if (x >= w0 && x < w1) sobSort(sob(view.list(), columnLT(proof_sz))), did_sort = true;

                w0 = w1; w1 += col_uses.width() + 1;  // <<== +1 here is temporary
                if (x >= w0 && x < w1) sobSort(sob(view.list(), columnLT(n_uses))), did_sort = true;

                w0 = w1; w1 += col_depth.width() + 1;  // <<== +1 here is temporary
                if (x >= w0 && x < w1) sobSort(sob(view.list(), columnLT(depth))), did_sort = true;

                w0 = w1; w1 += col_wdepth.width();
                if (x >= w0 && x < w1) sobSort(sob(view.list(), columnLT(wdepth))), did_sort = true;

                if (did_sort && vecEqual(old_order, view.list()))
                    reverse(view.list());
                moved_cursor = did_sort;
            }
        }

        // Update 'info_thm_line' from cursor (if should):
        if (moved_cursor)
            info_thm_line = (cursor == UIND_MAX) ? 0 : view[cursor];
    }
}


void PremiseViewer::oneOff(String theorem)
{
    for (line_t n = 0; n < P.size(); n++){
        if (P.is_markedThm(n) && eq(Str(P.thmName(n)), theorem)){
            prems.clear();
            seen.clear();
            getPremises(n, prems, seen);

            con_init();
            detailedViewer(P, Vec<line_t>{n}, prems);
            con_close();
            return;
        }
    }
    shoutLn("ERROR! No such theorem: %_", theorem);
    exit(1);
}


void PremiseViewer::run()
{
    // Calculate good width for left pane (theorem names):
    Vec<uint> lens;
    for (line_t n = 0; n < P.size(); n++)
        if (P.is_humanThm(n))     // -- assume marked theorems have short names
            lens.push(Str(P.thmName(n)).size());
    sort(lens);
    col_theorem.name_width = (lens.size() == 0) ? 1u : max_(1u, lens[uind(lens.size() * 0.97)]);
    newMax(col_theorem.name_width, 10u);    // -- make sure big enough to fit label

    // Launch:
    computeColumns();   // <<== delay this
    viewAll();

    con_init();
    eventLoop();
    con_close();
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Main:


void viewProof(String filename, String theorem)
{
    ProofStore P(filename, !true);
    PremiseViewer V(P);

    if (theorem != "") V.oneOff(theorem);
    else               V.run();
}


/*
premise mode
  - 'E' evaluate all, 'F' update multi-fanout marks?

  - default: named theorem (submodes: !-theorems and multi-fanount theorems)
      'h' = human named, 'n' = named, 'm' = multi-fanout

  - make selection: 's'
      i = one fanin
      i* = transitive
      i# = upto human-named theorem  }
      i! = upto marked theorem       }- corresponds to the three resolutions we can be in
      i^ = upto multi-fanout nodes   }
      o = fanout
      RETURN = select i? where ? is the current mode
      BACKSPACE = go back to previous selection

  - add to current selection: 'a'
      '+', or S-RETURN = add

  - TAB = switch to proof mode (take selection from current premise mode)

  - C-L redraw (position cursor at bottom of screen, if enough lines)

  - Need a way to step into the premises and then jump to that premise

  - Perhaps have a way to hide common premises to make lists shorter? In particular
    definitions (~= premise-free theorems) may be on a different footing (show in different color; perhaps calculate "level" for thm).

  - Special theorems: New_Ax, New_Def, New_TDef, TDef_Ex1, TDef_Ex2

  - fanout count: may change if fully deduping proof.

proof mode

  ^  type
  %  term
  #  theorem

  %65
  #123,456 = BETA(%12,001)
  %143 = Cnst(`=`, ^14)



*/


/*
Top line

Theorems (12,432)    ProofSize  |  Outputs  |   Inputs (premises):

Premise info:
  - proof-size (upto premises of theorem)
  - #fanouts
  - some hash of definition support?
  - premise list: abbreviated (no _DEF or _THM), without definitions, without theorems

Concepts:
  - cursor theorem (just one; clicked on or moved cursor to) Can be nulled by ESC or closing info-pane
      RETURN = if in right-pane, jump to theorem under cursor (if not in selection, do nothing)

  - granularity  (which 'Thm's to include)
  - view selection (see "make selection" above; also add regexp for theorem names...)
       thm | in(thm, steps) | out(thm, steps) |       -- steps=1 if omitted, *=infinity
       and(thm1, thm2) | or(thm1, thm2) | not(thm1) | xor(thm1, thm2) |
       name(glob)

         'i' = ... in(thm, 1)
         'o' = ... out(thm, 1)
         ',' = decrease param of last
         '.' = increase param of last
         '*' = infinity on last
         '|' = OR two last (also implict once we commit selection)
         '&' = AND two last
         '^' = XOR two last
         '~' = INV last
         BACKSPACE = undo last change

       Print this expression at bottom; if info-pane, with underscores: "Selection: in(LEFT_ADD_DISTRIB) & out(BIT0,*)"

         r = toggle "restrict to current selection"
         ESC = remove marked line (will hide bottom pane)
         F1/F3 = PREMISE mode
         F2/F4 = DETAILED mode

         TAB - switch between left and right pane (can also use cursor keys to move around)
         s = search for theorem (

  - order (sort by id, proof-size, #fanouts)
  - theorem selection ("consideration")

In lower-pane:
  - Full name
  - "hyps" and "concl" for theorem.
  - Full premise list and definitional support
  - Fanout list (names of theorems)
  - Total fanin size?

Switch premise highlighting to show mark; instead show premise high-light by a '>' to the left of the theorem (need to leave room for it, maybe a
space on either size; on the right we can use the '~' to mark that not all premises fit)


Depth/weighted depth
*/


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

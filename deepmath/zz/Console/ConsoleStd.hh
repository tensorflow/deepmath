#ifndef ZZ__Console__ConsoleStd_hh
#define ZZ__Console__ConsoleStd_hh

#include "Console.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Colors:


uchar hilite(uchar color);
uchar darken(uchar color);

inline AChar hiliteBg(AChar c)   { c.bg = hilite(c.bg); return c; }
inline Attr  darkenFg(Attr attr) { attr.fg = darken(attr.fg); return attr; }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Printing:


// Note on coordinates: '~y == con_rows() - y', '~x == con_cols() - x'

void putAt(uint y, uint x, AChar c);
AChar getAt(uint y, uint x);
void printAt(uint y, uint x, String const& text, Attr attr);
void printAt(uint y, uint x, Array<AChar const> text);
inline void printAt(uint y, uint x, Vec<AChar> const& text) { printAt(y, x, text.slice()); }


void fill(uint y0, uint x0, uint y1, uint x1, AChar c);
inline void fillScreen(AChar c = AChar()) { fill(0, 0, con_rows(), con_cols(), c); }
inline void fillRow(uint y, uint x0, uint x1, AChar c) { fill(y, x0, y+1, x1, c); }
inline void fillCol(uint x, uint y0, uint y1, AChar c) { fill(y0, x, y1, x+1, c); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Attributed string formatting:


template<typename... Args>
void attrWrite(Vec<AChar>& out, Vec<Attr> const& attrs, cchar* fmt, Args const&... args)
{
    String text;
    formattedWrite_rec(text, fmt, args...);
    text.push(0);   // -- to prevent 'parseUInt()' from running past the end

    Vec<Attr> as;       // -- attribute stack
    as.push(attrs[0]);

    cchar* p = &text[0];
    while (*p){
        if (*p != '\a'){
            if (*p != 1){
                out.push(AChar(*p, as.last()));
                p++;
            }else{
                p++;
                out.push(AChar{(uchar)p[0], (uchar)p[1], (uchar)p[2], (uchar)p[3]});
                p += 4;
            }
        }else{
            p++;
            if (*p == '}'){
                if (as.size() <= 1)
                    shoutLn("INTERNAL ERROR! In 'attrWrite()', closing '}' without opening '{'."), assert(false);
                as.pop();
                p++;
            }else if (isDigit(*p)){
                uint n = parseUInt(p);
                if (n >= attrs.size())
                    shoutLn("INTERNAL ERROR! In 'attrWrite()', attribute number too big: %_ (list size: %_)", n, attrs.size()), assert(false);
                if (*p != '{')
                    shoutLn("INTERNAL ERROR! In 'attrWrite()', missing '{' after '\\a', instead: %_", *p), assert(false);
                as.push(attrs[n]);
                p++;
            }else
                shoutLn("INTERNAL ERROR! In 'attrWrite()', expecting '}' or number after '\\a', not: %_", *p), assert(false);
        }
    }
}


template<typename... Args>
Vec<AChar> attrFmt(Vec<Attr> const& attrs, cchar* fmt, Args const&... args) {
    Vec<AChar> ret;
    attrWrite(ret, attrs, fmt, args...);
    return ret;
}


// Embed attributed strings into normal output (reversed by 'attrWrite()')
template<> fts_macro void write_(Out& out, AChar const& v){
    out += (char)1, v.chr, v.fg, v.bg, v.alt; }

template<> fts_macro void write_(Out& out, Vec<AChar> const& v){
    for (AChar c : v) out += c; }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

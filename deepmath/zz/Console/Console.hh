#ifndef ZZ__Console__Console_hh
#define ZZ__Console__Console_hh

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Setup:


void con_init();
void con_close();

void con_setPollSpeed(uint tenth_of_seconds);   // -- default is '1'; affects 'con_getEvent()'.
void con_showCursor(uint row, uint col);
void con_hideCursor();


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Events:


enum ConEventType {
    ev_NULL,
    ev_KEY,
    ev_MOUSE,
    ev_RESIZE,
};


struct ConEvent {
    ConEventType type;
    uint64  key;      // -- for CLICK events: u, d, l, m, r
    uint    row, col; // -- for MOUSE and RESIZE events
};


constexpr uint64 chr_CSI  = 1ull << 63;
constexpr uint64 chr_META = 1ull << 62;
constexpr uint64 chr_MASK = ~(chr_CSI | chr_META);


ConEvent con_getEvent(double timeout_secs = DBL_MAX, bool interpreted = true);
    // -- Will call 'con_redraw()' first. A timeout of 'DBL_MAX' is means "blocking" (will always
    // return a non-NULL event). NOTE! setting 'interpreted' to FALSE gives the raw underlying
    // stream of characters, including mouse codes. You probably don't want to do that.


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Attributed Characters:


inline uchar color_std(uint index) {
    assert(index < 16);
    return index; }

inline uchar color_rgb(uint r, uint g, uint b) {
    assert(r < 6); assert(g < 6); assert(b < 6);
    return 16 + 36*r + 6*g + b; }

inline uchar color_gray(uint v) {
    assert(v < 24);
    return 232 + v; }

constexpr char color_wtext = 7;     // -- white text from index part of palette
constexpr char color_white = 255;
constexpr char color_black = 232;


//=================================================================================================
// -- Attributes only:


enum AltStyle : uchar {sty_BOLD=1, sty_ITAL=2, sty_UNDER=4};     // -- or together to form 'alt'


struct BgAttr {
    uchar bg;
    BgAttr(uchar bg = color_black) : bg(bg) {}
    bool operator==(BgAttr const& other) const { return bg == other.bg; }
};
inline BgAttr bgStd (uint index)             { return BgAttr(color_std(index)); }
inline BgAttr bgRgb (uint r, uint g, uint b) { return BgAttr(color_rgb(r, g, b)); }
inline BgAttr bgGray(uint v)                 { return BgAttr(color_gray(v)); }


struct FgAttr {
    uchar fg;
    uchar alt;
    FgAttr(uchar fg = color_wtext, uchar alt = 0) : fg(fg), alt(alt) {}
    bool operator==(FgAttr const& other) const { return fg == other.fg && alt == other.alt; }
};
inline FgAttr fgStd (uint index, uchar alt = 0)             { return FgAttr(color_std(index), alt); }
inline FgAttr fgRgb (uint r, uint g, uint b, uchar alt = 0) { return FgAttr(color_rgb(r, g, b), alt); }
inline FgAttr fgGray(uint v, uchar alt = 0)                 { return FgAttr(color_gray(v), alt); }


struct Attr {
    uchar fg;
    uchar bg;
    uchar alt;
    Attr(FgAttr fg_attr = FgAttr(), BgAttr bg_attr = BgAttr()) : fg(fg_attr.fg), bg(bg_attr.bg), alt(fg_attr.alt) {}
    Attr(BgAttr bg_attr           , FgAttr fg_attr = FgAttr()) : fg(fg_attr.fg), bg(bg_attr.bg), alt(fg_attr.alt) {}
    bool operator==(Attr const& other) const { return bg == other.bg && fg == other.fg && alt == other.alt; }
};

inline Attr operator*(FgAttr f, BgAttr b) { return Attr(f, b); }    // -- sub-attributes are combined with '*'
inline Attr operator*(BgAttr b, FgAttr f) { return Attr(f, b); }


//=================================================================================================
// -- Full character:


struct AChar {
    uchar chr;
    uchar fg;
    uchar bg;
    uchar alt;          // -- bit-or of 'AltStyle's

    AChar(uchar chr, uchar fg, uchar bg, uchar alt) : chr(chr), fg(fg), bg(bg), alt(alt) {}     // -- low-level

    AChar(uchar chr = ' ', Attr attr = Attr()) : chr(chr), fg(attr.fg), bg(attr.bg), alt(attr.alt) {}
    AChar(Attr attr, uchar chr = ' ')          : chr(chr), fg(attr.fg), bg(attr.bg), alt(attr.alt) {}
    AChar(uchar chr, FgAttr fg_attr)           : chr(chr), fg(fg_attr.fg), bg(color_black), alt(fg_attr.alt) {}
    AChar(FgAttr fg_attr, uchar chr = ' ')     : chr(chr), fg(fg_attr.fg), bg(color_black), alt(fg_attr.alt) {}
    AChar(BgAttr bg_attr)                      : chr(' '), fg(color_wtext), bg(bg_attr.bg), alt(0) {}
    bool operator==(AChar const& other) const { return chr == other.chr && bg == other.bg && fg == other.fg && alt == other.alt; }
};

inline AChar operator+(uchar chr, Attr attr) { return AChar(chr, attr); }    // -- char and attribute are combined with '+'
inline AChar operator+(Attr attr, uchar chr) { return AChar(chr, attr); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// The canvas:


extern Vec<Vec<AChar>> con;

inline uint con_rows() { return con.size(); }
inline uint con_cols() { return con[0].size(); }

void con_redraw();  // -- redraw characters that have changed in 'con' (will also hide cursor)


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

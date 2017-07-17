//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : ConsoleStd.cc
//| Author(s)   : Niklas Een
//| Module      : Console
//| Description :
//|
//| (C) Copyright 2010-2016, Niklas Een
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//|
//|________________________________________________________________________________________________

#include ZZ_Prelude_hh
#include "ConsoleStd.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Colors:


uchar hilite(uchar color) {
    if (color < 16)
        return (color < 8) ? color + 8 : color;
    else if (color >= 232){
        uint v = color - 232;
        return color_gray(min_(23u, v + 4u));
    }else{
        color -= 16;
        uint r = color / 36; color -= r * 36;
        uint g = color /  6; color -= g *  6;
        uint b = color;
        return color_rgb(min_(5u, r+1), min_(5u, g+1), min_(5u, b+1));
    }
}


uchar darken(uchar color) {
    if (color < 16)
        return (color < 8) ? color : color - 8;
    else if (color >= 232)
        return 232 + (color - 232) / 2;
    else{
        color -= 16;
        uint r = color / 36; color -= r * 36;
        uint g = color /  6; color -= g *  6;
        uint b = color;
        return color_rgb(r/2, g/2, b/2);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Printing:


inline void xlatX(uint& x) { if ((int)x < 0) x = con_cols() - int(~x); }
inline void xlatY(uint& y) { if ((int)y < 0) y = con_rows() - int(~y); }


void putAt(uint y, uint x, AChar c)
{
    xlatY(y); xlatX(x);
    if (y < con_rows() && x < con_cols())
        con[y][x] = c;
}


AChar getAt(uint y, uint x)
{
    xlatY(y); xlatX(x);
    return (y < con_rows() && x < con_cols()) ? con[y][x] : AChar();
}


void printAt(uint y, uint x, String const& text, Attr attr)
{
    xlatY(y); xlatX(x);

    if (y >= con_rows()) return;
    for (uint i = 0; i < text.size(); i++){
        if (x + i >= con_cols()) return;
        con[y][x+i] = text[i] + attr;
    }
}


void printAt(uint y, uint x, Array<AChar const> text)
{
    xlatY(y); xlatX(x);

    if (y >= con_rows()) return;
    for (uint i = 0; i < text.size(); i++){
        if (x + i >= con_cols()) return;
        con[y][x+i] = text[i];
    }
}


void fill(uint y0, uint x0, uint y1, uint x1, AChar c)
{
    xlatY(y0); xlatY(y1); xlatX(x0); xlatX(x1);

    if (y0 > con_rows() || x0 > con_cols()) return;
    newMin(y1, con_rows());
    newMin(x1, con_cols());
    for (uint y = y0; y < y1; y++)
    for (uint x = x0; x < x1; x++)
        con[y][x] = c;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

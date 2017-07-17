//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Atom.cc
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : Global string -> uint (aka "atom") map.
//|
//| (C) Copyright 2010-2015, Niklas Een
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//|
//|________________________________________________________________________________________________

#include ZZ_Prelude_hh
#include "Atom.hh"
#include "Map.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


static StackAlloc<char> atom_data;
Map<Str, uint> text2atom;
Vec<Str> atom2text;

ZZ_Initializer(Atom, -10) {
    text2atom.set(Str(), 0);
    atom2text.push();
}


uint Atom_getOrCreate(Str text) {
    //**/WriteLn "-- lookup(%_)", text;
    uint ret;
    if (!text2atom.peek(text, ret)){
        ret = atom2text.size();

        uind sz = text.size();
        char* base = atom_data.alloc(sz + 1);
        memcpy(base, text.base(), sz);
        base[sz] = '\0';        // -- store string as zero-terminated string for old-style string usage
        Str str(base, sz);

        text2atom.set(str, ret);
        atom2text.push(str);
    }
    return ret;
}


Atom::Atom(cchar* text)          { id = Atom_getOrCreate(slize(text)); }
Atom::Atom(cchar* t0, cchar* t1) { id = Atom_getOrCreate(Str(t0, t1)); }
Atom::Atom(Str text)             { id = Atom_getOrCreate(text); }
Atom::Atom(Array<char> text)     { id = Atom_getOrCreate((Str)text); }
Atom::Atom(const String& text)   { id = Atom_getOrCreate(text.slice()); }

Str Atom::str() const { return atom2text[id]; }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

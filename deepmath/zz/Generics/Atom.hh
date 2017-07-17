#ifndef ZZ__Generics__Atom_hh
#define ZZ__Generics__Atom_hh
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


class Atom {
    uint id;

public:
    Atom(uint id) : id(id) {}

    Atom(cchar* text);
    Atom(cchar* begin, cchar* end);
    Atom(Str text);
    Atom(Array<char> text);
    Atom(const String& text);

  #if (__cplusplus >= 201103L)
    constexpr Atom() : id(0) {}
  #else
    Atom() : id(0) {}
  #endif
    Atom(const Atom& other) { id = other.id; }

    Atom& operator=(Atom other) { id = other.id; return *this; }
    bool operator==(Atom other) const { return id == other.id; }
    bool operator!=(Atom other) const { return id != other.id; }
    bool operator< (Atom other) const { return id <  other.id; }

    Str str() const;
    operator Str() const { return str(); }
    cchar* c_str() const { return str().base(); }

    uint operator+() const { return id; }
    Null_Method(Atom) { return id == 0; }
};

const Atom Atom_NULL;

template <> struct Hash_default<Atom> {
    uint64 hash (const Atom& key) const { return defaultHash(Str(key)); }
    bool   equal(const Atom& key1, const Atom& key2) const { return key1 == key2; }
};

template<> struct MkIndex_default<Atom> {
    typedef Atom Key;
    uind operator()(Key const& x) const { return +x; }
    Key inv(uind i) const { return Key(i); }
};


template<> fts_macro void write_(Out& out, const Atom& v) { write_(out, (Str)v); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

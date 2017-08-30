//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Arr.ihh
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : An 'Arr<T>' is a reference counted array of 'T's.
//|
//| (C) Copyright 2010-2016, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| Static sized, reference counted array which can be passed by value (like a handle).
//|________________________________________________________________________________________________

#ifndef ZZ__Generics__Arr_hh
#define ZZ__Generics__Arr_hh
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


template<class T>
struct Arr_data {
    uind refC;
    uind sz;
    T    data[1];
};

template<class T>
class Arr  {
    Arr_data<T>* ptr;

    void alloc(uind size) {
        ptr = (Arr_data<T>*)xmalloc<char>(sizeof(Arr_data<T>) - sizeof(T) + sizeof(T) * size);
        ptr->refC = 1;
        ptr->sz = size; }

    void init(uint i)             { new (&ptr->data[i]) T(); }
    void init(uint i, T const& t) { new (&ptr->data[i]) T(t); }

    static void ref_   (Arr_data<T>* p) { if (p) p->refC++; }
    static void unref_ (Arr_data<T>* p) { if (p){ assert(p->refC != 0); p->refC--; if (p->refC == 0){ dispose(p); } } }
    static void dispose(Arr_data<T>* p) { for (uind i = 0; i < p->sz; i++) p->data[i].~T(); xfree(p); }

public:
    typedef T Elem;

    Arr() : ptr(nullptr) {}
    explicit operator bool() const { return ptr; }

    Arr(Tag_reserve, uind size)       { alloc(size); for (uind i = 0; i < size; i++) init(i); }
    Arr(initializer_list<T> const& ts){ alloc(ts.size()); uind i = 0; for (T const& t: ts) init(i, t), i++; }
    Arr(Vec<T> const& ts)             { alloc(ts.size()); uind i = 0; for (T const& t: ts) init(i, t), i++; }

    // Reference counting:
   ~Arr()                            { unref_(ptr); }
    Arr(Arr const& other)            { ptr = other.ptr; ref_(other.ptr); }
    Arr& operator=(const Arr& other) { if (this == &other) return *this; unref_(ptr); ptr = other.ptr; ref_(ptr); return *this; }
    Arr(Arr&& other)                 { ptr = other.ptr; other.ptr = NULL; }
    Arr& operator=(Arr&& other)      { if (this == &other) return *this; unref_(ptr); ptr = other.ptr; other.ptr = NULL; return *this; }

    // Array operations:
    uint     size      ()       const { return ptr->sz; }
    T*       base      ()             { return ptr->data; }
    T const* base      ()       const { return ptr->data; }
    T&       operator[](uint i)       { return ptr->data[i]; }
    T const& operator[](uint i) const { return ptr->data[i]; }
    Std_Array_Funcs(T);

    Array<T>       operator+()       { return ptr ? this->slice() : Array<T>(nullptr, (uind)0); }
    Array<T const> operator+() const { return ptr ? this->slice() : Array<T const>(nullptr, (uind)0); }
        // -- unary '+' allows you to treat nil arrays as empty arrays: 'for (auto&& elem : +my_arr){ ... }'
    uint psize() const { return ptr ? ptr->sz : 0; }    // -- like 'size()' but returns 0 for null arrays

    // Shallow comparison:
    bool operator==(Arr<T> const& other) const { return ptr == other.ptr; }
};


template<class T> fts_macro void write_(Out& out, const Arr<T>& v) {
    if (!v) out += "<null-arr>";
    else out += slice(v[0], v[v.size()]); }

template<class T>
struct Hash_default<Arr<T> > {
    uint64 hash (Arr<T> const& key)                      const { return key ? vecHash(key.slice()) : 0ull; }
    bool   equal(Arr<T> const& key1, Arr<T> const& key2) const { return (key1 == key2) || (key1 && key2 && vecEqual(key1, key2)); }
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

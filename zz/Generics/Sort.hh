//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Sort.h
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : Combinator library for sorting.
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| 
//| Generic sort functions and combinators. Work on any class ("sorting object") with the following
//| methods:
//| 
//|     struct Sob {
//|         // Mandatory:
//|         //
//|         bool lessThan(uind i, uind j) const; // Is i:th element strictly less than the 'j:th'?
//|         void swap(uind i, uind j);           // Swap i:th and j:th element.
//|         uind size() const;                   // Legal indices are from '0' to 'size() - 1'.
//| 
//|         // For 'sortUnique()':
//|         //
//|         void shrinkTo(uind new_size);        // Shrink the final container to this size
//|         void dispose(uind i);                // If defined, called on the discarded elements.
//|     };
//| 
//| If 'dispose()' is not defined, it is up to the container's 'shrinkTo()' method to call
//| the destructor or do the approprieate thing for the last 'size() - new_size' elements lost
//| in the shrink operation.
//| 
//| Sorting objects for vector-like types 'V<T>' (including 'Vec<T>' and 'Array<T>') is predefined
//| through the function 'sob()':
//| 
//|     sob ( V<T>& t,  [optional:] LessThan lt ,  [optional:] Disposer disposer )
//|     
//| A less-than object should implement:    
//| 
//|     struct MyLessThan {
//|         typedef T Key;
//|         bool operator()(const Key& x, const Key& y) const { return "my x < y expression"; } };
//|     };    
//| 
//| Objects can then be combined using combinators:
//| 
//|     ordReverse  (s)
//|     ordByFirst  (s0, s1)
//|     ordLexico   (s0, s1)
//|     ordStabilize(s)
//|     
//| Finally, two sorting functions are defined:
//| 
//|     sobSort(s)
//|     sobSortUnique(s)
//|     
//| Examples:
//| 
//|     sobSort(ordByFirst(sob(my_vec), sob(my_other_vec)));
//|     sobSortUnique(sob(my_int_vec, LessThan_default<int>, int_disposer));
//|     
//| For convenience, the following short-cuts (functions) are defined for vector-like objects 
//| v and w:
//| 
//|     sort        (v)
//|     sort_reverse(v)
//|     sort_byFirst(v, w)
//|     sort_lexico (v, w)
//| 
//|     sortUnique        (v)
//|     sortUnique_reverse(v)
//|     sortUnique_byFirst(v, w)
//|     sortUnique_lexico (v, w)
//|________________________________________________________________________________________________

#ifndef ZZ__Generics__Sort_h
#define ZZ__Generics__Sort_h
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Default sorting objects:


template<class T, class LT, class D> struct Sob_default;

template<class T, class LT, class D>
macro Sob_default<T,LT,D> sob(T& t, LT lt, D disposer) {
    return Sob_default<T,LT,D>(t, lt, disposer); }

// Types 'V<T>' with vector interface:
template<template<class> class V, class T, class LT, class D>
struct Sob_default<V<T>, LT, D> {
    V<T>*   v;
    LT      cmp;
    D       disposer;
    Sob_default(V<T>& v_, LT lt, D d) : v(&v_), cmp(lt), disposer(d) {}

    bool    lessThan(uind i, uind j) const { return cmp((*v)[i], (*v)[j]); }
    void    swap    (uind i, uind j)       { swp((*v)[i], (*v)[j]); }
    uind    size    ()               const { return v->size(); }
    void    shrinkTo(uind sz)              { v->shrinkTo(sz); }
    void    dispose (uind i)               { disposer((*v)[i]); }
};


template<template<class> class V, class T>
macro      Sob_default< V<T>, LessThan_default<T>, void(*)(T) > sob(V<T>& v) {
    return Sob_default< V<T>, LessThan_default<T>, void(*)(T) >(v, LessThan_default<T>(), nop<T>); }


template<template<class> class V, class T, class LT>
macro      Sob_default<V<T>, LT, void(*)(T) > sob(V<T>& v, LT lt) {
    return Sob_default<V<T>, LT, void(*)(T) >(v, lt, nop<T>); }


template<template<class> class V, class T, class LT, class D>
macro      Sob_default<V<T>, LT, D > sob(V<T>& v, LT lt, D disposer) {
    return Sob_default<V<T>, LT, D >(v, lt, disposer); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Combinators:


template<class Sob> struct Sob_reverse {
    Sob     s;
    Sob_reverse(Sob s_) : s(s_) {}
    bool    lessThan(uind i, uind j) const { return s.lessThan(j, i); }
    void    swap    (uind i, uind j)       { s.swap(i, j); }
    uind    size()                   const { return s.size(); }
    void    shrinkTo(uind sz)              { s.shrinkTo(sz); }
    void    dispose (uind i)               { s.dispose(i); }
};

template<class S>
macro Sob_reverse<S> ordReverse(S s) { return Sob_reverse<S>(s); }


template<class Sob0, class Sob1> struct Sob_first {
    Sob0    s0;
    Sob1    s1;
    Sob_first(Sob0 s0_, Sob1 s1_) : s0(s0_), s1(s1_) {}
    bool    lessThan(uind i, uind j) const { return s0.lessThan(i, j); }
    void    swap    (uind i, uind j)       { s0.swap(i, j); s1.swap(i, j); }
    uind    size()                   const { assert(s0.size() == s1.size()); return s0.size(); }
    void    shrinkTo(uind sz)              { s0.shrinkTo(sz); s1.shrinkTo(sz); }
    void    dispose (uind i)               { s0.dispose(i); s1.dispose(i); }
};

template<class S0, class S1>
macro Sob_first<S0,S1> ordByFirst(S0 s0, S1 s1) { return Sob_first<S0,S1>(s0, s1); }


template<class Sob0, class Sob1> struct Sob_lexico {
    Sob0    s0;
    Sob1    s1;
    Sob_lexico(Sob0 s0_, Sob1 s1_) : s0(s0_), s1(s1_) {}
    bool    lessThan(uind i, uind j) const { return s0.lessThan(i, j) || (!s0.lessThan(j, i) && s1.lessThan(i, j)); }
    void    swap    (uind i, uind j)       { s0.swap(i, j); s1.swap(i, j); }
    uind    size()                   const { assert(s0.size() == s1.size()); return s0.size(); }
    void    shrinkTo(uind sz)              { s0.shrinkTo(sz); s1.shrinkTo(sz); }
    void    dispose (uind i)               { s0.dispose(i); s1.dispose(i); }
};

template<class S0, class S1>
macro Sob_lexico<S0,S1> ordLexico(S0 s0, S1 s1) { return Sob_lexico<S0,S1>(s0, s1); }


template<class Sob>
struct Sob_stabilize {
    Sob     s;
    uint*   order;

    Sob_stabilize(Sob s_) : s(s_) {
        assert(s.size() < UINT_MAX);   // -- only support stable sort for up to 4 billion elements
        order = xmalloc<uint>(s.size() + 1);
        order[0] = 1;
        order++;
        for (uint i = 0; i < s.size(); i++) order[i] = i;
    }

   ~Sob_stabilize() {
        order--;
        order[0]--;
        if (order[0] == 0) xfree(order); }

    Sob_stabilize(const Sob_stabilize& other) : s(other.s) {
        order = other.order;
        order[-1]++; }

    bool    lessThan(uind i, uind j) const { return s.lessThan(i, j) || (!s.lessThan(j, i) && order[i] < order[j]); }
    void    swap    (uind i, uind j)       { s.swap(i, j); swp(order[i], order[j]); }
    uind    size()                   const { return s.size(); }
    void    shrinkTo(uind sz)              { s.shrinkTo(sz); }
    void    dispose (uind i)               { s.dispose(i); }

private:
    Sob_stabilize& operator=(const Sob_stabilize& other);   // -- should not be needed
};

template<class S>
macro Sob_stabilize<S> ordStabilize(S s) { return Sob_stabilize<S>(s); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Sorting functions:


template<class Sob>
void insertionSort(Sob s, uind fst, uind end)
{
    for (uind i = fst+1; i < end; i++){
        for (uind j = i; j > fst && s.lessThan(j, j-1); j--)
            s.swap(j-1, j);
    }
}


template<class Sob>
macro void insertionSort(Sob s)
{
    insertionSort(s, 0, s.size());
}


template<class Sob>
void quickSort(Sob s, uind fst, uind end, uint64& seed)
{
    if (end - fst > 100){
        uind p = irand(seed, end - fst) + fst;
        uind i = fst - 1;
        uind j = end;

        for(;;){
            do i++; while(s.lessThan(i, p));
            do j--; while(s.lessThan(p, j));

            if (i >= j) break;

            s.swap(i, j);
            if      (p == i) p = j;
            else if (p == j) p = i;
        }

        quickSort(s, fst, i, seed);
        quickSort(s, i, end, seed);
    }
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


template<class Sob>
macro void sobSort(Sob s, ind fst, ind end)
{
    uint64 seed = DEFAULT_SEED;
    quickSort(s, fst, end, seed);
    insertionSort(s, fst, end);
}


template<class Sob>
macro void sobSort(Sob s)
{
    sobSort(s, 0, s.size());
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


template<class Sob>
void sobSortUnique(Sob s, uind sz, uind& new_sz)
{
    if (sz == 0){ new_sz = 0; return; }

    sobSort(s, 0, sz);

    uind i    = 1;
    uind last = 0;
    for (uind j = 1; j < sz; j++){
        if (s.lessThan(last, j)){
            s.swap(j, i);
            last = i;
            i++;
        }
    }

    new_sz = i;
}

template<class Sob>
macro void sobSortUnique(Sob s, uind sz)
{
    uind new_sz = 0;
    sobSortUnique(s, sz, new_sz);
    if (new_sz < sz){
        for (uind i = new_sz; i < sz; i++) s.dispose(i);
        s.shrinkTo(new_sz);
    }
}


template<class Sob>
macro void sobSortUnique(Sob s)
{
    sobSortUnique(s, s.size());
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Sorting function wrappers: (just for convenience...)


template<template<class> class V, class T> void sort        (V<T>& v) { sobSort(sob(v)); }
template<template<class> class V, class T> void sort_reverse(V<T>& v) { sobSort(ordReverse(sob(v))); }
template<template<class> class V, class T, template<class> class W, class S> void sort_byFirst(V<T>& v, W<S>& w) { sobSort(ordByFirst(sob(v), sob(w))); }
template<template<class> class V, class T, template<class> class W, class S> void sort_lexico (V<T>& v, W<S>& w) { sobSort(ordLexico (sob(v), sob(w))); }

template<template<class> class V, class T> void sortUnique        (V<T>& v) { sobSortUnique(sob(v)); }
template<template<class> class V, class T> void sortUnique_reverse(V<T>& v) { sobSortUnique(ordReverse(sob(v))); }
template<template<class> class V, class T, template<class> class W, class S> void sortUnique_byFirst(V<T>& v, W<S>& w) { sobSortUnique(ordByFirst(sob(v), sob(w))); }
template<template<class> class V, class T, template<class> class W, class S> void sortUnique_lexico (V<T>& v, W<S>& w) { sobSortUnique(ordLexico (sob(v), sob(w))); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

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

#ifndef ZZ__HolLight__List_hh
#define ZZ__HolLight__List_hh

#include "Hashing.hh"
#include "zz/Generics/Sort.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
/*

List<T> -- A list of type 'T' derived from 'IdBase'
SSet<T> -- A small set of type 'T', represented as a sorted list. Operations on 'SSet's are cached.

Examples 'List':
  *it   :=  it.head()
  ++it  :=  it = it.tail()
  it[2] :=  it.tail().tail().head()

  for (List<Var> it = list_of_vars; it; ++it)
      <code doing something with '*it' of type 'Var'>

Exampels 'SSet':
  set = aset & (bset << elem);      // '&' is intersection, '|' is union
  set = aset << elem;               // << is add an element (we want an asymmetric operator)
  set <<= elem;
  set |= other_set;
  small.subsetOf(big);

*/
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// List:


struct BaseList {
    id_t head;
    id_t tail;
    BaseList(id_t head = id_NULL, id_t tail = id_NULL) : head(head), tail(tail) {}
};
template<> struct Hash_default<BaseList> : Hash_mem<BaseList> {};


constexpr struct NilType { constexpr NilType(){} } nil_;


// 'T' must be inherited from 'IdBase'.
template<class T>
struct List : Composite<BaseList> {
    using P = Composite<BaseList>;     // -- parent type

    T    head() const { return T(me().head); }
    List tail() const { return List(me().tail); }

    T    operator*() const { return head(); }
    void operator++() { id = +tail(); }
    List operator++(int) { List ret = *this; id = +tail(); return ret; }
    T    operator[](size_t i) const { List it = *this; while (i > 0) ++it, --i; return *it; }     // -- NOTE! linear in 'i'

    T const* operator->() const = delete;   // -- prevent accidental use of these (different meaning in 'Composite')
    T*       operator->()       = delete;

    explicit operator bool() const { return id != id_NULL; }
    bool empty() const { return id == id_NULL; }

    size_t size() const {
        size_t c = 0;
        for (List<T> it(*this); it; ++it) c++;
        return c; }

    Vec<T> toVec() const {
        Vec<T> ret;
        for (List<T> it(*this); it; ++it) ret.push(*it);
        return ret; }

    List()        {}   // }- parent construct will set 'id = 0'
    List(NilType) {}   // }
    List(id_t id) : P(id) {}
    List(T head, List tail) : P({+head, +tail}) {}
    List(T head, NilType) : P({+head, id_NULL}) {}
};


template<class T> inline List<T> cons(T head, List<T> tail) { return List<T>(head, tail); }
template<class T> inline List<T> cons(T head, NilType)      { return List<T>(head, nil_); }


// USAGE: mkList(Array_or_Vec, [optional_map_fun])
template<class T>
inline List<T> mkList(Array<T const> xs) {
    List<T> ret;
    for (uind i = xs.size(); i > 0;) i--,
        ret = cons(xs[i], ret);
    return ret;
}

template<class T, class F>
inline auto mkList(Array<T const> xs, F f) -> List<decltype(f(T()))> {
    List< decltype(f(T())) > ret;
    for (uind i = xs.size(); i > 0;) i--,
        ret = cons(f(xs[i]), ret);
    return ret;
}

template<class T>
inline List<T> mkList(Vec<T> const& xs) {
    return mkList(xs.slice()); }

template<class T, class F>
inline auto mkList(Vec<T> const& xs, F f) -> List<decltype(f(T()))> {
    return mkList(xs.slice(), f); }


// USAGE: mkList({elem1, elem2, ..., elemN})
template<class T>
inline List<T> mkList(const initializer_list<T>& elems){
    Vec<T> xs(reserve_, elems.size());
    for (const auto& x: elems)
        xs.push(x);
    return mkList(xs); }


// Pretty-printing: ("%b" means include braces (default), "%n" no braces)
template<class T> fts_macro void write_(Out& out, List<T> const& list, Str flags) {
    assert(flags.size() == 1);
    if (flags[0] == 'b') out += '{';
    bool first = true;
    for (List<T> it = list; it; ++it){
        if (first) first = false;
        else       out += ", ";
        out += *it;
    }
    if (flags[0] == 'b') out += '}';
}

template<class T> fts_macro void write_(Out& out, List<T> const& list) {
    write_(out, list, slize("b")); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// SSet:


template <class T>
struct SSet : List<T> {
    using P = List<T>;     // -- parent type
    SSet(id_t id) : P(id) {}

    SSet() {}
    SSet(T sing) : P(sing, nil_) {}

    SSet operator<<(T x)    const { return SSet(setAdd(Lst(*this), x)); }
    SSet operator|(SSet ys) const { return SSet(setUnion(Lst(*this), Lst(ys))); }
    SSet operator&(SSet ys) const { return SSet(setIntersection(Lst(*this), Lst(ys))); }
    SSet exclude(T x)       const { return SSet(setExclude(Lst(*this), x)); }   // -- remove 'x' if exists (identity function otherwise)

    SSet& operator<<=(T x)     { this->id = (*this << x).id; return *this; }
    SSet& operator|= (SSet ys) { this->id = (*this | ys).id; return *this; }
    SSet& operator&= (SSet ys) { this->id = (*this & ys).id; return *this; }

    bool has     (T x)      const { return setHas(Lst(*this), x); }
    bool subsetOf(SSet big) const { return setSubsetOf(Lst(*this), big); }

    template<class FUN> auto map(FUN f) const -> SSet<decltype(f(T()))> {   // -- awkward because of C++11
        typedef decltype(f(T())) Ret;
        Vec<Ret> tmp;
        List<T> xs(*this);
        while (xs) tmp.push(f(*xs++));
        sortUnique(tmp);
        return SSet<Ret>(+mkList(tmp));
    }

private:
    using Lst = List<T>;
    SSet(Lst list) { this->id = +list; }

    static Lst setAdd(Lst xs, T x) {
        if (xs.empty())      return Lst(x, nil_);
        if (x == xs.head())  return xs;
        if (+x < +xs.head()) return cons(x, xs);
        // <<== memo goes here (also, if setHas is memoized, use that one first)
        return cons(xs.head(), setAdd(xs.tail(), x));
    }

    static Lst setExclude(Lst xs, T x) {
        if (xs.empty())      return xs;
        if (x == xs.head())  return xs.tail();
        if (+x < +xs.head()) return xs;
        // <<== memo goes here (also, if setHas is memoized, use that one first)
        return cons(xs.head(), setExclude(xs.tail(), x));
    }

    static Lst setUnion(Lst xs, Lst ys) {
        if (xs.empty()) return ys;
        if (ys.empty()) return xs;
        if (xs == ys)   return xs;
        //if (xs.subsetOf(ys))  or vice versa (if memoized)
        if (+xs.head() < +ys.head()) return cons(xs.head(), setUnion(xs.tail(), ys));
        if (+xs.head() > +ys.head()) return cons(ys.head(), setUnion(xs, ys.tail()));
        return cons(xs.head(), setUnion(xs.tail(), ys.tail()));
    }

    static Lst setIntersection(Lst xs, Lst ys) {
        if (xs.empty()) return xs;
        if (ys.empty()) return ys;
        if (xs == ys)   return xs;
        //if (xs.subsetOf(ys))  or vice versa (if memoized)
        if (+xs.head() < +ys.head()) return setIntersection(xs.tail(), ys);
        if (+xs.head() > +ys.head()) return setIntersection(xs, ys.tail());
        return cons(xs.head(), setIntersection(xs.tail(), ys.tail()));
    }

    static bool setHas(Lst xs, T x) {
        if (xs.empty()) return false;
        if (x == xs.head()) return true;
        return setHas(xs.tail(), x);
    }

    static bool setSubsetOf(Lst sml, Lst big) {
        if (sml.empty()) return true;
        if (big.empty()) return false;
        if (+sml.head() < +big.head()) return false;
        if (+sml.head() > +big.head()) return setSubsetOf(sml, big.tail());
        return setSubsetOf(sml.tail(), big.tail());
    }
};


template<class T> fts_macro void write_(Out& out, SSet<T> const& set) {
    out += static_cast<List<T> const&>(set); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

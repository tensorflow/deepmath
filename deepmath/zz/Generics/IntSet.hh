//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : IntSet.hh
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : Sets of integer-like types.
//|
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| struct MyKey2Index {
//|     typedef ... Key;
//|
//|     uind operator()(const Key& x) const { return ...; }
//|         // -- This operator HAS to be defined for a 'MkIndex' template paramter.
//|
//|     Key inv(uind i) const { return ...); }
//|         // -- This "inverse" function MAY be defined.
//| }
//|________________________________________________________________________________________________


#ifndef ZZ__Generics__IntSet_h
#define ZZ__Generics__IntSet_h

#include "IntMap.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// IntSet:


template<class Key_, class Key2Index = MkIndex_default<Key_> >
class IntSet /*: public NonCopyable*/ {
protected:
    Vec<Key_>                   elems;       // -- List of distinct elements (keys) that makes up the set.
    IntMap<Key_,uind,Key2Index> pos;         // -- Position of element in 'elems[]'.

public:
    typedef Key_ Key;

    Key2Index index;    // The indexing object. Overloads '()' operator.

    IntSet(Key2Index index_ = Key2Index()) : pos(UIND_MAX, index_) {}

    void reserve(uind cap) { pos.reserve(cap); }
    uind size() const { return elems.size(); }
    void clear(bool dispose = false);

    void copyTo(IntSet& copy) const { elems.copyTo(copy.elems); pos.copyTo(copy.pos); }
    void moveTo(IntSet& dest)       { elems.moveTo(dest.elems); pos.moveTo(dest.pos); }

    bool          add    (const Key_& key);       // -- Add or replace 'key'. Returns TRUE if element already existed.
    Key_&         addWeak(const Key_& key);       // -- Add an element if does not already exist. Return reference to old or new element.
    Key_*         get    (const Key_& key);       // -- Returns the representative element of 'key', or NULL if none.
    const Key_*   get    (const Key_& key) const; // -- Returns the representative element of 'key', or NULL if none.
    bool          has    (const Key_& key) const; // -- Returns TRUE if element exists.
    bool          exclude(const Key_& key);       // -- Returns TRUE if element existed and was excluded.

    Vec<Key_>&       list()       { return elems; }
    const Vec<Key_>& list() const { return elems; }

  #if (__cplusplus >= 201103L)
    IntSet(IntSet&& other) : elems(other.elems), pos(other.pos), index(other.index) {}
    IntSet& operator=(IntSet&& other) { other.moveTo(*this); return *this; }

    IntSet(const IntSet& other) { other.copyTo(*this); }
    IntSet& operator=(const IntSet& other) { other.copyTo(*this); return *this; }
  #endif
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// IntZet -- a compreZZed 'IntSet' with fewer operations:


template<class Key_, class Key2Index = MkIndex_default<Key_> >
class IntZet /*: public NonCopyable*/ {
protected:
    uind                sz;
    Vec<Key_>           elems;  // List of elements (posibly with duplicates and deleted elements unless 'compact()' is first called).
    Vec<uint32>         bitvec; // A bit-vector representation of the set.

public:
    typedef Key_ Key;

    Key2Index index;            // The indexing object. Overloads '()' operator.

    IntZet(Key2Index index_ = Key2Index()) : sz(0), index(index_) {}

    void reserve(uind cap) { bitvec.growTo((cap + 31) >> 5, 0); }
    uind size   () const { return sz; }
    void clear  (bool dispose = false);

    void copyTo(IntZet& copy) const { copy.sz = sz; elems.copyTo(copy.elems); bitvec.copyTo(copy.bitvec); }
    void moveTo(IntZet& dest)       { dest.sz = sz; elems.moveTo(dest.elems); bitvec.moveTo(dest.bitvec); sz = 0; }

    bool add    (const Key_& key);          // -- Add element unless already in set. Returns TRUE if element already existed.
    bool has    (const Key_& key) const;    // -- Check for existence of element.
    bool exclude(const Key_& key);          // -- Exclude element if exists (returns TRUE if found). Element is still in 'elems' (lazy deletion) until 'compact()' is called, but 'has()' will return FALSE.

    const Key& peekLast();      // -- non-const because removed elements missing from 'bitvec' may be popped from 'elems'
    void       popLast () { exclude(peekLast()); list().pop(); }
    Key        popLastC() { Key ret = peekLast(); popLast(); return ret; }

    Vec<Key_>&       list()       { return elems; }
    const Vec<Key_>& list() const { return elems; }
    void compact();
        // -- The list will contain excluded elements unless 'compact()' is called first.

    // Vector interface:
    bool push(const Key_& key) { return add(key); }
    Key  operator[](uint i) const { assert_debug(elems.size() == sz); return elems[i]; }

  #if (__cplusplus >= 201103L)
    IntZet(IntZet&& other) : sz(other.sz), elems(other.elems), bitvec(other.bitvec), index(other.index) {}
    IntZet& operator=(IntZet&& other) { other.moveTo(*this); return *this; }

    IntZet(const IntZet& other) { other.copyTo(*this); }
    IntZet& operator=(const IntZet& other) { other.copyTo(*this); return *this; }
  #endif
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// IntSeen -- Most compact set representation (just a bit-vector):


template<class Key_, class Key2Index = MkIndex_default<Key_> >
class IntSeen : public NonCopyable {
protected:
    Vec<uint32> bitvec; // A bit-vector representation of the set.

public:
    typedef Key_ Key;

    Key2Index index;            // The indexing object. Overloads '()' operator.

    IntSeen(Key2Index index_ = Key2Index()) : index(index_) {}
    IntSeen(Tag_copy, IntSeen const& s) : bitvec(copy_, s.bitvec) {}

    void reserve(uind cap) { bitvec.growTo((cap + 31) >> 5, 0); }
    void clear  (bool dispose = false) { bitvec.clear(dispose); }

    void copyTo(IntSeen& copy) const { bitvec.copyTo(copy.bitvec); }
    void moveTo(IntSeen& dest)       { bitvec.moveTo(dest.bitvec); }

    bool add    (const Key_& key);          // -- Add element unless already in set. Returns TRUE if element already existed.
    bool has    (const Key_& key) const;    // -- Check for existence of element.
    bool exclude(const Key_& key);          // -- Exclude element if exists (returns TRUE if found). Element is still in 'elems' (lazy deletion) until 'compact()' is called, but 'has()' will return FALSE.

  #if 0 // (__cplusplus >= 201103L)   // -- Google does not allow 'return move(...)'
    Vec<uind> list() const {  // -- returns the set a vector of 'Index's, not 'Key's.
        Vec<uind> is;
        for (uind i = 0; i < bitvec.size() * 32; i++)
            if (bitvec[i>>5] & (1 << (i&31))) is.push(i);
        return move(is); }
  #endif
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// IntSet Implementation:


template<class K, class I>
inline bool IntSet<K,I>::add(const K& key)
{
    uind i = pos[key];
    if (i == UIND_MAX){
        pos(key) = elems.size();
        elems.push(key);
        return false;
    }else{
        elems[i] = key;
        return true;
    }
}


template<class K, class I>
inline K& IntSet<K,I>::addWeak(const K& key)
{
    uind i = pos[key];
    if (i == UIND_MAX){
        i = pos(key) = elems.size();
        elems.push(key);
    }
    return elems[i];
}


template<class K, class I>
inline K* IntSet<K,I>::get(const K& key)
{
    uind i = pos[key];
    return (i == UIND_MAX) ? NULL : &elems[i];
}


template<class K, class I>
inline bool IntSet<K,I>::has(const K& key) const
{
    return pos[key] != UIND_MAX;
}


template<class K, class I>
inline bool IntSet<K,I>::exclude(const K& key)
{
    uind i = pos[key];
    if (i == UIND_MAX)
        return false;
    else{
        pos(key) = UIND_MAX;
        if (i != elems.size() - 1){
            elems[i] = elems.last();
            pos(elems[i]) = i; }
        elems.pop();
        return true;
    }
}


template<class K, class I>
inline void IntSet<K,I>::clear(bool dispose)
{
    if (dispose){
        pos  .clear(true);
        elems.clear(true);
    }else{
        for (uind i = 0; i < elems.size(); i++)
            pos(elems[i]) = UIND_MAX;
        elems.clear(false);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// IntZet Implementation:


template<class K, class I>
inline void IntZet<K,I>::clear(bool dispose)
{
    sz = 0;
    for (uind i = 0; i < elems.size(); i++)
        bitvec[index(elems[i]) >> 5] = 0;
    elems.clear(dispose);
}


template<class K, class I>
inline bool IntZet<K,I>::add(const K& key)
{
    uind i = index(key);
    bitvec.growTo((i>>5) + 1, 0);
    if (bitvec[i>>5] & (1 << (i&31)))
        return true;
    else{
        bitvec[i>>5] |= 1 << (i&31);
        elems.push(key);
        sz++;
        return false;
    }
}


template<class K, class I>
inline bool IntZet<K,I>::has(const K& key) const
{
    uind i = index(key);
    if ((i>>5) >= (uind)bitvec.size()) return false;
    return bitvec[i>>5] & (1 << (i&31));
}


template<class K, class I>
inline bool IntZet<K,I>::exclude(const K& key)
{
    uind i = index(key);
    bitvec.growTo((i>>5) + 1, 0);
    bool ret = bitvec[i>>5] & (1 << (i&31));
    if (ret){
        sz--;
        bitvec[i>>5] &= ~(1 << (i&31));
    }
    return ret;
}


template<class K, class I>
inline const K& IntZet<K,I>::peekLast()
{
    for(;;){
        const K& key = list().last();
        if (has(key))
            return key;
        list().pop();
    }
}


template<class K, class I>
inline void IntZet<K,I>::compact()
{
    if (sz == elems.size()) return;

    uind j = 0;
    for (uind i = 0; i < elems.size(); i++)
        if (has(elems[i]))
            elems[j++] = elems[i];
    elems.shrinkTo(j);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// IntSeen Implementation:


template<class K, class I>
inline bool IntSeen<K,I>::add(const K& key)
{
    uind i = index(key);
    bitvec.growTo((i>>5) + 1, 0);
    if (bitvec[i>>5] & (1 << (i&31)))
        return true;
    else{
        bitvec[i>>5] |= 1 << (i&31);
        return false;
    }
}


template<class K, class I>
inline bool IntSeen<K,I>::has(const K& key) const
{
    uind i = index(key);
    if ((i>>5) >= (uind)bitvec.size()) return false;
    return bitvec[i>>5] & (1 << (i&31));
}


template<class K, class I>
inline bool IntSeen<K,I>::exclude(const K& key)
{
    uind i = index(key);
    if ((i>>5) >= (uind)bitvec.size()) return false;
    bool ret = bitvec[i>>5] & (1 << (i&31));
    bitvec[i>>5] &= ~(1 << (i&31));
    return ret;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

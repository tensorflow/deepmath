//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : IntMap.hh
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : Map from an integer-like type to some arbitrary type 'T'.
//|
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//|
//|________________________________________________________________________________________________


#ifndef ZZ__Generics__IntMap_h
#define ZZ__Generics__IntMap_h
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// IntMap:


// Maps an integer like type 'Key' into an arbitrary type 'Value' by wrapping a 'Vec<Value>'. The
// conversion of a 'Key' to an unsigned integer 'uind' is provided by 'Key2Index' which defaults to
// 'MkIndex_default<Key>'.
//
template<class Key_, class Value_, class Key2Index = MkIndex_default<Key_>, class MapNorm = MapNorm_default<Key2Index,Value_> >
class IntMap /*: public NonCopyable*/ {
public:
    typedef Key_   Key;
    typedef Value_ Value;
    typedef typename MapNorm::RetValue RetValue;
        // -- operator[] is either of type "Value" or "const Value&" (and the normalizer makes this choice)

protected:
    Vec<Value> data;

    Value& update_(True_t , Key k) { return data(index(k), nil); }
    Value& update_(False_t, Key k) { return data(index(k)); }

public:
    Value     nil;      // The null-value (used for padding when the underlying vector is grown). It is sound to change this value as long as the map is empty.
    Key2Index index;    // The indexing object. Overloads '()' operator.
    MapNorm   norm;

    IntMap(            Key2Index index_ = Key2Index(), MapNorm norm_ = MapNorm()) : nil()    , index(index_), norm(norm_) {}
    IntMap(Value nil_, Key2Index index_ = Key2Index(), MapNorm norm_ = MapNorm()) : nil(nil_), index(index_), norm(norm_) {}

    void reserve(uind cap)          { data.growTo(cap, nil); }
    void reserve(uind cap, Value v) { data.growTo(cap, v); }

    RetValue operator[](Key k) const {
        // -- will return the null value for 'k's larger than the underlying vector size.
        uind i = index(k);
        if (i >= data.size()) return norm.get(k, nil);
        else return norm.get(k, data[i]); }

    Value& operator()(Key k) { assert(norm.isNormal(k)); return update_(typename IsCopyable<Value>::Result(), k); }
        // -- will grow underlying vector if necessary

    void mkNull(Key k)        { uind i = index(k); if (i < data.size()) data[i] = nil; }
    bool null  (Key k)  const { return operator[](k) == nil; }
    Key  inv   (uind i) const { return index.inv(i); }

    void clear(bool dispose = false) { data.clear(dispose); }
        // -- only clears the map, does not change 'nil' or 'index' (so you can continue to use
        // the map for the same purpose).

    void copyTo(IntMap& dst) const { data.copyTo(dst.data); dst.nil = nil; dst.index = index; }
    void moveTo(IntMap& dst)       { data.moveTo(dst.data); dst.nil = nil; dst.index = index; }
        // -- 'moveTo()', just as 'clear()', leaves 'nil' and 'index' untouched.

    Vec<Value>&       base()       { return data; }
    const Vec<Value>& base() const { return data; }
    uind              size() const { return base().size(); }
        // -- ONLY use 'size()' if your 'index' functor is the identity function! (otherwise, iterate
        // using the vector returned by 'base()').

  #if (__cplusplus >= 201103L)
    IntMap(IntMap&& other) : data(other.data), nil(other.nil), index(other.index), norm(other.norm) {}
    IntMap& operator=(IntMap&& other) { other.moveTo(*this); return *this; }

    IntMap(const IntMap& other) { other.copyTo(*this); }
    IntMap& operator=(const IntMap& other) { other.copyTo(*this); return *this; }
  #endif
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// IntTmpMap:


// An 'IntMap' with slightly larger memory foot-print, but faster 'clear()' method. Its ideal use
// is if you have only a few non-null element and don't set them back to 'nil' repeatedly.
// NOTE! Does not support non-copyable types (unlike 'IntMap').
//
template<class Key_, class Value_, class Key2Index = MkIndex_default<Key_> >
class IntTmpMap : public IntMap<Key_,Value_,Key2Index> {
    Vec<Key_> elems;
    typedef IntMap<Key_,Value_,Key2Index> P;

public:
    typedef Key_   Key;
    typedef Value_ Value;

    IntTmpMap(Key2Index index_ = Key2Index()) : IntMap<Key_,Value_,Key2Index>(index_) {}
    IntTmpMap(Value_ nil_, Key2Index index_ = Key2Index()) : IntMap<Key_,Value_,Key2Index>(nil_, index_) {}

    Value_& operator()(Key_ k) {
        if (this->null(k)) elems.push(k);
        return P::operator()(k); }

    void clear(bool dispose = false) {
        if (dispose){
            P::clear(true);
            elems.clear(true);
        }else{
            for (uind i = 0; i < elems.size(); i++)
                P::data[P::index(elems[i])] = P::nil;
            elems.clear();
        }
    }

    void copyTo(IntTmpMap& dst) const { P::copyTo(dst); elems.copyTo(dst.elems); }
    void moveTo(IntTmpMap& dst)       { P::moveTo(dst); elems.moveTo(dst.elems); }

    Vec<Key_>&       list()       { return elems; }
    const Vec<Key_>& list() const { return elems; }
        // -- Returns the list of keys assigned to a non-null value since the last 'clear()'.
        // If a key is assigned several times, it will occur several times in this list.
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

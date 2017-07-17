//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Heap.hh
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : A simlpe binary heap implementation, with and withoud data associated with keys.
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| 
//| The base implementation is 'KeyHeap' which implements a priority queue on a type ordered by
//| template parameter 'LT' (less than, defaults to '<'). For convenience, class 'Heap' is 
//| provided which is implemented as 'KeyHeap< Pair<Key,Value> >' for given template parameters
//| 'Key' and 'Value' (it is such a common use case).
//|________________________________________________________________________________________________

#ifndef ZZ__Generics__Heap_h
#define ZZ__Generics__Heap_h
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// KeyHeap:


template<class Key_, bool max_heap = false, class LT = LessThan_default<Key_> >
class KeyHeap : public NonCopyable {
protected:
    Vec<Key_> heap;
    LT        lt;

    bool lessThan(Key_ x, Key_ y) {
        return max_heap ? lt(x, y) : lt(y, x); }

    inline void siftUp  (uind pos, Key_ key);
    inline void siftDown(uind pos, Key_ key);

public:
    typedef Key_ Key;

    KeyHeap()       : lt(LT()) {}
    KeyHeap(LT lt_) : lt(lt_)  {}

    void reserve(uind cap)           { heap.reserve(cap); }
    void clear(bool dispose = false) { heap.clear(dispose); }
    uind size() const                { return heap.size(); }

    Key  peek()        { assert(size() != 0); return heap[0]; }
    Key  pop()         { assert(size() != 0); Key ret = heap[0], tmp = heap.last(); heap.pop(); if (size() > 0) siftDown(0, tmp); return ret; }
    void add(Key_ key) { heap.push(); siftUp(heap.size() - 1, key); }

    const Vec<Key_>& base() const { return heap; }
};

/*
    void   heapify() {
        if (heap.size() > 0){
            for (uind i = heap.size() / 2; i > 0;) i--,
                siftDown(i, heap[i]); } }
    void   push(ElemT elem) { assert(!has(elem)); uind pos = heap.size(); heap.push(); set(pos, elem); }   // -- must be followed by a 'heapify()'
*/


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Heap:


template<class T1, class T2, class LT>
struct LessThan_PairFst {
    LT lt;
    LessThan_PairFst()       : lt(LT()) {}
    LessThan_PairFst(LT lt_) : lt(lt_)  {}
    bool operator()(Pair<T1,T2>& elem1, Pair<T1,T2>& elem2) { return lt(elem1.fst, elem2.fst); }
};


template<class Key_, class Value_, bool max_heap = false, class LT = LessThan_default<Key_> >
class Heap : public NonCopyable {
protected:
    KeyHeap<Pair<Key_,Value_>, max_heap, LessThan_PairFst<Key_,Value_,LT> > heap;

public:
    typedef Key_   Key;
    typedef Value_ Value;

    Heap()      : heap(LessThan_PairFst<Key_,Value_,LT>(LT())) {}
    Heap(LT lt) : heap(LessThan_PairFst<Key_,Value_,LT>(lt))   {}

    void reserve(uind cap)           { heap.reserve(cap); }
    void clear(bool dispose = false) { heap.clear(dispose); }
    uind size() const                { return heap.size(); }

    Pair<Key_,Value_> peek()                      { return heap.peek(); }
    Key               peekKey()                   { return heap.peek().fst; }
    Value             peekValue()                 { return heap.peek().snd; }
    Pair<Key_,Value_> pop()                       { return heap.pop(); }
    void              add(Key_ key, Value_ val)   { heap.add(tuple(key, val)); }
    void              add(Pair<Key_,Value_> elem) { heap.add(elem); }

    const Vec<Pair<Key_,Value_> >& base() const { return heap.base(); }
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Implementation:


template<class K, bool mx, class LT>
inline void KeyHeap<K,mx,LT>::siftUp(uind pos, K key)
{
    while (pos > 0){
        uind parent = (pos - 1) >> 1;
        if (lessThan(heap[parent], key)){
            heap[pos] = heap[parent];
            pos = parent;
        }else
            break;
    }
    heap[pos] = key;
}


template<class K, bool mx, class LT>
inline void KeyHeap<K,mx,LT>::siftDown(uind pos, K key)
{
    for (;;){
        uind child = (pos << 1) + 1;
        if (child >= heap.size() - 1){
            if (child == heap.size() - 1){
                if (lessThan(key, heap[child])){
                    heap[pos] = heap[child];
                    pos = child;
                }else
                    break;
            }else
                break;
        }else{
            if (lessThan(heap[child], heap[child+1]))
                child++;
            if (lessThan(key, heap[child])){
                heap[pos] = heap[child];
                pos = child;
            }else
                break;
        }
    }
    heap[pos] = key;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

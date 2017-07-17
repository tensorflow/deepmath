//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : IdHeap.hh
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : A heap with supports for increase/decrease of the key (priority) of its elements.
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| 
//| The elements of the heap is of type 'ElemT', typically an integer type. The template 
//| argument 'T2Id' should convert an 'ElemT' to an integer ID of type 'uind' by implementing:
//| 
//|    uind operator()(const ElemT& elem) const;
//| 
//| The default implementation is to just cast 'ElemT' to 'uind' which works in most cases.
//| 
//| The keys (here called priorities) of the heap is of type 'Prio_'. It must be a type with 
//| 'operator <' defined (as well as copy-constructor and assignment operator).
//| 
//| The priorities of the elements are stored (and updated) externally. Default is to assume
//| the priorities to be stored in a 'Vec', but you can pass a different type as 'Prio_Vec'.
//| Note that this vector is indexed by 'ElemT', so you need to make sure 'Prio_Vec' supports:
//| 
//|    Prio_ operator[](Elem elem) const;
//|________________________________________________________________________________________________

#ifndef ZZ__Generics__IdHeap_h
#define ZZ__Generics__IdHeap_h
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


template<class Prio_, bool max_heap = false, class Prio_Vec = Vec<Prio_>, class ElemT = uind, class T2Id = MkIndex_default<ElemT> >
class IdHeap : public NonCopyable {
protected:
    Vec<uind>   id2pos;
    Vec<ElemT>  heap;
    T2Id        get_id;

    bool lessThan(Prio_ x, Prio_ y) {
        return max_heap ? (x < y) : (y < x); }

    void siftUp(uind pos, ElemT elem) {
        Prio_  prio_elem = (*prio)[get_id(elem)];
        while (pos > 0){
            uind parent = (pos - 1) >> 1;
            if (lessThan((*prio)[get_id(heap[parent])], prio_elem)){
                set(pos, heap[parent]);
                pos = parent;
            }else
                break;
        }
        set(pos, elem); }

    void siftDown(uind pos, ElemT elem) {
        Prio_  prio_elem = (*prio)[get_id(elem)];
        for (;;){
            uind child = (pos << 1) + 1;
            if (child >= heap.size() - 1){
                if (child == heap.size() - 1){
                    // No right-child
                    if (lessThan(prio_elem, (*prio)[get_id(heap[child])])){
                        set(pos, heap[child]);
                        pos = child;
                    }else
                        break;
                }else
                    // No children at all
                    break;
            }else{
                // Two children
                if (lessThan((*prio)[get_id(heap[child])], (*prio)[get_id(heap[child+1])]))
                    child++;    // -- now index the larger child.
                if (lessThan(prio_elem, (*prio)[get_id(heap[child])])){
                    set(pos, heap[child]);
                    pos = child;
                }else
                    break;
            }
        }
        set(pos, elem); }

    void set(uind pos, ElemT elem) { uind id = get_id(elem); heap[pos] = elem; id2pos.growTo(id+1, UINT_MAX); id2pos[id] = pos; }

public:
    typedef Prio_ Prio;

    const Prio_Vec* prio;   // id->prio. Pointer to external priority table. If you change it, you must run 'heapify()'.

    IdHeap(T2Id i = T2Id())                    : get_id(i), prio(NULL) {}
    IdHeap(const Prio_Vec& p, T2Id i = T2Id()) : get_id(i), prio(&p)   {}

    // Major operations:
    void   clear(bool dealloc = false) {
        heap.clear(dealloc); id2pos.clear(dealloc); }

    void   moveTo(IdHeap& dst) {
        dst.prio = prio;
        id2pos.moveTo(dst.id2pos);
        heap  .moveTo(dst.heap  );
        dst.get_id = get_id; }

    void   copyTo(IdHeap& dst) const {
        dst.prio = prio;
        id2pos.copyTo(dst.id2pos);
        heap  .copyTo(dst.heap  );
        dst.get_id = get_id; }

    // Minor operations:
    uind   size    () const     { return heap.size(); }
    ElemT  peek    () const     { assert(size() > 0); return heap[0]; }
    Prio_  peekPrio() const     { assert(size() > 0); return (*prio)[get_id(heap[0])]; }
    ElemT  pop     ()           { assert(size() > 0); id2pos[get_id(heap[0])] = UINT_MAX; ElemT tmp = heap.last(), ret = heap[0]; heap.pop(); if (size() > 0) siftDown(0, tmp); return ret; }
    void   remove  (ElemT elem) { uind id = get_id(elem); assert(has(elem)); uind pos = id2pos[id]; id2pos[id] = UINT_MAX; ElemT tmp = heap.last(); heap.pop(); if (size() > pos){ siftUp(pos, tmp); siftDown(id2pos[get_id(tmp)], tmp); } }
    void   exclude (ElemT elem) { uind id = get_id(elem); if (!has(elem)) return; uind pos = id2pos[id]; id2pos[id] = UINT_MAX; ElemT tmp = heap.last(); heap.pop(); if (size() > pos){ siftUp(pos, tmp); siftDown(id2pos[get_id(tmp)], tmp); } }
    void   add     (ElemT elem) { assert(!has(elem)); heap.push(); siftUp(heap.size() - 1, elem); }
    bool   weakAdd (ElemT elem) { if (has(elem)) return false; heap.push(); siftUp(heap.size() - 1, elem); return true; }     // -- returns TRUE if element was inserted, FALSE if it already existed.
    bool   has     (ElemT elem) { uind id = get_id(elem); return (uind(id) < id2pos.size() && id2pos[id] != UINT_MAX); }

    // Update when at most one node has changed (pre-condition):
    void   updateIncreased(ElemT elem) { uind id = get_id(elem); assert(has(elem)); if (max_heap) siftUp  (id2pos[id], elem); else siftDown(id2pos[id], elem); }
    void   updateDecreased(ElemT elem) { uind id = get_id(elem); assert(has(elem)); if (max_heap) siftDown(id2pos[id], elem); else siftUp  (id2pos[id], elem); }
    void   update         (ElemT elem) { updateIncreased(elem); updateDecreased(elem); }

    // Update all nodes:
    void   heapify() {
        if (heap.size() > 0){
            for (uind i = heap.size() / 2; i > 0;) i--,
                siftDown(i, heap[i]); } }
    void   push(ElemT elem) { assert(!has(elem)); uind pos = heap.size(); heap.push(); set(pos, elem); }   // -- must be followed by a 'heapify()'

    // Access to underlying vectors implementing heap:
    const Vec<ElemT>& base     () const { return heap; }
    const Vec<uind>&  positions() const { return id2pos; }        // -- contains 'UINT_MAX' for non-existent elements.

    // Debug:
    void checkConsistency() {
        for (uind i = 0; i < heap.size() >> 1; i++){
            assert(!lessThan((*prio)[get_id(heap[i])], (*prio)[get_id(heap[(i << 1) + 1])]));
            assert((i << 1) + 2 == heap.size() || !lessThan((*prio)[get_id(heap[i])], (*prio)[get_id(heap[(i << 1) + 2])])); }
        for (uind i = 0; i < id2pos.size(); i++)
            assert(id2pos[i] == UINT_MAX || (id2pos[i] < heap.size() && uind(get_id(heap[id2pos[i]])) == i));
    }
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : PArr.hh
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : Persistent array.
//|
//| (C) Copyright 2010-2017, Niklas Een
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//|
//|________________________________________________________________________________________________

#ifndef ZZ__Generics__PArr_hh
#define ZZ__Generics__PArr_hh
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


template<class T, uint branch_factor, uint leaf_size>
struct PArr_data {
    struct {
        uint64 refC    : 63;
        bool   is_leaf : 1;
    };
    union {
        PArr_data* subs[branch_factor];
        T          vals[leaf_size];
    };

    PArr_data(bool is_leaf) : refC(1), is_leaf(is_leaf) {   // -- reference counter starts at '1'
        if (is_leaf) for (T& val : vals) new(&val) T();
        else         memClear(subs); }

    PArr_data(Tag_copy, PArr_data const& other) : refC(1), is_leaf(other.is_leaf) {
        if (is_leaf) for (uint i = 0; i < leaf_size; i++) new(&vals[i]) T(other.vals[i]);
        else         for (uint i = 0; i < branch_factor; i++){ subs[i] = other.subs[i]; if (subs[i]) subs[i]->refC++; } }

   ~PArr_data() {
        if (is_leaf) for (T& val : vals) val.~T();
        else         for (PArr_data* sub : subs){ if (sub){ assert(sub->refC != 0); sub->refC--; if (sub->refC == 0) delete sub; } } }

    // Iteration:
    template<class F> bool forAll(F f) {                        // -- F: bool f(T elem)
        if (is_leaf) for (uint i = 0; i < leaf_size    ; i++){ if (!f(vals[i])) return false; }
        else         for (uint i = 0; i < branch_factor; i++){ if (subs[i]){ if (!subs[i]->forAll(f)) return false; } }
        return true;
    }

    template<class F> bool forAllRev(F f) {
        if (is_leaf) for (uint i = leaf_size    ; i > 0;){ i--; if (!f(vals[i])) return false; }
        else         for (uint i = branch_factor; i > 0;){ i--; if (subs[i]){ if (!subs[i]->forAllRev(f)) return false; } }
        return true;
    }

    template<class F> bool enumAll(F f, uind off, uind cap) {   // -- F: bool f(uind idx, T elem)
        if (is_leaf){
            for (uint i = 0; i < leaf_size; i++){ if (!f(off + i, vals[i])) return false; }
        }else{
            cap /= branch_factor;
            for (uint i = 0; i < branch_factor; i++){ if (subs[i]){ if (!subs[i]->enumAll(f, off + cap*i, cap)) return false; } }
        }
        return true;
    }

    template<class F> bool enumAllRev(F f, uind off, uind cap) {
        if (is_leaf){
            for (uint i = leaf_size; i > 0;){ i--; if (!f(off + i, vals[i])) return false; }
        }else{
            cap /= branch_factor;
            for (uint i = branch_factor; i > 0;){ i--; if (subs[i]){ if (!subs[i]->enumAllRev(f, off + cap*i, cap)) return false; } }
        }
        return true;
    }

    // Debug:
    void dump(){
        wr("%%%_", refC);
        if (is_leaf)
            wr("(%_)", join(", ", slice(vals[0], vals[leaf_size])));
        else{
            wr("[");
            for (uint i = 0; i < branch_factor; i++){
                if (i != 0) wr(", ");
                if (!subs[i]) wr("-"); else subs[i]->dump();
            }
            wr("]");
        }
    }
};


//=================================================================================================
// -- PArr:


template<class T, uint branch_factor = 8, uint leaf_size = (branch_factor * sizeof(void*) + sizeof(T) - 1) / sizeof(T)>
class PArr {
    typedef PArr_data<T, branch_factor, leaf_size> Data;

    uind  sz;
    uind  cap;
    Data* ptr;

    PArr(uind sz, uind cap, Data* ptr) : sz(sz), cap(cap), ptr(ptr) {}
    static void ref_  (Data* p) { if (p) p->refC++; }
    static void unref_(Data* p) { if (p){ assert(p->refC != 0); p->refC--; if (p->refC == 0) delete p; } }

public:
    PArr() : sz(0), cap(0), ptr(nullptr) {}

    // Modifiers -- returns new modified array:
    PArr set(uind idx, T const& val);                   // -- will auto-extend array if 'idx >= size()'
    PArr push(T const& val) { return set(sz ,val); }
    PArr pop() { assert(sz > 0); PArr ret = set(sz-1, T()); ret.sz--; return ret;  }  // -- will not return allocated nodes

    // Readers:
    uind size() const { return sz; }
    T operator[](uind idx) const;

    template<class F> bool enumAllCond   (F f) const { if (ptr) return ptr->enumAll(f, 0, cap); else return true; } // -- F: bool f(uind idx, T elem)
    template<class F> bool forAllCond    (F f) const { if (ptr) return ptr->forAll (f);         else return true; } // -- F: bool f(T elem)
    template<class F> bool enumAllCondRev(F f) const { if (ptr) return ptr->enumAllRev(f, 0, cap); else return true; }
    template<class F> bool forAllCondRev (F f) const { if (ptr) return ptr->forAllRev (f);         else return true; }
    template<class F> void enumAll       (F f) const { enumAllCond   ([&](uint idx, T const& elem){ f(idx, elem); return true;}); }
    template<class F> void forAll        (F f) const { forAllCond    ([&](T const& elem)          { f(elem);      return true;}); }
    template<class F> void enumAllRev    (F f) const { enumAllCondRev([&](uint idx, T const& elem){ f(idx, elem); return true;}); }
    template<class F> void forAllRev     (F f) const { forAllCondRev ([&](T const& elem)          { f(elem);      return true;}); }
        // -- NOTE! May skip null-elements if array was extended leaving holes.

    // Reference counting:
   ~PArr()                             { unref_(ptr); }
    PArr(PArr const& other)            { memCpy(*this, other); ref_(other.ptr); }
    PArr& operator=(const PArr& other) { if (this == &other) return *this; unref_(ptr); memCpy(*this, other); ref_(ptr); return *this; }
    PArr(PArr&& other)                 { memCpy(*this, other); new(&other) PArr(); }
    PArr& operator=(PArr&& other)      { if (this == &other) return *this; unref_(ptr); memCpy(*this, other); new(&other) PArr(); return *this; }

    // Debug:
    void dump() { if (!ptr) wr("-"); else ptr->dump(); }
    uind capacity() const { return cap; }
};


//=================================================================================================
// -- implementation:


template<class T, uint bf, uint ls>
fts_macro void write_(Out& out, PArr<T,bf,ls> const& v) {
    wr(out, "[%_]", join(", ", v)); }


template<class T, uint bf, uint ls>
PArr<T,bf,ls> PArr<T,bf,ls>::set(uind idx, T const& val)
{
    // bf = 3
    // ls = 4
    // size = 14                                     root
    //                                                 |
    //              +----------------------------------+--------------------------------+
    //              |                                  |                               nil
    //    +---------+----------+            +----------+----------+
    //    |         |          |            |         nil        nil
    // 0 1 2 3   4 5 6 7   8 9 10 11    12 13 - -
    //
    // where '-' is the default value for type T

    uind  cap = this->cap;
    Data* ptr = this->ptr;

    // Extend depth to cover index:
    if (idx >= cap){
        if (cap == 0){
            assert(ptr == nullptr);
            if (idx < ls){
                ptr = new PArr_data<T,bf,ls>(true);
                cap = ls;
            }else{
                ptr = new PArr_data<T,bf,ls>(false);
                cap = ls * bf;
                while (idx >= cap) cap *= bf;
            }
        }else{
            do{
                Data* old_ptr = ptr;
                ptr = new PArr_data<T,bf,ls>(false);
                ptr->refC--;
                cap *= bf;
                ptr->subs[0] = old_ptr;
                old_ptr->refC++;
            }while (idx >= cap);
            ptr->refC++;
        }
    }else
        ptr = new PArr_data<T,bf,ls>(copy_, *ptr);

    // Insert element:
    Data* p = ptr;
    uind  c = cap;
    uind  i = idx;
    for(;;){
        if (p->is_leaf){
            assert(i < ls);
            p->vals[i] = val;
            break;

        }else{
            c /= bf;
            uint s = i / c;     // -- could force both 'bf' and 'ls' to be powers of 2, then represent 'cap' in log2-space
            i %= c;
            if (!p->subs[s]){
                p->subs[s] = new PArr_data<T,bf,ls>(c == ls);
            }else{
                Data* sub = new PArr_data<T,bf,ls>(copy_, *p->subs[s]);
                unref_(p->subs[s]);
                p->subs[s] = sub;
            }
            p = p->subs[s];
        }
    }

    return PArr(max_(sz, idx+1), cap, ptr);
}


template<class T, uint bf, uint ls>
inline T PArr<T,bf,ls>::operator[](uind idx) const
{
    if (idx >= sz) return T();
    Data* p = ptr;
    uind  c = cap;
    for(;;){
        if (!p) return T();
        if (p->is_leaf){
            assert(idx < ls);
            return p->vals[idx];

        }else{
            c /= bf;
            uint s = idx / c;
            idx %= c;
            p = p->subs[s];
        }
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

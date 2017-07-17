//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Set.hh
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : A generic hash set.
//|
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//|
//| If you don't want to use the default hash function (see 'Prelude/Hash.hh'), the second template
//| parameter 'Hash_' should be a struct of the following type:
//|
//|      struct Hash_Param {
//|          uint64 hash (K key)          const { return hash_(key); }
//|          bool   equal(K key1, K key2) const { return equal_(key1, key2); }
//|      }
//|________________________________________________________________________________________________

#ifndef ZZ__Generics__Set_h
#define ZZ__Generics__Set_h
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


template<class Key_, class Hash_ = Hash_default<Key_> >
class Set /*: public NonCopyable*/ {
protected:
    struct Cell {
        Key_    key;
        Cell*   next;
    };

    UniAlloc<Cell>  mem;
    Cell**          table;
    uind            cap;
    uind            sz;
    Hash_           param;

    // Internal helpers:
    void init(uind min_capacity);
    void dispose();
    void rehash(uind min_capacity);

public:
    // Types:
    typedef Key_  Key;
    typedef Hash_ Hash;

    // Constructors:
    Set()                   : param()  { init(1);   }
    Set(uind cap_)          : param()  { init(cap_); }
    Set(Hash_ p)            : param(p) { init(1);   }
    Set(uind cap_, Hash_ p) : param(p) { init(cap_); }
   ~Set() { dispose(); }

    void setParam(Hash_ p) { param = p; }       // -- If used, it must be called once before any hash operation is performed.
    void moveTo(Set& dst);

    // Size:
    uind     size    () const { return sz; }
    uind     capacity() const { return cap; }
    void     clear   ()       { dispose(); init(1); }   // <<== should add 'bool dispose' as parameter
    void     reserve (uind min_capacity) { rehash(min_capacity); }


    //---------------------------------------------------------------------------------------------
    // SET OPERATIONS:

    bool        add    (const Key_& key);       // -- Add or replace 'key'. Returns TRUE if element already existed.
    Key_&       addWeak(const Key_& key);       // -- Add an element if does not already exist. Return reference to old or new element.
    Key_*       get    (const Key_& key);       // -- Returns the representative element of 'key', or NULL if none.
    const Key_* get    (const Key_& key) const; // -- Returns the representative element of 'key', or NULL if none.
    bool        has    (const Key_& key) const; // -- Returns TRUE if element exists.
    bool        exclude(const Key_& key);       // -- Returns TRUE if element existed and was excluded.


    //---------------------------------------------------------------------------------------------
    // LOW-LEVEL HASH OPERATIONS:

    uind   index_(uint64 hash_value) const { return uind(hash_value % cap); }
    uind   index (const Key_& key)   const { return index_(param.hash(key)); }
    uint64 hash  (const Key_& key)   const { return param.hash(key); }
        // -- Return the bucket index of a key or the hash value of a key.

    template<class Eq>
    bool search(uind i, Eq& eq, Key_*& result);
        // -- Search in bucket 'i' for an element matched by 'eq'. If successful, TRUE is returned
        // and 'result' is pointed to the matching 'key'. If not, FALSE is returned (and result
        // is left untouched).

    bool lookup(uind i, const Key_& key, Key_*& current_key);
        // -- Search for 'key' in bucket 'i'. If found, the current key is returned through
        // reference and TRUE as return value. If not found, return value is FALSE.

    Key_& newEntry(uind i, const Key_& key);
        // -- Add a new cell at bucked index 'i' and return a pointer to the copy of 'key'
        // within the cell. PRE-CONDITION: 'key' does not exist in hash-table already. NOTE!
        // Bucket index 'i' is provided for efficiency. If the hash-table needs resizing, a
        // new index will be computed from 'key' after resizing has been done.

    // Low-level iteration: (prefer macros instead)
    void*        firstCell(uind i) const { return table[i]; }
    static void* nextCell (void* cell)   { return static_cast<Cell*>(cell)->next; }
    static const Key_& key(void* cell)   { return static_cast<Cell*>(cell)->key; }

  #if (__cplusplus >= 201103L)
    Set(Set&& other) { other.moveTo(*this); }
    Set& operator=(Set&& other) { other.moveTo(*this); return *this; }

    Set(const Set& other) { other.copyTo(*this); }
    Set& operator=(const Set& other) { other.copyTo(*this); return *this; }
  #endif
};


// Iteration macros:
#define Set_Key(set) set.key(c_)
#define For_Set(set) \
    for (uind i_ = 0; i_ < set.capacity(); i_++) \
        for (void* c_ = set.firstCell(i_); c_; c_ = set.nextCell(c_))

// Example of usage:
//
//     Set<String> s;
//     ...
//     For_Set(s){
//         String key = Set_Key(s);
//         ...
//     }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Implementation:


//=================================================================================================
// -- Internal:


template<class K, class H>
inline void Set<K,H>::init(uind min_capacity)
{
    uint64 cap_ = (prime_ >= min_capacity); assert(cap_ <= UIND_MAX);
    cap = uind(cap_);
    sz  = 0;
    table = xmalloc<Cell*>(cap);
    for (uind i = 0; i < cap; i++)
        table[i] = NULL;
}


template<class K, class H>
inline void Set<K,H>::dispose()
{
    for (uind i = 0; i < cap; i++){
        for (Cell* p = table[i]; p != NULL;){
            Cell* next = p->next;
            p->key  .~K();
            p = next;
        }
    }
    xfree(table);
    mem.clear();
}


template<class K, class H>
inline void Set<K,H>::rehash(uind min_capacity)
{
    uind   old_cap = cap;
    uint64 cap_ = (prime_ >= min_capacity); assert(cap_ <= UIND_MAX);
    cap = uind(cap_);
    Cell** new_table = xmalloc<Cell*>(cap);

    for (uind i = 0; i < cap; i++)
        new_table[i] = NULL;

    for (uind i = 0; i < old_cap; i++){
        for (Cell* p = table[i]; p != NULL;){
            Cell* next = p->next;
            uind j = index(p->key);
            p->next = new_table[j];
            new_table[j] = p;
            p = next;
        }
    }
    xfree(table);
    table = new_table;
}


//=================================================================================================
// -- Public:


template<class K, class H>
inline void Set<K,H>::moveTo(Set<K,H>& dst)
{
    dst.dispose();
    mem.moveTo(dst.mem);
    dst.table = table;
    dst.cap   = cap;
    dst.sz    = sz;
    dst.param.~Hash();
    new (&dst.param) Hash(param);
    init(1);
}


template<class K, class H>
template<class Eq>
inline bool Set<K,H>::search(uind i, Eq& eq, K*& result)
{
    for (Cell* p = table[i]; p != NULL; p = p->next){
        if (eq(p->key)){
            result = &p->key;
            return true;
        }
    }
    return false;
}


template<class K, class H>
inline bool Set<K,H>::lookup(uind i, const K& key, K*& current_key)
{
    for (Cell* p = table[i]; p != NULL; p = p->next){
        if (param.equal(p->key, key)){
            current_key = &p->key;
            return true;
        }
    }
    return false;
}


template<class K, class H>
inline K& Set<K,H>::newEntry(uind i, const K& key)
{
    if (sz > cap){
        rehash(cap * 2);
        i = index(key); }

    Cell* p = mem.alloc();
    new (&p->key) K(key);
    p->next = table[i];
    table[i] = p;
    sz++;
    return p->key;
}


template<class K, class H>
inline bool Set<K,H>::add(const K& key)
{
    uind  i = index(key);
    K* k;
    if (lookup(i, key, k)){
        *k = key;
        return true;
    }else{
        newEntry(i, key);
        return false;
    }
}


template<class K, class H>
inline K& Set<K,H>::addWeak(const K& key)
{
    uind  i = index(key);
    K* k;
    if (lookup(i, key, k))
        return *k;
    else
        return newEntry(i, key);
}


template<class K, class H>
inline K* Set<K,H>::get(const K& key)
{
    uind  i = index(key);
    K* k;
    if (lookup(i, key, k))
        return k;
    else
        return NULL;
}


template<class K, class H>
inline const K* Set<K,H>::get(const K& key) const
{
    Set<K,H>& me = *const_cast<Set<K,H>*>(this);
    return me.get(key);
}


template<class K, class H>
inline bool Set<K,H>::has(const K& key) const
{
    return get(key);
}


template<class K, class H>
inline bool Set<K,H>::exclude(const K& key)
{
    uind i = index(key);
    for (Cell** pp = &table[i]; *pp != NULL; pp = &(*pp)->next){
        if (param.equal((*pp)->key, key)){
            Cell* p = *pp;
            *pp = (*pp)->next;
            sz--;
            p->key.~K();
            mem.free(p);
            return true; }
    }
    return false;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Map.hh
//| Author(s)   : Niklas Een
//| Module      : Generics
//| Description : A generic hash map.
//|
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//|
//| Provides both high and low-level hashing methods that allows you to avoid double hash
//| computation under virtually all use patterns. If you don't want to use the default hash function
//| (see 'Prelude/Hash.hh'), the third template parameter 'Hash_' should be a struct of the
//| following type:
//|
//|      struct Hash_Param {
//|          uint64 hash (K key)          const { return hash_(key); }
//|          bool   equal(K key1, K key2) const { return equal_(key1, key2); }
//|      }
//|________________________________________________________________________________________________

#ifndef ZZ__Generics__Map_h
#define ZZ__Generics__Map_h
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


template<class Key_, class Value_, class Hash_ = Hash_default<Key_> >
class Map : public NonCopyable {
protected:
    struct  Cell {
        Key_    key;
        Value_  value;
        Cell*   next;
    };

    UniAlloc<Cell>  mem;
    Cell**          table;
    uind            cap;
    uind            sz;
    Hash_           param;

    // Internal helpers:
    void    init(uind min_capacity);
    void    dispose();
    void    rehash(uind min_capacity);

public:
    // Types:
    typedef Key_   Key;
    typedef Value_ Value;

    // Constructors:
    Map()                  : param()  { init(1);   }
    Map(uind cap)          : param()  { init(cap); }
    Map(Hash_ p)           : param(p) { init(1);   }
    Map(uind cap, Hash_ p) : param(p) { init(cap); }
   ~Map() { dispose(); }

    void setParam(Hash_ p) { param = p; }       // -- If used, it must be called once before any hash operation is performed.
    void moveTo(Map& dst);

    // Size:
    uind size    () const { return sz; }
    uind capacity() const { return cap; }
    void clear   ()       { dispose(); init(1); }
    void reserve (uind min_capacity) { rehash(min_capacity); }


    //---------------------------------------------------------------------------------------------
    // MAP OPERATIONS:

        // - "get" methods will create a new cell (key/value pair) if no match was found.
        // - "peek" methods will just return false upon no match (hence "const" declared).
        // - All methods return TRUE if the lookup was successful (a "match"), FALSE if not.

    bool get(const Key_& key, Value_*& result);
        // -- Lookup 'key' and, if found, return TRUE plus (through 'result') a pointer to the
        // value corresponding to 'key'. If no match, FALSE is returned, and 'result' will
        // point to UNINITIALIZED memory of 'sizeof(Value_)' bytes. Example of usage:
        //
        //      if (map.get(key, val))
        //          <key found, do something with val>
        //      else
        //          new (val) Value_(<constr. params>);      // (where 'Value_' is your value type)

    bool getI(const Key_& key, Value_*& result);
        // -- Same as 'get' but will *initialize* the memory by calling the default constructor
        // of 'Value_' if key is not found.

    bool peek(const Key_& key, Value_*& result) const;
        // -- Returns through 'result' a pointer to the value matching 'key' (if any).

    bool peek(const Key_& key, Value_& result) const;
        // -- Returns, by copying to 'result', the value matching 'key'.

    bool has(const Key_& key) const;
        // -- Does 'key' exist in the hash-table? Often you would want to use peek or get instead!

    bool set(const Key_& key, const Value_& value);
        // -- Make 'key' map to 'value'. If key was already present, only the value is updated
        // (so if you use an equivalent key, the old key will still be used in the table).
        // Returns TRUE if 'key' did already exist.

    bool exclude(const Key_& key);
        // Exclude 'key' from hash-table if exists (and return TRUE).


    //---------------------------------------------------------------------------------------------
    // LOW-LEVEL HASH OPERATIONS:

    uind index_(uint64 hash_value) const { return uind(hash_value % cap); }
    uind index (const Key_& key)   const { return index_(param.hash(key)); }
        // -- Return the bucket index of a key or the hash value of a key.

    template<class Eq>
    bool search(uind i, Eq& eq, Key_*& key_result, Value_*& value_result) const;
        // -- Low-level search: You give a bucket index and a equality definition. It allows you to
        // do a hash lookup without constructing a proper key. By returning, through reference,
        // both key and value (if a match is found), you can substitute the key for something
        // equivalent, or modify the value. If no match is found, FALSE is returned.

    template<bool new_entry>
    bool lookup(uind i, const Key_& key, Value_*& result);
        // -- If 'new_entry' is TRUE, this method behaves like a "get", otherwise as a "peek".
        // Its main purpose is to serve as a helper method to implement the map operations
        // below, but it can also be used to avoid computing hash values twice.

    Value_& newEntry(uind i, const Key_& key);
        // -- Add a new cell at bucked index 'i' and return a pointer to the uninitialized 'Value_'
        // allocated for it. PRE-CONDITION: 'key' does not exist in hash-table already. NOTE!
        // Bucket index 'i' is provided for efficiency. If the hash-table needs resizing, a
        // new index will be computed from 'key' after resizing has been done.


    // Low-level iteration: (prefer macros instead)
    void*        firstCell    (uind i) const { return table[i]; }
    static void* nextCell     (void* cell)   { return static_cast<Cell*>(cell)->next; }
    static const Key_&   key  (void* cell)   { return static_cast<Cell*>(cell)->key; }
    static const Value_& value(void* cell)   { return static_cast<Cell*>(cell)->value; }
};


// Iteration macros:
#define Map_Key(map) map.key(c_)
#define Map_Value(map) map.value(c_)
#define For_Map(map) \
    for (uind i_ = 0; i_ < map.capacity(); i_++) \
        for (void* c_ = map.firstCell(i_); c_; c_ = map.nextCell(c_))

// Example of usage:
//
//     Map<int,float> m;
//     ...
//     For_Map(m){
//         int   key   = Map_Key(m);
//         float value = Map_Value(m);
//         ...
//     }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Implementation:


//=================================================================================================
// -- Internal:


// Initializer -- called from constructors.
template<class K, class V, class H>
inline void Map<K,V,H>::init(uind min_capacity)
{
    uint64 cap_ = (prime_ >= min_capacity); assert(cap_ <= UIND_MAX);
    cap = uind(cap_);
    sz  = 0;
    table = xmalloc<Cell*>(cap);
    for (uind i = 0; i < cap; i++)
        table[i] = NULL;
}


// Destruct all keys and values, thenn free memory.
template<class K, class V, class H>
inline void Map<K,V,H>::dispose()
{
    for (uind i = 0; i < cap; i++){
        for (Cell* p = table[i]; p != NULL;){
            Cell* next = p->next;
            p->key  .~K();
            p->value.~V();
            p = next;
        }
    }
    xfree(table);
    mem.clear();
}


// Resize hash-table to at least 'min_capacity' number of buckets.
template<class K, class V, class H>
inline void Map<K,V,H>::rehash(uind min_capacity)
{
    uind old_cap = cap;
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


// Create a new cell (key/value pair) in the hash-table.
template<class K, class V, class H>
inline V& Map<K,V,H>::newEntry(uind i, const K& key)
{
    if (sz > cap){
        rehash(cap * 2);
        i = index(key); }

    Cell* p = mem.alloc();
    new (&p->key) K(key);
    p->next = table[i];
    table[i] = p;
    sz++;
    return p->value;
}


//=================================================================================================
// -- Public:


template<class K, class V, class H>
inline void Map<K,V,H>::moveTo(Map<K,V,H>& dst)
{
    dst.dispose();
    mem.moveTo(dst.mem);
    dst.table = table;
    dst.cap   = cap;
    dst.sz    = sz;
    const_cast<H&>(dst.param) = param;
    init(1);
}


template<class K, class V, class H>
template<class Eq>
inline bool Map<K,V,H>::search(uind i, Eq& eq, K*& key_result, V*& value_result) const
{
    for (Cell* p = table[i]; p != NULL; p = p->next){
        if (eq(p->key)){
            key_result   = &p->key;
            value_result = &p->value;
            return true;
        }
    }
    return false;
}

template<class K, class V, class H>
template<bool new_entry>
inline bool Map<K,V,H>::lookup(uind i, const K& key, V*& result)
{
    for (Cell* p = table[i]; p != NULL; p = p->next){
        if (param.equal(p->key, key)){
            result = &p->value;
            return true;
        }
    }
    if (new_entry)
        result = &newEntry(i, key);
    return false;
}


template<class K, class V, class H>
inline bool Map<K,V,H>::get(const K& key, V*& result)
{
    return lookup<true>(index(key), key, result);
}


template<class K, class V, class H>
inline bool Map<K,V,H>::getI(const K& key, V*& result)
{
    if (get(key, result))
        return true;
    else{
        new (result) V();
        return false;
    }
}


template<class K, class V, class H>
inline bool Map<K,V,H>::peek(const K& key, V*& result) const
{
    Map& me = *const_cast<Map*>(this);
    return me.lookup<false>(index(key), key, result);
}


template<class K, class V, class H>
inline bool Map<K,V,H>::peek(const K& key, V& result) const
{
    V* ptr;
    Map<K,V,H>& me = *const_cast<Map<K,V,H>*>(this);
    if (me.peek(key, ptr)){
        result = *ptr;
        return true;
    }else
        return false;
}


template<class K, class V, class H>
inline bool Map<K,V,H>::has(const K& key) const
{
    V* ptr;
    Map<K,V,H>& me = *const_cast<Map<K,V,H>*>(this);
    return me.peek(key, ptr);
}


template<class K, class V, class H>
inline bool Map<K,V,H>::set(const K& key, const V& value)
{
    V* ptr;
    bool ret = get(key, ptr);
    if (ret)
        ptr->~V();
    new (ptr) V(value);
    return ret;
}


template<class K, class V, class H>
inline bool Map<K,V,H>::exclude(const K& key)
{
    uind i = index(key);
    for (Cell** pp = &table[i]; *pp != NULL; pp = &(*pp)->next){
        if (param.equal((*pp)->key, key)){
            Cell* p = *pp;
            *pp = (*pp)->next;
            sz--;
            p->key  .~K();
            p->value.~V();
            mem.free(p);
            return true;
        }
    }
    return false;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

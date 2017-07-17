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

#ifndef ZZ__HolLight__Hashing_hh
#define ZZ__HolLight__Hashing_hh

#include "zz/Generics/Map.hh"
#include "zz/Generics/Set.hh"
#include "zz/Generics/CuckooSet.hh"

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Width of IDs (serial numbers for hashed objects):


typedef uind id_t;      // -- make this 'uint64' if need to scale beyond 4B elements
constexpr id_t id_NULL = 0;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// IdBase


// Everything that is hashed to an ID derives from this base class:
class IdBase {
protected:
    id_t id;
public:
    constexpr IdBase() : id(id_NULL) {}

    IdBase(IdBase const& other) { id = other.id; }
    IdBase& operator=(IdBase const& other) { id = other.id; return *this; }

    explicit IdBase(id_t id) : id(id) {}
    id_t operator+() const { return id; }   // -- unary + returns underlying id

    bool operator==(IdBase other) const { return id == other.id; }
    bool operator!=(IdBase other) const { return id != other.id; }
    bool operator< (IdBase other) const { return id <  other.id; }

    explicit operator bool() const { return id != id_NULL; }
};


template<> struct Hash_default<IdBase> {
    uint64 hash (const IdBase& key) const { return defaultHash(+key); }
    bool   equal(const IdBase& key1, const IdBase& key2) const { return key1 == key2; }
};


// This is unfortunate, but C++ is still lacking concept checking which would avoid this...
#define Make_IdBase_MkIndex(IdBaseSubtype)                  \
    template<> struct MkIndex_default<IdBaseSubtype> {      \
        typedef IdBaseSubtype Key;                          \
        uind operator()(Key const& x) const { return +x; }  \
        Key inv(uind i) const { return Key(i); }            \
    };


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Atomic<TAG>:


// String-pool hashing strings to integers 1, 2, 3...
template<class TAG>
class Atomic : public IdBase {
    id_t getOrCreate(Str text) {
        id_t ret;
        if (!text2atom.peek(text, ret)){
            ret = atom2text.size();

            uind sz = text.size();
            char* base = atom_data.alloc(sz + 1);
            memcpy(base, text.base(), sz);
            base[sz] = '\0';        // -- store string as zero-terminated string for old-style string usage
            Str str(base, sz);

            text2atom.set(str, ret);
            atom2text.push(str);
        }
        return ret+1;
    }

    static StackAlloc<char> atom_data;
    static Map<Str, id_t>   text2atom;
    static Vec<Str>         atom2text;

public:
    using IdBase::IdBase;
    Atomic() : IdBase() {}

    explicit Atomic(cchar* text)          { id = getOrCreate(slize(text)); }
    explicit Atomic(cchar* t0, cchar* t1) { id = getOrCreate(Str(t0, t1)); }
    explicit Atomic(Str text)             { id = getOrCreate(text); }
    explicit Atomic(const String& text)   { id = getOrCreate(text.slice()); }

    operator Str() const { return atom2text[id-1]; }
    cchar* c_str() const { return atom2text[id-1].base(); }

    // For profiling:
    static id_t   count() { return atom2text.size(); }
    static size_t strAlloc() { size_t sz = 0; for (Str s : atom2text) sz += s.size() + 1; return sz; }
};

template<typename TAG> StackAlloc<char> Atomic<TAG>::atom_data;
template<typename TAG> Map<Str, id_t>   Atomic<TAG>::text2atom;
template<typename TAG> Vec<Str>         Atomic<TAG>::atom2text;

template<class T> fts_macro void write_(Out& out, Atomic<T> const& v) { write_(out, (Str)v); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Composite<DATA>:


template<class DATA>
class Composite : public IdBase {
    id_t getOrCreate(DATA const& new_data) {
        id2data.push(new_data);
        id_t ret_new = id2data.size();
        id_t ret = data2id.addWeak(ret_new);
        if (ret != ret_new)
            id2data.pop();
        return ret;
    }

    struct Hash {
        Vec<DATA> const& data;
        Hash(Vec<DATA> const& data) : data(data) {}
        uint64 hash (id_t key)             const { return defaultHash(data[key-1]); }
        bool   equal(id_t key1, id_t key2) const { return defaultEqual(data[key1-1], data[key2-1]); }
    };

    static Vec<DATA>       id2data;
  #if 0
    static Set<id_t, Hash> data2id;
    typedef Set<id_t, Hash> SetType;    // -- just to be able to specify the type in the static variable definition of 'data2id' below
  #else
    static CuckooSet<id_t, Hash> data2id;
    typedef CuckooSet<id_t, Hash> SetType;
  #endif

public:
    typedef DATA Data;

    using IdBase::IdBase;
    constexpr Composite() : IdBase() {}
  #if (__cplusplus > 201103L)
    explicit Composite(DATA const& data) { id = getOrCreate(data); }
  #else
    Composite(DATA const& data) { id = getOrCreate(data); }
  #endif

    DATA const& operator* () const { return  id2data[id-1]; }
    DATA&       operator* ()       { return  id2data[id-1]; }
    DATA const* operator->() const { return &id2data[id-1]; }
    DATA*       operator->()       { return &id2data[id-1]; }

    // Returns the 'DATA' component this object points to (via 'id'):
    DATA&       me()       { return id2data[id-1]; }
    DATA const& me() const { return id2data[id-1]; }

    static id_t count() { return id2data.size(); }      // -- Example: for (id_t i = 1; i <= Term::count(); i++) <code>
    static id_t alloc() { return data2id.capacity(); }  // -- for profiling (slots allocated for hash set)
};

template<typename DATA> Vec<DATA> Composite<DATA>::id2data;
template<typename DATA> typename Composite<DATA>::SetType Composite<DATA>::data2id((Composite<DATA>::Hash(Composite<DATA>::id2data)));


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

#ifndef ZZ__Generics__CuckooSet_hh
#define ZZ__Generics__CuckooSet_hh
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


constexpr uint n_cuckoo_primes = 6962;
extern uint64 cuckoo_primes[n_cuckoo_primes];


// NOTE! The given 'nil' element is compared using '==', not 'hash.equal()'.
template<class Key_, class Hash_ = Hash_default<Key_>>
class CuckooSet {
    // Parameters controlloing behavior:
    static constexpr double allowed_avg_tries = 0.05;
    static constexpr double start_tries = 50;
    static constexpr double max_tries = 10000;
    static constexpr uint   allowed_grow_rec = 8;
    static constexpr uint   n_bins = 5;
//    static constexpr uint   grow_steps = 150;     // -- must be >= #regions; a step of 200 ~= a doubling
    static constexpr uint   grow_steps = 256;     // -- must be >= #regions; a step of 200 ~= a doubling

    struct Region {
        uint    prime_idx = 0;
        uint    hash_num = 0;
        size_t  cap = 0;        // -- must be prime number
        size_t  sz  = 0;
        Key_*   data = nullptr;
        uchar*  abst = nullptr;
    };

    enum { REGPOW  = 8 };    // -- update this one (#regions == '1 << REGPOW'); must be >= 2
    enum { N_REGS  = 1u << REGPOW };
    enum { R_SHIFT = REGPOW - 1 };
    enum { R_ADD   = 1u << R_SHIFT };
    enum { R_MASK  = R_ADD - 1 };

    Region  reg[N_REGS];
    Hash_   hash;

    double tries_left = start_tries;
    uint grow_rec = 0;
    uint sum_prime_idx = 0;     // -- sum of all 'reg[].prime_idx's

    static uint   reg0 (uint64 key) { return key & R_MASK; }
    static uint   reg1 (uint64 key) { return R_ADD + ((key >> R_SHIFT) & R_MASK); }
    static uint64 shift(uint64 key) { return key >> (2 * R_SHIFT); }

    void grow(uint r0, uint r1);
    void rehash();
    void addTo(uint side, uint n_tries, Key_ key);

public:
    typedef Key_  Key;
    typedef Hash_ Hash;

    CuckooSet()        : hash()  { for (uint i = 0; i < N_REGS; i++) grow(i, i); }
    CuckooSet(Hash_ h) : hash(h) { for (uint i = 0; i < N_REGS; i++) grow(i, i); }

    Key addWeak(Key key); // -- Returns existing equal key (if there is one), or newly created key (if there is not).
    size_t size()     const { size_t sum = 0; for (uint i = 0; i < N_REGS; i++) sum += reg[i].sz ; return sum; }
    size_t capacity() const { size_t sum = 0; for (uint i = 0; i < N_REGS; i++) sum += reg[i].cap; return sum; }
};


template<class Key, class Hash>
inline void CuckooSet<Key,Hash>::addTo(uint side, uint n_tries, Key key)
{
    uint64 hkey = shuffleHash(hash.hash(key));
    uint r0 = reg0(hkey);
    uint r1 = reg1(hkey);
    uint r = side ? r1 : r0;
    uchar abst = shift(hkey);
    abst += (abst == -abst);    // -- reserve '0' for no element.
    size_t i = (hkey % reg[r].cap);

    for (uint b = 0; b < n_bins; b++){
        if (reg[r].abst[i+b] == 0){
            reg[r].abst[i+b] = abst;
            reg[r].data[i+b] = key;
            reg[r].sz++;
            tries_left += allowed_avg_tries;
            newMin(tries_left, max_tries);
            return;
        }
    }

    if (tries_left <= 0 && (reg[r0].prime_idx <= (sum_prime_idx >> REGPOW) || reg[r1].prime_idx <= (sum_prime_idx >> REGPOW) || tries_left < -100)){
        /**/assert(tries_left > -100);      // -- not sure if this will ever happen; keep an assert for now
        grow_rec++;
        if (grow_rec >= allowed_grow_rec && capacity() > 10000){
            ShoutLn "INTERNAL ERROR! Bad hash function detected in 'CuckooSet'.";
            assert(false); }

        tries_left = start_tries;
        grow(r0, r1);
        addTo(0, 0u, key);
        grow_rec--;
    }else{
        tries_left -= 1;
        reg[r].abst[i] = abst;
        swp(key, reg[r].data[i]);
        addTo(1-side, n_tries+1, key);
    }
}


template<class Key, class Hash>
inline Key CuckooSet<Key,Hash>::addWeak(Key key)
{
    uint64 hkey = shuffleHash(hash.hash(key));
    uint r0 = reg0(hkey);
    uint r1 = reg1(hkey);
    uchar abst = shift(hkey);
    abst += (abst == -abst);    // -- reserve '0' for no element.
    size_t i0 = (hkey % reg[r0].cap);
    size_t i1 = (hkey % reg[r1].cap);

    for (uint b = 0; b < n_bins; b++) if (reg[r0].abst[i0+b] == abst && hash.equal(key, reg[r0].data[i0+b])) return reg[r0].data[i0+b];
    for (uint b = 0; b < n_bins; b++) if (reg[r1].abst[i1+b] == abst && hash.equal(key, reg[r1].data[i1+b])) return reg[r1].data[i1+b];

    for (uint b = 0; b < n_bins; b++) if (reg[r0].abst[i0+b] == 0){ reg[r0].abst[i0+b] = abst; reg[r0].data[i0+b] = key; reg[r0].sz++; return key; }
    for (uint b = 0; b < n_bins; b++) if (reg[r1].abst[i1+b] == 0){ reg[r1].abst[i1+b] = abst; reg[r1].data[i1+b] = key; reg[r1].sz++; return key; }

    Key ret = key;
    reg[r0].abst[i0] = abst;
    swp(key, reg[r0].data[i0]);
    addTo(1, 0u, key);
    return ret;
}


template<class Key, class Hash>
inline void CuckooSet<Key,Hash>::grow(uint r0, uint r1)
{
    // Pick the smallest region of 'r0' and 'r1' for grow operation; grow it beyond the largest of all regions:
    uint r = (reg[r0].prime_idx < reg[r1].prime_idx) ? r0 : r1;
    uint new_idx = (r0 == r1) ? r0 : reg[r].prime_idx + grow_steps;     // -- constructor will set 'r0 == r1'; initialize with small primes
    sum_prime_idx += new_idx - reg[r].prime_idx;
    Region& me = reg[r];
    assert(new_idx < n_cuckoo_primes);

    // Copy existing keys to small area:
    size_t keys_sz = me.sz;
    Key* keys = new Key[keys_sz];
    size_t j = 0;
    if (me.cap != 0){
        for (size_t i = 0; i < me.cap + n_bins-1; i++)
            if (me.abst[i] != 0)
                keys[j++] = me.data[i];
    }
    assert(j == keys_sz);

    // Allocate new region:
    me.prime_idx = new_idx;
    me.cap = cuckoo_primes[me.prime_idx];
    me.sz = 0;
    delete [] me.data;
    delete [] me.abst;
    me.data = new Key  [me.cap + n_bins-1];
    me.abst = new uchar[me.cap + n_bins-1]();

    // Populate the new region
    for (size_t i = 0; i < keys_sz; i++)
        addWeak(keys[i]);
    delete [] keys;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

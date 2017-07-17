#include ZZ_Prelude_hh
#include "PArr.hh"
#include <vector>
#include <memory>

using namespace ZZ;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


struct MyInt {
    uint n;
    uint waste[15];
    MyInt(uint n = 0) : n(n) {}
    operator uint() const { return n; }
};


int main(int argc, char** argv)
{
    ZZ_Init;

#if 0
    // Simple test:
    PArr<int> arr;
    arr = arr.push(1);
    arr = arr.push(2);
    arr = arr.push(3);
    PArr<int> arr2 = arr.pop();
    Dump(arr);
    Dump(arr2);
    arr2 = arr2.set(100, 42);
    Dump(arr2);

    wr("numbers:"); arr2.enumAll([](uind idx, int n){ wr(" %_=%_", idx, n); return true; }); newLn();
#endif

#if 0
    // Correctness test:
    uint64 seed = 0;
  #if 0
    Vec<PArr<int>> arrs(10);
  #else
    Vec<vector<int>> arrs(10, vector<int>(30));
  #endif
    for (uint n = 0; n < 1000000; n++){
        uint i = irand(seed, arrs.size());
        uint j = irand(seed, 30);
        uint v = irand(seed, 10000 - 1000) + 1000;
        //**/wrLn("arrs[%_].set(%_, %_)", i, j, v);
      #if 0
        arrs[i] = arrs[i].set(j, v);
      #else
        arrs[i][j] = v;
      #endif
        //**/wr("  -> "); arrs[i].dump(); newLn();

        if (n % 1000 == 0){
            uint i0 = irand(seed, arrs.size());
            uint i1 = irand(seed, arrs.size());
            arrs[i0] = arrs[i1];
        }
    }

    wrLn("%\n_", arrs);
#endif

#if 0
    uint64 seed = 0;
    Vec<PArr<MyInt, 8, 2>> arrs(10);

    // Speed test:
    uint N = 1000000;
    uint sz = 30000;
    double T0 = realTime();
    for (uint n = 0; n < N; n++){
        uint i = irand(seed, arrs.size());
        uint j = irand(seed, sz);
        uint v = irand(seed, 10000 - 1000) + 1000;
        arrs[i] = arrs[i].set(j, v);

        if (n % 1000 == 0){
            uint i0 = irand(seed, arrs.size());
            uint i1 = irand(seed, arrs.size());
            arrs[i0] = arrs[i1];
        }
    }
    double T1 = realTime();

    wrLn("Cycles/write: %_", (T1-T0) / N * 2.4e9);
#endif

    // Iteration speed test:
    uint N = 10000;
    PArr<MyInt,4> arr;
    for (uint i = 0; i < 20; i++)
        arr = arr.push(i);

    {
        double T0 = realTime();
        uint64 sum = 0;
        for (uint n = 0; n < N; n++){
            for (uint i = 0; i < arr.size(); i++)
                sum += arr[i];
        }
        double T1 = realTime();
        wrLn("LOOP:  sum=%_   time=%t (%.0f cycles)", sum, (T1-T0) / N, (T1-T0) / N * 2.4e9);
    }
    {
        double T0 = realTime();
        uint64 sum = 0;
        for (uint n = 0; n < N; n++){
            arr.forAllCondRev([&](MyInt const& v) {
                sum += v;
                return true;
            });
        }
        double T1 = realTime();
        wrLn("ITER:  sum=%_   time=%t (%.0f cycles)", sum, (T1-T0) / N, (T1-T0) / N * 2.4e9);
    }

    return 0;
}


/*
[1195, 8025, 2330, 6909, 6214, 9379, 9308, 8788, 6198, 4952, 1802, 1087, 2321, 4546, 8536, 4030, 8695, 5455, 8509, 4664, 7999, 1533, 6433, 2418, 1532, 7387, 5372, 7431, 6036, 1711]
[4240, 6760, 7950, 3579, 8589, 4009, 4333, 6233, 4888, 2592, 3177, 4712, 2216, 2231, 8506, 8830, 3885, 4125, 1849, 3144, 6444, 1598, 4723, 8338, 5192, 5247, 5897, 1121, 5161, 2486]
[6340, 1005, 2170, 1384, 8589, 4094, 5023, 7018, 2393, 2592, 2692, 6242, 9446, 5686, 8506, 9150, 3390, 7975, 5414, 3144, 4494, 9868, 8118, 6838, 5192, 4592, 5272, 5721, 3096, 2486]
[4960, 8950, 6005, 9649, 5194, 2984, 3743, 6043, 8173, 9317, 7917, 1697, 5041, 5531, 4526, 1140, 9185, 9435, 5454, 7469, 7579, 4693, 4933, 9068, 2067, 8467, 2997, 3251, 1281, 7426]
[9030, 5110, 5550, 3409, 2909, 8284, 5978, 1398, 5088, 4952, 4547, 9207, 9046, 9296, 1256, 9580, 5650, 5710, 9434, 9224, 7044, 3608, 2148, 8418, 5637, 9242, 7357, 1721, 5256, 4016]
[3940, 7485, 5575, 6754, 5044, 5739, 7138, 6693, 1053, 2162, 3827, 8352, 2326, 2851, 8001, 1225, 4220, 8410, 9444, 9059, 1129, 1023, 1913, 3618, 9047, 5147, 6047, 3556, 6601, 3261]
[2985, 5480, 5640, 8119, 9344, 1814, 6538, 7908, 8088, 1667, 8417, 5317, 7841, 5341, 5726, 4895, 9170, 1285, 1214, 6434, 2349, 4898, 4978, 1773, 8462, 2637, 5712, 5256, 1516, 5826]
[6585, 1125, 2105, 3629, 1579, 8169, 7858, 4408, 3173, 9277, 8087, 9712, 9541, 9271, 7776, 1260, 5275, 7905, 9459, 5024, 9349, 6183, 8963, 9993, 2837, 4602, 1357, 1581, 2661, 3931]
[1775, 8070, 9145, 4074, 4729, 3974, 8663, 1203, 1158, 2632, 8697, 8402, 1336, 1916, 4556, 5855, 2115, 5275, 1069, 3439, 3709, 8618, 5438, 3203, 4137, 8992, 2742, 9881, 7821, 7611]
[4960, 4425, 1625, 6909, 1364, 2984, 1343, 4713, 6198, 8852, 7917, 9827, 1441, 4546, 1831, 1140, 7655, 8180, 8509, 2579, 7579, 9238, 7268, 2418, 1222, 8467, 5837, 2051, 6036, 6851]
*/

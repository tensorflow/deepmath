#include "Prelude.hh"

using namespace ZZ;


int main(int argc, char** argv)
{
    ZZ_Init;

    wrLn("Hello world. Time to crash...");

    Vec<int> v(10);
    v[100] = 42;    // -- this should throw an assertion in debug mode but not in release mode

    return 0;
}

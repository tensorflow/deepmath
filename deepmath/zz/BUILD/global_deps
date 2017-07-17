#!/bin/bash

D=`dirname $0`
printf '#include "Prelude.hh"\nusing namespace ZZ;\nint main(int argc, char** argv){ ZZ_Init; return 0; }' > Main___INIT__.cc
$D/zb q
ZZ_EMIT_GLOBAL_DEP= ./__INIT__
rm -f Main___INIT__.cc
//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Prelude.hh
//| Author(s)   : Niklas Een
//| Module      : Prelude
//| Description : Background library included in every ZZ project.
//|
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//|
//|________________________________________________________________________________________________

//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
/*


Defines that control compilation:
  ZZ_DEBUG            -- Compile in debug mode ('assert_debug' is defined as 'assert', else empty).
  ZZ_BIG_MODE         -- Let 'ind' be 64-bit to support large container classes.
  ZZ_PTHREADS         -- Enable multi-threading.
  ZZ_NO_EXCEPTIONS    -- Don't throw exception in prelude (have failing 'assert()' instead)
  ZZ_NO_SMART_YMALLOC -- Forward 'ymalloc()' (used eg. by 'Vec<T>') to regular 'malloc()'


  LessThan_default<T>    }
  MkIndex_default<T>     }- Passed by value
  Hash_default<T>        }

  NonCopyable (inherit from)

  swp(x, y), mov(x, y), cpy(x, y)


*/
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


#ifndef ZZ__Prelude__Main_hh
#define ZZ__Prelude__Main_hh


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Include core of standard C++ library:


#ifdef GOOGLE_CODE
  #include /*no-mangling*/ "base/integral_types.h"
#else
  #if !defined(_MSC_VER)
    #ifndef __STDC_LIMIT_MACROS
      #define __STDC_LIMIT_MACROS
    #endif
    #include <stdint.h>
  #endif
#endif

#if defined(__BAZEL_BUILD__)
  #if defined(NDEBUG)
    #undef NDEBUG
  #else
    #define ZZ_DEBUG
  #endif
#endif

#include <cassert>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <cfloat>
#include <cctype>
#include <ctime>
#include <cmath>
#include <csignal>
#include <new>
#include <memory>   // -- for 'shared_ptr'

#if __cplusplus >= 201103L
  #include <functional>
  template <class F>
  using Fun = std::function<F>;
#endif

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>

#if !defined(_MSC_VER)
  #include <unistd.h>
  #include <libgen.h>
  #include <sys/resource.h>
  #include <sys/ioctl.h>
  #include <sys/time.h>
  #if defined(__linux__)
    #include <fpu_control.h>
    #include <sys/prctl.h>
  #endif
  #if defined(sun)
    #include <procfs.h>
    #include <strings.h>
    #include <stdlib.h>
  #endif
#else
  #define NOMINMAX
  #include <Windows.h>
  #include <WinBase.h>
  #include <io.h>
  #include <psapi.h>
  #include <process.h>
  #include <time.h>
  #include <direct.h>
  typedef int ssize_t;
#endif

#if defined(ZZ_PTHREADS)
  #include <pthread.h>
  #if !defined(ZZ_NO_SMART_YMALLOC)
    #define ZZ_NO_SMART_YMALLOC             // -- not thread-safe
  #endif
#endif

#include <zlib.h>


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Include ZZ prelude:


#include "Init.ihh"
#include "Macros.ihh"
#include "CompilerExtensions.ihh"
#include "PrimitiveTypes.ihh"
#include "Traits.ihh"
#include "SwapMoveCopy.ihh"
#include "Compare.ihh"
#include "LBool.ihh"
#include "Threads.ihh"
#include "Signals.ihh"
#include "Mem.ihh"
#include "Hash.ihh"
#include "Tuples.ihh"
#include "Array.ihh"
#include "Vec.ihh"
#include "VecAlgo.ihh"
#include "Streams.ihh"
#include "String.ihh"
#include "Exceptions.ihh"
#include "File.ihh"
#include "Resources.ihh"
#include "Console.ihh"
#include "Utility.ihh"
#include "Format.ihh"
#include "Profile.ihh"
#if !defined(__BAZEL_BUILD__)
#include "ScopeGuard.ihh"
#include "ScopedPtr.ihh"
#endif

//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


/*
- Prefix all prelude initializers/finalizers with 'PRELUDE__'.
- Program exit handler
- Build system
(- pthreads stuff)
*/


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
#endif

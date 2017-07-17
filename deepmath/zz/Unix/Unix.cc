//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Unix.cc
//| Author(s)   : Niklas Een
//| Module      : Unix
//| Description : Wrappers for miscellaneous Unix functions.
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| 
//|________________________________________________________________________________________________

#include ZZ_Prelude_hh
#include "Unix.hh"
#include <pwd.h>

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


String homeDir()
{
    int bufsize = sysconf(_SC_GETPW_R_SIZE_MAX); assert(bufsize > 0);
    char* buf = xmalloc<char>(bufsize);
    On_Scope_Exit(xfree<char>, buf);

    struct passwd  pw;
    struct passwd* result = NULL;
    if (getpwuid_r(getuid(), &pw, buf, bufsize, &result) != 0 || result == NULL)
        return String();
    else
        return String(pw.pw_dir);
}


String homeDir(String username)
{
    int bufsize = sysconf(_SC_GETPW_R_SIZE_MAX); assert(bufsize > 0);
    char* buf = xmalloc<char>(bufsize);
    On_Scope_Exit(xfree<char>, buf);

    struct passwd  pw;
    struct passwd* result = NULL;
    if (getpwnam_r(username.c_str(), &pw, buf, bufsize, &result) != 0 || result == NULL)
        return String();
    else
        return String(pw.pw_dir);
}


String userName()
{
    int bufsize = sysconf(_SC_GETPW_R_SIZE_MAX); assert(bufsize > 0);
    char* buf = xmalloc<char>(bufsize);
    On_Scope_Exit(xfree<char>, buf);

    struct passwd  pw;
    struct passwd* result = NULL;
    if (getpwuid_r(getuid(), &pw, buf, bufsize, &result) != 0 || result == NULL)
        return String();
    else
        return String(pw.pw_name);
}


bool statFileExists(String filename)
{
    struct stat buf;
    int ret = stat(filename.c_str(), &buf);
    return ret == 0;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Unix.hh
//| Author(s)   : Niklas Een
//| Module      : Unix
//| Description : Wrappers for miscellaneous Unix functions.
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| 
//|________________________________________________________________________________________________

#ifndef ZZ__Unix__Unix_hh
#define ZZ__Unix__Unix_hh

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


String userName();
String homeDir();
String homeDir(String username);
bool   statFileExists(String filename);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

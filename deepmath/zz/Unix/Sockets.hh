//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Sockets.hh
//| Author(s)   : Niklas Een
//| Module      : Unix
//| Description : 
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| 
//|________________________________________________________________________________________________

#ifndef ZZ__Unix__Sockets_hh
#define ZZ__Unix__Sockets_hh

#include <netinet/in.h>

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


bool getHostIpaddr(String hostname, struct in_addr* out_addr);
int  setupSocket(int port);
int  acceptConnection(int sock_fd);
int  connectToSocket(char* hostname, int port);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

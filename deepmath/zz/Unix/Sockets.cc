//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Sockets.cc
//| Author(s)   : Niklas Een
//| Module      : Unix
//| Description : 
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| 
//|________________________________________________________________________________________________

#include ZZ_Prelude_hh
#include "Sockets.hh"

#include <netdb.h>
#if defined(__APPLE__)
#include <sys/socket.h>
#endif

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


bool getHostIpaddr(String hostname, struct in_addr* out_addr)
{
    struct hostent* entry;

    entry = gethostbyname(hostname.c_str());
    if (entry == NULL)
        return false;
    else{
        *out_addr = *(struct in_addr*)entry->h_addr;
        return true;
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Server side -- hosting a connection:


// Sets up a socket for incoming connection on the given port. Use 'select()' on the returned
// file descriptor to wait for a connection, the call 'acceptConnection()' on that descriptor.
int setupSocket(int port)
{
    int                 sock_fd;
    struct sockaddr_in  addr;
    struct in_addr      iaddr;
    int                 ret;

    sock_fd = socket(PF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0)
        perror("Call to 'socket()'"),
        exit(-1);

    int value = 1;
    if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &value, 4) < 0){
        perror("Call to 'setsockopt()'");
        exit(-1); }

    iaddr.s_addr    = INADDR_ANY;
    addr.sin_family = AF_INET;
    addr.sin_addr   = iaddr;
    addr.sin_port   = htons(port);

    ret = ::bind(sock_fd, (struct sockaddr *)&addr, sizeof(struct sockaddr_in));
    if (ret < 0)
        perror("Call to 'bind()'"),
        exit(-1);

    ret = listen(sock_fd, 5);
    if (ret < 0)
        perror("Call to 'listen()'"),
        exit(-1);

    return sock_fd;
}


// Accepts an incoming connection and returns a full-duplex file descriptor.
int acceptConnection(int sock_fd)
{
    struct sockaddr addr;
    socklen_t       addr_len = sizeof(addr);
    int             ret;

    ret = accept(sock_fd, &addr, &addr_len);   // (blocking)
    if (ret < 0)
        perror("Call to 'accept()'"),
        exit(-1);

    return ret;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Client side -- connecting to the server:


// Returns a file descriptor or -1 on failure.
int connectToSocket(char* hostname, int port)
{
    int                 sock;
    struct sockaddr_in  addr;
    struct in_addr      iaddr;
    int                 ret;

    sock = socket(PF_INET, SOCK_STREAM, 0);
    if (sock < 0)
        perror("Call to 'socket()'"),
        exit(-1);

    if (!getHostIpaddr(hostname, &iaddr))
        return -1;

    addr.sin_family = AF_INET;
    addr.sin_addr   = iaddr;
    addr.sin_port   = htons(port);

    ret = connect(sock, (struct sockaddr *)&addr, sizeof(struct sockaddr_in));
    if (ret >= 0)
        return sock;
    else{
        close(sock);
        return -1;
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

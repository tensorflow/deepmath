//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Processes.hh
//| Author(s)   : Niklas Een
//| Module      : Unix
//| Description : Wrappers for process handling.
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| 
//|________________________________________________________________________________________________

#ifndef ZZ__Unix__Processes_hh
#define ZZ__Unix__Processes_hh

#include <sys/types.h>
#include <sys/wait.h>

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


struct ProcMode {
    String username;        // Try to become this user (makes sense if you are running as root)
    String dir;             // Will set current working directory
    bool   own_group;       // Process will be the head of a new process group
    int    pdeath_sig;      // Linux only: send this signal when parent dies
    double timeout;         // Set rlimit to this time
    uint64 memout;          // Set rlimit to this amount of memory
    String stdin_file;      // If provided, 'out_std[0]' is set to '-1'. Non-absolute paths are relative to 'dir'.
    String stdout_file;     // If provided, 'out_std[1]' is set to '-1'. }
    String stderr_file;     // If provided, 'out_std[2]' is set to '-1'. }- can be the same file 
                            // Use filename "-" to keep parents file descriptor

    ProcMode(String username_ = "", String dir_ = "", bool own_group_ = false, int pdeath_sig_ = 0,
             double timeout_ = DBL_MAX, uint64 memout_ = UINT64_MAX,
             String stdin_file_ = "", String stdout_file_ = "", String stderr_file_ = "") :
        username(username_), dir(dir_), own_group(own_group_), pdeath_sig(pdeath_sig_),
        timeout(timeout_), memout(memout_),
        stdin_file(stdin_file_), stdout_file(stdout_file_), stderr_file(stderr_file_) {}
};


// NOTE! Don't forget to 'waitpid(pid, &status, 0)' for process!
//
char startProcess(const String& exec, const Vec<String>& args, int& out_pid, int out_std[3], const ProcMode& mode = ProcMode());
char startProcess(const String& exec, const Vec<String>& args, int& out_pid, int out_std[3], const Vec<String>& env, const ProcMode& mode = ProcMode());
    // Prefix 'exec' with a '*' to search the path for executable.
    // Return values:
    //   '\0' = successful
    //   'x'  = could not run executable
    //   'u'  = could not change user to 'username'
    //   'd'  = could not set current directory to 'dir'
    //   't'  = could not set timeout
    //   'm'  = could not set memout
    //   'i'  = could not redirect stdin to file
    //   'o'  = could not redirect stdout to file
    //   'e'  = could not redirect stderr to file

void closeChildIo(int out_std[3]);

struct Process {
    int   fd[3];    // -- to store 'out_std'
    pid_t pid;      // -- to store 'out_pid'
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Processes.cc
//| Author(s)   : Niklas Een
//| Module      : Unix
//| Description : Wrappers for process handling.
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| 
//|________________________________________________________________________________________________

#include ZZ_Prelude_hh
#include "Processes.hh"
#include <pwd.h>
#include <grp.h>

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


static
bool setUser(const String& username)
{
    int bufsize = sysconf(_SC_GETPW_R_SIZE_MAX); assert(bufsize > 0);
    char* buf = xmalloc<char>(bufsize);
    On_Scope_Exit(xfree<char>, buf);

    struct passwd  pw;
    struct passwd* result = NULL;
    if (getpwnam_r(username.c_str(), &pw, buf, bufsize, &result) != 0 || result == NULL)
        return false;
    if (setegid(pw.pw_gid) == -1)
        return false;
    if (initgroups(pw.pw_name, pw.pw_gid) == -1)
        return false;
    if (seteuid(pw.pw_uid) == -1)
        return false;
    if (chdir(pw.pw_dir) == -1)
        return false;
    return true;
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


static
void setCloseOnExec(int fd)
{
    int old_flags = fcntl(fd, F_GETFD, 0);
    if (old_flags != -1)
        fcntl(fd, F_SETFD, old_flags | FD_CLOEXEC);
}


// Close all file descriptors except 'fd' and: STDIN, STDOUT, STDERR
static
void closeAllBut(int fd)
{
    int n = sysconf(_SC_OPEN_MAX);
    for (int i = 3; i < n; i++)
        if (i != fd)
            close(i);
}


static
bool connectFd(int fd, const String& filename, bool write)
{
    if (eq(filename, "-"))      // -- keep 'fd' intact
        return true;

    int new_fd = write ? open(filename.c_str(), O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH)
                       : open(filename.c_str(), O_RDONLY);

    if (new_fd == -1)
        return false;

    dup2(new_fd, fd);
    close(new_fd);
    return true;
}


static
void fail(int signal_pipe[2], char err)
{
    int tmp ___unused = write(signal_pipe[1], &err, 1);
    _exit(255);
}


// If 'cmd' starts with a '*' then 'execvp' is used (i.e. search the PATH). Output argument
// 'out_pid' will be set to the process ID of the child. 'out_std' will contain pipes connected to
// the child processes' stdin, stdout and stderr (in that order). NOTE! All three file descriptors
// need to be closed by parent.
//
char startProcess(const String& cmd, const Vec<String>& args, int& out_pid, int out_std[3], const ProcMode& mode)
{
    int res;
    int stdin_pipe [2] = {-1, -1};
    int stdout_pipe[2] = {-1, -1};
    int stderr_pipe[2] = {-1, -1};
    int signal_pipe[2];

    if (mode.stdin_file  == ""){ res = pipe(stdin_pipe ); assert(res == 0); }
    if (mode.stdout_file == ""){ res = pipe(stdout_pipe); assert(res == 0); }
    if (mode.stderr_file == ""){ res = pipe(stderr_pipe); assert(res == 0); }
    res = pipe(signal_pipe); assert(res == 0);

    setCloseOnExec(signal_pipe[0]);
    setCloseOnExec(signal_pipe[1]);

    pid_t child_pid = fork(); assert(child_pid != -1);

    if (child_pid == 0){
        //
        // CHILD:
        //
      #if defined(__linux__)
        if (mode.pdeath_sig != 0)
            prctl(PR_SET_PDEATHSIG, mode.pdeath_sig);
      #endif

        // Setup process mode:
        if (mode.username != "" && !setUser(mode.username)) fail(signal_pipe, 'u');
        if (mode.dir != "" && chdir(mode.dir.c_str()) == -1) fail(signal_pipe, 'd');

        struct rlimit lim;
        if (mode.timeout != DBL_MAX){
            lim.rlim_cur = lim.rlim_max = (mode.timeout > ~rlim_t(0)) ? ~rlim_t(0) : mode.timeout;
            if (setrlimit(RLIMIT_CPU, &lim) == -1) fail(signal_pipe, 't'); }

        if (mode.memout != UINT64_MAX){
            lim.rlim_cur = lim.rlim_max = (mode.memout > ~rlim_t(0)) ? ~rlim_t(0) : mode.memout;
            if (setrlimit(RLIMIT_AS, &lim) == -1) fail(signal_pipe, 'm'); }

        lim.rlim_cur = lim.rlim_max = 0;    // -- no core-files
        setrlimit(RLIMIT_CORE, &lim);

        if (mode.own_group)
            setpgrp();  // -- move to its own process group

        // Redirect standard streams:
        if (mode.stdin_file == ""){
            dup2(stdin_pipe [0], STDIN_FILENO);
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
        }else if (!connectFd(STDIN_FILENO, mode.stdin_file, false))
            fail(signal_pipe, 'i');

        if (mode.stdout_file == ""){
            dup2(stdout_pipe[1], STDOUT_FILENO);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
        }else if (!connectFd(STDOUT_FILENO, mode.stdout_file, true))
            fail(signal_pipe, 'o');

        if (mode.stderr_file == ""){
            dup2(stderr_pipe[1], STDERR_FILENO);
            close(stderr_pipe[0]);
            close(stderr_pipe[1]);
        }else if (eq(mode.stdout_file, mode.stderr_file)){
            dup2(STDOUT_FILENO, STDERR_FILENO);
            close(stderr_pipe[0]);
            close(stderr_pipe[1]);
        }else if (!connectFd(STDERR_FILENO, mode.stderr_file, true))
            fail(signal_pipe, 'e');

        closeAllBut(signal_pipe[1]);

        // Call executable:
        bool use_path = (cmd.size() > 0 && cmd[0] == '*');
        char* exec_cmd = cmd.c_str() + use_path;
        char** exec_args = xmalloc<char*>(args.size() + 2);
        exec_args[0] = exec_cmd;
        for (uint i = 0; i < args.size(); i++)
            exec_args[i+1] = args[i].c_str();
        exec_args[args.size() + 1] = NULL;

        if (use_path)
            res = execvp(exec_cmd, exec_args);
        else
            res = execv(exec_cmd, exec_args);

        // Exec failed, let parent know:
        assert(res == -1);
        fail(signal_pipe, 'x');
    }

    // Parent continues:
    if (stdin_pipe [0] != -1) close(stdin_pipe [0]);
    if (stdout_pipe[1] != -1) close(stdout_pipe[1]);
    if (stderr_pipe[1] != -1) close(stderr_pipe[1]);
    close(signal_pipe[1]);

    out_pid = child_pid;
    out_std[0] = stdin_pipe[1];
    out_std[1] = stdout_pipe[0];
    out_std[2] = stderr_pipe[0];

    // Did child execute 'exec' successfully?
    char buf[1];
    bool success = (read(signal_pipe[0], buf, sizeof(buf)) == 0);
    close(signal_pipe[0]);

    return success ? 0 : buf[0];
}


// Each element in 'env' should be on form "key=value" or just "key" (an entry "key" will unset that key).
//
char startProcess(const String& cmd, const Vec<String>& args, int& out_pid, int out_std[3], const Vec<String>& env, const ProcMode& mode)
{
    Vec<String> clear;
    Vec<Pair<String,String> > restore;

    for (uint i = 0; i < env.size(); i++){
        char* key = const_cast<String&>(env[i]).c_str();
        char* p = strchr(key, '=');
        if (p == NULL){
            char* val = getenv(key);
            if (val != NULL){
                restore.push(tuple(String(key), String(val)));
                unsetenv(key);
            }
        }else{
            *p = '\0';
            char* val = getenv(key);
            if (val != NULL)
                restore.push(tuple(String(key), String(val)));
            else
                clear.push(String(key));

            setenv(key, p+1, 1);
            *p = '=';
        }
    }

    char ret = startProcess(cmd, args, out_pid, out_std, mode);

    for (uint i = 0; i < clear.size(); i++){
        unsetenv(clear[i].c_str()); }
    for (uint i = 0; i < restore.size(); i++){
        setenv(restore[i].fst.c_str(), restore[i].snd.c_str(), 1); }

    return ret;
}


void closeChildIo(int out_std[3])
{
    if (out_std[0] != -1) close(out_std[0]);
    if (out_std[1] != -1) close(out_std[1]);
    if (out_std[2] != -1) close(out_std[2]);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

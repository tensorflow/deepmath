//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Linux.cc
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


namespace ZZ {
using namespace std;


/*
/proc/[pid]/stat

    Status information about the process.  This is used by ps(1).
    It is defined in /usr/src/linux/fs/proc/array.c.

    The fields, in order, with their proper scanf(3) format
    specifiers, are:

    pid %d      (1) The process ID.

    comm %s     (2) The filename of the executable, in
                parentheses.  This is visible whether or not the
                executable is swapped out.

    state %c    (3) One character from the string "RSDZTW" where R
                is running, S is sleeping in an interruptible
                wait, D is waiting in uninterruptible disk sleep,
                Z is zombie, T is traced or stopped (on a signal),
                and W is paging.

    ppid %d     (4) The PID of the parent.

    pgrp %d     (5) The process group ID of the process.

    session %d  (6) The session ID of the process.

    tty_nr %d   (7) The controlling terminal of the process.  (The
                minor device number is contained in the
                combination of bits 31 to 20 and 7 to 0; the major
                device number is in bits 15 to 8.)

    tpgid %d    (8) The ID of the foreground process group of the
                controlling terminal of the process.

    flags %u (%lu before Linux 2.6.22)
                (9) The kernel flags word of the process.  For bit
                meanings, see the PF_* defines in the Linux kernel
                source file include/linux/sched.h.  Details depend
                on the kernel version.

    minflt %lu  (10) The number of minor faults the process has
                made which have not required loading a memory page
                from disk.

    cminflt %lu (11) The number of minor faults that the process's
                waited-for children have made.

    majflt %lu  (12) The number of major faults the process has
                made which have required loading a memory page
                from disk.

    cmajflt %lu (13) The number of major faults that the process's
                waited-for children have made.

    utime %lu   (14) Amount of time that this process has been
                scheduled in user mode, measured in clock ticks
                (divide by sysconf(_SC_CLK_TCK)).  This includes
                guest time, guest_time (time spent running a
                virtual CPU, see below), so that applications that
                are not aware of the guest time field do not lose
                that time from their calculations.

    stime %lu   (15) Amount of time that this process has been
                scheduled in kernel mode, measured in clock ticks
                (divide by sysconf(_SC_CLK_TCK)).

    cutime %ld  (16) Amount of time that this process's waited-for
                children have been scheduled in user mode,
                measured in clock ticks (divide by
                sysconf(_SC_CLK_TCK)).  (See also times(2).)  This
                includes guest time, cguest_time (time spent
                running a virtual CPU, see below).

    cstime %ld  (17) Amount of time that this process's waited-for
                children have been scheduled in kernel mode,
                measured in clock ticks (divide by
                sysconf(_SC_CLK_TCK)).

    priority %ld
                (18) (Explanation for Linux 2.6) For processes
                running a real-time scheduling policy (policy
                below; see sched_setscheduler(2)), this is the
                negated scheduling priority, minus one; that is, a
                number in the range -2 to -100, corresponding to
                real-time priorities 1 to 99.  For processes
                running under a non-real-time scheduling policy,
                this is the raw nice value (setpriority(2)) as
                represented in the kernel.  The kernel stores nice
                values as numbers in the range 0 (high) to 39
                (low), corresponding to the user-visible nice
                range of -20 to 19.

                Before Linux 2.6, this was a scaled value based on
                the scheduler weighting given to this process.

    nice %ld    (19) The nice value (see setpriority(2)), a value
                in the range 19 (low priority) to -20 (high
                priority).

    num_threads %ld
                (20) Number of threads in this process (since
                Linux 2.6).  Before kernel 2.6, this field was
                hard coded to 0 as a placeholder for an earlier
                removed field.

    itrealvalue %ld
                (21) The time in jiffies before the next SIGALRM
                is sent to the process due to an interval timer.
                Since kernel 2.6.17, this field is no longer
                maintained, and is hard coded as 0.

    starttime %llu (was %lu before Linux 2.6)
                (22) The time the process started after system
                boot.  In kernels before Linux 2.6, this value was
                expressed in jiffies.  Since Linux 2.6, the value
                is expressed in clock ticks (divide by
                sysconf(_SC_CLK_TCK)).

    vsize %lu   (23) Virtual memory size in bytes.

    rss %ld     (24) Resident Set Size: number of pages the
                process has in real memory.  This is just the
                pages which count toward text, data, or stack
                space.  This does not include pages which have not
                been demand-loaded in, or which are swapped out.

    rsslim %lu  (25) Current soft limit in bytes on the rss of the
                process; see the description of RLIMIT_RSS in
                getrlimit(2).

    startcode %lu
                (26) The address above which program text can run.

    endcode %lu (27) The address below which program text can run.

    startstack %lu
                (28) The address of the start (i.e., bottom) of
                the stack.

    kstkesp %lu (29) The current value of ESP (stack pointer), as
                found in the kernel stack page for the process.

    kstkeip %lu (30) The current EIP (instruction pointer).

    signal %lu  (31) The bitmap of pending signals, displayed as a
                decimal number.  Obsolete, because it does not
                provide information on real-time signals; use
                /proc/[pid]/status instead.

    blocked %lu (32) The bitmap of blocked signals, displayed as a
                decimal number.  Obsolete, because it does not
                provide information on real-time signals; use
                /proc/[pid]/status instead.

    sigignore %lu
                (33) The bitmap of ignored signals, displayed as a
                decimal number.  Obsolete, because it does not
                provide information on real-time signals; use
                /proc/[pid]/status instead.

    sigcatch %lu
                (34) The bitmap of caught signals, displayed as a
                decimal number.  Obsolete, because it does not
                provide information on real-time signals; use
                /proc/[pid]/status instead.

    wchan %lu   (35) This is the "channel" in which the process is
                waiting.  It is the address of a system call, and
                can be looked up in a namelist if you need a
                textual name.  (If you have an up-to-date
                /etc/psdatabase, then try ps -l to see the WCHAN
                field in action.)

    nswap %lu   (36) Number of pages swapped (not maintained).

    cnswap %lu  (37) Cumulative nswap for child processes (not
                maintained).

    exit_signal %d (since Linux 2.1.22)
                (38) Signal to be sent to parent when we die.

    processor %d (since Linux 2.2.8)
                (39) CPU number last executed on.

    rt_priority %u (since Linux 2.5.19; was %lu before Linux
    2.6.22)
                (40) Real-time scheduling priority, a number in
                the range 1 to 99 for processes scheduled under a
                real-time policy, or 0, for non-real-time
                processes (see sched_setscheduler(2)).

    policy %u (since Linux 2.5.19; was %lu before Linux 2.6.22)
                (41) Scheduling policy (see
                sched_setscheduler(2)).  Decode using the SCHED_*
                constants in linux/sched.h.

    delayacct_blkio_ticks %llu (since Linux 2.6.18)
                (42) Aggregated block I/O delays, measured in
                clock ticks (centiseconds).

    guest_time %lu (since Linux 2.6.24)
                (43) Guest time of the process (time spent running
                a virtual CPU for a guest operating system),
                measured in clock ticks (divide by
                sysconf(_SC_CLK_TCK)).

    cguest_time %ld (since Linux 2.6.24)
                (44) Guest time of the process's children,
                measured in clock ticks (divide by
                sysconf(_SC_CLK_TCK)).

Example:

    23238 (my term) S 23237 23232 7056 34821 23232 4202496 1878 661 0 0 37 69 0 0 20 0 1 0 206087792 10289152 1061 4294967295 134512640 134892160 3220396928 3220395856 3077469220 0 0 2097153 86022 3239404686 0 0 17 2 0 0 0 0 0

*/


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
#if !defined(__linux__)


uint64 getProcStatField(int pid, uint field)
{
    assert(false);
}


//=================================================================================================
#else


// Fields are numbered as above (starting from 1). Specifying field 2 is illegal since it is not
// numerical. Returns 'UINT64_MAX' if illegal pid or field is given.
uint64 getProcStatField(int pid, uint field)
{
    if (field == 2)
        return UINT64_MAX;

    String filename;
    FWrite(filename) "/proc/%_/stat", pid;
    Str text = readFile(filename);
    if (!text)
        return UINT64_MAX;

    if (field == 1){
        uind p = search(text, ' ');
        if (p == UIND_MAX) return UINT64_MAX;
        return stringToUInt64(text.slice(0, p));

    }else{
        assert(field >= 3);
        uind p = search(text, ')');
        if (p == UIND_MAX) return UINT64_MAX;

        Vec<Str> fs;
        splitArray(text.slice(p+1), " \n", fs);

        field -= 3;
        if (field == 0)
            return fs[0][0];
        else if (field >= fs.size())
            return UINT64_MAX;
        else
            return stringToUInt64(fs[field]);
    }
}


#endif
//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

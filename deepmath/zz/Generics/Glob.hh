#ifndef ZZ__Chameleon__Glob_hh
#define ZZ__Chameleon__Glob_hh
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


// If 'abort_on_error' is set to FALSE in the below functions, this exception will be thrown
// on error.
struct Excp_GlobError : Excp_Msg {
    Excp_GlobError(String msg) : Excp_Msg(msg) {}
};


String getMatchingDir(String dir, bool abort_on_error = true);
    // -- expand '~' or environment variables in 'dir' (if using wildcards, which is not advised,
    // the first match is returned).

void getMatchingFiles(String dir, String glob, Vec<String>& out_files, bool abort_on_error = true);
    // -- expand globs ('~', environment variables and wildcards) as well as lists of form
    // '{<glob>,<glob>,...,<glob>}'. The purpose of 'dir' is to return filenames without the
    // directory part. It can be left empty.

inline Vec<String> getMatchingFiles(String dir, String glob, bool abort_on_error = true) {
    Vec<String> out_files;
    getMatchingFiles(dir, glob, out_files, abort_on_error);
    return out_files; }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

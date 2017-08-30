#include ZZ_Prelude_hh
#include "Glob.hh"
#include <wordexp.h>


namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Globbing:


// 'dir' can contain '~' or environment variables, which are expanded.
String getMatchingDir(String dir, bool abort_on_error)
{
    wordexp_t p;
    int st = wordexp(dir.c_str(), &p, WRDE_NOCMD | WRDE_UNDEF);
    if (st != 0 || p.we_wordc != 1)
        throwOrAbort(abort_on_error, "GLOB ERROR! ", Excp_GlobError(fmt("Invalid directory: %_", dir)));

    dir = p.we_wordv[0];    // -- expanded directory
    wordfree(&p);
    return dir;
}


// Will append matching files. Expansions corresponding to non-files (directories or missing files)
// are ignored. Program is terminated on error.
void getMatchingFiles(String dir, String glob, Vec<String>& out_files, bool abort_on_error)
{
    uind p0 = search(glob, '{');
    if (p0 != UIND_MAX){
        Str pfx = glob.slice(0, p0);
        Vec<Str> segs;

        uint open = 1;
        uind p1 = p0 + 1;
        for (; p1 < glob.size(); p1++){
            if (p1 >= glob.size()){ wrLn("GLOB ERROR! Mismatching '{' and '}': %_", glob); exit(1); }
            else if (glob[p1] == '{') open++;
            else if (glob[p1] == '}'){
                open--;
                if (open == 0){
                    segs.push(glob.slice(p0+1, p1));
                    break; }
            }else if (open == 1 && glob[p1] == ','){
                segs.push(glob.slice(p0+1, p1));
                p0 = p1;
            }
        }
        Str sfx = glob.slice(p1+1);

        for (Str seg : segs)
            getMatchingFiles(dir, fmt("%_%_%_", pfx, seg, sfx), out_files);

    }else{
        wordexp_t p;
        int st;
        if (dir.size() > 0){
            dir = getMatchingDir(dir);
            st = wordexp(fmt("%_/%_", dir, glob).c_str(), &p, WRDE_NOCMD | WRDE_UNDEF);
        }else
            st = wordexp(glob.c_str(), &p, WRDE_NOCMD | WRDE_UNDEF);
        if (st != 0){
            switch (st){
            case WRDE_BADCHAR: throwOrAbort(abort_on_error, "GLOB ERROR!", Excp_GlobError(fmt("Illegal occurrence of newline or one of: | & ; < > ( ) { }")));
            case WRDE_BADVAL:  throwOrAbort(abort_on_error, "GLOB ERROR!", Excp_GlobError(fmt("An undefined shell variable was referenced.")));
            case WRDE_CMDSUB:  throwOrAbort(abort_on_error, "GLOB ERROR!", Excp_GlobError(fmt("Command substitution requested.")));
            case WRDE_SYNTAX:  throwOrAbort(abort_on_error, "GLOB ERROR!", Excp_GlobError(fmt("Shell syntax error, such as unbalanced parentheses or unmatched quotes.")));
            default:           throwOrAbort(abort_on_error, "GLOB ERROR!", Excp_GlobError(fmt("Unknown return code: %_", st)));
            }
            wrLn("  - when expanding: %_", glob);
            exit(1);
        }

        char** w = p.we_wordv;
        for (uint i = 0; i < p.we_wordc; i++){
            if (fileExists(w[i])){
                assert(pfx(w[i], dir));
                char* n = w[i] + dir.size();
                if (dir.size() > 0){
                    while (*n == '/') n++; }
                out_files.push(n);
            }
        }
        wordfree(&p);
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

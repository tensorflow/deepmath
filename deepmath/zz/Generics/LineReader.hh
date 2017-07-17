#ifndef ZZ__Generics__LineReader_hh
#define ZZ__Generics__LineReader_hh

#include <sys/mman.h>

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
/*
EXAMPLE:

    LineReader in("test.txt");
    if (!in){
        ShoutLn "ERROR! Could not open: test.txt";
        exit(1); }

    while (!in.eof()){
        Str text = in.getLine();
        WriteLn "`%_`", text;
    }
*/


class LineReader {
    int    fd   = -1;
    cchar* data0 = nullptr;
    cchar* data = nullptr;
    cchar* head = nullptr;
    cchar* end  = nullptr;

    size_t pagemask = ~size_t(sysconf(_SC_PAGE_SIZE) - 1);

public:
    LineReader(String const& filename);
   ~LineReader() { munmap((void*)data, end - data); }

    explicit operator bool() const { return fd != -1; }

    bool eof() const { return head == end; }
    Str getLine();  // -- return value does not include newline

    uint64 tell() const { return head - data0; }
    uint64 size() const { return end  - data0; }
};


inline LineReader::LineReader(String const& filename) {
    fd = open64(filename.c_str(), O_RDONLY, 0);
    if (fd == -1) return;

    size_t sz = lseek64(fd, 0, SEEK_END);
    data = data0 = (char*)mmap(0, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(data != MAP_FAILED);
    close(fd);
    head = data;
    end  = head + sz;
}


inline Str LineReader::getLine() {
    // Periodically unmap behind head:
    size_t unmap_sz = (head - data) & pagemask;
    if (unmap_sz > 65536){
        munmap((void*)data, unmap_sz);
        data += unmap_sz; }

    // Find end-of-line:
    char* p = (char*)memchr(head, '\n', end - head);
    Str ret;
    if (p){
        ret = Str(head, p - head);
        head = p + 1;
    }else{
        ret = Str(head, end - head);
        head = end;
    }
    return ret;
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

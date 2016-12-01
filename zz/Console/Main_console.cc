#include ZZ_Prelude_hh
#include "ConsoleStd.hh"

using namespace ZZ;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


// Get raw key-codes.
int main(int argc, char** argv)
{
    ZZ_Init;

    bool interpreted = false;
    if (argc == 2 && eq(argv[1], "-i"))
        interpreted = true;

    Vec<String> events;
    uint ev_width = 20;

    con_init();
    con_redraw();
    for(;;){
        // Accumulate key-codes:
        String text;
        for(;;){
            ConEvent ev = con_getEvent(0.01, interpreted);
            if (ev.type == ev_KEY){
                if (!interpreted){
                    if      (ev.key == 27)                  text += "ESC ";
                    else if (ev.key >= 32 && ev.key <= 126) FWrite(text) "%_ ", (char)ev.key;
                    else if (ev.key < 32)                   FWrite(text) "^%_ ", (char)(ev.key + 64);
                    else                                    FWrite(text) "#%_ ", (uint)ev.key;
                }else{
                    if (ev.key & chr_META) text += "META ";
                    if (ev.key & chr_CSI) text += "CSI ";
                    FWrite(text) "0x%X ", ev.key & chr_MASK;
                }
            }else if (ev.type == ev_MOUSE){
                FWrite(text) "y=%_  x=%_  b=%_ ", ev.col, ev.row, (char)ev.key;
            }else
                break;
            if (interpreted) break;
        }

        if (text.size() > 0){
            if (text.last() == ' ') text.pop();
            events.push(text);
        }

        // Display them:
        uint max_displayed = con_rows() * (con_cols() / ev_width);
        uint start = (events.size() > max_displayed) ? events.size() - max_displayed : 0;

        fillScreen();
        uint y = 0, x = 0;
        for (uint i = start; i < events.size(); i++){
            printAt(y, x, events[i], fgRgb(5, 5, 1));
            y++;
            if (y >= con_rows()){
                y = 0;
                x += ev_width;
            }
        }
        con_redraw();
    }

    con_redraw();
    con_close();

    return 0;
}

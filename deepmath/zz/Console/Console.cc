//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : Console.cc
//| Author(s)   : Niklas Een
//| Module      : Console
//| Description : Console handling with mouse and extended color support.
//|
//| (C) Copyright 2010-2016, Niklas Een
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//|
//| Works best with 'xterm' or 'Eterm', slow with 'konsole', really sluggish with 'gnome-terminal'.
//|________________________________________________________________________________________________

#include ZZ_Prelude_hh
#include "Console.hh"
#include <termios.h>

namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Global state:


static bool            console_active = false;
static bool            first_init = true;
static struct termios  saved_attributes;
static Vec<Vec<AChar>> old_con;

static uint cursor_row = UINT_MAX;
static uint cursor_col = UINT_MAX;

Vec<Vec<AChar>> con;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Helpers:


static void getTermSize(uint& rows, uint& cols)
{
    struct winsize ws;
    ioctl(0, TIOCGWINSZ, &ws);
    rows = (uint)ws.ws_row;
    cols = (uint)ws.ws_col;
}


static void allocCanvas(uint rows, uint cols)
{
    con.reset(rows);
    for (auto& r : con) r.reset(cols);
    old_con.clear();
}


inline void write_safe(int fd, cchar* buf, size_t count)
{
    while (count > 0){
        ssize_t n = write(fd, buf, count);
        assert(n != -1);
        buf += n;
        count -= n;
    }
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Keyboard input:


static bool inputAvail(double timeout)
{
    struct timeval tv;
    fd_set fds;
    tv.tv_sec = (uint)timeout;
    tv.tv_usec = uint(1000000 * (timeout - (uint)timeout));
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    select(STDIN_FILENO+1, &fds, NULL, NULL, (timeout != DBL_MAX) ? &tv : NULL);
    return FD_ISSET(STDIN_FILENO, &fds);
}


static Vec<char> key_buf;
static uind      key_pos = 0;


static int getChar(double timeout)
{

    if (key_pos < key_buf.size())
        return key_buf[key_pos++];

    else{
        if (!inputAvail(timeout))
            return -1;
        else{
            key_buf.setSize(4096);
            key_pos = 0;
            ssize_t n = read(STDIN_FILENO, &key_buf[0], 4096);
            assert(n > 0);
            key_buf.shrinkTo(n);
            key_buf.push(-1);

            return key_buf[key_pos++];
        }
    }
}


static void putBackChar(char c)
{
    assert(key_pos > 0);
    key_pos--;
    assert(key_buf[key_pos] == c);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


void con_close()
{
    if (console_active){
        write_safe(STDOUT_FILENO, "\x1B[?1000l", 8);     // -- turn off mouse support
        write_safe(STDOUT_FILENO, "\x1B[?25h", 6);       // -- turn on cursor
        tcsetattr(STDIN_FILENO, TCSANOW, &saved_attributes);
        console_active = false;
    }
    write_safe(STDOUT_FILENO, "\n" , 1);
}


void con_showCursor(uint row, uint col)
{
    if ((int)row < 0) row = con_rows() - int(~row);
    if ((int)col < 0) col = con_cols() - int(~col);     // -- allow for '~n' to mean 'con_cols() - n'
    cursor_row = row;
    cursor_col = col;

    String out;
    FWrite(out) "\x1b[%_;%_H", row+1, col+1;
    FWrite(out) "\x1B[?25h";
    write_safe(STDOUT_FILENO, out.base(), out.size());
}


void con_hideCursor()
{
    write_safe(STDOUT_FILENO, "\x1B[?25l", 6);
    cursor_row = UINT_MAX;
    cursor_col = UINT_MAX;
}


void con_init()
{
    struct termios tattr;

    // Make sure stdin is a terminal:
    if (!isatty(STDIN_FILENO)){
        ShoutLn "ERROR! Standard input is not connected to a terminal.";
        exit(-1); }

    // Save the terminal attributes so we can restore them later:
    if (first_init){
        tcgetattr(STDIN_FILENO, &saved_attributes);
        atExit(x_Always, (ExitFun0)con_close);
        first_init = false; }

    // Set the terminal mode:
    tcgetattr(STDIN_FILENO, &tattr);    // -- could use 'cfmakeraw(tattr)' for even rawer input
    tattr.c_lflag &= ~(ICANON|ECHO);    // -- clear ICANON and ECHO
  #if defined(__APPLE__)
    tattr.c_cc[VMIN]  = 1;
  #else
    tattr.c_cc[VMIN]  = 0;
  #endif
    tattr.c_cc[VTIME] = 1;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &tattr);

    uint rows, cols;
    getTermSize(rows, cols);
    allocCanvas(rows, cols);

    write_safe(STDOUT_FILENO, "\x1B[?1000h", 8);     // -- turn on mouse support
    write_safe(STDOUT_FILENO, "\x1B[?25l", 6);       // -- turn off cursor

    // Scroll down one screen to keep old terminal content in scroll-back buffer:
    Vec<char> tmp(rows, '\n');
    write_safe(STDOUT_FILENO, tmp.base(), tmp.size());

    console_active = true;
    con_hideCursor();
}


ConEvent con_getEvent(double timeout, bool interpreted)
{
    con_redraw();

    for(;;){
        double this_timeout = (timeout < 0.05) ? timeout : 0.05;    // -- 0.05 means 20 times a second we will check if window has been resized.
        timeout -= this_timeout;
        int c = getChar(this_timeout);
        if (c == -1){
            uint rows, cols;
            getTermSize(rows, cols);    // <<== 0.04 behövs för den här pollningen...
            if (rows != con_rows() || cols != con_cols()){
                if (rows == 0){ rows = 1; cols = 0; }
                allocCanvas(rows, cols);
                return ConEvent{ev_RESIZE, 0ull, rows, cols};

            }else if  (timeout == 0)
                return ConEvent{ev_NULL, 0ull, 0u, 0u};

        }else{
            if (!interpreted)
                return ConEvent{ev_KEY, (uint64)(uchar)c, 0u, 0u};
            else{
                if (c == 27){
                    auto encode = [](Array<char> arr) {
                        uint64 ret = 0;
                        for (char c : arr) ret = (ret << 8) | (uchar)c;
                        return ret;
                    };

                    auto button = [](char b) {
                        return (b == 0x60) ? 'u' :  // -- wheel up
                               (b == 0x61) ? 'd' :  // -- wheel down
                               (b == 0x20) ? 'l' :  // -- left button
                               (b == 0x21) ? 'm' :  // -- middle button
                               (b == 0x22) ? 'r' :  // -- right button
                               (b == 0x23) ? 'R' :  // -- release button
                               /*otherwise*/  0  ;
                    };

                    Vec<char> buf;
                    do buf.push(getChar(0.0)); while (buf.last() != char(-1) && buf.last() != 27);
                    if (buf.last() == 27) putBackChar(27);
                    buf.pop();

                    if (buf.size() == 0)
                        return ConEvent{ev_KEY, (uint64)(uchar)c, 0u, 0u};

                    if (buf[0] == '['){
                        if (buf.size() == 5 && buf[1] == 'M'){   // -- mouse
                            return ConEvent{ev_MOUSE, (uchar)button(buf[2]), uchar(buf[4]-33), uchar(buf[3]-33)}; }
                        else
                            return ConEvent{ev_KEY, chr_CSI | encode(buf.slice(1)), 0u, 0u};
                    }else
                        return ConEvent{ev_KEY, chr_META | encode(buf.slice(0)), 0u, 0u};
                }
                return ConEvent{ev_KEY, (uint64)(uchar)c, 0u, 0u};
            }
        }
    }
}


void con_redraw()
{
    auto eq = [](AChar a, AChar b) {
        if (a.chr == ' ' && b.chr == ' ' && !(a.alt & sty_UNDER) && !(b.alt & sty_UNDER))
            a.fg = 0, b.fg = 0;     // -- a non-underscored space looks the same for all foreground colors...
        return a == b;
    };

    uint  x = 0;
    uint  y = 0;
    uchar fg = AChar().fg;
    uchar bg = AChar().bg;
    uchar alt = 0;

    String out;
    FWrite(out) "\x1B[?25l" "\x1b[1;1H" "\x1b[0m" "\x1b[38;5;%_m" "\x1b[48;5;%_m", (uint)fg, (uint)bg;

    for (uint row = 0; row < con.size(); row++){
        for (uint col = 0; col < con[row].size(); col++){
            if (row < old_con.size() && col < old_con[row].size() && eq(con[row][col], old_con[row][col]))
                continue;

            if (y != row || x != col){
                FWrite(out) "\x1b[%_;%_H", row+1, col+1;
                y = row; x = col; }

            AChar c = con[row][col];
            if (alt != c.alt){
                // This is tricky only because there are no codes for turning off bold/italic/underscore:
                if ((alt | c.alt) != c.alt){
                    alt = 0;
                    fg = c.fg;
                    bg = c.bg;
                    FWrite(out) "\x1b[0m" "\x1b[38;5;%_m" "\x1b[48;5;%_m", (uint)fg, (uint)bg;
                }
                if ((c.alt & sty_BOLD ) && !(alt & sty_BOLD )) FWrite(out) "\x1b[1m";
                if ((c.alt & sty_ITAL ) && !(alt & sty_ITAL )) FWrite(out) "\x1b[3m";
                if ((c.alt & sty_UNDER) && !(alt & sty_UNDER)) FWrite(out) "\x1b[4m";
                alt = c.alt;
            }

            if (fg != c.fg){
                FWrite(out) "\x1b[38;5;%_m", (uint)c.fg;
                fg = c.fg; }

            if (bg != c.bg){
                FWrite(out) "\x1b[48;5;%_m", (uint)c.bg;
                bg = c.bg; }

            if (c.chr < 32)
                out += ' ';
            else if (c.chr >= 0xA0 && c.chr <= 0xBF)
                out += (char)0xC2, (char)c.chr;
            else if (c.chr >= 0xC0)
                out += (char)0xC3, (char)(c.chr - 64);
            else
                out += (char)c.chr;
            x++;
        }
    }
    con.copyTo(old_con);

    FWrite(out) "\x1b[0m" "\x1b[%_;1H", (uint)con_rows();   // -- position cursor at bottom in case CTRL-C is pressed
    write_safe(STDOUT_FILENO, out.base(), out.size());

    if (cursor_row != UINT_MAX)
        con_showCursor(cursor_row, cursor_col);
}


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : CmdLine.hh
//| Author(s)   : Niklas Een
//| Module      : CmdLine
//| Description : Command line parsing.
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| Full documentation can be found in CmdLine_README.txt
//| 
//| To add an argument to the global command-line interface object 'cli', use:
//| 
//|     cli.add(arg_name, arg_sig, default_value, help_text [, arg_pos])
//|     
//| Argument position is optional and can be left out. Special default value 'arg_REQUIRED'
//| denotes a required argument. It should be used with positional arguments only.
//| 
//| Example of argument signatures:
//| 
//|     int, int[1:10], ufloat, string, {one,two,three}, (uint,string), [string], int | {none}, any
//|________________________________________________________________________________________________

#ifndef ZZ__CmdLine__CmdLine_hh
#define ZZ__CmdLine__CmdLine_hh
namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Universal Type:


enum CLI_UnivEnum {
    cli_ERROR,
    cli_Bool,
    cli_Int,
    cli_Float,
    cli_Enum,
    cli_String,
    cli_Tuple,
    cli_List,
    cli_Any,            // -- only present in signatures 
    cli_Or,             // -- only present in signatures 

    CLI_UnivEnum_size
};


struct CLI_IntRange {
    int64 lo;         // -- 'INT64_MIN' and 'INT64_MAX' denotes negative and positive infinity
    int64 hi;
};


struct CLI_FloatRange {
    double lo;          // -- '-DBL_MAX' and '+DBL_MAX' denotes negative and positive infinity
    double hi;
    bool   lo_strict;
    bool   hi_strict;
};


//=================================================================================================
// -- Signatures:


struct CLI_Sig {
    // Element type and range:
    CLI_UnivEnum type;
    union {
        CLI_IntRange    int_range;
        CLI_FloatRange  float_range;
        Vec<String>*    enum_range;
        String*         error_msg;      // -- may be defined for 'type == cli_ERROR'
    };
    Vec<CLI_Sig>* sub;      // -- defined for tuples, list (always length 1), and disjunctions ('cli_Or')

    CLI_Sig() : type(cli_ERROR), error_msg(NULL), sub(NULL) {}
};


void dispose(CLI_Sig sig);


//=================================================================================================
// -- Values:


struct CLI_Val {
    // Is this a value matched with a signature?
    bool   matched;
    Null_Method(CLI_Val){ return !matched; }

    // For atoms, the unquoted string:
    String string_val;      // -- for unmatched values, this is guaranteed to be the empty string

    // Element type and range:
    CLI_UnivEnum type;
    union {
        CLI_IntRange    int_range;
        CLI_FloatRange  float_range;
        Vec<String>*    enum_range;
        String*         error_msg;
    };

    // Parsed value:
    union {
        bool    bool_val;
        int64   int_val;
        double  float_val;
        uint    enum_val;
    };
    Vec<CLI_Val>* sub;      // -- defined for tuples and lists

    // Convenience access to 'sub':
    uind           size      ()       const { assert(sub); return sub->size(); }
    CLI_Val&       operator[](uind i)       { assert(sub); return sub->get(i); }
    const CLI_Val& operator[](uind i) const { assert(sub); return sub->get(i); }

    // Which alternative did we match? ('-1' means there was no choice to be made (no "Or" node in signature))
    int     choice;

    // Default constructor:
    CLI_Val() : matched(false), type(cli_ERROR), error_msg(NULL), sub(NULL), choice(-1) {}
};


void dispose(CLI_Val val);


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Parsing and matching:


CLI_Sig CLI_parseSignature(cchar* text);
CLI_Val CLI_parseValue    (cchar* text);
    // -- you probably do not need to call these directly; the CLI object will use them.


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Command line interface:


// Helper types, real stuff is below...

struct CLI;

struct CLI_Arg {
    String  name;
    CLI_Sig sig;
    String  default_;
    int     pos;        // -1 is used for "no position" (must use name).
    String  help;

    CLI_Arg() : pos(-1) {}
    CLI_Arg(String name_, CLI_Sig sig_, String def_, int pos_, String help_) : name(name_), sig(sig_), default_(def_), pos(pos_), help(help_) {}
};

struct CLI_Cmd {
    String  name;
    String  help;
    CLI*    sub_cli;
    CLI_Cmd(String name_, String help_, CLI* sub_cli_) : name(name_), help(help_), sub_cli(sub_cli_) {}
};

typedef String (*CLI_PostProcessFun0)(CLI& cli);
typedef String (*CLI_PostProcessFun1)(CLI& cli, void* data);

struct CLI_PostProcessCB {
    int type;
    union {
        CLI_PostProcessFun0 fun0;
        CLI_PostProcessFun1 fun1;
    };
    void* data;
};


//=================================================================================================
// -- The CLI class:


#define arg_REQUIRED String("\x01")      // -- use this as 'default_value' for required arguments


// Here is where the action is!

struct CLI {
    Vec<Pair<CLI_Arg,CLI_Val> >  args;              // Command line argument: declaration and values (the latter filled in after 'parseCmdLine()' is called).
    Vec<CLI_Cmd>                 cmds;              // Commands declared. Can be empty if program does only one thing.
    Vec<Pair<CLI*,String> >      embedded;          // Embedded arguments.
    Vec<CLI_PostProcessCB>       post_process;      // Callbacks after parsing is done (can do further checks and report errors). Should throw a 'String' on error.

   ~CLI();

    void     add(String arg_name, String arg_signature, String default_value = "", String help_text = "", int arg_pos = -1);
        // -- Add an argument declaration. 'arg_name' must not contain '=', '\' or white spaces.

    void     addCommand(String command_name, String help_text = "", CLI* sub_cli = NULL);
        // -- Extends the special switch '-command={...}', which is always of enum type.

    void     parseCmdLine(String command_line, String prog_name, bool embed_cmd_cli = true);
    void     parseCmdLine(int argc, char** argv, bool embed_cmd_cli = true);
    bool     parseCmdLine(String command_line  , String& out_error_msg, uint columns, String prog_name, bool embed_cmd_cli = true);
    bool     parseCmdLine(int argc, char** argv, String& out_error_msg, uint columns, bool embed_cmd_cli = true);
        // -- Parse commandline (this is how arguments get their values). All arguments must be
        // given values (either from command line or from default values), otherwise program
        // is aborted and an error message is printed (top two functions), or alternatively
        // the error message is returned through 'error_msg' (bottom two functions; FALSE is
        // returned on failure). If 'embed_cmd_cli' is TRUE (default), then if a command is given,
        // its CLI is embedded automatically (if there is an error, or if 'embed_cmd_cli' is
        // FALSE, this CLI is left untouched).

    void     postProcess(CLI_PostProcessFun0 fun);
    void     postProcess(CLI_PostProcessFun1 fun, void* data);
        // -- Callbacks are applied after command line has been parsed. On success, the
        // call-backs should return an empty string, otherwise an error message.

    bool     has(String arg_name);
        // -- does this CLI contain a declaration of 'arg_name' (not necessarily assigned a value)

    void     embed(CLI& cli, String prefix);
        // -- Embed, through reference, argument definitions of another CLI object into this one,
        // prefixing all names with 'prefix' (which can be empty). Calling 'parseCmdLine()' on this
        // object will modified embedded objects as well.
        //
        // IMPORTANT! Ownership is NOT taken. Caller must make sure embedded CLIs outlive this CLI
        // and are then disposed of afterwards.

    bool     unbed(CLI& cli);
        // -- Remove embedded CLI. Returns TRUE if 'cli' was found (and removed).

    //
    // TO READ RESULT:
    //
    const CLI_Val& get(String arg_name) const;  // -- read a value
    String       cmd;                           // -- what command was selected? [read only]
    Vec<CLI_Val> tail;                          // -- contain all arguments parsed as part of special switch "..." [read only]

private:
    Pair<CLI_Arg,CLI_Val>* getIfExist(String arg_name);
    const Pair<CLI_Arg,CLI_Val>* getIfExist(String arg_name) const { return const_cast<CLI*>(this)->getIfExist(arg_name); }
    bool setArg(String name, String val, Vec<String>& defined, String& error_msg);
    void printHelp(String& error_msg, uint columns, String prog_name, String prefix = "", bool show_hidden = false);

    bool verifyRequired(const Vec<String>& is_defined, String prefix, String& error_msg) const;
};


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Debug:


void write_CLI_Sig(Out& out, const CLI_Sig& sig);
void write_CLI_Val(Out& out, const CLI_Val& sig);

template<> fts_macro void write_(Out& out, const CLI_Sig& sig) { write_CLI_Sig(out, sig); }
template<> fts_macro void write_(Out& out, const CLI_Val& val) { write_CLI_Val(out, val); }


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
// Global CLIs:


extern CLI cli;             // -- Use this CLI in 'main()'.
extern CLI cli_hidden;      // -- Automatically embedded in 'cli' but hidden from documentation.


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}
#endif

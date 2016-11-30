//_________________________________________________________________________________________________
//|                                                                                      -- INFO --
//| Name        : CmdLine.cc
//| Author(s)   : Niklas Een
//| Module      : CmdLine
//| Description : Command line parsing.
//| 
//| (C) Copyright 2010-2014, The Regents of the University of California
//|________________________________________________________________________________________________
//|                                                                                  -- COMMENTS --
//| Documentation can be found in CmdLine_README.txt
//|________________________________________________________________________________________________

#include "Prelude.hh"
#include "CmdLine.hh"


namespace ZZ {
using namespace std;


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm


CLI cli;
CLI cli_hidden;

ZZ_Initializer(cli_hidden, -10050) {
    cli.embed(cli_hidden, "");
}

ZZ_Finalizer(cli_clear, -10050) {
    // Must clear vectors before global variables are destroyed (because allocation in Prelude is destroyed first)
    cli.~CLI();
    new (&cli) CLI();
    cli_hidden.~CLI();
    new (&cli_hidden) CLI();
}


#include "CmdLine_UnivType.icc"
#include "CmdLine_Parse.icc"
#include "CmdLine_Match.icc"
#include "CmdLine_Debug.icc"
#include "CmdLine_CLI.icc"


//mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
}

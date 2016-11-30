###
###  Makefile for building a ZZ module: lib-file, executables, dependency files
###  (C) Copyright 2012, Niklas Een
###

ifndef VERBOSE
.SILENT:
endif


##=================================================================================================
## Parameters:  (should be setup by caller; below is an example)


# BUILD_PATH       = /home/een/ZZ/BUILD
# ROOT_PATH        = /home/een/ZZ
# OUTPUT_SUBDIR    = out
# TARGET_QUALIFIER = turbantad/quick
# MODULE_PREFIX    = ZZ_
#
# MOD_DEPS = ZZ_Prelude ZZ_CmdLine ZZ_MiniSat ZZ_Netlist    # should NOT include the this module
# SYS_LIBS = rt z dl abc
# SYS_PATH = /home/een/new/ZZ/External/Abc        # path of 'libabc.a'
# SYS_INCL = /home/een/new/ZZ/External/Abc/src    # include path
#
# CXXFLAGS  = -I ~/new/ZZ -I ~/new/ZZ/Generics -I ~/new/ZZ/Prelude -I ~/new/ZZ/External/Abc/src
# CXXFLAGS += -O0 -ggdb -Wall
#
# LFLAGS = -pg


##=================================================================================================
## Determine sources: 


# Each directory defines a single library together with one or more executable targets. All files
# matching '*.cc' in the current directory make up the source files for the library, except those
# that match 'Main_*.cc' (or 'Main_*.qcc' for QT). Each of these files produce an executable by
# compiling that file individually and then linking it to the local library as well as all external
# libraries.

EXE_SRCS   = $(wildcard Main_*.cc)  
EXE_SRCS  += $(wildcard Main_*.C)  
EXE_SRCS  += $(wildcard Main_*.c)  
LIB_SRCS   = $(filter-out $(EXE_SRCS), $(wildcard *.cc))
LIB_SRCS  += $(filter-out $(EXE_SRCS), $(wildcard *.C))
LIB_SRCS  += $(filter-out $(EXE_SRCS), $(wildcard *.c))
#LOCAL_PATH = $(CURDIR:$(ROOT_PATH)/%=%)
MODULE     = $(MODULE_PREFIX)$(subst /,.,$(LOCAL_PATH))
OUT        = $(CURDIR)/$(OUTPUT_SUBDIR)/$(TARGET_QUALIFIER)
LIB_OUT    = $(ROOT_PATH)/lib/$(TARGET_QUALIFIER)
BIN_OUT    = $(OUT)

#LINK = g++
LINK = $(BUILD_PATH)/zz_gdep_link	
    # -- "g++" works fine under Linux, but this script handles MacOS as well

ifdef USE_CCACHE
  CXX = ccache g++
  CC  = ccache gcc
else
  CXX = g++
  CC  = gcc
endif

COMMA=,

#$(warning $(BIN_OUT)  $(OUT)/Main_%.o  $(LIB_BIN)  $(LMOD))

# Quick-ref:
#
#   $(wildcard shell-pattern)      -- list of names of existing files matching the pattern
#   $(filter-out pattern...,text)  -- keep words in 'text' that do NOT match any pattern
#   $(patsubst pattern,repl,text)  -- words in 'text' matching 'pattern' becomes 'repl'
#   $(foreach var,list,text)       -- for each element in 'list', create 'text' (using 'var')


##=================================================================================================
## Main:


ifdef QT_MODULE
include $(BUILD_PATH)/qt_module.mak
endif

.PHONY: all lib bin libdep bindep
all : lib bin

THIS_MAKEFILE = $(firstword $(MAKEFILE_LIST))


##=================================================================================================
## Dependencies:


INCL = $(addprefix -I,$(SYS_INCL))
$(OUT)/%.dep : %.cc
	mkdir -p $(dir $@)
	$(CXX) -M $(CXXFLAGS) $(INCL) $(CURDIR)/$< | $(BUILD_PATH)/filter_deps $(OUT) > $@
  # NOTE! Dependencies are generated with full absolute paths

$(OUT)/%.dep : %.C
	mkdir -p $(dir $@)
	$(CXX) -M $(CXXFLAGS) $(INCL) $(CURDIR)/$< | $(BUILD_PATH)/filter_deps $(OUT) > $@

$(OUT)/%.dep : %.c 
	mkdir -p $(dir $@)
	$(CC) -M $(CFLAGS) $(INCL) $(CURDIR)/$< | $(BUILD_PATH)/filter_deps $(OUT) > $@

LIB_DEPS = $(foreach F, $(LIB_SRCS), $(OUT)/$(basename $(F)).dep)
$(OUT)/LIB.deps: $(LIB_DEPS)
	mkdir -p $(dir $@)
	cat $^ /dev/null > $@
#	echo "\`\` Module library dependencies built:" $(MODULE)


EXE_DEPS = $(foreach F, $(EXE_SRCS), $(OUT)/$(basename $(F)).dep)
$(OUT)/BIN.deps: $(EXE_DEPS)
	mkdir -p $(dir $@)
	cat $^ /dev/null > $@
#	echo "\`\` Module executable dependencies built:" $(MODULE)

libdep : $(OUT)/LIB.deps
bindep : $(OUT)/BIN.deps


##=================================================================================================
## Library:


$(OUT)/%.o : %.cc 
	mkdir -p $(dir $@)
	echo "\`\` Compiling:" $(LOCAL_PATH)/$<
	$(CXX) -c $(CXXFLAGS) $(INCL) $< -o $@
	rm -f $(basename $@).dep      # (remove stale dependency file)

$(OUT)/%.o : %.C
	mkdir -p $(dir $@)
	echo "\`\` Compiling:" $(LOCAL_PATH)/$<
	$(CXX) -c $(CXXFLAGS) $(INCL) $< -o $@
	rm -f $(basename $@).dep      # (remove stale dependency file)

$(OUT)/%.o : %.c
	mkdir -p $(dir $@)
	echo "\`\` Compiling:" $(LOCAL_PATH)/$<
	$(CC) -c $(CFLAGS) $(INCL) $< -o $@
	rm -f $(basename $@).dep      # (remove stale dependency file)


LIB_BIN  = $(LIB_OUT)/lib$(MODULE).a
LIB_OBJS = $(foreach F, $(LIB_SRCS), $(OUT)/$(basename $(F)).o)
$(LIB_BIN) : $(LIB_OBJS)
	mkdir -p $(dir $@)
	printf "\`\` Linking library: %s\n" "$(notdir $@)" #; printf "  - %s\n" $(notdir $^)
	rm -f $@
	echo "int __zb_dummy_var__;" > $(OUT)/__dummy__.cc; $(CXX) -c $(OUT)/__dummy__.cc -o $(OUT)/__dummy__.o   # (needed for Mac who doesn't like empty libraries)
	ar cqs $@ $^ $(OUT)/__dummy__.o 2>&1

lib : $(LIB_BIN)
	$(MAKE) NO_DEP_INCLUDES=1 -f $(THIS_MAKEFILE) $(OUT)/LIB.deps   # (remake module dependencies)

ifndef NO_DEP_INCLUDES
-include $(OUT)/LIB.deps
endif


##=================================================================================================
## Executables:


LSYS0  = $(addprefix -l,$(SYS_LIBS))
LPATHS = $(addprefix -L,$(SYS_PATH))
LMOD   = $(patsubst %,$(LIB_OUT)/lib%.a,$(MOD_DEPS))

LSYS   = $(patsubst %:static,-Wl$(COMMA)-Bstatic % -Wl$(COMMA)-Bdynamic,$(LSYS0))
    # -- wrap libraries ending in ':static' with appropriate linker flags

$(BIN_OUT)/%.exe : $(OUT)/Main_%.o $(LIB_BIN) $(LMOD)
	mkdir -p $(dir $@)
	echo "\`\` Linking binary:" $@
	$(LINK) $(LFLAGS) $(LPATHS) -Wl,--start-group $^ -Wl,--end-group $(LSYS) -o $@
#	$(LINK) $(LFLAGS) $(LPATHS) -Wl,-whole-archive $^ -Wl,-no-whole-archive $(LSYS) -o $@

bin : $(patsubst Main_%.cc,$(BIN_OUT)/%.exe,$(EXE_SRCS)) $(patsubst Main_%.qcc,$(BIN_OUT)/%.exe,$(EXE_SRCS)) $(patsubst Main_%.c,$(BIN_OUT)/%.exe,$(EXE_SRCS)) $(patsubst Main_%.C,$(BIN_OUT)/%.exe,$(EXE_SRCS))
	$(MAKE) NO_DEP_INCLUDES=1 -f $(THIS_MAKEFILE) $(OUT)/BIN.deps   # (remake executable dependencies)

ifndef NO_DEP_INCLUDES
-include $(OUT)/BIN.deps
endif

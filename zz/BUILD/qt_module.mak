QT_INCLUDE_DIR = $(shell qmake -query QT_INSTALL_HEADERS)
QT_INCL = -I$(QT_INCLUDE_DIR) -I$(QT_INCLUDE_DIR)/QtGui -I$(QT_INCLUDE_DIR)/QtCore
QT_DEFINES = -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB -DQT_SHARED


EXE_SRCS += $(wildcard Main_*.qcc)
LIB_SRCS += $(filter-out $(EXE_SRCS), $(wildcard *.qcc))


$(OUT)/%.dep : %.qcc
	mkdir -p $(dir $@)
	g++ -x c++ -M $(CXXFLAGS) $(QT_DEFINES) $(INCL) $(QT_INCL) $(CURDIR)/$< | $(BUILD_PATH)/filter_deps $(OUT) > $@


$(OUT)/%.o : %.qcc $(OUT)/%.moc
	mkdir -p $(dir $@)
	echo "\`\` QT Compiling:" $(LOCAL_PATH)/$<
	export TMP=$(OUT)/$<.__tmp__.$$PPID.cc; \
	echo "#line 1 \"$<\"" > $$TMP; cat $< >> $$TMP; \
	echo "#include \"$(OUT)/$(basename $<).moc\"" >> $$TMP; \
	g++ -c $(CXXFLAGS) $(QT_DEFINES) $(INCL) $(QT_INCL) $$TMP -o $@; \
	rm -f $$TMP
	rm -f $(basename $@).dep      # (remove stale dependency file)


.PRECIOUS: $(OUT)/%.moc 
$(OUT)/%.moc : %.qcc
	mkdir -p $(dir $@)
	moc -nw $< > $@

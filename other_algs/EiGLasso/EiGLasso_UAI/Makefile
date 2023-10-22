MKLROOT = /opt/intel/mkl
MACOSSDK = /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk

MKLIDIR = $(MKLROOT)/include
MKLLDIR = $(MKLROOT)/lib



CXXFLAGS = -std=c++11 -Wall -m64 -I$(MKLIDIR)
LDFLAGS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl


UNAME = $(shell uname)

ifeq ($(UNAME), Linux)
	CXX = g++
	CXXFLAGS += -O3
	MKLLDIR := $(MKLLDIR)/intel64
	LDFLAGS += -Wl,--no-as-needed -L$(MKLLDIR)
	
endif

ifeq ($(UNAME), Darwin)
	CXX = clang++
	CXXFLAGS += -Ofast -I$(MACOSSDK)/usr/include
	LDFLAGS += -Wl,-rpath,$(MKLLDIR) -L$(MKLLDIR) -L$(MACOSSDK)/usr/lib
	
endif



eiglasso_flip: eiglasso_flip.cpp
	 $(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: clean

clean:
	$(RM) eiglasso_flip

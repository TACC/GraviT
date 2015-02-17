
CXX = g++
GVT_SRC = /Users/pnav/dev/gravit/src
BOOST_INC = /opt/local/include
BOOST_LIB = /opt/local/lib

DEFINES = -DGVT_USE_DEBUG
CFLAGS = -g -std=c++11 
LFLAGS = 

SOURCES = Context.cpp Database.cpp DatabaseNode.cpp
OBJECTS = $(SOURCES:cpp=o)
HEADERS = $(SOURCES:cpp=h)
HEADERS += Types.h

all: driver

driver: driver.o $(OBJECTS)
	$(CXX) $(LFLAGS) -o $@ $^

%.cpp : %.h

%.o : %.cpp
	$(CXX) -c $(CFLAGS) $(DEFINES) -I$(GVT_SRC) -I$(BOOST_INC) $<

clean:
	rm -f driver driver.o $(OBJECTS)

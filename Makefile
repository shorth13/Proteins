CXX = g++
CXXFLAGS = -Wall -ggdb3 -O5
LDFLAGS = -L. 

all: libenergy.so grad_w_armijo

libenergy.so: energy.cpp energy.hpp
	$(CXX) $(CXXFLAGS) -shared -o libenergy.so -fPIC energy.cpp

grad_w_armijo: libenergy.so grad_w_armijo.o 
	$(CXX) $(CXXFLAGS) grad_w_armijo.o -o grad_w_armijo  $(LDFLAGS) -lenergy


clean: FORCE
	@-rm libenergy.so
	@-rm grad_w_armijo

FORCE:

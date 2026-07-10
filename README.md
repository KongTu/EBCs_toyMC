# EBCs_toyMC

run pythia:

compile:

'''g++ -std=c++17 -O2 main_dijet_pp_full.cc -o run_pythia   -I/Users/zhoudunmingtu/bnl_work/Work/EIC/Pythia/Pythia8/pythia8235/include $(root-config --cflags)   -L/Users/zhoudunmingtu/bnl_work/Work/EIC/Pythia/Pythia8/pythia8235/lib -lpythia8   $(root-config --glibs)'''

Run:

'''export PYTHIA8DATA=/Users/zhoudunmingtu/bnl_work/Work/EIC/Pythia/Pythia8/pythia8235/share/Pythia8/xmldoc'''

'''./run_pythia'''
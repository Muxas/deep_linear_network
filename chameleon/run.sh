rm a.out

g++ test.cc -I/workspace/Install/chameleon/include \
    -I/workspace/Code/chameleon -I/workspace/Code/chameleon/build \
    -I/usr/local/cuda/include \
    -L/workspace/Install/chameleon/lib -lchameleon -lchameleon_starpu \
    -L/workspace/Install/starpu/lib -lstarpu-1.3 \
    -L/usr/local/cuda/lib64 -lcudart -L/opt/conda/lib -lmkl_gf_lp64 \
    -lmkl_sequential -lmkl_core -lpthread -lm -ldl -lcublas

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/Install/chameleon/lib:/workspace/Install/starpu/lib:/opt/conda/lib \
    STARPU_NCPU=4 ./a.out 65536 8192 16 4096

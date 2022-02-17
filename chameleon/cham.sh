LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib:/workspace/Install/starpu/lib \
    /workspace/Install/chameleon/bin/chameleon_stesting -o gemm -H -t 1 -g 1 \
    --forcegpu -n 20480 -b 4096 -l 3 --mtxfmt 1

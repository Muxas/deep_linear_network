#include <string>
#include <iostream>
#include <sstream>
#include <chameleon.h>

int main(int argc, char **argv)
{
    std::cout << "test B N D NB\n"
        " B is the number of input samples\n"
        " N is the number of inputs/outputs of each linear layer\n"
        " D is the number of linear layers\n"
        " NB is the tile size (parameter of the CHAMELEON library)\n"
        "\n"
        " 1. Generate random BxN matrix X_0 and D random NxN matrices W_0,"
        " ..., W_{D-1}\n"
        " 2. Compute BxN matrices X_1 = X_0 W_0, ..., X_D = X_{D-1} W_{D-1}\n"
        " 3. Set BxN matrix G_D = X_D\n"
        " 4. Multiply G_{D-1} = G_D W_{D-1}', ..., G_1 = G_2 W_1'\n"
        " 5. Compute NxN matrices Y_D = X_D' G_D, ..., Y_1 = X_1' G_1\n"
        " 6. Update W_i += 1e-16 Y_{i+1}\n";
    if(argc != 5)
    {
        return 0;
    }
    // Initialize CHAMELEON
    int ncores = -1, ngpus = -1;
    int rc = CHAMELEON_Init(ncores, ngpus);
    if(rc != CHAMELEON_SUCCESS)
    {
        std::cout << "CHAMELEON_Init failed and returned " << rc << "\n";
        return -1;
    }
    // Force all computations on GPUs only
    RUNTIME_slocality_allrestrict(RUNTIME_CUDA);

    // Read arguments
    int B, N, D, NB;
    std::stringstream args;
    args << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4];
    args >> B >> N >> D >> NB;
    std::cout << "B=" << B << " N=" << N << " D=" << D << " NB=" << NB << "\n";
    size_t nbytes_X = sizeof(float) * (D+1) * B * N;
    size_t nbytes_W = sizeof(float) * D * N * N;
    size_t nbytes_G = sizeof(float) * (D+1) * B * N;
    size_t nbytes_Y = sizeof(float) * (D+1) * N * N;
    float gbytes = (nbytes_X + nbytes_W + nbytes_G + nbytes_Y) / 1024.0 /
        1024.0 / 1024.0;

    // Allocate memory and init descriptors
    std::cout << "Allocating memory (" << gbytes << " GBytes)...\n";
    CHAM_desc_t **desc_X, **desc_W, **desc_G, **desc_Y;
    desc_X = (CHAM_desc_t **)malloc((D+1) * sizeof(CHAM_desc_t *));
    desc_W = (CHAM_desc_t **)malloc(D * sizeof(CHAM_desc_t *));
    desc_G = (CHAM_desc_t **)malloc((D+1) * sizeof(CHAM_desc_t *));
    desc_Y = (CHAM_desc_t **)malloc((D+1) * sizeof(CHAM_desc_t *));

    int ldX = B, ldW = N, ldG = B, ldY = N, P = 1, Q = 1;

    for(int i = 0; i <= D; ++i)
    {
        CHAMELEON_Desc_Create(&desc_X[i], nullptr, ChamRealFloat, NB, NB,
                NB * NB, ldX, N, 0, 0, B, N, P, Q);
        CHAMELEON_Desc_Create(&desc_G[i], nullptr, ChamRealFloat, NB, NB,
                NB * NB, ldG, N, 0, 0, B, N, P, Q);
        CHAMELEON_Desc_Create(&desc_Y[i], nullptr, ChamRealFloat, NB, NB,
                NB * NB, ldY, N, 0, 0, N, N, P, Q);
    }
    for(int i = 0; i < D; ++i)
    {
        CHAMELEON_Desc_Create(&desc_W[i], nullptr, ChamRealFloat, NB, NB,
                NB * NB, ldW, N, 0, 0, N, N, P, Q);
    }

    std::cout << "Total memory allocated on RAM is " << gbytes <<
        " GBytes\n\n";

    // Fill random data
    std::cout << "Initializing random data...\n";
    //for(int i = 0; i <= D; ++i)
    //{
    //    int seed_X = rand();
    //    CHAMELEON_splrnt_Tile(desc_X[i], seed_X);
    //    int seed_G = rand();
    //    CHAMELEON_splrnt_Tile(desc_G[i], seed_G);
    //    int seed_Y = rand();
    //    CHAMELEON_splrnt_Tile(desc_Y[i], seed_Y);
    //}
    int seed_X = rand();
    for(int i = 0; i < D; ++i)
    {
        int seed_W = rand();
        CHAMELEON_splrnt_Tile(desc_W[i], seed_W);
    }
    std::cout << "Initialization Complete\n";
    // Create runtime sequence for parallel execution of tasks
    RUNTIME_sequence_t *sequence;
    CHAMELEON_Sequence_Create(&sequence);
    RUNTIME_request_t *request;
    CHAMELEON_Request_Create(&request);
    // Get a workspace for GEMM (TODO: make it work for MPI)
    const cham_trans_t NoTrans = ChamNoTrans;
    const float alpha = 1.0, beta = 0.0;
    void *user_ws = CHAMELEON_sgemm_WS_Alloc(NoTrans, NoTrans, desc_X[0],
            desc_W[0], desc_X[1]);
    // Synchronize to start
    CHAMELEON_Sequence_Wait(sequence);
    std::cout << "Starting the process\n";
    double t_start = RUNTIME_get_time();
    // Epochs
    for(int epoch = 0; epoch < 10; ++epoch)
    {
        // Generate input
        CHAMELEON_splrnt_Tile_Async(desc_X[0], seed_X, sequence, request);
        // Compute all X
        for(int i = 0; i < D; ++i)
        {
            // X_{i+1} = X_i W_i
            CHAMELEON_sgemm_Tile_Async(NoTrans, NoTrans, alpha, desc_X[i],
                    desc_W[i], beta, desc_X[i+1], user_ws, sequence, request);
        }
        // Set G_D = X_D
        const cham_uplo_t UpperLower = ChamUpperLower;
        CHAMELEON_slacpy_Tile_Async(UpperLower, desc_X[D], desc_G[D], sequence,
                request);
        // Compute all other G
        const cham_trans_t Trans = ChamTrans;
        for(int i = D; i > 1; --i)
        {
            // G_{i-1} = G_i W_{i-1}'
            CHAMELEON_sgemm_Tile_Async(NoTrans, Trans, alpha, desc_G[i],
                    desc_W[i-1], beta, desc_G[i-1], user_ws, sequence, request);
        }
        // Compute all Y
        for(int i = D; i > 0; --i)
        {
            // Y_i = X_i' G_i
            CHAMELEON_sgemm_Tile_Async(Trans, NoTrans, alpha, desc_X[i],
                    desc_G[i], beta, desc_Y[i], user_ws, sequence, request);
        }
        // Update all W_i
        for(int i = 0; i < D; ++i)
        {
            CHAMELEON_sgeadd_Tile_Async(NoTrans, 1e-16, desc_Y[i+1], 1.0,
                    desc_W[i], sequence, request);
        }
    }
    // Get all the matrices back to CPU
    for(int i = 0; i <= D; ++i)
    {
        //CHAMELEON_Desc_Flush(desc_X[i], sequence);
        //CHAMELEON_Desc_Flush(desc_G[i], sequence);
        //CHAMELEON_Desc_Flush(desc_Y[i], sequence);
    }
    // Synchronize to stop
    CHAMELEON_Sequence_Wait(sequence);
    CHAMELEON_Sequence_Destroy(sequence);
    // Measure time
    double t_end = RUNTIME_get_time();
    double t = t_end - t_start;
    size_t flops = (size_t)2 * B * N * N * (D + D + D - 1) * 10;
    float gflops = flops * 1e-9;
    std::cout << "Time, s: " << t << "\n"
        "GFLOPS : " << gflops << "\n"
        "GFLOP/s: " << gflops / t << "\n";
    // Destroy descriptors and free memory
    CHAMELEON_sgemm_WS_Free(user_ws);
    for(int i = 0; i <= D; ++i)
    {
        CHAMELEON_Desc_Destroy(&desc_X[i]);
        CHAMELEON_Desc_Destroy(&desc_G[i]);
        CHAMELEON_Desc_Destroy(&desc_Y[i]);
    }
    for(int i = 0; i < D; ++i)
    {
        CHAMELEON_Desc_Destroy(&desc_W[i]);
    }
    free(desc_X);
    free(desc_W);
    free(desc_G);
    free(desc_Y);
    CHAMELEON_Finalize();
    return 0;
}


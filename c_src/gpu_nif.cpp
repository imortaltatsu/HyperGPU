#include <erl_nif.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

// Error handling
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t err = call; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("CUDNN error: ") + cudnnGetErrorString(err)); \
    } \
}

// Global handles
static cudnnHandle_t cudnn_handle = nullptr;
static cudaStream_t stream = nullptr;

// Convert Erlang binary to C++ vector
static std::vector<float> binary_to_vector(ErlNifBinary* bin) {
    return std::vector<float>(reinterpret_cast<float*>(bin->data),
                            reinterpret_cast<float*>(bin->data + bin->size));
}

// Convert C++ vector to Erlang binary
static ERL_NIF_TERM vector_to_binary(ErlNifEnv* env, const std::vector<float>& vec) {
    ErlNifBinary bin;
    enif_alloc_binary(vec.size() * sizeof(float), &bin);
    memcpy(bin.data, vec.data(), bin.size);
    return enif_make_binary(env, &bin);
}

// Initialize GPU and CuDNN
static ERL_NIF_TERM gpu_init(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    try {
        // Initialize CUDA
        CHECK_CUDA(cudaSetDevice(0));
        
        // Create CUDA stream
        CHECK_CUDA(cudaStreamCreate(&stream));
        
        // Initialize CuDNN
        CHECK_CUDNN(cudnnCreate(&cudnn_handle));
        CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));
        
        // Set deterministic mode
        CHECK_CUDNN(cudnnSetRNNDescriptor_v8(cudnn_handle, CUDNN_RNN_ALGO_STANDARD, 
                                            CUDNN_RNN_SINGLE_INF_DEVICE_MODE, 
                                            CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, 
                                            CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, 
                                            CUDNN_DEFAULT_MATH, CUDNN_DETERMINISTIC));
        
        return enif_make_atom(env, "ok");
    } catch (const std::exception& e) {
        return enif_make_tuple2(env,
                              enif_make_atom(env, "error"),
                              enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    }
}

// Perform GPU computation
static ERL_NIF_TERM gpu_compute(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    try {
        ErlNifBinary input_bin, params_bin;
        
        if (!enif_inspect_binary(env, argv[0], &input_bin) ||
            !enif_inspect_binary(env, argv[1], &params_bin)) {
            return enif_make_badarg(env);
        }
        
        // Convert input data
        std::vector<float> input = binary_to_vector(&input_bin);
        std::vector<float> params = binary_to_vector(&params_bin);
        
        // Allocate GPU memory
        float *d_input, *d_output;
        CHECK_CUDA(cudaMalloc(&d_input, input.size() * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output, input.size() * sizeof(float)));
        
        // Copy data to GPU
        CHECK_CUDA(cudaMemcpy(d_input, input.data(), 
                            input.size() * sizeof(float), 
                            cudaMemcpyHostToDevice));
        
        // TODO: Implement your specific CuDNN operations here
        // This is a placeholder for the actual computation
        
        // Copy result back
        std::vector<float> result(input.size());
        CHECK_CUDA(cudaMemcpy(result.data(), d_output,
                            result.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
        
        // Cleanup
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_output));
        
        return enif_make_tuple2(env,
                              enif_make_atom(env, "ok"),
                              vector_to_binary(env, result));
    } catch (const std::exception& e) {
        return enif_make_tuple2(env,
                              enif_make_atom(env, "error"),
                              enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    }
}

// Terminate GPU resources
static ERL_NIF_TERM gpu_terminate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    try {
        if (cudnn_handle) {
            CHECK_CUDNN(cudnnDestroy(cudnn_handle));
            cudnn_handle = nullptr;
        }
        
        if (stream) {
            CHECK_CUDA(cudaStreamDestroy(stream));
            stream = nullptr;
        }
        
        return enif_make_atom(env, "ok");
    } catch (const std::exception& e) {
        return enif_make_tuple2(env,
                              enif_make_atom(env, "error"),
                              enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    }
}

// Get GPU state
static ERL_NIF_TERM gpu_get_state(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    // TODO: Implement state serialization
    return enif_make_tuple2(env,
                          enif_make_atom(env, "ok"),
                          enif_make_binary(env, nullptr));
}

// Set GPU state
static ERL_NIF_TERM gpu_set_state(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    // TODO: Implement state deserialization
    return enif_make_atom(env, "ok");
}

// NIF function definitions
static ErlNifFunc nif_funcs[] = {
    {"gpu_init", 0, gpu_init},
    {"gpu_compute", 2, gpu_compute},
    {"gpu_terminate", 0, gpu_terminate},
    {"gpu_get_state", 0, gpu_get_state},
    {"gpu_set_state", 1, gpu_set_state}
};

ERL_NIF_INIT(dev_gpu_nif, nif_funcs, NULL, NULL, NULL, NULL) 
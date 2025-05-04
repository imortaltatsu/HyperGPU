#include <erl_nif.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_fp16.h>  // For half precision support
#include "stable-diffusion.cpp/stable-diffusion.h"  // From stable-diffusion.cpp

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
static sd_ctx_t* sd_ctx = nullptr;

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

// Initialize GPU and Stable Diffusion
static ERL_NIF_TERM gpu_init(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    try {
        // Initialize CUDA
        CHECK_CUDA(cudaSetDevice(0));
        
        // Create CUDA stream
        CHECK_CUDA(cudaStreamCreate(&stream));
        
        // Initialize CuDNN
        CHECK_CUDNN(cudnnCreate(&cudnn_handle));
        CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));
        
        // Initialize Stable Diffusion context
        sd_ctx = new_sd_ctx(
            "models/sd-v1-4.ckpt",  // model_path
            nullptr,  // vae_path
            nullptr,  // taesd_path
            nullptr,  // control_net_path
            nullptr,  // lora_model_dir
            nullptr,  // embed_dir
            nullptr,  // stacked_id_embed_dir
            nullptr,  // vae_decode_only
            nullptr,  // vae_tiling
            nullptr,  // free_params_immediately
            nullptr,  // thread_count
            nullptr,  // wtype
            nullptr,  // rng_type
            nullptr,  // schedule
            nullptr,  // clip_skip
            nullptr,  // control_net_cpu
            nullptr,  // normalize_input
            nullptr,  // vae_on_cpu
            nullptr   // verbose
        );
        
        if (!sd_ctx) {
            throw std::runtime_error("Failed to create Stable Diffusion context");
        }
        
        return enif_make_atom(env, "ok");
    } catch (const std::exception& e) {
        return enif_make_tuple2(env,
                              enif_make_atom(env, "error"),
                              enif_make_string(env, e.what(), ERL_NIF_LATIN1));
    }
}

// Perform image generation with Stable Diffusion
static ERL_NIF_TERM gpu_compute(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    try {
        ErlNifBinary prompt_bin, params_bin;
        
        if (!enif_inspect_binary(env, argv[0], &prompt_bin) ||
            !enif_inspect_binary(env, argv[1], &params_bin)) {
            return enif_make_badarg(env);
        }
        
        // Convert input data
        std::string prompt(reinterpret_cast<char*>(prompt_bin.data), prompt_bin.size);
        std::vector<float> params_vec = binary_to_vector(&params_bin);
        
        // Default parameters matching the repository's examples
        int width = 512;
        int height = 512;
        int steps = 20;
        float cfg_scale = 7.0f;
        int seed = -1;  // Random seed
        const char* sampler = "euler_a";  // Default sampler from repository
        const char* schedule = "discrete";  // Default schedule from repository
        
        // Generate image using txt2img
        sd_image_t* image = txt2img(
            sd_ctx,
            prompt.c_str(),
            "",  // negative prompt (empty string as default)
            steps,
            cfg_scale,
            0.0f,  // strength (unused for txt2img)
            0.0f,  // noise (unused for txt2img)
            width,
            height,
            sampler,
            schedule,
            seed,
            -1,  // clip_skip (unspecified)
            nullptr,  // input image (unused for txt2img)
            0.0f,  // control_strength (unused)
            0.0f,  // style_strength (unused)
            false,  // normalize_input
            nullptr,  // control_net_path
            nullptr,  // control_net_cond
            0,  // control_net_cond_size
            0.0f,  // control_net_guidance_start
            0.0f,  // control_net_guidance_end
            0.0f   // control_net_guidance_scale
        );
        
        if (!image) {
            throw std::runtime_error("Failed to generate image");
        }
        
        // Convert image to vector
        std::vector<float> result(image->data, image->data + image->width * image->height * 3);
        
        // Free image
        free_image(image);
        
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
        if (sd_ctx) {
            free_sd_ctx(sd_ctx);
            sd_ctx = nullptr;
        }
        
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
    // TODO: Implement state serialization for Stable Diffusion model
    return enif_make_tuple2(env,
                          enif_make_atom(env, "ok"),
                          enif_make_binary(env, nullptr));
}

// Set GPU state
static ERL_NIF_TERM gpu_set_state(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    // TODO: Implement state deserialization for Stable Diffusion model
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

ERL_NIF_INIT(dev_gpu, nif_funcs, NULL, NULL, NULL, NULL) 
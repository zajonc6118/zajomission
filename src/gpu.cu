#include "gpu.h"
#include "Random.h"

#include <cstdint>
#include <cinttypes>
#include <cstdio>
#include <array>
#include <chrono>
#include <bit>
#include <thread>
#include <mutex>
#include <utility>

#define PANIC(...) { \
    std::fprintf(stderr, __VA_ARGS__); \
    std::abort(); \
}

#define TRY_CUDA(expr) try_cuda(expr, __FILE__, __LINE__)

void try_cuda(cudaError_t error, const char *file, uint64_t line) {
    if (error == cudaSuccess) return;

    PANIC("CUDA error at %s:%" PRIu64 ": %s (%d)\n", file, line, cudaGetErrorString(error), error);
}

// from cubiomes
constexpr XrsrForkHash hash_continentalness { 0x83886c9d0ae3a662, 0xafa638a61b42e8ad }; // md5 "minecraft:continentalness"
constexpr XrsrForkHash hash_continentalness_large { 0x9a3f51a113fce8dc, 0xee2dbd157e5dcdad }; // md5 "minecraft:continentalness_large"
constexpr XrsrForkHash hash_octave[] {
    { 0xb198de63a8012672, 0x7b84cad43ef7b5a8 }, // md5 "octave_-12"
    { 0x0fd787bfbc403ec3, 0x74a4a31ca21b48b8 }, // md5 "octave_-11"
    { 0x36d326eed40efeb2, 0x5be9ce18223c636a }, // md5 "octave_-10"
    { 0x082fe255f8be6631, 0x4e96119e22dedc81 }, // md5 "octave_-9"
    { 0x0ef68ec68504005e, 0x48b6bf93a2789640 }, // md5 "octave_-8"
    { 0xf11268128982754f, 0x257a1d670430b0aa }, // md5 "octave_-7"
    { 0xe51c98ce7d1de664, 0x5f9478a733040c45 }, // md5 "octave_-6"
    { 0x6d7b49e7e429850a, 0x2e3063c622a24777 }, // md5 "octave_-5"
    { 0xbd90d5377ba1b762, 0xc07317d419a7548d }, // md5 "octave_-4"
    { 0x53d39c6752dac858, 0xbcd1c5a80ab65b3e }, // md5 "octave_-3"
    { 0xb4a24d7a84e7677b, 0x023ff9668e89b5c4 }, // md5 "octave_-2"
    { 0xdffa22b534c5f608, 0xb9b67517d3665ca9 }, // md5 "octave_-1"
    { 0xd50708086cef4d7c, 0x6e1651ecc7f43309 }, // md5 "octave_0"
};

struct ImprovedNoise {
    uint8_t p[256];
    float xo;
    float yo;
    float zo;
};

struct Octave {
    ImprovedNoise noise;
    double input_factor;
    double value_factor;
};

template<size_t N>
struct NoiseParameters {
    int32_t first_octave;
    std::array<double, N> amplitudes;
};

template<size_t N>
constexpr NoiseParameters<N> make_noise_parameters(int32_t first_octave, const double (&amplitudes)[N]) {
    std::array<double, N> amp {};
    std::copy(std::begin(amplitudes), std::end(amplitudes), amp.begin());
    return { first_octave, amp };
}

constexpr auto continentalness_parameters = make_noise_parameters(-9, { 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0 });
constexpr auto continentalness_large_parameters = make_noise_parameters(-11, { 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0 });

struct OctaveConfig {
    XrsrForkHash fork_hash;
    double input_factor;
    double value_factor;
};

template<size_t N>
struct NormalNoiseConfig {
    XrsrForkHash fork_hash;
    std::array<OctaveConfig, N> octaves_a;
    std::array<OctaveConfig, N> octaves_b;
};

template<size_t N>
constexpr NormalNoiseConfig<N> make_normal_noise_config(const NoiseParameters<N> &noise_parameters, const XrsrForkHash &fork_hash) {
    NormalNoiseConfig<N> res { fork_hash };

    const auto first_octave = noise_parameters.first_octave;
    const auto &amplitudes = noise_parameters.amplitudes;

    double root_value_factor = 0.16666666666666666 / (0.1 * (1.0 + 1.0 / amplitudes.size()));

    double input_factor = 1.0 / (1 << -first_octave);
    double value_factor = (1 << (amplitudes.size() - 1)) / ((1 << amplitudes.size()) - 1.0) * root_value_factor;

    for (size_t i = 0; i < amplitudes.size(); i++) {
        res.octaves_a[i] = { hash_octave[first_octave + 12 + i], input_factor, value_factor * amplitudes[i] };
        res.octaves_b[i] = { hash_octave[first_octave + 12 + i], input_factor * 1.0181268882175227, value_factor * amplitudes[i] };
        input_factor *= 2.0;
        value_factor *= 0.5;
    }

    return res;
}

constexpr auto continentalness_config = make_normal_noise_config(continentalness_parameters, hash_continentalness);
constexpr auto continentalness_large_config = make_normal_noise_config(continentalness_large_parameters, hash_continentalness_large);
constexpr auto chosen_continentalness_config = large_biomes ? continentalness_large_config : continentalness_config;
__device__ constexpr auto device_chosen_continentalness_config = chosen_continentalness_config;

// switch - 4.745 Gsps
// int8_t[3][16] - 5.293 Gsps
// float[3][16] - 5.324 Gsps
// uint32_t[16] - 5.306 Gsps

struct GradDotTable {
    float x[16];
    float y[16];
    float z[16];
};

__device__ GradDotTable device_grad_dot_table;

void init_grad_dot_table() {
    GradDotTable table;
    table.x[ 0] =  1; table.y[ 0] =  1; table.z[ 0] =  0; // { 1,  1,  0}
    table.x[ 1] = -1; table.y[ 1] =  1; table.z[ 1] =  0; // {-1,  1,  0}
    table.x[ 2] =  1; table.y[ 2] = -1; table.z[ 2] =  0; // { 1, -1,  0}
    table.x[ 3] = -1; table.y[ 3] = -1; table.z[ 3] =  0; // {-1, -1,  0}
    table.x[ 4] =  1; table.y[ 4] =  0; table.z[ 4] =  1; // { 1,  0,  1}
    table.x[ 5] = -1; table.y[ 5] =  0; table.z[ 5] =  1; // {-1,  0,  1}
    table.x[ 6] =  1; table.y[ 6] =  0; table.z[ 6] = -1; // { 1,  0, -1}
    table.x[ 7] = -1; table.y[ 7] =  0; table.z[ 7] = -1; // {-1,  0, -1}
    table.x[ 8] =  0; table.y[ 8] =  1; table.z[ 8] =  1; // { 0,  1,  1}
    table.x[ 9] =  0; table.y[ 9] = -1; table.z[ 9] =  1; // { 0, -1,  1}
    table.x[10] =  0; table.y[10] =  1; table.z[10] = -1; // { 0,  1, -1}
    table.x[11] =  0; table.y[11] = -1; table.z[11] = -1; // { 0, -1, -1}
    table.x[12] =  1; table.y[12] =  1; table.z[12] =  0; // { 1,  1,  0}
    table.x[13] =  0; table.y[13] = -1; table.z[13] =  1; // { 0, -1,  1}
    table.x[14] = -1; table.y[14] =  1; table.z[14] =  0; // {-1,  1,  0}
    table.x[15] =  0; table.y[15] = -1; table.z[15] = -1; // { 0, -1, -1}

    void *device_grad_dot_table_addr;
    TRY_CUDA(cudaGetSymbolAddress(&device_grad_dot_table_addr, device_grad_dot_table));
    TRY_CUDA(cudaMemcpy(device_grad_dot_table_addr, &table, sizeof(GradDotTable), cudaMemcpyHostToDevice));
}

__device__ float gradDot(const GradDotTable &table, uint8_t p, float x, float y, float z) {
    return x * table.x[p & 0xF] + y * table.y[p & 0xF] + z * table.z[p & 0xF];
}

__device__ float smoothstep(float value) {
    return value * value * value * (value * (value * 6.0f - 15.0f) + 10.0f);
}

__device__ float lerp1(float fx, float v0, float v1) {
    return v0 + fx * (v1 - v0);
}

__device__ float lerp2(float fx, float fy, float v00, float v10, float v01, float v11) {
    return lerp1(fy, lerp1(fx, v00, v10), lerp1(fx, v01, v11));
}

__device__ float lerp3(float fx, float fy, float fz, float v000, float v100, float v010, float v110, float v001, float v101, float v011, float v111) {
    return lerp1(fz, lerp2(fx, fy, v000, v100, v010, v110), lerp2(fx, fy, v001, v101, v011, v111));
}

__device__ float sample_noise(const GradDotTable &table, const ImprovedNoise &noise, float x, float y, float z) {
    x += noise.xo;
    y += noise.yo;
    z += noise.zo;
    float floor_x = std::floor(x);
    float floor_y = std::floor(y);
    float floor_z = std::floor(z);
    float frac_x = x - floor_x;
    float frac_y = y - floor_y;
    float frac_z = z - floor_z;
    int32_t int_x = floor_x;
    int32_t int_y = floor_y;
    int32_t int_z = floor_z;
    uint8_t p0 = noise.p[(int_x    ) & 0xFF];
    uint8_t p1 = noise.p[(int_x + 1) & 0xFF];
    uint8_t p00 = noise.p[(p0 + int_y    ) & 0xFF];
    uint8_t p01 = noise.p[(p0 + int_y + 1) & 0xFF];
    uint8_t p10 = noise.p[(p1 + int_y    ) & 0xFF];
    uint8_t p11 = noise.p[(p1 + int_y + 1) & 0xFF];
    float n000 = gradDot(table, noise.p[(p00 + int_z    ) & 0xFF], frac_x       , frac_y       , frac_z       );
    float n100 = gradDot(table, noise.p[(p10 + int_z    ) & 0xFF], frac_x - 1.0f, frac_y       , frac_z       );
    float n010 = gradDot(table, noise.p[(p01 + int_z    ) & 0xFF], frac_x       , frac_y - 1.0f, frac_z       );
    float n110 = gradDot(table, noise.p[(p11 + int_z    ) & 0xFF], frac_x - 1.0f, frac_y - 1.0f, frac_z       );
    float n001 = gradDot(table, noise.p[(p00 + int_z + 1) & 0xFF], frac_x       , frac_y       , frac_z - 1.0f);
    float n101 = gradDot(table, noise.p[(p10 + int_z + 1) & 0xFF], frac_x - 1.0f, frac_y       , frac_z - 1.0f);
    float n011 = gradDot(table, noise.p[(p01 + int_z + 1) & 0xFF], frac_x       , frac_y - 1.0f, frac_z - 1.0f);
    float n111 = gradDot(table, noise.p[(p11 + int_z + 1) & 0xFF], frac_x - 1.0f, frac_y - 1.0f, frac_z - 1.0f);
    float fx = smoothstep(frac_x);
    float fy = smoothstep(frac_y);
    float fz = smoothstep(frac_z);
    return lerp3(fx, fy, fz, n000, n100, n010, n110, n001, n101, n011, n111);
}

__device__ float wrap(float value) {
    // return value - std::floor(value / 256.0) * 256.0;
    return value;
}

template<OctaveConfig config>
__device__ float sample_octave(const GradDotTable &table, const ImprovedNoise &noise, int32_t x, int32_t y, int32_t z) {
    return sample_noise(table, noise, wrap(x * (float)config.input_factor), wrap(y * (float)config.input_factor), wrap(z * (float)config.input_factor)) * (float)config.value_factor;
}

__device__ void init_noise(ImprovedNoise &noise, XrsrRandom &&random) {
    noise.xo = random.nextFloat() * 256.0f;
    noise.yo = random.nextFloat() * 256.0f;
    noise.zo = random.nextFloat() * 256.0f;

    for (uint32_t i = 0; i < 256; i++) {
        noise.p[i] = i;
    }
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t j = random.nextInt(256 - i);
        uint8_t b = noise.p[i];
        noise.p[i] = noise.p[i + j];
        noise.p[i + j] = b;
    }
}

struct DeviceBuffer {
    void *data;
    size_t size;

    DeviceBuffer(size_t size) : size(size) {
        TRY_CUDA(cudaMalloc(&data, size));
    }

    ~DeviceBuffer() {
        TRY_CUDA(cudaFree(data));
    }
};

template<typename T>
struct OutputBuffer {
    T *data;
    uint32_t &len;
    uint32_t max_len;

    OutputBuffer(T *data, uint32_t &len, uint32_t max_len) : data(data), len(len), max_len(max_len) {

    }

    OutputBuffer(const DeviceBuffer &buffer, uint32_t &len) : data((T*)buffer.data), len(len), max_len(buffer.size / sizeof(T)) {

    }

    OutputBuffer(const OutputBuffer<T> &other) : data(other.data), len(other.len), max_len(other.max_len) {

    }
};

template<typename T>
struct InputBuffer {
    const T *data;
    const uint32_t &len;

    InputBuffer(const T *data, const uint32_t &len) : data(data), len(len) {

    }

    InputBuffer(const OutputBuffer<T> &buffer) : data(buffer.data), len(buffer.len) {

    }

    InputBuffer(const InputBuffer<T> &other) : data(other.data), len(other.len) {

    }
};

namespace KernelFilterSeeds {
    constexpr uint32_t threads_per_block = 256;
    // constexpr uint32_t threads_per_run = UINT64_C(1) << 10;
    // constexpr uint32_t threads_per_run = UINT64_C(1) << 22;
    // constexpr uint32_t threads_per_run = UINT64_C(1) << 25;
    constexpr uint32_t threads_per_run = UINT64_C(1) << 28;

    __device__ XrsrRandomFork noise_yo_fork(XrsrRandomFork noise_fork) {
        XrsrRandom rng { noise_fork.lo, noise_fork.hi };
        rng.nextInternal();
        return { rng.lo, rng.hi };
    }

    constexpr XrsrForkHash octave_yo_fork_hash(XrsrForkHash hash) {
        XrsrRandom rng { hash.lo, hash.hi };
        rng.nextInternal();
        return { rng.lo, rng.hi };
    }

    template<OctaveConfig octave_config>
    __device__ float octave_yo_mod1(const XrsrRandomFork &noise_yo_fork) {
        constexpr auto fork_hash = octave_yo_fork_hash(octave_config.fork_hash);

        return ((uint32_t)noise_yo_fork.from(fork_hash).nextBits(32) & 0xFFFFFF) * 5.9604645E-8f;
    }

    __global__ __launch_bounds__(threads_per_block) void kernel(uint64_t start_seed, OutputBuffer<uint64_t> outputs) {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t seed = start_seed + index;

        const auto seed_fork = XrsrRandom(seed).fork();
        auto noise_random = seed_fork.from(device_chosen_continentalness_config.fork_hash);

        auto noise_a_yo_fork = noise_yo_fork(noise_random.fork());
        float c_0A_yo = octave_yo_mod1<chosen_continentalness_config.octaves_a[0]>(noise_a_yo_fork);
        float c_1A_yo = octave_yo_mod1<chosen_continentalness_config.octaves_a[1]>(noise_a_yo_fork);
        float c_2A_yo = octave_yo_mod1<chosen_continentalness_config.octaves_a[2]>(noise_a_yo_fork);

        auto noise_b_yo_fork = noise_yo_fork(noise_random.fork());
        float c_0B_yo = octave_yo_mod1<chosen_continentalness_config.octaves_b[0]>(noise_b_yo_fork);
        float c_1B_yo = octave_yo_mod1<chosen_continentalness_config.octaves_b[1]>(noise_b_yo_fork);
        float c_2B_yo = octave_yo_mod1<chosen_continentalness_config.octaves_b[2]>(noise_b_yo_fork);

        float score =
            .35f * abs(c_0A_yo - .5f) + .35f * abs(c_0B_yo - .5f) +
            .11f * abs(c_1A_yo - .5f) + .11f * abs(c_1B_yo - .5f) +
            .04f * abs(c_2A_yo - .5f) + .04f * abs(c_2B_yo - .5f);
        // if (score >= 0.045f) return; // 1 in 2700
        if (score >= 0.035f) return; // 1 in 9400
        // if (score >= 0.03f) return; // 1 in 26000
        // if (score >= 0.025f) return; // 1 in 54000

        uint32_t result_index = atomicAdd(&outputs.len, 1);
        if (result_index >= outputs.max_len) return;
        outputs.data[result_index] = seed;
    }

    void run(uint64_t start_seed, OutputBuffer<uint64_t> outputs) {
        kernel<<<threads_per_run / threads_per_block, threads_per_block>>>(start_seed, outputs);
        TRY_CUDA(cudaGetLastError());
    }
}

namespace KernelSeed1 {
    constexpr uint32_t threads_per_run = UINT64_C(1) << 16;
    constexpr uint32_t threads_per_block = 32;

    struct Result {
        ImprovedNoise continentalness_0A;
        ImprovedNoise continentalness_0B;
        ImprovedNoise continentalness_1A;
        ImprovedNoise continentalness_1B;
        ImprovedNoise continentalness_2A;
        ImprovedNoise continentalness_2B;
        ImprovedNoise continentalness_3A;
        ImprovedNoise continentalness_3B;
        ImprovedNoise continentalness_4A;
        ImprovedNoise continentalness_4B;
        ImprovedNoise continentalness_5A;
        ImprovedNoise continentalness_5B;
        ImprovedNoise continentalness_6A;
        ImprovedNoise continentalness_6B;
        ImprovedNoise continentalness_7A;
        ImprovedNoise continentalness_7B;
        ImprovedNoise continentalness_8A;
        ImprovedNoise continentalness_8B;
    };

    template<size_t Octaves>
    struct ResultSampler {
        ImprovedNoise octaves[Octaves];

        __device__ float sample(const GradDotTable &table, int32_t x, int32_t y, int32_t z) const {
            float val = 0;
            if constexpr (Octaves >=  1) val += sample_octave<chosen_continentalness_config.octaves_a[0]>(table, octaves[ 0], x, y, z);
            if constexpr (Octaves >=  2) val += sample_octave<chosen_continentalness_config.octaves_b[0]>(table, octaves[ 1], x, y, z);
            if constexpr (Octaves >=  3) val += sample_octave<chosen_continentalness_config.octaves_a[1]>(table, octaves[ 2], x, y, z);
            if constexpr (Octaves >=  4) val += sample_octave<chosen_continentalness_config.octaves_b[1]>(table, octaves[ 3], x, y, z);
            if constexpr (Octaves >=  5) val += sample_octave<chosen_continentalness_config.octaves_a[2]>(table, octaves[ 4], x, y, z);
            if constexpr (Octaves >=  6) val += sample_octave<chosen_continentalness_config.octaves_b[2]>(table, octaves[ 5], x, y, z);
            if constexpr (Octaves >=  7) val += sample_octave<chosen_continentalness_config.octaves_a[3]>(table, octaves[ 6], x, y, z);
            if constexpr (Octaves >=  8) val += sample_octave<chosen_continentalness_config.octaves_b[3]>(table, octaves[ 7], x, y, z);
            if constexpr (Octaves >=  9) val += sample_octave<chosen_continentalness_config.octaves_a[4]>(table, octaves[ 8], x, y, z);
            if constexpr (Octaves >= 10) val += sample_octave<chosen_continentalness_config.octaves_b[4]>(table, octaves[ 9], x, y, z);
            if constexpr (Octaves >= 11) val += sample_octave<chosen_continentalness_config.octaves_a[5]>(table, octaves[10], x, y, z);
            if constexpr (Octaves >= 12) val += sample_octave<chosen_continentalness_config.octaves_b[5]>(table, octaves[11], x, y, z);
            if constexpr (Octaves >= 13) val += sample_octave<chosen_continentalness_config.octaves_a[6]>(table, octaves[12], x, y, z);
            if constexpr (Octaves >= 14) val += sample_octave<chosen_continentalness_config.octaves_b[6]>(table, octaves[13], x, y, z);
            if constexpr (Octaves >= 15) val += sample_octave<chosen_continentalness_config.octaves_a[7]>(table, octaves[14], x, y, z);
            if constexpr (Octaves >= 16) val += sample_octave<chosen_continentalness_config.octaves_b[7]>(table, octaves[15], x, y, z);
            if constexpr (Octaves >= 17) val += sample_octave<chosen_continentalness_config.octaves_a[8]>(table, octaves[16], x, y, z);
            if constexpr (Octaves >= 18) val += sample_octave<chosen_continentalness_config.octaves_b[8]>(table, octaves[17], x, y, z);
            return val;
        }
    };

    __device__ Result results[threads_per_run];
    // constexpr size_t a = sizeof(results);

    __device__ void copy_noise(ImprovedNoise (&shared_noise)[threads_per_block], ImprovedNoise Result::* result_member) {
        for (uint32_t result_index = 0; result_index < threads_per_block; result_index++) {
            ImprovedNoise &src = shared_noise[result_index];
            ImprovedNoise &dst = results[blockIdx.x * blockDim.x + result_index].*result_member;
            for (uint32_t i = threadIdx.x; i < sizeof(ImprovedNoise) / sizeof(uint32_t); i += threads_per_block) {
                reinterpret_cast<uint32_t*>(&dst)[i] = reinterpret_cast<uint32_t*>(&src)[i];
            }
        }
    }

    __device__ void init_octave(const XrsrRandomFork &noise_fork, const XrsrForkHash &fork_hash, ImprovedNoise Result::* result_member) {
        __shared__ ImprovedNoise shared_noise[threads_per_block];

        init_noise(shared_noise[threadIdx.x], noise_fork.from(fork_hash));
        __syncthreads();
        copy_noise(shared_noise, result_member);
        __syncthreads();
    }

    __global__ __launch_bounds__(threads_per_block) void kernel(InputBuffer<uint64_t> input) {
        uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index / threads_per_block * threads_per_block >= input.len) return;
        uint64_t seed = input.data[index];

        const auto seed_fork = XrsrRandom(seed).fork();
        auto noise_random = seed_fork.from(device_chosen_continentalness_config.fork_hash);

        const auto noise_a_fork = noise_random.fork();
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[0].fork_hash, &Result::continentalness_0A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[1].fork_hash, &Result::continentalness_1A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[2].fork_hash, &Result::continentalness_2A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[3].fork_hash, &Result::continentalness_3A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[4].fork_hash, &Result::continentalness_4A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[5].fork_hash, &Result::continentalness_5A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[6].fork_hash, &Result::continentalness_6A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[7].fork_hash, &Result::continentalness_7A);
        init_octave(noise_a_fork, device_chosen_continentalness_config.octaves_a[8].fork_hash, &Result::continentalness_8A);

        const auto noise_b_fork = noise_random.fork();
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[0].fork_hash, &Result::continentalness_0B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[1].fork_hash, &Result::continentalness_1B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[2].fork_hash, &Result::continentalness_2B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[3].fork_hash, &Result::continentalness_3B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[4].fork_hash, &Result::continentalness_4B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[5].fork_hash, &Result::continentalness_5B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[6].fork_hash, &Result::continentalness_6B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[7].fork_hash, &Result::continentalness_7B);
        init_octave(noise_b_fork, device_chosen_continentalness_config.octaves_b[8].fork_hash, &Result::continentalness_8B);
    }
}

struct SeedPos {
    uint32_t seed_index;
    int32_t x;
    int32_t z;
};

constexpr int32_t large_biomes_pos_mul = large_biomes ? 4 : 1;

#include "kernel_0A.h"
__device__ float device_kernel_0A[2][6][6][16];
static_assert(sizeof(host_kernel_0A) == sizeof(device_kernel_0A));
#include "kernel_0B.h"
__device__ float device_kernel_0B[2][6][6][16];
static_assert(sizeof(host_kernel_0B) == sizeof(device_kernel_0B));

void init_conv_kernels() {
    void *device_kernel_0A_addr;
    TRY_CUDA(cudaGetSymbolAddress(&device_kernel_0A_addr, device_kernel_0A));
    TRY_CUDA(cudaMemcpy(device_kernel_0A_addr, host_kernel_0A, sizeof(host_kernel_0A), cudaMemcpyHostToDevice));

    void *device_kernel_0B_addr;
    TRY_CUDA(cudaGetSymbolAddress(&device_kernel_0B_addr, device_kernel_0B));
    TRY_CUDA(cudaMemcpy(device_kernel_0B_addr, host_kernel_0B, sizeof(host_kernel_0B), cudaMemcpyHostToDevice));
}

namespace KernelFilterGradVecs1 {
    constexpr uint32_t block_dim_x = 256;

    // __managed__ float scores[256][256];

    __global__ __launch_bounds__(block_dim_x) void kernel(InputBuffer<uint64_t> seeds, OutputBuffer<SeedPos> outputs) {
        __shared__ ImprovedNoise oct_0A;
        __shared__ float shared_kernel_0A[2][6][6][16];
        __shared__ float conv_z[2][6][256];
        __shared__ int32_t idx_xy[2][256];

        for (uint32_t i = threadIdx.x; i < sizeof(shared_kernel_0A) / sizeof(uint32_t); i += block_dim_x) {
            reinterpret_cast<uint32_t*>(&shared_kernel_0A)[i] = reinterpret_cast<uint32_t*>(device_kernel_0A)[i];
        }

        uint32_t seeds_len = seeds.len;
        for (uint32_t seed_index = blockIdx.x; seed_index < seeds_len; seed_index += gridDim.x) {
            __syncthreads();
            if (threadIdx.x < sizeof(oct_0A) / sizeof(uint32_t)) {
                reinterpret_cast<uint32_t*>(&oct_0A)[threadIdx.x] = reinterpret_cast<uint32_t*>(&KernelSeed1::results[seed_index].continentalness_0A)[threadIdx.x];
            }
            __syncthreads();

            for (int32_t dny = 0; dny < 2; dny++) {
                for (int32_t dnx = 0; dnx < 6; dnx++) {
                    int32_t nz = threadIdx.x;
                    float conv = 0;
                    for (int32_t dnz = 0; dnz < 6; dnz++) {
                        conv += shared_kernel_0A[dny][dnx][dnz][oct_0A.p[(nz + dnz) & 0xFF] & 0xF];
                    }
                    conv_z[dny][dnx][nz] = conv;
                }
            }

            int32_t cell_size = 512 * large_biomes_pos_mul;

            int32_t x_center = (2.5f - oct_0A.xo) * cell_size;
            int32_t ny = oct_0A.yo;
            int32_t z_center = (2.5f - oct_0A.zo) * cell_size;

            int32_t idx_x = oct_0A.p[threadIdx.x];
            idx_xy[0][threadIdx.x] = oct_0A.p[(idx_x + ny) & 0xFF];
            idx_xy[1][threadIdx.x] = oct_0A.p[(idx_x + ny + 1) & 0xFF];
            __syncthreads();

            for (int32_t nx = 0; nx < 256; nx++) {
                int32_t nz = threadIdx.x;
                float conv = 0;
                for (int32_t dny = 0; dny < 2; dny++) {
                    for (int32_t dnx = 0; dnx < 6; dnx++) {
                        int32_t idx_xy0 = idx_xy[dny][(nx + dnx) & 0xFF];
                        conv += conv_z[dny][dnx][(idx_xy0 + nz) & 0xFF];
                    }
                }
                // if (seeds.data[seed_index] == 123) {
                //     scores[nx][nz] = conv;
                // }
                // if (blockIdx.x == 42) {
                //     printf("GV %" PRIi64 " %i %i %.3f\n", seeds.data[seed_index], nx, nz, conv);
                // }

                int32_t x = x_center + nx * cell_size;
                int32_t z = z_center + nz * cell_size;

                if (conv >= -21) {
                    uint32_t result_index = atomicAdd(&outputs.len, 1);
                    if (result_index >= outputs.max_len) continue;
                    outputs.data[result_index] = { seed_index, x, z };
                }
            }
        }
    }

    void run(InputBuffer<uint64_t> seeds, OutputBuffer<SeedPos> outputs) {
        kernel<<<2048, block_dim_x>>>(seeds, outputs);
        TRY_CUDA(cudaGetLastError());
    }
}

namespace KernelFilterGradVecs2 {
    constexpr uint32_t block_dim_x = 128;

    __global__ __launch_bounds__(block_dim_x) void kernel(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {
        __shared__ ImprovedNoise oct_0B;
        __shared__ float shared_kernel_0B[2][6][6][16];

        for (uint32_t i = threadIdx.x; i < sizeof(shared_kernel_0B) / sizeof(uint32_t); i += block_dim_x) {
            reinterpret_cast<uint32_t*>(&shared_kernel_0B)[i] = reinterpret_cast<uint32_t*>(device_kernel_0B)[i];
        }

        constexpr int32_t cell_size_0A = (int32_t)(1 / chosen_continentalness_config.octaves_a[0].input_factor) * 256;
        int32_t tile_dx = (threadIdx.x % 12 - 6) * cell_size_0A;
        int32_t tile_dz = (threadIdx.x / 12 - 6) * cell_size_0A;

        uint32_t inputs_len = inputs.len;
        for (uint32_t input_index = blockIdx.x; input_index < inputs_len; input_index += gridDim.x) {
            SeedPos input = inputs.data[input_index];
            int32_t x = input.x + tile_dx;
            int32_t z = input.z + tile_dz;

            __syncthreads();
            if (threadIdx.x < sizeof(oct_0B) / sizeof(uint32_t)) {
                reinterpret_cast<uint32_t*>(&oct_0B)[threadIdx.x] = reinterpret_cast<uint32_t*>(&KernelSeed1::results[input.seed_index].continentalness_0B)[threadIdx.x];
            }
            __syncthreads();

            int32_t nx = std::floor(x * (float)chosen_continentalness_config.octaves_b[0].input_factor + oct_0B.xo - 2.0f);
            int32_t ny = oct_0B.yo;
            int32_t nz = std::floor(z * (float)chosen_continentalness_config.octaves_b[0].input_factor + oct_0B.zo - 2.0f);

            float conv = 0;
            for (int32_t dnx = 0; dnx < 6; dnx++) {
                int32_t idx_x = oct_0B.p[(nx + dnx) & 0xFF];
                for (int32_t dny = 0; dny < 2; dny++) {
                    int32_t idx_xy = oct_0B.p[(idx_x + ny + dny) & 0xFF];
                    for (int32_t dnz = 0; dnz < 6; dnz++) {
                        int32_t idx_xyz = oct_0B.p[(idx_xy + nz + dnz) & 0xFF];
                        conv += shared_kernel_0B[dny][dnx][dnz][idx_xyz & 0xF];
                    }
                }
            }

            if (conv > -21.5) {
                uint32_t result_index = atomicAdd(&outputs.len, 1);
                if (result_index >= outputs.max_len) continue;
                outputs.data[result_index] = { input.seed_index, x, z };
            }
        }
    }

    void run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {
        kernel<<<2048, block_dim_x>>>(inputs, outputs);
        TRY_CUDA(cudaGetLastError());
    }
}

namespace KernelFilter1 {
    constexpr uint32_t threads_per_block = 256;
    constexpr uint32_t threads_per_seed_sqrt = UINT64_C(1) << 10;
    constexpr uint32_t threads_per_seed = threads_per_seed_sqrt * threads_per_seed_sqrt;
    // noise (1:4) coords
    constexpr int32_t pos_step = 14600 * large_biomes_pos_mul / 4;
    constexpr int32_t pos_range = (int32_t)threads_per_seed_sqrt * pos_step;
    static_assert(pos_range <= 60'000'000 / 4);

    __global__ __launch_bounds__(threads_per_block) void kernel(InputBuffer<uint64_t> seeds, OutputBuffer<SeedPos> outputs) {
        __shared__ GradDotTable shared_grad_dot_table;
        if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
            reinterpret_cast<uint32_t*>(&shared_grad_dot_table)[threadIdx.x] = reinterpret_cast<uint32_t*>(&device_grad_dot_table)[threadIdx.x];
        }

        __shared__ KernelSeed1::ResultSampler<2> shared_octaves;

        uint32_t seeds_len = seeds.len;
        for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < seeds_len * threads_per_seed; index += gridDim.x * blockDim.x) {
            uint32_t seed_index = index / threads_per_seed;
            uint32_t pos_index = index % threads_per_seed;

            __syncthreads();
            if (threadIdx.x < sizeof(shared_octaves) / sizeof(uint32_t)) {
                reinterpret_cast<uint32_t*>(&shared_octaves)[threadIdx.x] = reinterpret_cast<uint32_t*>(&KernelSeed1::results[seed_index])[threadIdx.x];
            }
            __syncthreads();

            uint32_t x_index = pos_index % threads_per_seed_sqrt;
            uint32_t z_index = pos_index / threads_per_seed_sqrt;

            int32_t x = (int32_t)x_index * pos_step - pos_range / 2;
            int32_t z = (int32_t)z_index * pos_step - pos_range / 2;

            float val = shared_octaves.sample(shared_grad_dot_table, x, 0, z);

            if (val >= -0.515f) continue; // 1 in 27.7
            // if (val >= -0.7f) continue; // 1 in 176
            // if (val >= -0.8f) continue;
            // if (val >= -1.48f) continue;

            uint32_t result_index = atomicAdd(&outputs.len, 1);
            if (result_index >= outputs.max_len) continue;
            outputs.data[result_index] = { seed_index, x, z };
        }
    }

    void run(InputBuffer<uint64_t> seeds, OutputBuffer<SeedPos> outputs) {
        kernel<<<16 * 1024, threads_per_block>>>(seeds, outputs);
        TRY_CUDA(cudaGetLastError());
    }
}

constexpr bool is_pow2(uint32_t val) {
    return (val & (val - 1)) == 0;
}

constexpr uint32_t log2(uint32_t val) {
    return 31 - std::countl_zero(val);
}

template<typename T>
__device__ T warp_reduce_add(T val) {
#if __CUDA_ARCH__ >= 800
    return __reduce_add_sync(0xFFFFFFFF, val);
#else
    val += __shfl_down_sync(-1u, val, 1);
    val += __shfl_down_sync(-1u, val, 2);
    val += __shfl_down_sync(-1u, val, 4);
    val += __shfl_down_sync(-1u, val, 8);
    val += __shfl_down_sync(-1u, val, 16);
    return val;
#endif
}

namespace KernelFilter2 {
    template<int32_t NoiseThreshold, size_t Octaves, uint32_t PosRange, uint32_t Samples, uint32_t MinCount, bool FlippedSparseSamples, bool MoveCenter>
    struct Template {
        static constexpr float noise_threshold = NoiseThreshold / 10000.0f;
        static constexpr size_t octaves = Octaves;
        static constexpr uint32_t pos_range = PosRange;
        static constexpr uint32_t samples = Samples;
        static constexpr uint32_t min_count = MinCount;
        static constexpr bool flipped_sparse_samples = FlippedSparseSamples;
        static constexpr bool move_center = MoveCenter;

        static constexpr uint32_t threads_per_block = 256;
        static_assert(samples >= 32 && samples <= threads_per_block * threads_per_block && is_pow2(samples));
        static constexpr uint32_t samples_square_size = UINT32_C(1) << (log2(samples) + 1) / 2;
        static constexpr bool samples_square_sparse = log2(samples) % 2 == 1;
        static_assert(!flipped_sparse_samples || samples_square_sparse);
        static_assert(pos_range * large_biomes_pos_mul % (samples_square_size * 2 * 4) == 0);
        // noise (1:4) coords
        static constexpr uint32_t pos_step = pos_range * large_biomes_pos_mul / 4 / samples_square_size;
        static constexpr int32_t pos_offset = -(int32_t)(pos_step * (samples_square_size - 1) / 2);

        static constexpr uint32_t threads_per_input = std::min(samples, threads_per_block);
        static constexpr uint32_t loops = samples / threads_per_input;
        static constexpr uint32_t inputs_per_block = threads_per_block / threads_per_input;

        static void run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs);
    };

    template<typename T>
    __global__ __launch_bounds__(T::threads_per_block) void kernel(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {
        __shared__ GradDotTable shared_grad_dot_table;
        if (threadIdx.x < sizeof(shared_grad_dot_table) / sizeof(uint32_t)) {
            reinterpret_cast<uint32_t*>(&shared_grad_dot_table)[threadIdx.x] = reinterpret_cast<uint32_t*>(&device_grad_dot_table)[threadIdx.x];
        }

        uint32_t inputs_len = inputs.len;
        for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x; index < inputs_len * T::threads_per_input; index += gridDim.x * blockDim.x) {
            __syncthreads();
            uint32_t input_index = index / T::threads_per_input;
            uint32_t pos_index = index % T::threads_per_input;
            uint32_t block_input_index = threadIdx.x / T::threads_per_input;

            const auto input = inputs.data[input_index];

            __shared__ KernelSeed1::ResultSampler<T::octaves> shared_octaves[T::inputs_per_block];
            for (uint32_t i = pos_index; i < sizeof(shared_octaves[0]) / sizeof(uint32_t); i += T::threads_per_input) {
                reinterpret_cast<uint32_t*>(&shared_octaves[block_input_index])[i] = reinterpret_cast<uint32_t*>(&KernelSeed1::results[input.seed_index])[i];
            }
            __syncthreads();

            uint32_t x_index = pos_index % T::samples_square_size;
            uint32_t z_index = pos_index / T::samples_square_size;
            if constexpr (T::samples_square_sparse) {
                z_index = z_index * 2 + ((x_index & 1) ^ T::flipped_sparse_samples);
            }

            int32_t x = input.x + (int32_t)(x_index * T::pos_step) + T::pos_offset;
            int32_t z = input.z + (int32_t)(z_index * T::pos_step) + T::pos_offset;

            uint32_t total_valid = 0;
            int32_t sum_dx = 0;
            int32_t sum_dz = 0;

            for (uint32_t i = 0; i < T::loops; i++) {
                float val = shared_octaves[block_input_index].sample(shared_grad_dot_table, x, 0, z);

                bool valid = val < T::noise_threshold;

                total_valid += warp_reduce_add((uint32_t)valid);

                if constexpr (T::move_center) {
                    if (valid) {
                        sum_dx += x - input.x;
                        sum_dz += z - input.z;
                    }
                }

                z += (int32_t)(T::pos_step * (T::samples_square_size / T::loops));
            }

            if constexpr (T::samples > 32) {
                __shared__ uint32_t shared_counts[T::inputs_per_block];
                if (threadIdx.x < T::inputs_per_block) {
                    shared_counts[threadIdx.x] = 0;
                }
                __syncthreads();
                if (threadIdx.x % 32 == 0) {
                    atomicAdd(&shared_counts[block_input_index], total_valid);
                }
                __syncthreads();
                total_valid = shared_counts[block_input_index];
            }

            if constexpr (T::move_center) {
                sum_dx = warp_reduce_add(sum_dx);
                sum_dz = warp_reduce_add(sum_dz);
                if constexpr (T::samples > 32) {
                    __shared__ int32_t shared_sums[T::inputs_per_block][2];
                    if (threadIdx.x < T::inputs_per_block) {
                        shared_sums[threadIdx.x][0] = 0;
                        shared_sums[threadIdx.x][1] = 0;
                    }
                    __syncthreads();
                    if (threadIdx.x % 32 == 0) {
                        atomicAdd(&shared_sums[block_input_index][0], sum_dx);
                        atomicAdd(&shared_sums[block_input_index][1], sum_dz);
                    }
                    __syncthreads();
                    sum_dx = shared_sums[block_input_index][0];
                    sum_dz = shared_sums[block_input_index][1];
                }
                if (total_valid != 0) {
                    sum_dx /= (int32_t)total_valid;
                    sum_dz /= (int32_t)total_valid;
                }
            }

            if (total_valid < T::min_count) continue;

            if (pos_index == 0) {
                uint32_t result_index = atomicAdd(&outputs.len, 1);
                if (result_index >= outputs.max_len) continue;
                outputs.data[result_index] = { input.seed_index, input.x + sum_dx, input.z + sum_dz };
            }
        }
    }

    template<int32_t NoiseThreshold, size_t Octaves, uint32_t PosRange, uint32_t Samples, uint32_t MinCount, bool FlippedSparseSamples, bool MoveCenter>
    void Template<NoiseThreshold, Octaves, PosRange, Samples, MinCount, FlippedSparseSamples, MoveCenter>::run(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs) {
        using T = Template<NoiseThreshold, Octaves, PosRange, Samples, MinCount, FlippedSparseSamples, MoveCenter>;
        kernel<T><<<32 * 256, T::threads_per_block>>>(inputs, outputs);
        TRY_CUDA(cudaGetLastError());
    }
}

struct CudaEventWrapper {
    cudaEvent_t event;

    CudaEventWrapper() : event(nullptr) {
        TRY_CUDA(cudaEventCreate(&event));
    }

    CudaEventWrapper(CudaEventWrapper &&other) : event(other.event) {
        other.event = nullptr;
    }

    ~CudaEventWrapper() {
        if (event == nullptr) return;
        TRY_CUDA(cudaEventDestroy(event));
    }

    void record(cudaStream_t stream = 0) const {
        TRY_CUDA(cudaEventRecord(event, stream));
    }

    float elapsed(const CudaEventWrapper &end) const {
        float ms;
        TRY_CUDA(cudaEventElapsedTime(&ms, event, end.event));
        return ms;
    }

    void synchronize() const {
        TRY_CUDA(cudaEventSynchronize(event));
    }
};

struct StageStats {
    std::string name;
    uint32_t *inputs_len;
    uint32_t *outputs_len;
    uint64_t inputs_multiplier;
    uint32_t max_outputs_len;
    CudaEventWrapper event;
    double total_time;
    uint64_t total_inputs;
    uint64_t total_outputs;

    StageStats(std::string name, uint32_t *inputs_len, uint32_t *outputs_len, uint64_t inputs_len_multiplier, uint32_t max_outputs_len) :
        name(std::move(name)),
        inputs_len(inputs_len),
        outputs_len(outputs_len),
        inputs_multiplier(inputs_len_multiplier),
        max_outputs_len(max_outputs_len),
        event(),
        total_time(),
        total_inputs(),
        total_outputs()
    {

    }

    StageStats(StageStats &&other) :
        name(std::move(other.name)),
        inputs_len(other.inputs_len),
        outputs_len(other.outputs_len),
        inputs_multiplier(other.inputs_multiplier),
        max_outputs_len(other.max_outputs_len),
        event(std::move(other.event)),
        total_time(other.total_time),
        total_inputs(other.total_inputs),
        total_outputs(other.total_outputs)
    {

    }

    void record() {
        event.record();
    }

    void update(CudaEventWrapper &prev_event) {
        total_time += prev_event.elapsed(event) * 1e-3;
        total_inputs += inputs_len ? *inputs_len : 1;
        total_outputs += *outputs_len;
        if (*outputs_len > max_outputs_len) {
            std::printf("%s outputs overflow: len = %" PRIu32 " max_len = %" PRIu32 "\n", name.c_str(), *outputs_len, max_outputs_len);
        }
    }

    void reset() {
        total_time = 0;
        total_inputs = 0;
        total_outputs = 0;
    }
};

std::pair<double, char> scale_si(double val) {
    std::pair<double, char> units[] = {
        { 1e9 , 'G' },
        { 1e6 , 'M' },
        { 1e3 , 'k' },
    };
    for (auto [unit_scale, unit] : units) {
        if (val >= unit_scale) {
            return { val / unit_scale, unit };
        }
    }
    return { val, ' ' };
}

struct BufferLens {
    uint32_t results_len_filter_seeds;
    uint32_t results_len_filter_gradvecs_1;
    uint32_t results_len_filter_gradvecs_2;
    uint32_t results_len_filter_1;
    uint32_t results_len_filter_2[7];
};

GpuThread::GpuThread(int device, SeedIterator &input, GpuOutputs &outputs) : Thread(), device(device), input(input), outputs(outputs) {
    start();
}

void GpuThread::run() {
    std::printf("Initializing device %d\n", device);

    TRY_CUDA(cudaSetDevice(device));
    init_grad_dot_table();
    init_conv_kernels();

    BufferLens host_buffer_lens;
    BufferLens *device_buffer_lens;
    TRY_CUDA(cudaMalloc(&device_buffer_lens, sizeof(*device_buffer_lens)));

    std::printf("Running device %d\n", device);

    DeviceBuffer buffer_seeds(sizeof(uint64_t) * KernelSeed1::threads_per_run);
    DeviceBuffer buffer_1(UINT32_C(1) << 31);
    DeviceBuffer buffer_2(UINT32_C(1) << 29);
    std::vector<SeedPos> h_buffer;
    std::vector<StageStats> stage_stats;
    stage_stats.reserve(16);

    CudaEventWrapper event_start;

    OutputBuffer<uint64_t> outputs_filter_seeds(buffer_seeds, device_buffer_lens->results_len_filter_seeds);
    auto &stage_filter_seeds = stage_stats.emplace_back("filter_seeds", nullptr, &host_buffer_lens.results_len_filter_seeds, KernelFilterSeeds::threads_per_run, outputs_filter_seeds.max_len);

    auto &stage_init_seeds = stage_stats.emplace_back("init_seeds", stage_filter_seeds.outputs_len, stage_filter_seeds.outputs_len, 1, KernelSeed1::threads_per_run);

    OutputBuffer<SeedPos> outputs_filter_gradvecs_1(buffer_2, device_buffer_lens->results_len_filter_gradvecs_1);
    auto &stage_filter_gradvecs_1 = stage_stats.emplace_back("filter_gradvecs_1", stage_filter_seeds.outputs_len, &host_buffer_lens.results_len_filter_gradvecs_1, 256 * 256, outputs_filter_gradvecs_1.max_len);

    OutputBuffer<SeedPos> outputs_filter_gradvecs_2(buffer_1, device_buffer_lens->results_len_filter_gradvecs_2);
    auto &stage_filter_gradvecs_2 = stage_stats.emplace_back("filter_gradvecs_2", stage_filter_gradvecs_1.outputs_len, &host_buffer_lens.results_len_filter_gradvecs_2, 128, outputs_filter_gradvecs_2.max_len);

    // namespace Filter1 = KernelFilter1;
    // OutputBuffer<SeedPos> outputs_filter_1(buffer_1, device_buffer_lens->results_len_filter_1);
    // auto &stage_filter_1 = stage_stats.emplace_back("filter_1", stage_filter_seeds.outputs_len, &host_buffer_lens.results_len_filter_1, KernelFilter1::threads_per_seed, outputs_filter_1.max_len);

    using Kernel2RunFunc = void (*)(InputBuffer<SeedPos> inputs, OutputBuffer<SeedPos> outputs);
    struct Filter2Stage {
        Kernel2RunFunc run;
        OutputBuffer<SeedPos> outputs;
        StageStats &stage;

        Filter2Stage(Kernel2RunFunc run, OutputBuffer<SeedPos> outputs, StageStats &stage) : run(run), outputs(outputs), stage(stage) {

        }
    };
    Kernel2RunFunc filter_2_runs[] = {
        // KernelFilter2::Template<-7400, 2, 6 * 1024, 32, 3, false, true>::run, // 6m | P(X < x) = 0.0973
        // KernelFilter2::Template<-7400, 2, 6 * 1024, 128, 11, true, true>::run, // 6m | P(X < x) = <0.01
        // KernelFilter2::Template<-7400, 2, 6 * 1024, 512, 63, false, true>::run, // 6m | P(X < x) = <0.01
        KernelFilter2::Template<-10500, 18, 8 * 1024, 256, 13, false, true>::run, // 6m | P(X < x) = <0.01
        KernelFilter2::Template<-10500, 18, 8 * 1024, 1024, 71, false, true>::run, // 6m | P(X < x) = <0.01
        KernelFilter2::Template<-10500, 18, 8 * 1024, 16384, 1380, false, false>::run, // 6m | P(X < x) = <0.01
        KernelFilter2::Template<-10500, 18, 8 * 1024, 65536, 5690, false, false>::run, // 6m | P(X < x) = <0.01
    };
    std::vector<Filter2Stage> filter_2;
    {
        // uint32_t *inputs_len = stage_filter_1.outputs_len;
        uint32_t *inputs_len = stage_filter_gradvecs_2.outputs_len;
        for (size_t i = 0; i < sizeof(filter_2_runs) / sizeof(*filter_2_runs); i++) {
            OutputBuffer<SeedPos> outputs(i % 2 == 0 ? buffer_2 : buffer_1, device_buffer_lens->results_len_filter_2[i]);

            uint32_t *outputs_len = &host_buffer_lens.results_len_filter_2[i];
            auto &stage = stage_stats.emplace_back(std::string("filter_2") + (char)('a' + i), inputs_len, outputs_len, 1, outputs.max_len);
            inputs_len = outputs_len;

            filter_2.emplace_back(filter_2_runs[i], outputs, stage);
        }
    }

    int print_interval = 16;

    // for (int32_t nx = 0; nx < 256; nx++) {
    //     for (int32_t nz = 0; nz < 256; nz++) {
    //         KernelFilterGradVecs1::scores[nx][nz] = -123;
    //     }
    // }

    auto start = std::chrono::steady_clock::now();

    for (uint32_t i = 0; !should_stop(); i++) {
        uint64_t start_seed = input.next(KernelFilterSeeds::threads_per_run);

        TRY_CUDA(cudaMemsetAsync(device_buffer_lens, 0, sizeof(*device_buffer_lens)));

        event_start.record();

        KernelFilterSeeds::run(start_seed, outputs_filter_seeds);
        stage_filter_seeds.record();

        KernelSeed1::kernel<<<KernelSeed1::threads_per_run / KernelSeed1::threads_per_block, KernelSeed1::threads_per_block>>>(outputs_filter_seeds);
        TRY_CUDA(cudaGetLastError());
        stage_init_seeds.record();

        KernelFilterGradVecs1::run(outputs_filter_seeds, outputs_filter_gradvecs_1);
        stage_filter_gradvecs_1.record();

        KernelFilterGradVecs2::run(outputs_filter_gradvecs_1, outputs_filter_gradvecs_2);
        stage_filter_gradvecs_2.record();

        // Filter1::run(outputs_filter_seeds, outputs_filter_1);
        // stage_filter_1.record();

        {
            // OutputBuffer<SeedPos> *inputs = &outputs_filter_1;
            OutputBuffer<SeedPos> *inputs = &outputs_filter_gradvecs_2;
            for (auto &filter : filter_2) {
                filter.run(*inputs, filter.outputs);
                filter.stage.record();
                inputs = &filter.outputs;
            }
        }

        TRY_CUDA(cudaMemcpyAsync(&host_buffer_lens, device_buffer_lens, sizeof(host_buffer_lens), cudaMemcpyDeviceToHost));

        TRY_CUDA(cudaDeviceSynchronize());

        {
            CudaEventWrapper *prev_event = &event_start;
            for (auto &stage : stage_stats) {
                stage.update(*prev_event);
                prev_event = &stage.event;
            }
        }

        const auto &final_outputs = filter_2.back().outputs;
        const auto &final_outputs_len = *filter_2.back().stage.outputs_len;
        if (final_outputs_len > 0) {
            // uint32_t len = std::min(final_outputs_len, UINT32_C(10));
            uint32_t len = final_outputs_len;
            h_buffer.resize(len);
            TRY_CUDA(cudaMemcpy(h_buffer.data(), final_outputs.data, sizeof(*h_buffer.data()) * len, cudaMemcpyDeviceToHost));

            {
                // auto lock_start = std::chrono::steady_clock::now();
                std::lock_guard lock(outputs.mutex);
                for (const auto &result : h_buffer) {
                    uint64_t seed;
                    TRY_CUDA(cudaMemcpy(&seed, &outputs_filter_seeds.data[result.seed_index], sizeof(seed), cudaMemcpyDeviceToHost));
                    // std::printf("seed = %" PRIi64 " seed_index = %" PRIu32 " x = %" PRIi32 " z = %" PRIi32 "\n", seed, result.seed_index, result.x, result.z);
                    outputs.queue.push({ seed, result.x * 4, result.z * 4 });
                }
                // auto lock_end = std::chrono::steady_clock::now();
                // double lock_time = std::chrono::duration_cast<std::chrono::nanoseconds>(lock_end - lock_start).count() * 1e-9;
                // std::printf("Lock took %.3f s\n", lock_time);
            }
        }

        if ((i + 1) % print_interval == 0) {
            auto end = std::chrono::steady_clock::now();
            double host_total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() * 1e-9;

            std::printf("\n");
            std::printf("start_seed = %" PRIi64 "\n", start_seed);

            double kernel_total_time = 0;
            for (auto &stage : stage_stats) {
                uint64_t scaled_total_inputs = stage.total_inputs * stage.inputs_multiplier;
                auto [scaled_input_speed, input_speed_unit] = scale_si(scaled_total_inputs / stage.total_time);
                auto [scaled_output_speed, output_speed_unit] = scale_si(stage.total_outputs / stage.total_time);
                std::printf("%-20s - %9.3f ms | %7.3f %% | %12" PRIu64 " -> %12" PRIu64 " | 1 in %11.3f | %7.3f %cips | %7.3f %cops\n", stage.name.c_str(), stage.total_time * 1e3, stage.total_time / host_total_time * 100.0, scaled_total_inputs, stage.total_outputs, (double)scaled_total_inputs / stage.total_outputs, scaled_input_speed, input_speed_unit, scaled_output_speed, output_speed_unit);
                kernel_total_time += stage.total_time;
            }

            uint64_t total_inputs = stage_filter_seeds.total_inputs * stage_filter_seeds.inputs_multiplier;
            uint64_t total_outputs = filter_2.back().stage.total_outputs;
            auto [scaled_input_speed, input_speed_unit] = scale_si(total_inputs / host_total_time);
            auto [scaled_output_speed, output_speed_unit] = scale_si(total_outputs / host_total_time);
            std::printf("total                - %9.3f ms | %7.3f %% | %12" PRIu64 " -> %12" PRIu64 " |                  | %7.3f %cips | %7.3f %cops\n", host_total_time * 1e3, kernel_total_time / host_total_time * 100.0, total_inputs, total_outputs, scaled_input_speed, input_speed_unit, scaled_output_speed, output_speed_unit);

            size_t gpu_outputs_size;
            {
                std::lock_guard lock(outputs.mutex);
                gpu_outputs_size = outputs.queue.size();
            }
            std::printf("gpu_outputs.size() = %zu\n", gpu_outputs_size);

            for (auto &stage : stage_stats) {
                stage.reset();
            }
            start = end;
        }

        // FILE *scores_file = std::fopen("scores.txt", "w");
        // if (!scores_file) std::abort();
        // std::fprintf(scores_file, "123");
        // for (int32_t nx = 0; nx < 256; nx++) {
        //     for (int32_t nz = 0; nz < 256; nz++) {
        //         std::fprintf(scores_file, " %.6f", KernelFilterGradVecs1::scores[nx][nz]);
        //     }
        // }
        // std::fclose(scores_file);
    }

    TRY_CUDA(cudaFree(device_buffer_lens));
}
// gpu_filler.cu
//
// Idle-fill GPU utilization to a configurable target percentage per card,
// without crowding out real workloads.
//
// Approach:
//   * One worker thread per GPU. Each thread runs cuBLAS SGEMM in a loop on a
//     LOW-PRIORITY CUDA stream so the hardware scheduler lets real work jump
//     ahead.
//   * Small matrices (default 1024x1024 fp32, ~12 MiB total for A/B/C) so we
//     don't crowd VRAM. Configurable via --size.
//   * After each "work burst" we sleep for `idle_ms`. A simple proportional
//     controller adjusts work_ms / idle_ms each tick so the NVML-reported total
//     utilization tracks the target. When a real workload appears, total util
//     rises above target -> filler shrinks work_ms / grows idle_ms -> yields.
//
// Build:  make
// Run:    ./gpu-filler --gpus 0,1,2,3 --target 70
//         ./gpu-filler --spec 0:70,1:50,2:80
//
// Notes / caveats:
//   * NVML utilization is the % of the past sample window during which any
//     kernel ran, NOT compute density. That's exactly the metric most
//     dashboards display, which is what you want to "lift".
//   * If your monitoring averages over long windows, the controller's overshoot
//     is invisible anyway. For tight SLOs, drop the proportional gain.
//   * VRAM is not yielded back when a real task arrives. Keep --size small.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>

// ----------------------------- error helpers --------------------------------

#define CUDA_CHECK(stmt)                                                       \
    do {                                                                       \
        cudaError_t _e = (stmt);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "[gpu-filler] CUDA error %s at %s:%d: %s\n",  \
                         #stmt, __FILE__, __LINE__, cudaGetErrorString(_e));   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(stmt)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (stmt);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            std::fprintf(stderr, "[gpu-filler] cuBLAS error %s at %s:%d: %d\n",\
                         #stmt, __FILE__, __LINE__, (int)_s);                  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

#define NVML_CHECK(stmt)                                                       \
    do {                                                                       \
        nvmlReturn_t _r = (stmt);                                              \
        if (_r != NVML_SUCCESS) {                                              \
            std::fprintf(stderr, "[gpu-filler] NVML error %s at %s:%d: %s\n",  \
                         #stmt, __FILE__, __LINE__, nvmlErrorString(_r));      \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ----------------------------- config / cli ---------------------------------

struct Config {
    std::vector<int>   gpu_ids;        // GPUs to drive
    std::map<int,int>  per_gpu_target; // gpu_id -> target util %
    int                default_target = 70;
    int                matrix_size    = 1024;   // N for NxN SGEMM
    int                tick_ms        = 200;    // controller tick
    int                min_work_ms    = 5;
    int                max_work_ms    = 200;
    int                min_idle_ms    = 5;
    int                max_idle_ms    = 500;
    bool               verbose        = false;
};

static std::atomic<bool> g_stop{false};

static void on_signal(int) { g_stop.store(true); }

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "  --gpus  LIST      Comma-separated GPU indices, e.g. 0,1,2,3\n"
        "                    or a count like '4' meaning 0..3.\n"
        "                    Default: all visible GPUs.\n"
        "  --target PCT      Default target utilization %% per GPU (1..99).\n"
        "                    Default: 70.\n"
        "  --spec  LIST      Per-GPU targets, overrides --target for listed\n"
        "                    GPUs. Format: id:pct,id:pct,...\n"
        "                    e.g. --spec 0:70,1:50,3:85\n"
        "  --size  N         SGEMM matrix size (NxN fp32). Default: 1024.\n"
        "                    Larger -> more SM occupancy per burst, more VRAM.\n"
        "  --tick  MS        Controller tick interval in ms. Default: 200.\n"
        "  --verbose         Log per-tick state.\n"
        "  -h, --help        Show this help.\n"
        "\n"
        "Examples:\n"
        "  %s --gpus 4 --target 70\n"
        "  %s --gpus 0,2,3 --target 60\n"
        "  %s --spec 0:80,1:80,2:50,3:50\n",
        prog, prog, prog, prog);
}

static std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, d)) if (!item.empty()) out.push_back(item);
    return out;
}

static bool parse_int(const std::string& s, int& v) {
    try { size_t p; v = std::stoi(s, &p); return p == s.size(); }
    catch (...) { return false; }
}

static bool parse_args(int argc, char** argv, Config& cfg) {
    std::string gpus_arg, spec_arg;
    bool has_gpus = false, has_spec = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[gpu-filler] %s requires a value\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-h" || a == "--help") { print_usage(argv[0]); std::exit(0); }
        else if (a == "--gpus")    { gpus_arg = need("--gpus"); has_gpus = true; }
        else if (a == "--spec")    { spec_arg = need("--spec"); has_spec = true; }
        else if (a == "--target")  {
            if (!parse_int(need("--target"), cfg.default_target) ||
                cfg.default_target < 1 || cfg.default_target > 99) {
                std::fprintf(stderr, "[gpu-filler] --target must be 1..99\n");
                return false;
            }
        }
        else if (a == "--size")    {
            if (!parse_int(need("--size"), cfg.matrix_size) ||
                cfg.matrix_size < 64 || cfg.matrix_size > 16384) {
                std::fprintf(stderr, "[gpu-filler] --size must be 64..16384\n");
                return false;
            }
        }
        else if (a == "--tick")    {
            if (!parse_int(need("--tick"), cfg.tick_ms) ||
                cfg.tick_ms < 50 || cfg.tick_ms > 5000) {
                std::fprintf(stderr, "[gpu-filler] --tick must be 50..5000\n");
                return false;
            }
        }
        else if (a == "--verbose") { cfg.verbose = true; }
        else {
            std::fprintf(stderr, "[gpu-filler] unknown arg: %s\n", a.c_str());
            return false;
        }
    }

    // Resolve GPU list.
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::fprintf(stderr, "[gpu-filler] no CUDA devices found\n");
        return false;
    }

    if (has_gpus) {
        // Either "N" (count) or "0,1,2".
        int as_count = 0;
        if (gpus_arg.find(',') == std::string::npos &&
            parse_int(gpus_arg, as_count) && as_count > 0) {
            if (as_count > device_count) {
                std::fprintf(stderr, "[gpu-filler] requested %d GPUs but only %d "
                             "are visible\n", as_count, device_count);
                return false;
            }
            for (int i = 0; i < as_count; ++i) cfg.gpu_ids.push_back(i);
        } else {
            for (auto& tok : split(gpus_arg, ',')) {
                int id;
                if (!parse_int(tok, id) || id < 0 || id >= device_count) {
                    std::fprintf(stderr, "[gpu-filler] invalid gpu id: %s\n",
                                 tok.c_str());
                    return false;
                }
                cfg.gpu_ids.push_back(id);
            }
        }
    } else {
        for (int i = 0; i < device_count; ++i) cfg.gpu_ids.push_back(i);
    }

    if (has_spec) {
        for (auto& tok : split(spec_arg, ',')) {
            auto kv = split(tok, ':');
            int id, pct;
            if (kv.size() != 2 || !parse_int(kv[0], id) || !parse_int(kv[1], pct)
                || pct < 1 || pct > 99) {
                std::fprintf(stderr, "[gpu-filler] invalid --spec entry: %s\n",
                             tok.c_str());
                return false;
            }
            cfg.per_gpu_target[id] = pct;
            // Auto-add to gpu_ids if user gave --spec without --gpus.
            if (!has_gpus &&
                std::find(cfg.gpu_ids.begin(), cfg.gpu_ids.end(), id) ==
                cfg.gpu_ids.end()) {
                cfg.gpu_ids.push_back(id);
            }
        }
    }

    if (cfg.gpu_ids.empty()) {
        std::fprintf(stderr, "[gpu-filler] no GPUs selected\n");
        return false;
    }
    return true;
}

// ----------------------------- worker ---------------------------------------

struct Worker {
    int  gpu_id;
    int  target_util;
    const Config* cfg;
};

static std::mutex g_log_mtx;
static void log_line(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_log_mtx);
    std::fprintf(stdout, "%s\n", s.c_str());
    std::fflush(stdout);
}

// Returns NVML utilization % for this device, or -1 on transient failure.
static int nvml_util(nvmlDevice_t dev) {
    nvmlUtilization_t u{};
    nvmlReturn_t r = nvmlDeviceGetUtilizationRates(dev, &u);
    if (r != NVML_SUCCESS) return -1;
    return (int)u.gpu;
}

static void worker_main(Worker w) {
    const Config& cfg = *w.cfg;

    CUDA_CHECK(cudaSetDevice(w.gpu_id));

    // Map this CUDA device to an NVML handle. Using PCI bus id is the robust
    // path; CUDA device order and NVML device order are not always identical.
    char pci_id[32];
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, w.gpu_id));
    CUDA_CHECK(cudaDeviceGetPCIBusId(pci_id, sizeof(pci_id), w.gpu_id));

    nvmlDevice_t nvml_dev;
    NVML_CHECK(nvmlDeviceGetHandleByPciBusId(pci_id, &nvml_dev));

    // Lowest-priority stream so real work preempts us at the SM scheduler.
    int lo_pri = 0, hi_pri = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                            lo_pri));

    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));
    CUBLAS_CHECK(cublasSetStream(blas, stream));

    const int N = cfg.matrix_size;
    const size_t bytes = (size_t)N * N * sizeof(float);
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // Initialize with a fixed pattern; values don't matter, we never read C.
    CUDA_CHECK(cudaMemsetAsync(dA, 0x3f, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dB, 0x3f, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dC, 0,    bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const float alpha = 1.0f, beta = 0.0f;

    // Controller state. Start at 50% duty cycle and let the loop converge.
    int work_ms = std::max(cfg.min_work_ms,
                           std::min(cfg.max_work_ms, cfg.tick_ms / 2));
    int idle_ms = std::max(cfg.min_idle_ms,
                           std::min(cfg.max_idle_ms, cfg.tick_ms / 2));

    log_line("[gpu-filler] gpu " + std::to_string(w.gpu_id) +
             " (" + prop.name + ") target=" + std::to_string(w.target_util) +
             "% size=" + std::to_string(N) +
             " stream_prio=" + std::to_string(lo_pri));

    while (!g_stop.load()) {
        // ---- work burst ----
        auto t_end = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(work_ms);
        int gemm_count = 0;
        while (std::chrono::steady_clock::now() < t_end && !g_stop.load()) {
            CUBLAS_CHECK(cublasSgemm(blas,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, N, N,
                                     &alpha, dA, N, dB, N,
                                     &beta,  dC, N));
            ++gemm_count;
            // Don't synchronize every iteration; let kernels queue up so the
            // scheduler has work to interleave with real tasks.
            if (gemm_count % 16 == 0) {
                cudaStreamQuery(stream);  // hint, ignore result
            }
        }
        // Drain so the burst ends close to work_ms wall time.
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (g_stop.load()) break;

        // ---- idle ----
        std::this_thread::sleep_for(std::chrono::milliseconds(idle_ms));

        // ---- measure + adjust ----
        int util = nvml_util(nvml_dev);
        if (util < 0) continue;  // transient NVML hiccup, skip this tick

        // Proportional controller. Error positive -> we're below target ->
        // grow work, shrink idle. Error negative -> shrink work, grow idle.
        int err = w.target_util - util;
        // Gain: 1ms per percent of error, scaled by current tick budget.
        int delta = err;  // straight 1:1; small enough to stay stable

        work_ms = std::max(cfg.min_work_ms,
                           std::min(cfg.max_work_ms, work_ms + delta));
        idle_ms = std::max(cfg.min_idle_ms,
                           std::min(cfg.max_idle_ms, idle_ms - delta));

        if (cfg.verbose) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "[gpu-filler] gpu=%d util=%d%% target=%d%% err=%+d "
                "work=%dms idle=%dms gemms=%d",
                w.gpu_id, util, w.target_util, err, work_ms, idle_ms, gemm_count);
            log_line(buf);
        }
    }

    // Cleanup.
    cublasDestroy(blas);
    cudaStreamDestroy(stream);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    log_line("[gpu-filler] gpu " + std::to_string(w.gpu_id) + " stopped");
}

// ----------------------------- main -----------------------------------------

int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 2;
    }

    NVML_CHECK(nvmlInit());

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    // Build worker list.
    std::vector<Worker> workers;
    workers.reserve(cfg.gpu_ids.size());
    for (int id : cfg.gpu_ids) {
        int t = cfg.default_target;
        auto it = cfg.per_gpu_target.find(id);
        if (it != cfg.per_gpu_target.end()) t = it->second;
        workers.push_back({id, t, &cfg});
    }

    log_line("[gpu-filler] starting on " + std::to_string(workers.size()) +
             " GPU(s); SIGINT/SIGTERM to stop");

    std::vector<std::thread> threads;
    for (auto& w : workers) threads.emplace_back(worker_main, w);
    for (auto& t : threads) t.join();

    nvmlShutdown();
    log_line("[gpu-filler] exited cleanly");
    return 0;
}

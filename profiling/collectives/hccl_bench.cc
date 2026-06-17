#include <acl/acl.h>
#include <hccl/hccl.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>

namespace {

struct Config {
    uint64_t count = 0;
    std::string dtype;
    std::vector<uint32_t> devices;
    uint32_t warmup_rounds = 0;
    uint32_t timed_rounds = 1;
};

struct ThreadResult {
    bool ok = false;
    std::string error;
    std::vector<double> warmup_walls;
    std::vector<double> timed_walls;
};

struct InitBarrier {
    std::mutex mu;
    std::condition_variable cv;
    uint32_t ready = 0;
    bool released = false;
    std::chrono::steady_clock::time_point setup_t0{};
    double setup_s = 0.0;
};

struct RoundBarrier {
    std::mutex mu;
    std::condition_variable cv;
    uint32_t count = 0;
    uint32_t generation = 0;

    void Wait(uint32_t world_size, ThreadResult* result)
    {
        std::unique_lock<std::mutex> lock(mu);
        const uint32_t gen = generation;
        count += 1;
        if (count == world_size) {
            count = 0;
            generation += 1;
            cv.notify_all();
            return;
        }
        cv.wait(lock, [&]() { return generation != gen; });
        if (generation == gen) {
            result->ok = false;
            result->error = "round barrier generation did not advance";
        }
    }
};

std::mutex g_log_mu;

void LogLine(const std::string& line)
{
    std::lock_guard<std::mutex> lock(g_log_mu);
    std::cout << line << std::endl;
}

std::vector<uint32_t> ParseDevices(const std::string& text)
{
    std::vector<uint32_t> devices;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        devices.push_back(static_cast<uint32_t>(std::stoul(item)));
    }
    return devices;
}

bool ParseArgs(int argc, char** argv, Config* config, std::string* error)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--count" && i + 1 < argc) {
            config->count = std::stoull(argv[++i]);
            continue;
        }
        if (arg == "--dtype" && i + 1 < argc) {
            config->dtype = argv[++i];
            continue;
        }
        if (arg == "--devices" && i + 1 < argc) {
            config->devices = ParseDevices(argv[++i]);
            continue;
        }
        if (arg == "--warmup-rounds" && i + 1 < argc) {
            config->warmup_rounds = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--timed-rounds" && i + 1 < argc) {
            config->timed_rounds = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        *error = "unknown or incomplete argument: " + arg;
        return false;
    }

    if (config->count == 0) {
        *error = "--count must be > 0";
        return false;
    }
    if (config->dtype != "fp32") {
        *error = "compiled HCCL helper currently supports only --dtype fp32";
        return false;
    }
    if (config->devices.empty()) {
        *error = "--devices must not be empty";
        return false;
    }
    if (config->timed_rounds == 0 && config->warmup_rounds == 0) {
        config->timed_rounds = 1;
    }
    return true;
}

std::string PtrHex(const void* ptr)
{
    std::ostringstream os;
    os << ptr;
    return os.str();
}

void CleanupBuffers(void* send_buf, void* recv_buf, void* host_buf, void* out_buf, aclrtStream stream, HcclComm comm)
{
    if (out_buf != nullptr) {
        (void)aclrtFreeHost(out_buf);
    }
    if (host_buf != nullptr) {
        (void)aclrtFreeHost(host_buf);
    }
    if (stream != nullptr) {
        (void)aclrtDestroyStream(stream);
    }
    if (comm != nullptr) {
        (void)HcclCommDestroy(comm);
    }
    if (send_buf != nullptr) {
        (void)aclrtFree(send_buf);
    }
    if (recv_buf != nullptr) {
        (void)aclrtFree(recv_buf);
    }
}

bool WaitForCollectiveStart(uint32_t rank, uint32_t world_size, InitBarrier* barrier, ThreadResult* result)
{
    std::unique_lock<std::mutex> lock(barrier->mu);
    barrier->ready += 1;
    if (barrier->ready == world_size) {
        barrier->setup_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - barrier->setup_t0).count();
        std::cout << "HCCL_COMM_SETUP_OK world_size=" << world_size << " setup_s=" << barrier->setup_s << std::endl;
        barrier->released = true;
        barrier->cv.notify_all();
        return true;
    }
    barrier->cv.wait(lock, [&]() { return barrier->released; });
    if (!barrier->released) {
        result->ok = false;
        result->error = "collective start barrier released unexpectedly";
        return false;
    }
    return true;
}

bool RunAllReduceTimed(void* send_buf, void* recv_buf, uint64_t count, HcclComm comm, aclrtStream stream,
                       double* wall_s, ThreadResult* result)
{
    const auto t0 = std::chrono::steady_clock::now();
    const HcclResult hccl_ret =
        HcclAllReduce(send_buf, recv_buf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, comm, stream);
    const aclError sync_ret = aclrtSynchronizeStream(stream);
    *wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (hccl_ret != HCCL_SUCCESS) {
        result->ok = false;
        result->error = "HcclAllReduce -> " + std::to_string(hccl_ret);
        return false;
    }
    if (sync_ret != ACL_SUCCESS) {
        result->ok = false;
        result->error = "aclrtSynchronizeStream -> " + std::to_string(sync_ret);
        return false;
    }
    return true;
}

bool VerifyGolden(uint32_t rank, uint32_t world_size, uint64_t count, void* recv_buf, size_t nbytes,
                  ThreadResult* result)
{
    void* out_buf = nullptr;
    const aclError alloc_ret = aclrtMallocHost(&out_buf, nbytes);
    if (alloc_ret != ACL_SUCCESS) {
        result->ok = false;
        result->error = "aclrtMallocHost(out_buf) -> " + std::to_string(alloc_ret);
        return false;
    }

    const aclError copy_ret = aclrtMemcpy(out_buf, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (copy_ret != ACL_SUCCESS) {
        (void)aclrtFreeHost(out_buf);
        result->ok = false;
        result->error = "aclrtMemcpy(D2H) -> " + std::to_string(copy_ret);
        return false;
    }

    const auto* out = static_cast<const float*>(out_buf);
    const float expected_base = static_cast<float>(100 * world_size * (world_size - 1) / 2);
    double max_err = 0.0;
    for (uint64_t index = 0; index < count; ++index) {
        const float expected = static_cast<float>(world_size * index) + expected_base;
        max_err = std::max(max_err, std::fabs(static_cast<double>(out[index]) - static_cast<double>(expected)));
    }
    (void)aclrtFreeHost(out_buf);

    if (max_err > 1e-5) {
        std::ostringstream os;
        os << "verification max_err=" << max_err;
        result->ok = false;
        result->error = os.str();
        return false;
    }
    return true;
}

void RunRank(uint32_t rank, uint32_t world_size, uint32_t device_id, const HcclRootInfo* root_info,
             const Config& config, InitBarrier* setup_barrier, RoundBarrier* round_barrier, ThreadResult* result)
{
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    void* host_buf = nullptr;
    aclrtStream stream = nullptr;
    HcclComm comm = nullptr;

    auto fail = [&](const std::string& message) {
        result->ok = false;
        result->error = message;
        CleanupBuffers(send_buf, recv_buf, host_buf, nullptr, stream, comm);
    };

    aclError acl_ret = aclrtSetDevice(static_cast<int32_t>(device_id));
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtSetDevice(" + std::to_string(device_id) + ") -> " + std::to_string(acl_ret));
        return;
    }

    const size_t nbytes = static_cast<size_t>(config.count * sizeof(float));
    acl_ret = aclrtMalloc(&send_buf, nbytes, ACL_MEM_MALLOC_HUGE_ONLY);
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMalloc(send_buf) -> " + std::to_string(acl_ret));
        return;
    }

    acl_ret = aclrtMalloc(&recv_buf, nbytes, ACL_MEM_MALLOC_HUGE_ONLY);
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMalloc(recv_buf) -> " + std::to_string(acl_ret));
        return;
    }

    acl_ret = aclrtMallocHost(&host_buf, nbytes);
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMallocHost(host_buf) -> " + std::to_string(acl_ret));
        return;
    }

    auto* host = static_cast<float*>(host_buf);
    for (uint64_t index = 0; index < config.count; ++index) {
        host[index] = static_cast<float>(index + rank * 100);
    }

    acl_ret = aclrtMemcpy(send_buf, nbytes, host_buf, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMemcpy(H2D) -> " + std::to_string(acl_ret));
        return;
    }

    HcclResult hccl_ret = HcclCommInitRootInfo(world_size, root_info, device_id, &comm);
    if (hccl_ret != HCCL_SUCCESS) {
        fail("HcclCommInitRootInfo -> " + std::to_string(hccl_ret));
        return;
    }

    acl_ret = aclrtCreateStream(&stream);
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtCreateStream -> " + std::to_string(acl_ret));
        return;
    }

    if (!WaitForCollectiveStart(rank, world_size, setup_barrier, result)) {
        CleanupBuffers(send_buf, recv_buf, host_buf, nullptr, stream, comm);
        return;
    }
    result->ok = true;

    result->warmup_walls.reserve(config.warmup_rounds);
    for (uint32_t round = 0; round < config.warmup_rounds; ++round) {
        double wall_s = 0.0;
        if (!RunAllReduceTimed(send_buf, recv_buf, config.count, comm, stream, &wall_s, result)) {
            CleanupBuffers(send_buf, recv_buf, host_buf, nullptr, stream, comm);
            return;
        }
        result->warmup_walls.push_back(wall_s);
        round_barrier->Wait(world_size, result);
        if (!result->ok) {
            CleanupBuffers(send_buf, recv_buf, host_buf, nullptr, stream, comm);
            return;
        }
    }

    result->timed_walls.reserve(config.timed_rounds);
    for (uint32_t round = 0; round < config.timed_rounds; ++round) {
        double wall_s = 0.0;
        if (!RunAllReduceTimed(send_buf, recv_buf, config.count, comm, stream, &wall_s, result)) {
            CleanupBuffers(send_buf, recv_buf, host_buf, nullptr, stream, comm);
            return;
        }
        result->timed_walls.push_back(wall_s);
        if (!VerifyGolden(rank, world_size, config.count, recv_buf, nbytes, result)) {
            CleanupBuffers(send_buf, recv_buf, host_buf, nullptr, stream, comm);
            return;
        }
        round_barrier->Wait(world_size, result);
        if (!result->ok) {
            CleanupBuffers(send_buf, recv_buf, host_buf, nullptr, stream, comm);
            return;
        }
    }

    result->ok = true;
    CleanupBuffers(send_buf, recv_buf, host_buf, nullptr, stream, comm);
}

std::string FormatWarmupRound(const std::vector<ThreadResult>& results, uint32_t round)
{
    std::ostringstream per_rank;
    per_rank << '[';
    for (size_t index = 0; index < results.size(); ++index) {
        if (index != 0) {
            per_rank << ", ";
        }
        per_rank << results[index].warmup_walls[round];
    }
    per_rank << ']';
    return per_rank.str();
}

std::string FormatTimedRound(const std::vector<ThreadResult>& results, uint32_t round)
{
    std::ostringstream per_rank;
    per_rank << '[';
    for (size_t index = 0; index < results.size(); ++index) {
        if (index != 0) {
            per_rank << ", ";
        }
        per_rank << results[index].timed_walls[round];
    }
    per_rank << ']';
    return per_rank.str();
}

}  // namespace

int main(int argc, char** argv)
{
    Config config;
    std::string arg_error;
    if (!ParseArgs(argc, argv, &config, &arg_error)) {
        std::cerr << arg_error << std::endl;
        return 2;
    }

    const aclError init_ret = aclInit(nullptr);
    if (init_ret != ACL_SUCCESS) {
        return 1;
    }

    uint32_t visible_count = 0;
    const aclError count_ret = aclrtGetDeviceCount(&visible_count);
    if (count_ret != ACL_SUCCESS) {
        (void)aclFinalize();
        return 1;
    }

    if (visible_count < config.devices.size()) {
        std::cerr << "visible device count " << visible_count << " is smaller than requested world size "
                  << config.devices.size() << std::endl;
        (void)aclFinalize();
        return 1;
    }

    const uint32_t root_device = config.devices.front();
    const aclError root_set_ret = aclrtSetDevice(static_cast<int32_t>(root_device));
    if (root_set_ret != ACL_SUCCESS) {
        (void)aclFinalize();
        return 1;
    }

    void* root_info_buf = nullptr;
    const aclError root_alloc_ret = aclrtMallocHost(&root_info_buf, sizeof(HcclRootInfo));
    if (root_alloc_ret != ACL_SUCCESS) {
        (void)aclFinalize();
        return 1;
    }

    auto* root_info = static_cast<HcclRootInfo*>(root_info_buf);
    const HcclResult root_info_ret = HcclGetRootInfo(root_info);
    if (root_info_ret != HCCL_SUCCESS) {
        (void)aclrtFreeHost(root_info_buf);
        (void)aclFinalize();
        return 1;
    }

    InitBarrier setup_barrier;
    setup_barrier.setup_t0 = std::chrono::steady_clock::now();
    RoundBarrier round_barrier;
    std::vector<std::thread> threads;
    std::vector<ThreadResult> results(config.devices.size());
    threads.reserve(config.devices.size());
    for (size_t index = 0; index < config.devices.size(); ++index) {
        threads.emplace_back(RunRank, static_cast<uint32_t>(index), static_cast<uint32_t>(config.devices.size()),
                             config.devices[index], root_info, std::cref(config), &setup_barrier, &round_barrier,
                             &results[index]);
    }
    for (auto& thread : threads) {
        thread.join();
    }

    bool all_ok = true;
    for (size_t index = 0; index < results.size(); ++index) {
        if (!results[index].ok) {
            all_ok = false;
            std::cerr << "HCCL rank " << index << ": " << results[index].error << std::endl;
        }
    }

    if (all_ok) {
        for (uint32_t round = 0; round < config.warmup_rounds; ++round) {
            std::cout << "HCCL_WARMUP round=" << (round + 1) << " per_rank=" << FormatWarmupRound(results, round)
                      << std::endl;
        }
        for (uint32_t round = 0; round < config.timed_rounds; ++round) {
            std::cout << "HCCL_TIMED round=" << (round + 1) << " per_rank=" << FormatTimedRound(results, round)
                      << std::endl;
        }
        std::cout << "HCCL_ALLREDUCE_OK setup_s=" << setup_barrier.setup_s << std::endl;
    }

    (void)aclrtFreeHost(root_info_buf);
    (void)aclFinalize();

    return all_ok ? 0 : 1;
}

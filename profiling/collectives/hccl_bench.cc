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
};

struct ThreadResult {
    bool ok = false;
    std::string error;
    double wall_s = 0.0;
};

struct InitBarrier {
    std::mutex mu;
    std::condition_variable cv;
    uint32_t ready = 0;
    bool released = false;
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
        std::cout << "HCCL_COMM_SETUP_OK world_size=" << world_size << std::endl;
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

void RunRank(uint32_t rank, uint32_t world_size, uint32_t device_id, const HcclRootInfo* root_info,
             uint64_t count, InitBarrier* barrier, ThreadResult* result)
{
    void* send_buf = nullptr;
    void* recv_buf = nullptr;
    void* host_buf = nullptr;
    void* out_buf = nullptr;
    aclrtStream stream = nullptr;
    HcclComm comm = nullptr;

    auto fail = [&](const std::string& message) {
        result->ok = false;
        result->error = message;
        CleanupBuffers(send_buf, recv_buf, host_buf, out_buf, stream, comm);
    };

    aclError acl_ret = aclrtSetDevice(static_cast<int32_t>(device_id));
    LogLine("[diag] rank " + std::to_string(rank) + " aclrtSetDevice(" + std::to_string(device_id) + ") -> "
            + std::to_string(acl_ret));
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtSetDevice(" + std::to_string(device_id) + ") -> " + std::to_string(acl_ret));
        return;
    }

    const size_t nbytes = static_cast<size_t>(count * sizeof(float));
    acl_ret = aclrtMalloc(&send_buf, nbytes, ACL_MEM_MALLOC_HUGE_ONLY);
    LogLine("[diag] rank " + std::to_string(rank) + " aclrtMalloc(send=" + PtrHex(send_buf) + ") -> "
            + std::to_string(acl_ret));
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMalloc(send_buf) -> " + std::to_string(acl_ret));
        return;
    }

    acl_ret = aclrtMalloc(&recv_buf, nbytes, ACL_MEM_MALLOC_HUGE_ONLY);
    LogLine("[diag] rank " + std::to_string(rank) + " aclrtMalloc(recv=" + PtrHex(recv_buf) + ") -> "
            + std::to_string(acl_ret));
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMalloc(recv_buf) -> " + std::to_string(acl_ret));
        return;
    }

    acl_ret = aclrtMallocHost(&host_buf, nbytes);
    LogLine("[diag] rank " + std::to_string(rank) + " aclrtMallocHost(host=" + PtrHex(host_buf) + ") -> "
            + std::to_string(acl_ret));
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMallocHost(host_buf) -> " + std::to_string(acl_ret));
        return;
    }

    auto* host = static_cast<float*>(host_buf);
    for (uint64_t index = 0; index < count; ++index) {
        host[index] = static_cast<float>(index + rank * 100);
    }

    acl_ret = aclrtMemcpy(send_buf, nbytes, host_buf, nbytes, ACL_MEMCPY_HOST_TO_DEVICE);
    LogLine("[diag] rank " + std::to_string(rank) + " aclrtMemcpy H2D src=" + PtrHex(host_buf)
            + " dst=" + PtrHex(send_buf) + " -> " + std::to_string(acl_ret));
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMemcpy(H2D) -> " + std::to_string(acl_ret));
        return;
    }

    HcclResult hccl_ret = HcclCommInitRootInfo(world_size, root_info, device_id, &comm);
    LogLine("[diag] rank " + std::to_string(rank) + " HcclCommInitRootInfo(world_size="
            + std::to_string(world_size) + ", device_id=" + std::to_string(device_id)
            + ", root_info=" + PtrHex(root_info) + ") comm=" + PtrHex(comm) + " -> "
            + std::to_string(hccl_ret));
    if (hccl_ret != HCCL_SUCCESS) {
        fail("HcclCommInitRootInfo -> " + std::to_string(hccl_ret));
        return;
    }

    acl_ret = aclrtCreateStream(&stream);
    LogLine("[diag] rank " + std::to_string(rank) + " aclrtCreateStream(stream=" + PtrHex(stream) + ") -> "
            + std::to_string(acl_ret));
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtCreateStream -> " + std::to_string(acl_ret));
        return;
    }

    if (!WaitForCollectiveStart(rank, world_size, barrier, result)) {
        CleanupBuffers(send_buf, recv_buf, host_buf, out_buf, stream, comm);
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    hccl_ret = HcclAllReduce(send_buf, recv_buf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, comm, stream);
    const aclError sync_ret = aclrtSynchronizeStream(stream);
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    result->wall_s = wall_s;
    LogLine("[diag] rank " + std::to_string(rank) + " HcclAllReduce(send=" + PtrHex(send_buf)
            + ", recv=" + PtrHex(recv_buf) + ", count=" + std::to_string(count)
            + ", data_type=FP32, reduce_op=SUM, comm=" + PtrHex(comm) + ", stream=" + PtrHex(stream)
            + ") -> " + std::to_string(hccl_ret) + "; aclrtSynchronizeStream -> "
            + std::to_string(sync_ret) + "; wall=" + std::to_string(wall_s) + "s");
    if (hccl_ret != HCCL_SUCCESS) {
        fail("HcclAllReduce -> " + std::to_string(hccl_ret));
        return;
    }
    if (sync_ret != ACL_SUCCESS) {
        fail("aclrtSynchronizeStream -> " + std::to_string(sync_ret));
        return;
    }

    acl_ret = aclrtMallocHost(&out_buf, nbytes);
    LogLine("[diag] rank " + std::to_string(rank) + " aclrtMallocHost(out=" + PtrHex(out_buf) + ") -> "
            + std::to_string(acl_ret));
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMallocHost(out_buf) -> " + std::to_string(acl_ret));
        return;
    }
    acl_ret = aclrtMemcpy(out_buf, nbytes, recv_buf, nbytes, ACL_MEMCPY_DEVICE_TO_HOST);
    LogLine("[diag] rank " + std::to_string(rank) + " aclrtMemcpy D2H src=" + PtrHex(recv_buf)
            + " dst=" + PtrHex(out_buf) + " -> " + std::to_string(acl_ret));
    if (acl_ret != ACL_SUCCESS) {
        fail("aclrtMemcpy(D2H) -> " + std::to_string(acl_ret));
        return;
    }

    const auto* out = static_cast<const float*>(out_buf);
    const float expected_base = static_cast<float>(100 * world_size * (world_size - 1) / 2);
    double max_err = 0.0;
    for (uint64_t index = 0; index < count; ++index) {
        const float expected = static_cast<float>(world_size * index) + expected_base;
        max_err = std::max(max_err, std::fabs(static_cast<double>(out[index]) - static_cast<double>(expected)));
    }
    if (max_err > 1e-5) {
        std::ostringstream os;
        os << "verification max_err=" << max_err;
        fail(os.str());
        return;
    }

    result->ok = true;
    CleanupBuffers(send_buf, recv_buf, host_buf, out_buf, stream, comm);
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

    const char* visible_devices = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    const char* expansion_mode = std::getenv("HCCL_OP_EXPANSION_MODE");
    LogLine(std::string("[diag] env ASCEND_RT_VISIBLE_DEVICES=")
            + (visible_devices == nullptr ? "<unset>" : visible_devices)
            + " HCCL_OP_EXPANSION_MODE="
            + (expansion_mode == nullptr ? "<unset>" : expansion_mode));

    const aclError init_ret = aclInit(nullptr);
    LogLine("[diag] aclInit -> " + std::to_string(init_ret));
    if (init_ret != ACL_SUCCESS) {
        return 1;
    }

    uint32_t visible_count = 0;
    const aclError count_ret = aclrtGetDeviceCount(&visible_count);
    LogLine("[diag] aclrtGetDeviceCount -> " + std::to_string(count_ret) + ", count=" + std::to_string(visible_count));
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
    LogLine("[diag] root aclrtSetDevice(" + std::to_string(root_device) + ") -> " + std::to_string(root_set_ret));
    if (root_set_ret != ACL_SUCCESS) {
        (void)aclFinalize();
        return 1;
    }

    void* root_info_buf = nullptr;
    const aclError root_alloc_ret = aclrtMallocHost(&root_info_buf, sizeof(HcclRootInfo));
    LogLine("[diag] aclrtMallocHost(root_info=" + PtrHex(root_info_buf) + ") -> " + std::to_string(root_alloc_ret));
    if (root_alloc_ret != ACL_SUCCESS) {
        (void)aclFinalize();
        return 1;
    }

    auto* root_info = static_cast<HcclRootInfo*>(root_info_buf);
    const HcclResult root_info_ret = HcclGetRootInfo(root_info);
    LogLine("[diag] HcclGetRootInfo(root_info=" + PtrHex(root_info) + ") -> " + std::to_string(root_info_ret));
    if (root_info_ret != HCCL_SUCCESS) {
        (void)aclrtFreeHost(root_info_buf);
        (void)aclFinalize();
        return 1;
    }

    std::vector<std::thread> threads;
    std::vector<ThreadResult> results(config.devices.size());
    InitBarrier barrier;
    threads.reserve(config.devices.size());
    for (size_t index = 0; index < config.devices.size(); ++index) {
        threads.emplace_back(RunRank, static_cast<uint32_t>(index), static_cast<uint32_t>(config.devices.size()),
                             config.devices[index], root_info, config.count, &barrier, &results[index]);
    }
    for (size_t index = 0; index < threads.size(); ++index) {
        threads[index].join();
        LogLine("[diag]   thread rank=" + std::to_string(index) + " joined (alive=false)");
    }

    bool all_ok = true;
    std::ostringstream per_rank;
    per_rank << '[';
    for (size_t index = 0; index < results.size(); ++index) {
        if (index != 0) {
            per_rank << ", ";
        }
        per_rank << results[index].wall_s;
        if (!results[index].ok) {
            all_ok = false;
            std::cerr << "HCCL rank " << index << ": " << results[index].error << std::endl;
        }
    }
    per_rank << ']';

    (void)aclrtFreeHost(root_info_buf);
    const aclError fin_ret = aclFinalize();
    LogLine("[diag] aclFinalize -> " + std::to_string(fin_ret));

    if (!all_ok) {
        return 1;
    }

    std::cout << "HCCL_ALLREDUCE_OK per_rank=" << per_rank.str() << std::endl;
    return 0;
}
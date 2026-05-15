/**
 * @file main.cpp
 */
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_baremix_custom.h"
#else
#include "tikicpulib.h"
extern "C" void baremix_custom(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                               uint8_t *, uint8_t *);
#endif

extern size_t GetTilingBufferSize();
extern void GenerateTiling(const char *socVersion, uint32_t n, uint8_t *tilingBuf, size_t tilingBufSize,
                           uint32_t &blockDim, size_t &workspaceSize);

namespace {
constexpr uint32_t GROUP_NUM = 17;
constexpr uint32_t AVG_NUM = 16;
constexpr uint32_t K_DIM = 256;
constexpr uint32_t USER_VEC = 8;

bool LoadFileToBuffer(const std::string &path, size_t size, uint8_t *buffer)
{
    size_t fileSize = size;
    return ReadFile(path, fileSize, buffer, size) && fileSize == size;
}

void WriteOutputs(uint8_t *b, uint8_t *bPlur, uint8_t *bPlui, uint8_t *bsum, uint8_t *csum, size_t bSize,
                  size_t matrixSize, size_t csumSize)
{
    WriteFile("./output/b.bin", b, bSize);
    WriteFile("./output/bplur.bin", bPlur, bSize);
    WriteFile("./output/bplui.bin", bPlui, bSize);
    WriteFile("./output/bsum.bin", bsum, matrixSize);
    WriteFile("./output/csum.bin", csum, csumSize);
}
}  // namespace

int32_t main(int32_t argc, char *argv[])
{
    const char *socVersion = SOC_VERSION;
    uint32_t n = 1;
    if (argc > 1) {
        n = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
    }
    if (n == 0) {
        std::cerr << "n must be greater than 0" << std::endl;
        return 1;
    }

    const uint32_t u = n * USER_VEC;
    const size_t aFileSize = static_cast<size_t>(GROUP_NUM) * AVG_NUM * K_DIM * u * sizeof(uint16_t);
    const size_t matrixFileSize = static_cast<size_t>(u) * u * sizeof(float);
    const size_t bFileSize = static_cast<size_t>(GROUP_NUM) * u * u * sizeof(float);
    const size_t csumFileSize = static_cast<size_t>(n) * n * sizeof(float);
    const size_t tilingFileSize = GetTilingBufferSize();
    uint8_t *tilingBuf = static_cast<uint8_t *>(std::malloc(tilingFileSize));
    if (tilingBuf == nullptr) {
        std::cerr << "malloc tiling buffer failed" << std::endl;
        return 1;
    }

    size_t workspaceSize = 0;
    uint32_t blockDim = 1;
    GenerateTiling(socVersion, n, tilingBuf, tilingFileSize, blockDim, workspaceSize);
    if (workspaceSize == 0) {
        std::free(tilingBuf);
        std::cerr << "GenerateTiling failed" << std::endl;
        return 1;
    }

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *ar = static_cast<uint8_t *>(AscendC::GmAlloc(aFileSize));
    uint8_t *ai = static_cast<uint8_t *>(AscendC::GmAlloc(aFileSize));
    uint8_t *b = static_cast<uint8_t *>(AscendC::GmAlloc(bFileSize));
    uint8_t *bPlur = static_cast<uint8_t *>(AscendC::GmAlloc(bFileSize));
    uint8_t *bPlui = static_cast<uint8_t *>(AscendC::GmAlloc(bFileSize));
    uint8_t *bsum = static_cast<uint8_t *>(AscendC::GmAlloc(matrixFileSize));
    uint8_t *csum = static_cast<uint8_t *>(AscendC::GmAlloc(csumFileSize));
    uint8_t *tiling = static_cast<uint8_t *>(AscendC::GmAlloc(tilingFileSize));
    uint8_t *workspace = static_cast<uint8_t *>(AscendC::GmAlloc(workspaceSize));

    if (!LoadFileToBuffer("./input/ar.bin", aFileSize, ar) || !LoadFileToBuffer("./input/ai.bin", aFileSize, ai)) {
        std::cerr << "read input failed" << std::endl;
        return 1;
    }
    std::memcpy(tiling, tilingBuf, tilingFileSize);

    ICPU_RUN_KF(baremix_custom, blockDim, ar, ai, b, bPlur, bPlui, bsum, csum, workspace, tiling);
    WriteOutputs(b, bPlur, bPlui, bsum, csum, bFileSize, matrixFileSize, csumFileSize);

    AscendC::GmFree(ar);
    AscendC::GmFree(ai);
    AscendC::GmFree(b);
    AscendC::GmFree(bPlur);
    AscendC::GmFree(bPlui);
    AscendC::GmFree(bsum);
    AscendC::GmFree(csum);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *arHost = nullptr;
    uint8_t *arDevice = nullptr;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&arHost), aFileSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&arDevice), aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!LoadFileToBuffer("./input/ar.bin", aFileSize, arHost)) {
        std::cerr << "read ar failed" << std::endl;
        return 1;
    }
    CHECK_ACL(aclrtMemcpy(arDevice, aFileSize, arHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *aiHost = nullptr;
    uint8_t *aiDevice = nullptr;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&aiHost), aFileSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&aiDevice), aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    if (!LoadFileToBuffer("./input/ai.bin", aFileSize, aiHost)) {
        std::cerr << "read ai failed" << std::endl;
        return 1;
    }
    CHECK_ACL(aclrtMemcpy(aiDevice, aFileSize, aiHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bHost = nullptr;
    uint8_t *bDevice = nullptr;
    uint8_t *bPlurHost = nullptr;
    uint8_t *bPlurDevice = nullptr;
    uint8_t *bPluiHost = nullptr;
    uint8_t *bPluiDevice = nullptr;
    uint8_t *bsumHost = nullptr;
    uint8_t *bsumDevice = nullptr;
    uint8_t *csumHost = nullptr;
    uint8_t *csumDevice = nullptr;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&bHost), bFileSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&bDevice), bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&bPlurHost), bFileSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&bPlurDevice), bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&bPluiHost), bFileSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&bPluiDevice), bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&bsumHost), matrixFileSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&bsumDevice), matrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&csumHost), csumFileSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&csumDevice), csumFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *tilingHost = nullptr;
    uint8_t *tilingDevice = nullptr;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&tilingDevice), tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice = nullptr;
    CHECK_ACL(aclrtMalloc(reinterpret_cast<void **>(&workspaceDevice), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(baremix_custom)
    (blockDim, stream, arDevice, aiDevice, bDevice, bPlurDevice, bPluiDevice, bsumDevice, csumDevice, workspaceDevice,
     tilingDevice);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(bHost, bFileSize, bDevice, bFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(bPlurHost, bFileSize, bPlurDevice, bFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(bPluiHost, bFileSize, bPluiDevice, bFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(bsumHost, matrixFileSize, bsumDevice, matrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(csumHost, csumFileSize, csumDevice, csumFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteOutputs(bHost, bPlurHost, bPluiHost, bsumHost, csumHost, bFileSize, matrixFileSize, csumFileSize);

    CHECK_ACL(aclrtFree(arDevice));
    CHECK_ACL(aclrtFreeHost(arHost));
    CHECK_ACL(aclrtFree(aiDevice));
    CHECK_ACL(aclrtFreeHost(aiHost));
    CHECK_ACL(aclrtFree(bDevice));
    CHECK_ACL(aclrtFreeHost(bHost));
    CHECK_ACL(aclrtFree(bPlurDevice));
    CHECK_ACL(aclrtFreeHost(bPlurHost));
    CHECK_ACL(aclrtFree(bPluiDevice));
    CHECK_ACL(aclrtFreeHost(bPluiHost));
    CHECK_ACL(aclrtFree(bsumDevice));
    CHECK_ACL(aclrtFreeHost(bsumHost));
    CHECK_ACL(aclrtFree(csumDevice));
    CHECK_ACL(aclrtFreeHost(csumHost));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(workspaceDevice));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif

    std::free(tilingBuf);
    return 0;
}

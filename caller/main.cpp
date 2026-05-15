#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_complex_gram.h"

#define CHECK_ACL(expr) do { auto _ret = (expr); if (_ret != ACL_SUCCESS) { \
    std::cerr << #expr << " failed, ret=" << _ret << std::endl; return 1; } } while (0)

static int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    int64_t s = 1;
    for (auto v : shape) s *= v;
    return s;
}

static std::vector<char> ReadBin(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("open failed: " + path);
    return std::vector<char>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

static void WriteBin(const std::string &path, const void *data, size_t bytes)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("write failed: " + path);
    ofs.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
}

static aclTensor *CreateTensorFromFile(const std::string &path, const std::vector<int64_t> &shape,
                                       aclDataType dtype, size_t elemBytes, void **dev)
{
    auto host = ReadBin(path);
    const size_t bytes = static_cast<size_t>(ShapeSize(shape)) * elemBytes;
    if (host.size() != bytes) {
        throw std::runtime_error("bad file size for " + path);
    }
    if (aclrtMalloc(dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return nullptr;
    if (aclrtMemcpy(*dev, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) return nullptr;
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), *dev);
}

static aclTensor *CreateOutputTensor(const std::vector<int64_t> &shape, aclDataType dtype,
                                     size_t elemBytes, void **dev)
{
    const size_t bytes = static_cast<size_t>(ShapeSize(shape)) * elemBytes;
    if (aclrtMalloc(dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) return nullptr;
    if (aclrtMemset(*dev, bytes, 0, bytes) != ACL_SUCCESS) return nullptr;
    return aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND,
                           shape.data(), shape.size(), *dev);
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        std::cerr << "Usage: execute_complex_gram_op <device_id> <n> <input_dir> <output_dir>\n";
        return 1;
    }
    const int32_t deviceId = std::atoi(argv[1]);
    const int32_t n = std::atoi(argv[2]);
    const int32_t k = n * 8;
    const std::string inputDir = argv[3];
    const std::string outputDir = argv[4];

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    std::vector<int64_t> aShape = {272, 256, k};
    std::vector<int64_t> bShape = {17, k, k};
    std::vector<int64_t> bsumShape = {k, k};
    std::vector<int64_t> csumShape = {n, n};

    void *arDev=nullptr, *aiDev=nullptr, *bDev=nullptr, *bplurDev=nullptr, *bpluiDev=nullptr, *bsumDev=nullptr, *csumDev=nullptr;
    aclTensor *ar = CreateTensorFromFile(inputDir + "/ar.bin", aShape, ACL_FLOAT16, sizeof(aclFloat16), &arDev);
    aclTensor *ai = CreateTensorFromFile(inputDir + "/ai.bin", aShape, ACL_FLOAT16, sizeof(aclFloat16), &aiDev);
    aclTensor *b = CreateOutputTensor(bShape, ACL_FLOAT, sizeof(float), &bDev);
    aclTensor *bplur = CreateOutputTensor(bShape, ACL_FLOAT, sizeof(float), &bplurDev);
    aclTensor *bplui = CreateOutputTensor(bShape, ACL_FLOAT, sizeof(float), &bpluiDev);
    aclTensor *bsum = CreateOutputTensor(bsumShape, ACL_FLOAT, sizeof(float), &bsumDev);
    aclTensor *csum = CreateOutputTensor(csumShape, ACL_FLOAT, sizeof(float), &csumDev);
    if (!ar || !ai || !b || !bplur || !bplui || !bsum || !csum) {
        std::cerr << "aclTensor create failed\n";
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACL(aclnnComplexGramGetWorkspaceSize(ar, ai, b, bplur, bplui, bsum, csum, &workspaceSize, &executor));
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnComplexGram(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<float> hostB(ShapeSize(bShape));
    std::vector<float> hostBplur(ShapeSize(bShape));
    std::vector<float> hostBplui(ShapeSize(bShape));
    std::vector<float> hostBsum(ShapeSize(bsumShape));
    std::vector<float> hostCsum(ShapeSize(csumShape));
    CHECK_ACL(aclrtMemcpy(hostB.data(), hostB.size()*sizeof(float), bDev, hostB.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(hostBplur.data(), hostBplur.size()*sizeof(float), bplurDev, hostBplur.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(hostBplui.data(), hostBplui.size()*sizeof(float), bpluiDev, hostBplui.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(hostBsum.data(), hostBsum.size()*sizeof(float), bsumDev, hostBsum.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(hostCsum.data(), hostCsum.size()*sizeof(float), csumDev, hostCsum.size()*sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));

    WriteBin(outputDir + "/b.bin", hostB.data(), hostB.size()*sizeof(float));
    WriteBin(outputDir + "/bplur.bin", hostBplur.data(), hostBplur.size()*sizeof(float));
    WriteBin(outputDir + "/bplui.bin", hostBplui.data(), hostBplui.size()*sizeof(float));
    WriteBin(outputDir + "/bsum.bin", hostBsum.data(), hostBsum.size()*sizeof(float));
    WriteBin(outputDir + "/csum.bin", hostCsum.data(), hostCsum.size()*sizeof(float));

    aclDestroyTensor(ar); aclDestroyTensor(ai); aclDestroyTensor(b); aclDestroyTensor(bplur); aclDestroyTensor(bplui); aclDestroyTensor(bsum); aclDestroyTensor(csum);
    aclrtFree(arDev); aclrtFree(aiDev); aclrtFree(bDev); aclrtFree(bplurDev); aclrtFree(bpluiDev); aclrtFree(bsumDev); aclrtFree(csumDev);
    if (workspace) aclrtFree(workspace);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    std::cout << "ComplexGram run finished, outputs written to " << outputDir << std::endl;
    return 0;
}

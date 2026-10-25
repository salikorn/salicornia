// salikorn / S8 spike — raw Direct3D 12 compute-shader GEMM on the iGPU.
// Motivation: the 2022 Intel driver hangs every DirectML dispatch, but its plain compute
// pipeline is game-tested. This spike runs a hand-written tiled fp32 GEMM (C = A * B^T)
// via a runtime-compiled SM5.1 compute shader (d3dcompiler_47, loaded dynamically).
// Events -> runs/<ts>-s8-d3d12/events.jsonl. Flags: --warp, --check N, --sizes a,b,c.
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "xlab/log.hpp"

using Microsoft::WRL::ComPtr;

static const char* kGemmHlsl = R"(
StructuredBuffer<float> bufA : register(t0);
StructuredBuffer<float> bufB : register(t1);   // B is [N, K] row-major; C = A * B^T
RWStructuredBuffer<float> bufC : register(u0);

uint4 dims : register(b0);   // x=M, y=N, z=K

groupshared float As[16][16];
groupshared float Bs[16][16];

[numthreads(16, 16, 1)]
void cs_main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y * 16 + tid.y;
    uint col = gid.x * 16 + tid.x;
    float acc = 0.0;
    for (uint k0 = 0; k0 < K; k0 += 16) {
        uint kk = k0 + tid.x;
        uint kr = k0 + tid.y;
        As[tid.y][tid.x] = (row < M && kk < K) ? bufA[row * K + kk] : 0.0;
        Bs[tid.x][tid.y] = (col < N && kr < K) ? bufB[col * K + kr] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint t = 0; t < 16; ++t) acc += As[tid.y][t] * Bs[tid.x][t];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N) bufC[row * N + col] = acc;
}
)";

static std::string hr_hex(HRESULT hr) {
    char b[16];
    std::snprintf(b, sizeof b, "0x%08X", (unsigned)hr);
    return b;
}
static void check(HRESULT hr, const char* what) {
    if (FAILED(hr)) throw std::runtime_error(std::string(what) + " failed: " + hr_hex(hr));
}

struct Env {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> dev;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;
    DXGI_ADAPTER_DESC1 adesc{};
    bool is_warp = false;

    void init(bool force_warp) {
        check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
        if (force_warp) {
            ComPtr<IDXGIFactory4> f4;
            check(factory.As(&f4), "QI factory4");
            check(f4->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
            adapter->GetDesc1(&adesc);
            is_warp = true;
        } else {
            ComPtr<IDXGIAdapter1> a;
            for (UINT i = 0; factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC1 d;
                a->GetDesc1(&d);
                if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                    adapter = a;
                    adesc = d;
                    break;
                }
            }
            if (!adapter) return init(true);
        }
        HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
        if (FAILED(hr))
            check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)),
                  "D3D12CreateDevice(11_0)");
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        check(dev->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&alloc)), "CreateCommandAllocator");
        check(dev->CreateCommandList(0, qd.Type, alloc.Get(), nullptr, IID_PPV_ARGS(&list)),
              "CreateCommandList");
        check(list->Close(), "Close(new list)");
        check(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        fence_event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event) throw std::runtime_error("CreateEvent");
    }
    void begin() {
        check(alloc->Reset(), "alloc reset");
        check(list->Reset(alloc.Get(), nullptr), "list reset");
    }
    void submit_and_wait() {
        check(list->Close(), "close");
        ID3D12CommandList* ls[] = {list.Get()};
        queue->ExecuteCommandLists(1, ls);
        check(queue->Signal(fence.Get(), ++fence_value), "Signal");
        check(fence->SetEventOnCompletion(fence_value, fence_event), "SetEventOnCompletion");
        DWORD w = WaitForSingleObject(fence_event, 120000);
        if (w != WAIT_OBJECT_0) throw std::runtime_error("fence wait failed");
        HRESULT rr = dev->GetDeviceRemovedReason();
        if (FAILED(rr)) throw std::runtime_error("device removed: " + hr_hex(rr));
    }
    void transition(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = r;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore = before;
        b.Transition.StateAfter = after;
        list->ResourceBarrier(1, &b);
    }
    void uav_barrier(ID3D12Resource* r) {
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.UAV.pResource = r;
        list->ResourceBarrier(1, &b);
    }
};

static ComPtr<ID3D12Resource> make_buffer(ID3D12Device* dev, uint64_t size, D3D12_HEAP_TYPE type,
                                          bool uav_flag) {
    D3D12_HEAP_PROPERTIES props{type};
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc = {1, 0};
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = uav_flag ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    D3D12_RESOURCE_STATES init = D3D12_RESOURCE_STATE_COMMON;
    if (type == D3D12_HEAP_TYPE_UPLOAD) init = D3D12_RESOURCE_STATE_GENERIC_READ;
    if (type == D3D12_HEAP_TYPE_READBACK) init = D3D12_RESOURCE_STATE_COPY_DEST;
    ComPtr<ID3D12Resource> r;
    check(dev->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &d, init, nullptr,
                                       IID_PPV_ARGS(&r)),
          "CreateCommittedResource");
    return r;
}
static void upload_bytes(ID3D12Resource* dst, const void* data, uint64_t size) {
    void* p = nullptr;
    D3D12_RANGE none{0, 0};
    check(dst->Map(0, &none, &p), "Map(upload)");
    std::memcpy(p, data, size_t(size));
    dst->Unmap(0, nullptr);
}

struct ShaderKit {
    ComPtr<ID3D12RootSignature> rootsig;
    ComPtr<ID3D12PipelineState> pso;
};
static ShaderKit kit;

static void build_shader(Env& env, xlab::Run& run) {
    // runtime compile via d3dcompiler_47 (dynamic load — no import lib in the Windows SDK)
    HMODULE dc = LoadLibraryW(L"d3dcompiler_47.dll");
    if (!dc) throw std::runtime_error("d3dcompiler_47.dll not found");
    using FnD3DCompile = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*,
                                          ID3DInclude*, LPCSTR, LPCSTR, UINT, UINT, ID3DBlob**,
                                          ID3DBlob**);
    auto fn = (FnD3DCompile)GetProcAddress(dc, "D3DCompile");
    if (!fn) throw std::runtime_error("D3DCompile not exported");
    ComPtr<ID3DBlob> code, errs;
    HRESULT hr = fn(kGemmHlsl, std::strlen(kGemmHlsl), nullptr, nullptr, nullptr, "cs_main",
                    "cs_5_1", 0, 0, &code, &errs);
    if (FAILED(hr)) {
        std::string msg = errs ? (const char*)errs->GetBufferPointer() : "no diagnostics";
        throw std::runtime_error("D3DCompile failed: " + msg);
    }
    run.event("hlsl_compiled", "\"bytecode\":" + std::to_string(code->GetBufferSize()));

    // root signature: [0]=SRV t0 (A), [1]=SRV t1 (B), [2]=UAV u0 (C), [3]=32-bit constants b0
    D3D12_ROOT_PARAMETER rp[4]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[0].Descriptor.ShaderRegister = 0;
    rp[0].Descriptor.RegisterSpace = 0;
    rp[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[1].Descriptor.ShaderRegister = 1;
    rp[1].Descriptor.RegisterSpace = 0;
    rp[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[2].Descriptor.ShaderRegister = 0;
    rp[2].Descriptor.RegisterSpace = 0;
    rp[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[3].Constants.ShaderRegister = 0;
    rp[3].Constants.RegisterSpace = 0;
    rp[3].Constants.Num32BitValues = 4;
    rp[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 4;
    rsd.pParameters = rp;
    rsd.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> sig, sigerr;
    check(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &sigerr),
          "D3D12SerializeRootSignature");
    check(env.dev->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                       IID_PPV_ARGS(&kit.rootsig)),
          "CreateRootSignature");

    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = kit.rootsig.Get();
    pd.CS.pShaderBytecode = code->GetBufferPointer();
    pd.CS.BytecodeLength = code->GetBufferSize();
    check(env.dev->CreateComputePipelineState(&pd, IID_PPV_ARGS(&kit.pso)),
          "CreateComputePipelineState");
}

static void run_gemm_cs(Env& env, xlab::Run& run, UINT M, UINT N, UINT K, UINT iters,
                        bool correctness) {
    const UINT64 asz = uint64_t(M) * K * 4, bsz = uint64_t(N) * K * 4, osz = uint64_t(M) * N * 4;
    std::mt19937 rng(321);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> A(size_t(M) * K), B(size_t(N) * K);
    for (auto& v : A) v = uni(rng);
    for (auto& v : B) v = uni(rng);

    auto st_a = make_buffer(env.dev.Get(), asz, D3D12_HEAP_TYPE_UPLOAD, false);
    auto st_b = make_buffer(env.dev.Get(), bsz, D3D12_HEAP_TYPE_UPLOAD, false);
    upload_bytes(st_a.Get(), A.data(), asz);
    upload_bytes(st_b.Get(), B.data(), bsz);
    auto buf_a = make_buffer(env.dev.Get(), asz, D3D12_HEAP_TYPE_DEFAULT, false);
    auto buf_b = make_buffer(env.dev.Get(), bsz, D3D12_HEAP_TYPE_DEFAULT, false);
    auto buf_c = make_buffer(env.dev.Get(), osz, D3D12_HEAP_TYPE_DEFAULT, true);
    auto rb = make_buffer(env.dev.Get(), osz, D3D12_HEAP_TYPE_READBACK, false);

    env.begin();
    env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    env.list->CopyBufferRegion(buf_a.Get(), 0, st_a.Get(), 0, asz);
    env.list->CopyBufferRegion(buf_b.Get(), 0, st_b.Get(), 0, bsz);
    env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    env.submit_and_wait();

    struct Dimsc {
        uint32_t M, N, K, pad;
    } dims{M, N, K, 0};

    auto record = [&]() {
        env.list->SetPipelineState(kit.pso.Get());
        env.list->SetComputeRootSignature(kit.rootsig.Get());
        env.list->SetComputeRootShaderResourceView(0, buf_a->GetGPUVirtualAddress());
        env.list->SetComputeRootShaderResourceView(1, buf_b->GetGPUVirtualAddress());
        env.list->SetComputeRootUnorderedAccessView(2, buf_c->GetGPUVirtualAddress());
        env.list->SetComputeRoot32BitConstants(3, 4, &dims, 0);
        env.list->Dispatch((N + 15) / 16, (M + 15) / 16, 1);
    };

    // warmup + correctness
    env.begin();
    env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    env.transition(buf_c.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    record();
    env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COMMON);
    env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COMMON);
    env.transition(buf_c.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COMMON);
    env.submit_and_wait();

    if (correctness) {
        env.begin();
        env.transition(buf_c.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        env.list->CopyBufferRegion(rb.Get(), 0, buf_c.Get(), 0, osz);
        env.transition(buf_c.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
        env.submit_and_wait();
        std::vector<float> got(size_t(M) * N);
        {
            void* p = nullptr;
            D3D12_RANGE rd{0, static_cast<SIZE_T>(osz)};
            check(rb->Map(0, &rd, &p), "Map(rb)");
            std::memcpy(got.data(), p, size_t(osz));
            D3D12_RANGE wr{0, 0};
            rb->Unmap(0, &wr);
        }
        // CPU ref: C = A B^T
        std::vector<float> Bt(size_t(K) * N);
        for (UINT j = 0; j < N; ++j)
            for (UINT k = 0; k < K; ++k) Bt[size_t(k) * N + j] = B[size_t(j) * K + k];
        double maxd = 0, sumd = 0;
        size_t cnt = 0;
        std::vector<double> crow(N, 0.0);
        for (UINT i = 0; i < M; ++i) {
            std::fill(crow.begin(), crow.end(), 0.0);
            for (UINT k = 0; k < K; ++k) {
                double a = A[size_t(i) * K + k];
                const float* bt = &Bt[size_t(k) * N];
                for (UINT j = 0; j < N; ++j) crow[j] += a * double(bt[j]);
            }
            for (UINT j = 0; j < N; ++j) {
                double d = std::fabs(double(got[size_t(i) * N + j]) - crow[j]);
                maxd = std::max(maxd, d);
                sumd += d;
                ++cnt;
            }
        }
        char b[192];
        std::snprintf(b, sizeof b, "\"m\":%u,\"max_abs_diff\":%.6e,\"mean_abs_diff\":%.6e", M, maxd,
                      sumd / double(cnt));
        run.event("cs_gemm_correct", b);
    }

    // timed batch
    env.begin();
    for (UINT i = 0; i < iters; ++i) {
        if (i) env.uav_barrier(buf_c.Get());
        env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        env.transition(buf_c.Get(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        record();
        env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_COMMON);
        env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_COMMON);
        env.transition(buf_c.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COMMON);
    }
    auto t0 = std::chrono::steady_clock::now();
    env.submit_and_wait();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    double tflops = 2.0 * double(M) * double(N) * double(K) / (ms / 1e3) / 1e12;
    char b[192];
    std::snprintf(b, sizeof b, "\"m\":%u,\"iters\":%u,\"ms\":%.3f,\"tflops\":%.4f", M, iters, ms,
                  tflops);
    run.event("cs_gemm_perf", b);
}

int main(int argc, char** argv) {
    bool warp = false;
    UINT check = 256;
    std::vector<UINT> sizes{512, 1024, 2048};
    UINT iters = 10;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return UINT(std::stoi(argv[++i])); };
        if (a == "--warp") warp = true;
        else if (a == "--check") check = next();
        else if (a == "--iters") iters = next();
        else if (a == "--sizes") {
            sizes.clear();
            std::string s = argv[++i];
            size_t p = 0;
            while ((p = s.find(',')) != std::string::npos) {
                sizes.push_back(UINT(std::stoul(s.substr(0, p))));
                s.erase(0, p + 1);
            }
            sizes.push_back(UINT(std::stoul(s)));
        }
    }
    xlab::Run run(warp ? "s8-d3d12-warp" : "s8-d3d12");
    Env env;
    try {
        env.init(warp);
        std::wstring wname(env.adesc.Description);
        std::string name(wname.begin(), wname.end());
        run.event("env", "\"adapter\":\"" + name + "\",\"warp\":" +
                             (env.is_warp ? "true" : "false"));
        build_shader(env, run);
        run_gemm_cs(env, run, check, check, check, 1, true);
        for (UINT s : sizes) run_gemm_cs(env, run, s, s, s, iters, false);
        run.event("done", "\"status\":\"ok\"");
        return 0;
    } catch (const std::exception& e) {
        run.event("fatal", std::string("\"error\":\"") + e.what() + "\"");
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}

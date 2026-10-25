/** \file
 *  \brief Implementations for the D3D12 compute backend declared in cs.hpp.
 *
 * Bodies are moved verbatim from the former header-only definitions; the
 * documented contracts live on the declarations in cs.hpp. Notable
 * implementation details not visible from the header:
 * - Kernel compilation binds d3dcompiler_47.dll at run time via LoadLibraryW /
 *   GetProcAddress("D3DCompile") — no link-time d3dcompiler dependency.
 * - submit_and_wait() enforces a 120 s fence timeout and validates
 *   GetDeviceRemovedReason() after every batch.
 * - Buffer::upload()/download() build their staging buffer on the fly and run a
 *   synchronous COMMON -> COPY_DEST/COPY_SOURCE -> COMMON cycle per call.
 */
#include "gpu/cs.hpp"

#include <d3dcompiler.h>

#include <cstring>
#include <stdexcept>

namespace sal::gpu {

void check(HRESULT hr, const char* what) {
    if (FAILED(hr)) {
        char b[64];
        std::snprintf(b, sizeof b, "0x%08X", (unsigned)hr);
        throw std::runtime_error(std::string(what) + " failed: " + b);
    }
}

void Device::init(bool force_warp) {
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory_)), "CreateDXGIFactory2");
    if (force_warp) {
        ComPtr<IDXGIFactory4> f4;
        check(factory_.As(&f4), "QI factory4");
        check(f4->EnumWarpAdapter(IID_PPV_ARGS(&adapter_)), "EnumWarpAdapter");
        adapter_->GetDesc1(&adesc_);
        warp_ = true;
    } else {
        ComPtr<IDXGIAdapter1> a;
        for (UINT i = 0; factory_->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d;
            a->GetDesc1(&d);
            if (!(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                adapter_ = a;
                adesc_ = d;
                break;
            }
        }
        if (!adapter_) return init(true);
    }
    HRESULT hr = D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev_));
    if (FAILED(hr))
        check(D3D12CreateDevice(adapter_.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev_)),
              "D3D12CreateDevice(11_0)");
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    check(dev_->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue_)), "CreateCommandQueue");
    check(dev_->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&alloc_)), "CreateCommandAllocator");
    check(dev_->CreateCommandList(0, qd.Type, alloc_.Get(), nullptr, IID_PPV_ARGS(&list_)),
          "CreateCommandList");
    check(list_->Close(), "Close(new list)");
    check(dev_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "CreateFence");
    fence_event_ = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event_) throw std::runtime_error("CreateEvent");
}

ID3D12Device* Device::dev() { return dev_.Get(); }
bool Device::warp() const { return warp_; }
std::string Device::adapter_name() const {
    return {adesc_.Description, adesc_.Description + std::wcslen(adesc_.Description)};
}

void Device::begin() {
    check(alloc_->Reset(), "alloc reset");
    check(list_->Reset(alloc_.Get(), nullptr), "list reset");
}
void Device::submit_and_wait() {
    check(list_->Close(), "close");
    ID3D12CommandList* ls[] = {list_.Get()};
    queue_->ExecuteCommandLists(1, ls);
    check(queue_->Signal(fence_.Get(), ++fence_value_), "Signal");
    check(fence_->SetEventOnCompletion(fence_value_, fence_event_), "SetEventOnCompletion");
    DWORD w = WaitForSingleObject(fence_event_, 120000);
    if (w != WAIT_OBJECT_0) throw std::runtime_error("GPU fence wait failed/timeout");
    HRESULT rr = dev_->GetDeviceRemovedReason();
    if (FAILED(rr)) throw std::runtime_error("device removed during compute");
}
void Device::transition(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list_->ResourceBarrier(1, &b);
}
void Device::uav_barrier(ID3D12Resource* r) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = r;
    list_->ResourceBarrier(1, &b);
}
ID3D12GraphicsCommandList* Device::list() { return list_.Get(); }

Buffer::Buffer(Device& d, uint64_t size, BufferKind kind) : dev_(&d), size_(size), kind_(kind) {
    D3D12_HEAP_TYPE ht;
    D3D12_RESOURCE_STATES init;
    bool uav = false;
    switch (kind) {
        case BufferKind::SrcRead: ht = D3D12_HEAP_TYPE_DEFAULT; init = D3D12_RESOURCE_STATE_COMMON; break;
        case BufferKind::Uav: ht = D3D12_HEAP_TYPE_DEFAULT; init = D3D12_RESOURCE_STATE_COMMON; uav = true; break;
        case BufferKind::CpuWrite: ht = D3D12_HEAP_TYPE_UPLOAD; init = D3D12_RESOURCE_STATE_GENERIC_READ; break;
        case BufferKind::Readback: ht = D3D12_HEAP_TYPE_READBACK; init = D3D12_RESOURCE_STATE_COPY_DEST; break;
    }
    D3D12_HEAP_PROPERTIES props{ht};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc = {1, 0};
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    check(d.dev()->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, init, nullptr,
                                           IID_PPV_ARGS(&res_)),
          "CreateCommittedResource");
}
uint64_t Buffer::size() const { return size_; }
ID3D12Resource* Buffer::res() { return res_.Get(); }
D3D12_GPU_VIRTUAL_ADDRESS Buffer::gpu_addr() { return res_->GetGPUVirtualAddress(); }

// CPU-side staging copy (uploads/downloads go via CopyBufferRegion on the device queue)
void Buffer::upload(const void* data, uint64_t bytes, uint64_t dst_offset) {
    if (kind_ == BufferKind::CpuWrite) {
        void* p = nullptr;
        D3D12_RANGE none{0, 0};
        check(res_->Map(0, &none, &p), "Map(upload)");
        std::memcpy(reinterpret_cast<uint8_t*>(p) + dst_offset, data, size_t(bytes));
        res_->Unmap(0, nullptr);
    } else {
        Buffer staging(*dev_, bytes, BufferKind::CpuWrite);
        staging.upload(data, bytes, 0);
        dev_->begin();
        dev_->transition(res_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
        dev_->list()->CopyBufferRegion(res_.Get(), dst_offset, staging.res(), 0, bytes);
        dev_->transition(res_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
        dev_->submit_and_wait();
    }
}
void Buffer::download(void* data, uint64_t bytes, uint64_t src_offset) {
    Buffer staging(*dev_, bytes, BufferKind::Readback);
    dev_->begin();
    dev_->transition(res_.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    dev_->list()->CopyBufferRegion(staging.res(), 0, res_.Get(), src_offset, bytes);
    dev_->transition(res_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    dev_->submit_and_wait();
    staging.map_read(data, bytes);
}
void Buffer::map_read(void* data, uint64_t bytes) {
    void* p = nullptr;
    D3D12_RANGE rd{0, static_cast<SIZE_T>(bytes)};
    check(res_->Map(0, &rd, &p), "Map(read)");
    std::memcpy(data, p, size_t(bytes));
    D3D12_RANGE wr{0, 0};
    res_->Unmap(0, &wr);
}

std::string substitute_tokens(
    const std::string& tmpl, const std::map<std::string, std::string>& defines) {
    std::string out = tmpl;
    for (const auto& [k, v] : defines) {
        std::string tok = "@" + k + "@";
        for (size_t pos; (pos = out.find(tok)) != std::string::npos;)
            out.replace(pos, tok.size(), v);
    }
    if (out.find('@') != std::string::npos)
        throw std::runtime_error("unresolved @token@ in shader template");
    return out;
}

Kernel::Kernel(Device& d, const char* hlsl, const char* entry, const char* target)
    : Kernel(d, hlsl, {}, entry, target) {}

// Template form: @TOKEN@ occurrences are replaced from `defines` before compilation;
// a leftover '@' after substitution is a hard error (catches typos in templates).
Kernel::Kernel(Device& d, const char* hlsl_template,
               const std::map<std::string, std::string>& defines, const char* entry,
               const char* target) {
    std::string src = substitute_tokens(hlsl_template, defines);
    const char* hlsl = src.c_str();
    HMODULE dc = LoadLibraryW(L"d3dcompiler_47.dll");
    if (!dc) throw std::runtime_error("d3dcompiler_47.dll not found");
    using Fn = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*,
                                LPCSTR, LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**);
    auto fn = (Fn)GetProcAddress(dc, "D3DCompile");
    if (!fn) throw std::runtime_error("D3DCompile not exported");
    ComPtr<ID3DBlob> code, errs;
    HRESULT hr = fn(hlsl, std::strlen(hlsl), nullptr, nullptr, nullptr, entry, target, 0, 0,
                    &code, &errs);
    if (FAILED(hr)) {
        std::string msg = errs ? (const char*)errs->GetBufferPointer() : "no diagnostics";
        throw std::runtime_error(std::string("D3DCompile: ") + msg);
    }
    D3D12_ROOT_PARAMETER rp[4]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[0].Descriptor.ShaderRegister = 0;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[1].Descriptor.ShaderRegister = 1;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[2].Descriptor.ShaderRegister = 0;
    for (int i = 0; i < 3; ++i) {
        rp[i].Descriptor.RegisterSpace = 0;
        rp[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    }
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    rp[3].Constants.ShaderRegister = 0;
    rp[3].Constants.RegisterSpace = 0;
    rp[3].Constants.Num32BitValues = 8;
    rp[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsd{};
    rsd.NumParameters = 4;
    rsd.pParameters = rp;
    ComPtr<ID3DBlob> sig, sigerr;
    check(D3D12SerializeRootSignature(&rsd, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &sigerr),
          "D3D12SerializeRootSignature");
    check(d.dev()->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                       IID_PPV_ARGS(&rootsig_)),
          "CreateRootSignature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = rootsig_.Get();
    pd.CS.pShaderBytecode = code->GetBufferPointer();
    pd.CS.BytecodeLength = code->GetBufferSize();
    check(d.dev()->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pso_)),
          "CreateComputePipelineState");
}

// Records one dispatch. Caller is responsible for resource states + UAV barriers
// between dispatches writing the same buffer, and for keeping chunks under ~2 s.
void Kernel::dispatch(Device& d, Buffer& a, Buffer& b, Buffer& out, const uint32_t consts[8],
                      uint32_t gx, uint32_t gy, uint32_t gz) {
    d.list()->SetPipelineState(pso_.Get());
    d.list()->SetComputeRootSignature(rootsig_.Get());
    d.list()->SetComputeRootShaderResourceView(0, a.gpu_addr());
    d.list()->SetComputeRootShaderResourceView(1, b.gpu_addr());
    d.list()->SetComputeRootUnorderedAccessView(2, out.gpu_addr());
    d.list()->SetComputeRoot32BitConstants(3, 8, consts, 0);
    d.list()->Dispatch(gx, gy, gz);
}

}  // namespace sal::gpu

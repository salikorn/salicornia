// salikorn / M1 spikes S1+S2 — DirectML feasibility on THIS machine.
//
// S1: FP16 GEMM correctness vs CPU reference + async throughput (TFLOPS) on the real adapter.
// S2: adapter identity, DirectML feature level, tensor-dtype support, operator availability
//     (probed via CreateOperator+CompileOperator — the 1.15.4 header exposes no cheaper query).
//
// Results go to runs/<ts>-s1s2-dml/events.jsonl (xlab). Run from the repo root.
// Flags: --warp (force software adapter), --m/--n/--k <int>, --iters <int>, --check <int>.

/// \file
/// \brief M1 spikes S1+S2: DirectML feasibility probes on this machine's adapter.
///
/// S2 answers "what does this driver support?" — adapter identity, DML feature level,
/// tensor-dtype support, and operator availability via CreateOperator+CompileOperator
/// (the 1.15.4 header exposes no cheaper query). S1 answers "does it actually run
/// correctly and fast?" — fp32 identity/softmax sanity ops, then FP16 GEMM correctness
/// against a CPU reference and timed TFLOP/s at 4096^3.
///
/// Findings baked into the design (comments inline): bare CompileOperator dispatches
/// misbehave on this driver (zeros/hangs), so execution always goes through
/// compile_single_op_graph(); initializer outputs must bind as a BUFFER_ARRAY; buffer
/// states are managed with fully explicit transitions — no reliance on implicit
/// promotion/decay, which old/strict drivers mishandle (Env::transition comment).
/// Optional diagnostics: SAL_DML_DEBUG enables the D3D12 debug layer, SAL_TRACE prints
/// per-stage progress, and the info queue is dumped on fatal errors.
///
/// Flags: --warp (force software adapter), --compute (compute queue), --skip-gemm,
/// --notrans (B row-major instead of B^T), --f32check, --single (bare CompileOperator),
/// --m/--n/--k, --iters, --check.

#define DML_TARGET_VERSION 0x6400
#include <DirectML.h>

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <d3d12sdklayers.h>  // optional debug-layer diagnostics (Graphics Tools)

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

// ---------------------------------------------------------------- fp16 helpers
/// \brief Local fp32 -> fp16 converter (round-to-nearest-even), identical to
///        xlab/fp16.hpp's sal::f32_to_f16 but duplicated here so the spike builds
///        standalone against raw DML data. See xlab/fp16.hpp for the documented
///        special cases.
/// \param f Value to convert. \return 16-bit half pattern.
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = int32_t((x >> 23) & 0xFFu);
    uint32_t m = x & 0x7FFFFFu;
    if (e == 255) return uint16_t(sign | 0x7C00u | (m ? 0x200u : 0u));  // Inf/NaN
    e = e - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7C00u);                       // overflow -> Inf
    if (e <= 0) {
        if (e < -10) return uint16_t(sign);                             // underflow -> 0
        m |= 0x800000u;                                                 // implicit bit -> 24-bit
        uint32_t shift = 14u - uint32_t(e);                             // e in [-10, 0]
        uint32_t kept = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (kept & 1u))) kept += 1;
        return uint16_t(sign | kept);
    }
    uint32_t kept = m >> 13;
    uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (kept & 1u))) {
        kept += 1;
        if (kept == 0x400u) {
            kept = 0;
            e += 1;
            if (e >= 31) return uint16_t(sign | 0x7C00u);
        }
    }
    return uint16_t(sign | (uint32_t(e) << 10) | kept);
}

/// \brief Local exact fp16 -> fp32 converter, identical to xlab/fp16.hpp's
///        sal::f16_to_f32 but duplicated here so the spike builds standalone against
///        raw DML data. Subnormals are normalized; see that header for semantics.
/// \param h 16-bit half pattern. \return Exact float value.
static float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h >> 15) & 1u;
    uint32_t e = uint32_t(h >> 10) & 0x1Fu;
    uint32_t m = uint32_t(h) & 0x3FFu;
    uint32_t x;
    if (e == 0) {
        if (m == 0) {
            x = sign << 31;
        } else {  // subnormal: normalize
            int k = -1;
            while (!(m & 0x400u)) {
                m <<= 1;
                k += 1;
            }
            m &= 0x3FFu;
            x = (sign << 31) | (uint32_t(127 - 15 - k) << 23) | (m << 13);
        }
    } else if (e == 31) {
        x = (sign << 31) | 0x7F800000u | (m << 13);
    } else {
        x = (sign << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ---------------------------------------------------------------- helpers
/// \brief Formats an HRESULT as \c 0x%08X for error messages/events.
static std::string hr_hex(HRESULT hr) {
    char b[16];
    std::snprintf(b, sizeof b, "0x%08X", (unsigned)hr);
    return b;
}
/// \brief Throws std::runtime_error naming \p what and the HRESULT if FAILED(hr).
/// \param hr   Result to test.
/// \param what API call name for the error message.
static void check(HRESULT hr, const char* what) {
    if (FAILED(hr)) throw std::runtime_error(std::string(what) + " failed: " + hr_hex(hr));
}
/// \brief Naive wchar -> char narrowing of an adapter description (ASCII names only).
/// \param w Wide string. \return Narrowed copy.
static std::string w2s(const std::wstring& w) { return std::string(w.begin(), w.end()); }

// ---------------------------------------------------------------- device env
/// \brief D3D12 + DirectML environment: adapter selection (hardware, WARP fallback),
///        device/queue/allocator/list/fence, optional D3D12 debug layer (SAL_DML_DEBUG),
///        DML device + command recorder, and an optional info queue for diagnostics.
///
/// Same resource-state discipline as spike_d3d12.cpp: transition() records explicit
/// COMMON <-> shader/UAV/COPY barriers because implicit promotion/decay is not relied
/// on — old/strict drivers mishandle it. submit_and_wait() fences with a 120 s timeout
/// (hang detector) and then checks GetDeviceRemovedReason.
struct Env {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> dev;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<IDMLDevice> dml;
    ComPtr<IDMLCommandRecorder> recorder;
    ComPtr<ID3D12InfoQueue> info;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;
    DXGI_ADAPTER_DESC1 adesc{};
    bool is_warp = false;

    void dump_info_queue(const char* where) {
        if (!info) return;
        UINT64 n = info->GetNumStoredMessages();
        for (UINT64 i = 0; i < n && i < 40; ++i) {
            SIZE_T sz = 0;
            if (FAILED(info->GetMessage(i, nullptr, &sz)) || sz == 0) continue;
            std::vector<char> buf(sz);
            D3D12_MESSAGE* msg = (D3D12_MESSAGE*)buf.data();
            if (SUCCEEDED(info->GetMessage(i, msg, &sz)))
                std::fprintf(stderr, "[d3d12:%s] %s\n", where, msg->pDescription ? msg->pDescription : "");
        }
    }

    void init(bool force_warp, bool compute_queue = false) {
        // Best-effort debug layer (needs Graphics Tools; silently absent otherwise).
        if (getenv("SAL_DML_DEBUG")) {
            ComPtr<ID3D12Debug> dbg;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
                dbg->EnableDebugLayer();
                std::fprintf(stderr, "[spike] D3D12 debug layer enabled\n");
            }
        }
        check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
        if (force_warp) {
            ComPtr<IDXGIFactory4> f4;
            check(factory.As(&f4), "QI IDXGIFactory4");
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
            if (!adapter) return init(true);  // no hardware adapter; fall back
        }
        HRESULT hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev));
        if (FAILED(hr)) check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev)),
                              "D3D12CreateDevice(11_0)");
        (void)dev.As(&info);
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = compute_queue ? D3D12_COMMAND_LIST_TYPE_COMPUTE : D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        check(dev->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&alloc)), "CreateCommandAllocator");
        check(dev->CreateCommandList(0, qd.Type, alloc.Get(), nullptr, IID_PPV_ARGS(&list)),
              "CreateCommandList");
        check(list->Close(), "Close(new list)");
        check(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
        fence_event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event) throw std::runtime_error("CreateEvent failed");

        check(DMLCreateDevice(dev.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dml)),
              "DMLCreateDevice");
        check(dml->CreateCommandRecorder(IID_PPV_ARGS(&recorder)), "CreateCommandRecorder");
    }

    void begin() {
        check(alloc->Reset(), "CommandAllocator::Reset");
        check(list->Reset(alloc.Get(), nullptr), "CommandList::Reset");
    }
    // Fully explicit buffer state machine — no reliance on implicit promotion/decay,
    // which old/strict drivers mishandle.
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
    void submit_and_wait() {
        check(list->Close(), "CommandList::Close");
        ID3D12CommandList* lists[] = {list.Get()};
        queue->ExecuteCommandLists(1, lists);
        check(queue->Signal(fence.Get(), ++fence_value), "Signal");
        check(fence->SetEventOnCompletion(fence_value, fence_event), "SetEventOnCompletion");
        DWORD w = WaitForSingleObject(fence_event, 120000);
        if (w != WAIT_OBJECT_0) throw std::runtime_error("GPU fence wait failed/timeout (w=" + std::to_string(w) + ")");
        HRESULT rr = dev->GetDeviceRemovedReason();
        if (FAILED(rr)) throw std::runtime_error("device removed: " + hr_hex(rr));
    }
};

// ---------------------------------------------------------------- buffers
/// \brief Creates a committed buffer resource, choosing the initial state from the
///        heap type (UPLOAD -> GENERIC_READ, READBACK -> COPY_DEST, else COMMON).
///        Same contract as spike_d3d12.cpp's make_buffer.
/// \param dev       Device to create the resource on.
/// \param size      Buffer width in bytes.
/// \param type      Heap type (DEFAULT/UPLOAD/READBACK).
/// \param allow_uav True to add D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.
/// \return The created resource.
static ComPtr<ID3D12Resource> make_buffer(ID3D12Device* dev, uint64_t size, D3D12_HEAP_TYPE type,
                                          bool allow_uav) {
    D3D12_HEAP_PROPERTIES props{type};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_UNKNOWN;
    desc.SampleDesc = {1, 0};
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = allow_uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    D3D12_RESOURCE_STATES initial = D3D12_RESOURCE_STATE_COMMON;
    if (type == D3D12_HEAP_TYPE_UPLOAD) initial = D3D12_RESOURCE_STATE_GENERIC_READ;
    if (type == D3D12_HEAP_TYPE_READBACK) initial = D3D12_RESOURCE_STATE_COPY_DEST;
    ComPtr<ID3D12Resource> r;
    check(dev->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, initial, nullptr,
                                       IID_PPV_ARGS(&r)),
          "CreateCommittedResource");
    return r;
}

/// \brief Memcpy-uploads \p size bytes into a mapped UPLOAD-heap buffer.
/// \param dst  Upload resource. \param data Source bytes. \param size Byte count.
static void upload_bytes(ID3D12Resource* dst, const void* data, uint64_t size) {
    void* p = nullptr;
    D3D12_RANGE none{0, 0};
    check(dst->Map(0, &none, &p), "Map(upload)");
    std::memcpy(p, data, size_t(size));
    dst->Unmap(0, nullptr);
}
/// \brief Copies \p size bytes out of a READBACK-heap buffer after GPU completion.
/// \param src  Readback resource. \param size Byte count.
/// \return Copy of the read region.
static std::vector<uint8_t> readback_bytes(ID3D12Resource* src, uint64_t size) {
    void* p = nullptr;
    D3D12_RANGE read{0, static_cast<SIZE_T>(size)};
    check(src->Map(0, &read, &p), "Map(readback)");
    std::vector<uint8_t> out(static_cast<size_t>(size));
    std::memcpy(out.data(), p, static_cast<size_t>(size));
    D3D12_RANGE wr{0, 0};
    src->Unmap(0, &wr);
    return out;
}

// ---------------------------------------------------------------- tensor desc
/// \brief Buffer-tensor description with owning strides/sizes storage (DML's
///        DML_BUFFER_TENSOR_DESC points into these arrays, so they must outlive it).
struct TensorInfo {
    DML_BUFFER_TENSOR_DESC buf{};   ///< Buffer tensor descriptor.
    DML_TENSOR_DESC desc{};         ///< Generic descriptor wrapping \c buf.
    UINT sizes[6]{};                ///< Dimension sizes backing buf.Sizes.
    UINT strides[6]{};              ///< Row-major strides backing buf.Strides (optional).

    /// \brief Returns the generic DML_TENSOR_DESC pointer for operator descs.
    const DML_TENSOR_DESC* desc_ptr() const { return &desc; }
};

/// \brief Bytes per element for the DML tensor data types this spike uses
///        (2/4/8 for 16/32/64-bit ints and floats, 1 as the fallback).
/// \param t DML data type. \return Element size in bytes.
static UINT bytes_per_elem(DML_TENSOR_DATA_TYPE t) {
    switch (t) {
        case DML_TENSOR_DATA_TYPE_FLOAT16:
        case DML_TENSOR_DATA_TYPE_INT16:
        case DML_TENSOR_DATA_TYPE_UINT16: return 2;
        case DML_TENSOR_DATA_TYPE_FLOAT32:
        case DML_TENSOR_DATA_TYPE_INT32:
        case DML_TENSOR_DATA_TYPE_UINT32: return 4;
        case DML_TENSOR_DATA_TYPE_FLOAT64:
        case DML_TENSOR_DATA_TYPE_INT64:
        case DML_TENSOR_DATA_TYPE_UINT64: return 8;
        default: return 1;
    }
}

/// \brief Fills a TensorInfo for a contiguous row-major buffer tensor (optionally
///        computing explicit row-major strides instead of a null Strides pointer).
/// \param ti               Target to fill.
/// \param sizes            Dimensions.
/// \param dt               Data type (fixes TotalTensorSizeInBytes).
/// \param flags            DML_TENSOR_FLAGS (default none).
/// \param explicit_strides True to write row-major strides into ti.strides; false for
///                         a contiguous tensor (Strides = nullptr).
static void make_tensor(TensorInfo& ti, std::initializer_list<UINT> sizes, DML_TENSOR_DATA_TYPE dt,
                        DML_TENSOR_FLAGS flags = DML_TENSOR_FLAG_NONE, bool explicit_strides = false) {
    ti.buf.DataType = dt;
    ti.buf.Flags = flags;
    ti.buf.DimensionCount = UINT(sizes.size());
    UINT i = 0;
    uint64_t elems = 1;
    for (UINT s : sizes) {
        ti.sizes[i++] = s;
        elems *= s;
    }
    ti.buf.Sizes = ti.sizes;
    ti.buf.Strides = nullptr;  // row-major contiguous
    if (explicit_strides) {
        // row-major strides per dimension
        UINT64 acc = 1;
        UINT n = ti.buf.DimensionCount;
        for (UINT d = 0; d < n; ++d) {
            UINT idx = n - 1 - d;
            ti.strides[idx] = UINT(acc);
            acc *= ti.sizes[idx];
        }
        ti.buf.Strides = ti.strides;
    }
    ti.buf.TotalTensorSizeInBytes = elems * bytes_per_elem(dt);
    ti.buf.GuaranteedBaseOffsetAlignment = 0;
    ti.desc.Type = DML_TENSOR_TYPE_BUFFER;
    ti.desc.Desc = &ti.buf;
}

// ---------------------------------------------------------------- op probe
/// \brief S2 operator-availability probe: tries CreateOperator + CompileOperator
///        (with meta-commands disabled) and reports which stage failed.
/// \param dml  DML device.
/// \param od   Operator descriptor to probe.
/// \param note Receives "CreateOperator <hr>" / "CompileOperator <hr>" on failure.
/// \return True iff the operator both created and compiled.
static bool probe_op(IDMLDevice* dml, const DML_OPERATOR_DESC& od, std::string& note) {
    ComPtr<IDMLOperator> op;
    HRESULT hr = dml->CreateOperator(&od, IID_PPV_ARGS(&op));
    if (FAILED(hr)) {
        note = "CreateOperator " + hr_hex(hr);
        return false;
    }
    ComPtr<IDMLCompiledOperator> compiled;
    hr = dml->CompileOperator(op.Get(), DML_EXECUTION_FLAG_DISABLE_META_COMMANDS, IID_PPV_ARGS(&compiled));
    if (FAILED(hr)) {
        note = "CompileOperator " + hr_hex(hr);
        return false;
    }
    return true;
}

/// \brief Binds an operator initializer's outputs. Per DML docs this must be a
///        BUFFER_ARRAY binding (one persistent buffer per compiled operator), never
///        a plain buffer binding.
/// \param table           Binding table targeting the initializer.
/// \param persistent      Persistent-resource buffer (DEFAULT heap, UAV-capable).
/// \param persistent_size Exact PersistentResourceSize of the compiled operator.
static void bind_initializer_outputs(IDMLBindingTable* table, ID3D12Resource* persistent,
                                     uint64_t persistent_size) {
    DML_BUFFER_BINDING pb{persistent, 0, persistent_size};
    DML_BUFFER_ARRAY_BINDING array{1, &pb};
    DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER_ARRAY, &array};
    table->BindOutputs(1, &d);
}

/// \brief Compiles a single operator as a one-node DML graph. On this system bare
///        CompileOperator dispatches are broken (zeros/hangs) while CompileGraph
///        works — the engine always uses this path (--single opts out for A/B runs).
/// \param env   Environment with an initialized DML device.
/// \param op    Operator to wrap in the graph node.
/// \param n_in  Graph input count.
/// \param n_out Graph output count.
/// \param flags Execution flags passed to CompileGraph.
/// \return The compiled graph operator.
static ComPtr<IDMLCompiledOperator> compile_single_op_graph(Env& env, IDMLOperator* op,
                                                            UINT n_in, UINT n_out,
                                                            DML_EXECUTION_FLAGS flags) {
    ComPtr<IDMLDevice1> dev1;
    check(env.dml.As(&dev1), "QI IDMLDevice1");
    DML_OPERATOR_GRAPH_NODE_DESC nodeOp{op, nullptr};
    DML_GRAPH_NODE_DESC nodes[1] = {{DML_GRAPH_NODE_TYPE_OPERATOR, &nodeOp}};
    std::vector<DML_INPUT_GRAPH_EDGE_DESC> ie(n_in);
    std::vector<DML_GRAPH_EDGE_DESC> ied(n_in);
    for (UINT i = 0; i < n_in; ++i) {
        ie[i] = {i, 0, i, nullptr};
        ied[i] = {DML_GRAPH_EDGE_TYPE_INPUT, &ie[i]};
    }
    DML_OUTPUT_GRAPH_EDGE_DESC oe{0, 0, 0, nullptr};
    DML_GRAPH_EDGE_DESC oed[1] = {{DML_GRAPH_EDGE_TYPE_OUTPUT, &oe}};
    DML_GRAPH_DESC g{};
    g.InputCount = n_in;
    g.OutputCount = n_out;
    g.NodeCount = 1;
    g.Nodes = nodes;
    g.InputEdgeCount = n_in;
    g.InputEdges = ied.data();
    g.OutputEdgeCount = n_out;
    g.OutputEdges = oed;
    ComPtr<IDMLCompiledOperator> compiled;
    check(dev1->CompileGraph(&g, flags, IID_PPV_ARGS(&compiled)), "CompileGraph(single op)");
    return compiled;
}

// ---------------------------------------------------------------- unary op exec test
// Executes a 1-in/1-out fp32 op over 4 floats with the full correct DML pattern:
// staging copies, initializer with BUFFER_ARRAY persistent output, persistent/temp
// bound at exec, explicit transitions throughout, readback.
/// \brief Executes a 1-in/1-out fp32 op over 4 floats through the complete correct
///        DML execution pattern, as an S1 sanity check: staging copy-in, initializer
///        dispatch (BUFFER_ARRAY persistent output), exec dispatch, readback. Each
///        submit wraps every resource in explicit transitions per the spike's
///        resource-state discipline. Logs \c \<name\>_exec with got[]/ok.
/// \param env     Environment.
/// \param run     Event sink.
/// \param name    Op name for the event key and SAL_TRACE lines.
/// \param od      Operator descriptor.
/// \param in_data 4 input floats.
/// \param ref     4 expected output floats.
/// \param tol     Per-element absolute tolerance for the ok verdict.
static void run_unary_op(Env& env, xlab::Run& run, const char* name,
                         const DML_OPERATOR_DESC& od, const float in_data[4],
                         const float ref[4], float tol) {
    auto trace = [&](const char* s) {
        if (getenv("SAL_TRACE")) {
            std::fprintf(stderr, "[%s] %s\n", name, s);
            std::fflush(stderr);
        }
    };
    ComPtr<IDMLOperator> op;
    check(env.dml->CreateOperator(&od, IID_PPV_ARGS(&op)), "CreateOperator");
    ComPtr<IDMLCompiledOperator> compiled =
        compile_single_op_graph(env, op.Get(), 1, 1, DML_EXECUTION_FLAG_DISABLE_META_COMMANDS);
    ComPtr<IDMLOperatorInitializer> init;
    IDMLCompiledOperator* ops[] = {compiled.Get()};
    check(env.dml->CreateOperatorInitializer(1, ops, IID_PPV_ARGS(&init)), "CreateOperatorInitializer");

    DML_BINDING_PROPERTIES ip = init->GetBindingProperties();
    DML_BINDING_PROPERTIES ep = compiled->GetBindingProperties();
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = std::max(ip.RequiredDescriptorCount, ep.RequiredDescriptorCount) + 4;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(env.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap");
    auto round_up = [](uint64_t v, uint64_t a) { return (v + a - 1) / a * a; };
    ComPtr<ID3D12Resource> persistent, temporary;
    if (ep.PersistentResourceSize > 0)
        persistent = make_buffer(env.dev.Get(), round_up(ep.PersistentResourceSize, 256),
                                 D3D12_HEAP_TYPE_DEFAULT, true);
    const uint64_t tempsz = std::max(ip.TemporaryResourceSize, ep.TemporaryResourceSize);
    if (tempsz > 0)
        temporary = make_buffer(env.dev.Get(), round_up(tempsz, 256), D3D12_HEAP_TYPE_DEFAULT, true);

    const UINT64 sz = sizeof(float) * 4;
    auto stage = make_buffer(env.dev.Get(), sz, D3D12_HEAP_TYPE_UPLOAD, false);
    upload_bytes(stage.Get(), in_data, sz);
    auto buf_in = make_buffer(env.dev.Get(), sz, D3D12_HEAP_TYPE_DEFAULT, true);
    auto buf_out = make_buffer(env.dev.Get(), sz, D3D12_HEAP_TYPE_DEFAULT, true);
    auto rb = make_buffer(env.dev.Get(), sz, D3D12_HEAP_TYPE_READBACK, false);
    trace("copy-in submit");
    env.begin();
    env.transition(buf_in.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    env.list->CopyBufferRegion(buf_in.Get(), 0, stage.Get(), 0, sz);
    env.transition(buf_in.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    env.submit_and_wait();

    DML_BINDING_TABLE_DESC btd{};
    btd.Dispatchable = init.Get();
    btd.CPUDescriptorHandle = heap->GetCPUDescriptorHandleForHeapStart();
    btd.GPUDescriptorHandle = heap->GetGPUDescriptorHandleForHeapStart();
    btd.SizeInDescriptors = hd.NumDescriptors;
    ComPtr<IDMLBindingTable> table;
    check(env.dml->CreateBindingTable(&btd, IID_PPV_ARGS(&table)), "CreateBindingTable");
    if (ip.TemporaryResourceSize > 0 && temporary) {
        DML_BUFFER_BINDING tb{temporary.Get(), 0, ip.TemporaryResourceSize};
        DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &tb};
        table->BindTemporaryResource(&d);
    }
    if (ep.PersistentResourceSize > 0 && persistent)
        bind_initializer_outputs(table.Get(), persistent.Get(), ep.PersistentResourceSize);
    trace("init dispatch submit");
    env.begin();
    if (persistent) env.transition(persistent.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (temporary) env.transition(temporary.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    env.recorder->RecordDispatch(env.list.Get(), init.Get(), table.Get());
    if (persistent) env.transition(persistent.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    if (temporary) env.transition(temporary.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    env.submit_and_wait();

    btd.Dispatchable = compiled.Get();
    check(table->Reset(&btd), "BindingTable::Reset");
    DML_BUFFER_BINDING ib{buf_in.Get(), 0, sz}, ob{buf_out.Get(), 0, sz};
    DML_BINDING_DESC ind{DML_BINDING_TYPE_BUFFER, &ib};
    DML_BINDING_DESC outd{DML_BINDING_TYPE_BUFFER, &ob};
    table->BindInputs(1, &ind);
    table->BindOutputs(1, &outd);
    if (ep.TemporaryResourceSize > 0 && temporary) {
        DML_BUFFER_BINDING tb{temporary.Get(), 0, ep.TemporaryResourceSize};
        DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &tb};
        table->BindTemporaryResource(&d);
    }
    if (ep.PersistentResourceSize > 0 && persistent) {
        DML_BUFFER_BINDING pb{persistent.Get(), 0, ep.PersistentResourceSize};
        DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &pb};
        table->BindPersistentResource(&d);
    }
    trace("exec dispatch submit");
    env.begin();
    env.transition(buf_in.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    env.transition(buf_out.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (persistent) env.transition(persistent.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (temporary) env.transition(temporary.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    env.recorder->RecordDispatch(env.list.Get(), compiled.Get(), table.Get());
    env.transition(buf_in.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    env.transition(buf_out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    if (persistent) env.transition(persistent.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    if (temporary) env.transition(temporary.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    env.submit_and_wait();

    trace("readback submit");
    env.begin();
    env.transition(buf_out.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    env.list->CopyBufferRegion(rb.Get(), 0, buf_out.Get(), 0, sz);
    env.transition(buf_out.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    env.submit_and_wait();
    trace("readback done");

    float got[4] = {-9e9f, -9e9f, -9e9f, -9e9f};
    void* p = nullptr;
    D3D12_RANGE rd{0, sz};
    check(rb->Map(0, &rd, &p), "Map(unary)");
    std::memcpy(got, p, size_t(sz));
    D3D12_RANGE wr{0, 0};
    rb->Unmap(0, &wr);
    bool ok = true;
    for (int i = 0; i < 4; ++i)
        if (std::fabs(got[i] - ref[i]) > tol) ok = false;
    char buf[256];
    std::snprintf(buf, sizeof buf, "\"got\":[%g,%g,%g,%g],\"ok\":%s", got[0], got[1], got[2], got[3],
                  ok ? "true" : "false");
    run.event(std::string(name) + "_exec", buf);
}

/// \brief S1 sanity op: DML_ELEMENT_WISE_IDENTITY over [1,2,2,1] fp32 (must return
///        the input bit-for-bit).
static void run_identity(Env& env, xlab::Run& run) {
    TensorInfo ti, to;
    make_tensor(ti, {1, 2, 2, 1}, DML_TENSOR_DATA_TYPE_FLOAT32);
    make_tensor(to, {1, 2, 2, 1}, DML_TENSOR_DATA_TYPE_FLOAT32);
    DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC id{ti.desc_ptr(), to.desc_ptr()};
    DML_OPERATOR_DESC od{DML_OPERATOR_ELEMENT_WISE_IDENTITY, &id};
    const float in_data[4] = {0.f, 1.f, 2.f, 3.f};
    run_unary_op(env, run, "identity", od, in_data, in_data, 1e-6f);
}

// Real compute kernel (no metacommand path): softmax1 over [1,2,3,4] -> [0.0321,0.0871,0.2369,0.6439]
/// \brief S1 sanity op: real compute-kernel softmax1 over [1,2,3,4] -> known
///        reference (no metacommand path involved).
static void run_softmax(Env& env, xlab::Run& run) {
    TensorInfo ti, to;
    make_tensor(ti, {1, 1, 1, 4}, DML_TENSOR_DATA_TYPE_FLOAT32);
    make_tensor(to, {1, 1, 1, 4}, DML_TENSOR_DATA_TYPE_FLOAT32);
    UINT ax = 3;
    DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC sd{ti.desc_ptr(), to.desc_ptr(), 1, &ax};
    DML_OPERATOR_DESC od{DML_OPERATOR_ACTIVATION_SOFTMAX1, &sd};
    const float in_data[4] = {1.f, 2.f, 3.f, 4.f};
    const float ref[4] = {0.0320586f, 0.0871443f, 0.2368828f, 0.6439143f};
    run_unary_op(env, run, "softmax", od, in_data, ref, 1e-4f);
}

// ---------------------------------------------------------------- GEMM
// Computes C = A * B^T (TransB=TRANSPOSE — the x @ W^T shape every LLM uses).
/// \brief Aggregated GEMM run statistics for one run_gemm call.
struct GemmStats {
    double max_abs_diff = 0.0, mean_abs_diff = 0.0;  ///< Max / mean |gpu - cpu_ref| element when check_vs_cpu.
    double ms = 0.0, tflops = 0.0;                   ///< Per-dispatch wall-clock ms and 2*M*N*K throughput (timed batch).
};

/// \brief Runs one DML GEMM (C = A * B^T by default — the x @ W^T shape every LLM
///        uses): staging uploads, graph/bare compilation, initializer + exec
///        dispatches with explicit transitions throughout, warmup + timed batch
///        (UAV barrier between iterations), optional CPU-reference check from the
///        same rounded inputs with fp32 accumulation.
/// \param env          Environment.
/// \param M,N,K        GEMM dimensions.
/// \param iters        Timed dispatch count on one command list.
/// \param check_vs_cpu Read back and compare vs CPU reference (small sizes only).
/// \param dt           DML_TENSOR_DATA_TYPE_FLOAT16 or _FLOAT32.
/// \param trans_b      True: B is [N,K] with TransB=TRANSPOSE; false: B is [K,N].
/// \param use_graph    True: compile via CompileGraph (the working path); false: bare
///                     CompileOperator (the broken-on-this-driver path, for A/B).
/// \return GemmStats with correctness and/or timing fields filled.
static GemmStats run_gemm(Env& env, UINT M, UINT N, UINT K, UINT iters, bool check_vs_cpu,
                          DML_TENSOR_DATA_TYPE dt, bool trans_b = true, bool use_graph = true) {
    GemmStats st;
    const bool half = (dt == DML_TENSOR_DATA_TYPE_FLOAT16);
    const UINT64 elem = half ? 2 : 4;
    const UINT64 asz = uint64_t(M) * K * elem,
                 bsz = trans_b ? uint64_t(N) * K * elem : uint64_t(K) * N * elem,
                 osz = uint64_t(M) * N * elem;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    const bool tiny = (M <= 8);
    std::vector<uint16_t> ah, bh;
    std::vector<float> af, bf;
    auto bval = [&]() { return tiny ? 0.f : uni(rng); };
    if (half) {
        ah.resize(size_t(M) * K);
        bh.resize(size_t(bsz / elem));
        for (UINT i = 0; i < M; ++i)
            for (UINT k = 0; k < K; ++k)
                ah[size_t(i) * K + k] = f32_to_f16(tiny ? float(i + 1) / 16.0f : uni(rng));
        for (UINT j = 0; j < N; ++j)
            for (UINT k = 0; k < K; ++k) {
                float v = tiny ? float(j + 1) / 64.0f : uni(rng);
                uint64_t off = trans_b ? uint64_t(j) * K + k : uint64_t(k) * N + j;
                bh[size_t(off)] = f32_to_f16(v);
            }
    } else {
        af.resize(size_t(M) * K);
        bf.resize(size_t(bsz / elem));
        for (auto& v : af) v = uni(rng);
        for (auto& v : bf) v = uni(rng);
        (void)bval;
    }

    // Inputs must live in DEFAULT-heap buffers (UAV-capable); stage via UPLOAD copies.
    auto stage_a = make_buffer(env.dev.Get(), asz, D3D12_HEAP_TYPE_UPLOAD, false);
    auto stage_b = make_buffer(env.dev.Get(), bsz, D3D12_HEAP_TYPE_UPLOAD, false);
    upload_bytes(stage_a.Get(), half ? (void*)ah.data() : (void*)af.data(), asz);
    upload_bytes(stage_b.Get(), half ? (void*)bh.data() : (void*)bf.data(), bsz);
    auto buf_a = make_buffer(env.dev.Get(), asz, D3D12_HEAP_TYPE_DEFAULT, true);
    auto buf_b = make_buffer(env.dev.Get(), bsz, D3D12_HEAP_TYPE_DEFAULT, true);
    auto buf_o = make_buffer(env.dev.Get(), osz, D3D12_HEAP_TYPE_DEFAULT, true);
    auto buf_rb = make_buffer(env.dev.Get(), osz, D3D12_HEAP_TYPE_READBACK, false);
    env.begin();
    env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    env.list->CopyBufferRegion(buf_a.Get(), 0, stage_a.Get(), 0, asz);
    env.list->CopyBufferRegion(buf_b.Get(), 0, stage_b.Get(), 0, bsz);
    env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    env.submit_and_wait();

    TensorInfo ta, tb, to;
    make_tensor(ta, {1, 1, M, K}, dt, DML_TENSOR_FLAG_NONE, true);
    if (trans_b) make_tensor(tb, {1, 1, N, K}, dt, DML_TENSOR_FLAG_NONE, true);
    else make_tensor(tb, {1, 1, K, N}, dt, DML_TENSOR_FLAG_NONE, true);
    make_tensor(to, {1, 1, M, N}, dt, DML_TENSOR_FLAG_NONE, true);
    DML_GEMM_OPERATOR_DESC gd{};
    gd.ATensor = &ta.desc;
    gd.BTensor = &tb.desc;
    gd.CTensor = nullptr;
    gd.OutputTensor = &to.desc;
    gd.TransA = DML_MATRIX_TRANSFORM_NONE;
    gd.TransB = trans_b ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
    gd.Alpha = 1.0f;
    gd.Beta = 0.0f;
    gd.FusedActivation = nullptr;
    DML_OPERATOR_DESC od{DML_OPERATOR_GEMM, &gd};

    ComPtr<IDMLOperator> op;
    check(env.dml->CreateOperator(&od, IID_PPV_ARGS(&op)), "CreateOperator(GEMM)");
    ComPtr<IDMLCompiledOperator> compiled;
    if (use_graph) {
        compiled = compile_single_op_graph(env, op.Get(), 2, 1, DML_EXECUTION_FLAG_DISABLE_META_COMMANDS);
    } else {
        check(env.dml->CompileOperator(op.Get(), DML_EXECUTION_FLAG_DISABLE_META_COMMANDS,
                                       IID_PPV_ARGS(&compiled)),
              "CompileOperator(GEMM)");
    }
    ComPtr<IDMLOperatorInitializer> init;
    IDMLCompiledOperator* ops[] = {compiled.Get()};
    check(env.dml->CreateOperatorInitializer(1, ops, IID_PPV_ARGS(&init)), "CreateOperatorInitializer");

    DML_BINDING_PROPERTIES ip = init->GetBindingProperties();
    DML_BINDING_PROPERTIES ep = compiled->GetBindingProperties();
    const UINT ndesc = std::max(ip.RequiredDescriptorCount, ep.RequiredDescriptorCount) + 4;

    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = ndesc;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(env.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap");

    auto round_up = [](uint64_t v, uint64_t a) { return (v + a - 1) / a * a; };
    ComPtr<ID3D12Resource> persistent, temporary;
    if (ep.PersistentResourceSize > 0)
        persistent = make_buffer(env.dev.Get(), round_up(ep.PersistentResourceSize,
                                                         DML_PERSISTENT_BUFFER_ALIGNMENT),
                                 D3D12_HEAP_TYPE_DEFAULT, true);
    if (std::max(ip.TemporaryResourceSize, ep.TemporaryResourceSize) > 0)
        temporary = make_buffer(env.dev.Get(),
                                round_up(std::max(ip.TemporaryResourceSize, ep.TemporaryResourceSize),
                                         DML_TEMPORARY_BUFFER_ALIGNMENT),
                                D3D12_HEAP_TYPE_DEFAULT, true);

    DML_BINDING_TABLE_DESC btd{};
    btd.Dispatchable = init.Get();
    btd.CPUDescriptorHandle = heap->GetCPUDescriptorHandleForHeapStart();
    btd.GPUDescriptorHandle = heap->GetGPUDescriptorHandleForHeapStart();
    btd.SizeInDescriptors = ndesc;
    ComPtr<IDMLBindingTable> table;
    check(env.dml->CreateBindingTable(&btd, IID_PPV_ARGS(&table)), "CreateBindingTable");

    // --- initializer dispatch (no owned-by-DML weights => no inputs)
    if (ip.TemporaryResourceSize > 0 && temporary) {
        DML_BUFFER_BINDING tb{temporary.Get(), 0, ip.TemporaryResourceSize};
        DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &tb};
        table->BindTemporaryResource(&d);
    }
    if (ep.PersistentResourceSize > 0 && persistent)
        bind_initializer_outputs(table.Get(), persistent.Get(), ep.PersistentResourceSize);
    env.begin();
    if (persistent)
        env.transition(persistent.Get(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (temporary)
        env.transition(temporary.Get(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    env.recorder->RecordDispatch(env.list.Get(), init.Get(), table.Get());
    if (persistent)
        env.transition(persistent.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COMMON);
    if (temporary)
        env.transition(temporary.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COMMON);
    env.submit_and_wait();

    // --- execute
    btd.Dispatchable = compiled.Get();
    check(table->Reset(&btd), "BindingTable::Reset");
    DML_BUFFER_BINDING ab{buf_a.Get(), 0, asz}, bb{buf_b.Get(), 0, bsz}, ob{buf_o.Get(), 0, osz};
    DML_BINDING_DESC in[2] = {{DML_BINDING_TYPE_BUFFER, &ab}, {DML_BINDING_TYPE_BUFFER, &bb}};
    DML_BINDING_DESC out{DML_BINDING_TYPE_BUFFER, &ob};
    table->BindInputs(2, in);
    table->BindOutputs(1, &out);
    if (ep.TemporaryResourceSize > 0 && temporary) {
        DML_BUFFER_BINDING tb{temporary.Get(), 0, ep.TemporaryResourceSize};
        DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &tb};
        table->BindTemporaryResource(&d);
    }
    if (ep.PersistentResourceSize > 0 && persistent) {
        DML_BUFFER_BINDING pb{persistent.Get(), 0, ep.PersistentResourceSize};
        DML_BINDING_DESC d{DML_BINDING_TYPE_BUFFER, &pb};
        table->BindPersistentResource(&d);
    }

    // warmup + timed run: `iters` dispatches on one command list, one fence wait.
    auto dispatch_once = [&]() {
        env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        env.transition(buf_o.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (persistent)
            env.transition(persistent.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (temporary)
            env.transition(temporary.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        env.recorder->RecordDispatch(env.list.Get(), compiled.Get(), table.Get());
        env.transition(buf_a.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        env.transition(buf_b.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        env.transition(buf_o.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        if (persistent)
            env.transition(persistent.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
        if (temporary)
            env.transition(temporary.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    };
    env.begin();
    dispatch_once();
    D3D12_RESOURCE_BARRIER uav{};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = buf_o.Get();
    for (UINT i = 0; i + 1 < iters; ++i) {
        env.list->ResourceBarrier(1, &uav);
        env.recorder->RecordDispatch(env.list.Get(), compiled.Get(), table.Get());
    }
    auto t0 = std::chrono::steady_clock::now();
    env.submit_and_wait();
    auto t1 = std::chrono::steady_clock::now();
    double warm_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();  // 1 op

    // timed batch (explicit transitions per iteration; overhead accepted for the spike)
    env.begin();
    for (UINT i = 0; i < iters; ++i) {
        if (i) env.list->ResourceBarrier(1, &uav);
        dispatch_once();
    }
    t0 = std::chrono::steady_clock::now();
    env.submit_and_wait();
    t1 = std::chrono::steady_clock::now();
    st.ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    st.tflops = 2.0 * double(M) * double(N) * double(K) / (st.ms / 1e3) / 1e12;
    (void)warm_ms;

    if (!check_vs_cpu) return st;

    env.begin();
    env.transition(buf_o.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    env.list->CopyBufferRegion(buf_rb.Get(), 0, buf_o.Get(), 0, osz);
    env.transition(buf_o.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    env.submit_and_wait();

    // CPU reference from the same (rounded) inputs, fp32 accumulate.
    std::vector<float> A(size_t(M) * K), B(size_t(N) * K), ref(size_t(M) * N), got(size_t(M) * N);
    if (half) {
        for (size_t i = 0; i < A.size(); ++i) A[i] = f16_to_f32(ah[i]);
        for (size_t i = 0; i < B.size(); ++i) B[i] = f16_to_f32(bh[i]);
        const uint16_t* o = (const uint16_t*)readback_bytes(buf_rb.Get(), osz).data();
        for (size_t i = 0; i < got.size(); ++i) got[i] = f16_to_f32(o[i]);
    } else {
        A = af;
        B = bf;
        const float* o = (const float*)readback_bytes(buf_rb.Get(), osz).data();
        for (size_t i = 0; i < got.size(); ++i) got[i] = o[i];
    }
    std::vector<float> Bt(size_t(K) * N);  // transpose for cache-friendly ikj loop
    for (UINT j = 0; j < N; ++j)
        for (UINT k = 0; k < K; ++k) Bt[size_t(k) * N + j] = B[size_t(j) * K + k];
    std::fill(ref.begin(), ref.end(), 0.0f);
    for (UINT i = 0; i < M; ++i)
        for (UINT k = 0; k < K; ++k) {
            float a = A[size_t(i) * K + k];
            float* row = &ref[size_t(i) * N];
            const float* bt = &Bt[size_t(k) * N];
            for (UINT j = 0; j < N; ++j) row[j] += a * bt[j];
        }
    double sum = 0;
    size_t cnt = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double d = std::fabs(double(got[i]) - double(ref[i]));
        st.max_abs_diff = std::max(st.max_abs_diff, d);
        sum += d;
        ++cnt;
    }
    st.mean_abs_diff = sum / double(cnt);
    if (tiny) {
        std::printf("tiny debug — C = A * B^T (A[i,k]=(i+1)/16, B[j,k]=(j+1)/64):\n");
        for (UINT i = 0; i < std::min<UINT>(M, 4); ++i) {
            std::printf("  row %u gpu:", i);
            for (UINT j = 0; j < std::min<UINT>(N, 4); ++j)
                std::printf(" %10.5f", got[size_t(i) * N + j]);
            std::printf("   ref:");
            for (UINT j = 0; j < std::min<UINT>(N, 4); ++j)
                std::printf(" %10.5f", ref[size_t(i) * N + j]);
            std::printf("\n");
        }
    }
    return st;
}

// ---------------------------------------------------------------- main
/// \brief Entry point: parses flags, initializes the environment (optionally WARP /
///        compute queue), runs the S2 probes (feature level, dtypes, operator
///        availability), the S1 sanity ops, then fp16/fp32 GEMM correctness and
///        perf runs per flags. Fatal errors dump the info queue and log \c "fatal".
/// \return 0 on success, 1 on fatal error.
int main(int argc, char** argv) {
    bool warp = false, compute = false, skip_gemm = false, notrans = false, f32check = false,
         single = false;
    UINT M = 4096, N = 4096, K = 4096, iters = 10, ckn = 1024;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return UINT(std::stoi(argv[++i])); };
        if (a == "--warp") warp = true;
        else if (a == "--compute") compute = true;
        else if (a == "--skip-gemm") skip_gemm = true;
        else if (a == "--notrans") notrans = true;
        else if (a == "--f32check") f32check = true;
        else if (a == "--single") single = true;
        else if (a == "--m") M = next();
        else if (a == "--n") N = next();
        else if (a == "--k") K = next();
        else if (a == "--iters") iters = next();
        else if (a == "--check") ckn = next();
    }

    xlab::Run run(warp ? "s1s2-dml-warp" : "s1s2-dml");
    Env env;
    try {
        env.init(warp, compute);

        char vram[3][32];
        std::snprintf(vram[0], 32, "%llu", (unsigned long long)(env.adesc.DedicatedVideoMemory / (1024 * 1024)));
        std::snprintf(vram[1], 32, "%llu", (unsigned long long)(env.adesc.SharedSystemMemory / (1024 * 1024)));
        std::snprintf(vram[2], 32, "%04x", env.adesc.Revision);
        run.event("env", std::string("\"adapter\":\"") + w2s(env.adesc.Description) +
                       "\",\"warp\":" + (env.is_warp ? "true" : "false") +
                       ",\"dedicated_mb\":" + vram[0] + ",\"shared_mb\":" + vram[1] +
                       ",\"rev\":\"" + vram[2] + "\"");

        // S2a: feature level
        const DML_FEATURE_LEVEL levels[] = {
            DML_FEATURE_LEVEL_6_3, DML_FEATURE_LEVEL_6_2, DML_FEATURE_LEVEL_6_1,
            DML_FEATURE_LEVEL_6_0, DML_FEATURE_LEVEL_5_1, DML_FEATURE_LEVEL_5_0,
            DML_FEATURE_LEVEL_4_1, DML_FEATURE_LEVEL_4_0, DML_FEATURE_LEVEL_3_0,
            DML_FEATURE_LEVEL_2_1, DML_FEATURE_LEVEL_2_0, DML_FEATURE_LEVEL_1_0};
        DML_FEATURE_QUERY_FEATURE_LEVELS fq{UINT(std::size(levels)), levels};
        DML_FEATURE_DATA_FEATURE_LEVELS fd{};
        check(env.dml->CheckFeatureSupport(DML_FEATURE_FEATURE_LEVELS, sizeof fq, &fq, sizeof fd,
                                           &fd),
              "CheckFeatureSupport(FEATURE_LEVELS)");
        run.event("dml_feature_level", "\"max_supported\":" + std::to_string(int(fd.MaxSupportedFeatureLevel)));

        // S2b: dtype support
        const std::pair<DML_TENSOR_DATA_TYPE, const char*> dts[] = {
            {DML_TENSOR_DATA_TYPE_FLOAT16, "float16"},
            {DML_TENSOR_DATA_TYPE_FLOAT32, "float32"},
            {DML_TENSOR_DATA_TYPE_INT8, "int8"},
            {DML_TENSOR_DATA_TYPE_UINT32, "uint32"},
            {DML_TENSOR_DATA_TYPE_INT32, "int32"},
            {DML_TENSOR_DATA_TYPE_INT64, "int64"},
        };
        for (auto& [dt, name] : dts) {
            DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT q{dt};
            DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT d{};
            HRESULT hr = env.dml->CheckFeatureSupport(DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT,
                                                      sizeof q, &q, sizeof d, &d);
            run.event("dtype_support",
                      std::string("\"dtype\":\"") + name + "\",\"supported\":" +
                          (SUCCEEDED(hr) && d.IsSupported ? "true" : "false"));
        }

        // S2c: operator availability probes
        auto log_probe = [&](const char* name, bool ok, const std::string& note) {
            std::string f = std::string("\"op\":\"") + name + "\",\"ok\":" + (ok ? "true" : "false");
            if (!ok && !note.empty()) f += ",\"note\":\"" + note + "\"";
            run.event("op_probe", f);
        };
        {
            TensorInfo a, b, o;
            make_tensor(a, {1, 1, 64, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(b, {1, 1, 64, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(o, {1, 1, 64, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            DML_GEMM_OPERATOR_DESC g{&a.desc, &b.desc, nullptr, &o.desc, DML_MATRIX_TRANSFORM_NONE,
                                     DML_MATRIX_TRANSFORM_TRANSPOSE, 1.f, 0.f, nullptr};
            DML_OPERATOR_DESC od{DML_OPERATOR_GEMM, &g};
            std::string note;
            log_probe("gemm_f16", probe_op(env.dml.Get(), od, note), note);
        }
        {
            TensorInfo a, b, o;
            make_tensor(a, {1, 1, 64, 64}, DML_TENSOR_DATA_TYPE_FLOAT32);
            make_tensor(b, {1, 1, 64, 64}, DML_TENSOR_DATA_TYPE_FLOAT32);
            make_tensor(o, {1, 1, 64, 64}, DML_TENSOR_DATA_TYPE_FLOAT32);
            DML_GEMM_OPERATOR_DESC g{&a.desc, &b.desc, nullptr, &o.desc, DML_MATRIX_TRANSFORM_NONE,
                                     DML_MATRIX_TRANSFORM_TRANSPOSE, 1.f, 0.f, nullptr};
            DML_OPERATOR_DESC od{DML_OPERATOR_GEMM, &g};
            std::string note;
            log_probe("gemm_f32", probe_op(env.dml.Get(), od, note), note);
        }
        {
            TensorInfo i, o;
            UINT ax = 3;
            make_tensor(i, {1, 4, 8, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(o, {1, 4, 8, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC d{i.desc_ptr(), o.desc_ptr(), 1, &ax};
            DML_OPERATOR_DESC od{DML_OPERATOR_ACTIVATION_SOFTMAX1, &d};
            std::string note;
            log_probe("softmax1_f16", probe_op(env.dml.Get(), od, note), note);
        }
        {
            TensorInfo i, v, x;
            make_tensor(i, {1, 1, 1, 288}, DML_TENSOR_DATA_TYPE_FLOAT32);
            make_tensor(v, {1, 1, 1, 8}, DML_TENSOR_DATA_TYPE_FLOAT32);
            make_tensor(x, {1, 1, 1, 8}, DML_TENSOR_DATA_TYPE_UINT32);
            DML_TOP_K1_OPERATOR_DESC d{i.desc_ptr(), v.desc_ptr(), x.desc_ptr(), 3, 8,
                                       DML_AXIS_DIRECTION_DECREASING};
            DML_OPERATOR_DESC od{DML_OPERATOR_TOP_K1, &d};
            std::string note;
            log_probe("topk1_router", probe_op(env.dml.Get(), od, note), note);
        }
        {
            TensorInfo i, o;
            make_tensor(i, {1, 32}, DML_TENSOR_DATA_TYPE_FLOAT32);
            make_tensor(o, {1, 32}, DML_TENSOR_DATA_TYPE_FLOAT32);
            DML_CUMULATIVE_SUMMATION_OPERATOR_DESC d{i.desc_ptr(), o.desc_ptr(), 1,
                                                     DML_AXIS_DIRECTION_INCREASING, FALSE};
            DML_OPERATOR_DESC od{DML_OPERATOR_CUMULATIVE_SUMMATION, &d};
            std::string note;
            log_probe("cumsum", probe_op(env.dml.Get(), od, note), note);
        }
        {
            TensorInfo i, x, o;
            make_tensor(i, {8, 4}, DML_TENSOR_DATA_TYPE_FLOAT32);
            make_tensor(x, {4, 4}, DML_TENSOR_DATA_TYPE_UINT32);
            make_tensor(o, {4, 4}, DML_TENSOR_DATA_TYPE_FLOAT32);
            DML_GATHER_ELEMENTS_OPERATOR_DESC d{i.desc_ptr(), x.desc_ptr(), o.desc_ptr(), 1};
            DML_OPERATOR_DESC od{DML_OPERATOR_GATHER_ELEMENTS, &d};
            std::string note;
            log_probe("gather_elements", probe_op(env.dml.Get(), od, note), note);
        }
        {
            TensorInfo i, o;
            make_tensor(i, {1, 16}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(o, {1, 16}, DML_TENSOR_DATA_TYPE_FLOAT16);
            DML_ACTIVATION_SWISH_OPERATOR_DESC d{i.desc_ptr(), o.desc_ptr(), 1.0f};
            DML_OPERATOR_DESC od{DML_OPERATOR_ACTIVATION_SWISH, &d};
            std::string note;
            log_probe("swish_f16", probe_op(env.dml.Get(), od, note), note);
        }
        {
            TensorInfo a, b, o;
            make_tensor(a, {1, 1, 64, 64}, DML_TENSOR_DATA_TYPE_INT8);
            make_tensor(b, {1, 1, 64, 64}, DML_TENSOR_DATA_TYPE_INT8);
            make_tensor(o, {1, 1, 64, 64}, DML_TENSOR_DATA_TYPE_INT32);
            DML_MATRIX_MULTIPLY_INTEGER_OPERATOR_DESC d{a.desc_ptr(), nullptr, b.desc_ptr(),
                                                        nullptr, o.desc_ptr()};
            DML_OPERATOR_DESC od{DML_OPERATOR_MATRIX_MULTIPLY_INTEGER, &d};
            std::string note;
            log_probe("matmul_int8", probe_op(env.dml.Get(), od, note), note);
        }
        {
            TensorInfo q, k, v, o;
            make_tensor(q, {1, 64, 16, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(k, {1, 8, 16, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);   // GQA: 8 KV heads
            make_tensor(v, {1, 8, 16, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(o, {1, 64, 16, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC d{};
            d.QueryTensor = q.desc_ptr();
            d.KeyTensor = k.desc_ptr();
            d.ValueTensor = v.desc_ptr();
            d.OutputTensor = o.desc_ptr();
            d.Scale = 0.125f;
            d.MaskFilterValue = -10000.0f;
            d.QueryHeadCount = 64;
            d.KeyValueHeadCount = 8;
            d.MaskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
            DML_OPERATOR_DESC od{DML_OPERATOR_MULTIHEAD_ATTENTION1, &d};
            std::string note;
            log_probe("mha1_gqa_f16", probe_op(env.dml.Get(), od, note), note);
        }
        {
            TensorInfo p, m1, m2, g, t, op, om1, om2;
            make_tensor(p, {1, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(m1, {1, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(m2, {1, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(g, {1, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(t, {1, 1}, DML_TENSOR_DATA_TYPE_FLOAT32);
            make_tensor(op, {1, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(om1, {1, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            make_tensor(om2, {1, 64}, DML_TENSOR_DATA_TYPE_FLOAT16);
            DML_ADAM_OPTIMIZER_OPERATOR_DESC d{p.desc_ptr(), m1.desc_ptr(), m2.desc_ptr(),
                                               g.desc_ptr(), t.desc_ptr(), op.desc_ptr(),
                                               om1.desc_ptr(), om2.desc_ptr(),
                                               1e-3f, 0.9f, 0.999f, 1e-8f};
            DML_OPERATOR_DESC od{DML_OPERATOR_ADAM_OPTIMIZER, &d};
            std::string note;
            log_probe("adam_f16", probe_op(env.dml.Get(), od, note), note);
        }

        // S1: execution-path sanity (identity + softmax), then GEMM correctness + perf
        run_identity(env, run);
        if (skip_gemm) {
            run_softmax(env, run);
            run.event("done", "\"status\":\"no-gemm\"");
            return 0;
        }
        run_softmax(env, run);
        if (f32check) {
            auto c32 = run_gemm(env, ckn, ckn, ckn, 1, true, DML_TENSOR_DATA_TYPE_FLOAT32, !notrans);
            run.event("gemm_correct_f32",
                      "\"m\":" + std::to_string(ckn) + ",\"trans_b\":" + (notrans ? "false" : "true") +
                          ",\"max_abs_diff\":" + std::to_string(c32.max_abs_diff) +
                          ",\"mean_abs_diff\":" + std::to_string(c32.mean_abs_diff));
            return 0;
        }
        auto cf16 = run_gemm(env, ckn, ckn, ckn, 1, true, DML_TENSOR_DATA_TYPE_FLOAT16, !notrans,
                             !single);
        run.event("gemm_correct_f16",
                  "\"m\":" + std::to_string(ckn) + ",\"trans_b\":" + (notrans ? "false" : "true") +
                      ",\"graph\":" + (single ? "false" : "true") + ",\"max_abs_diff\":" +
                      std::to_string(cf16.max_abs_diff) + ",\"mean_abs_diff\":" +
                      std::to_string(cf16.mean_abs_diff));
        auto pf16 = run_gemm(env, M, N, K, iters, false, DML_TENSOR_DATA_TYPE_FLOAT16);
        run.event("gemm_perf_f16",
                  "\"m\":" + std::to_string(M) + ",\"iters\":" + std::to_string(iters) +
                      ",\"ms\":" + std::to_string(pf16.ms) + ",\"tflops\":" +
                      std::to_string(pf16.tflops));
        auto pf32 = run_gemm(env, ckn, ckn, ckn, iters, false, DML_TENSOR_DATA_TYPE_FLOAT32);
        run.event("gemm_perf_f32",
                  "\"m\":" + std::to_string(ckn) + ",\"ms\":" + std::to_string(pf32.ms) +
                      ",\"tflops\":" + std::to_string(pf32.tflops));

        run.event("done", "\"status\":\"ok\"");
        std::printf("\nS1/S2 complete — events in %s\n", run.dir().string().c_str());
        return 0;
    } catch (const std::exception& e) {
        run.event("fatal", std::string("\"error\":\"") + e.what() + "\"");
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        env.dump_info_queue("fatal");
        return 1;
    }
}

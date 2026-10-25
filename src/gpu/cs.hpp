/** \file
 *  \brief salikorn / gpu — minimal Direct3D 12 compute backend (raw compute shaders).
 *
 * Proven on the Intel UHD (spike_d3d12): runtime HLSL compile via d3dcompiler_47 (SM5.1),
 * root-descriptor binding, explicit transitions. Constraint: keep a single dispatch under
 * ~2 s (TDR — Timeout Detection & Recovery kills the device if a GPU call stalls longer);
 * use the chunked helpers.
 *
 * Declarations only; implementations live in cs.cpp (nothing here is a template).
 */
#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdint>
#include <map>
#include <string>

namespace sal::gpu {

using Microsoft::WRL::ComPtr;

/// \brief Throws std::runtime_error("<what> failed: 0x........") when `hr` is a
///        failure code; returns without effect on success.
void check(HRESULT hr, const char* what);

/** \brief Owning D3D12 device context: one adapter/device plus a single reusable
 *         compute command queue, allocator, command list and fence.
 *
 * Resource-state discipline: the Device does no state tracking. transition()
 * records exactly the StateBefore/StateAfter pair the caller passes, so the
 * caller must know each buffer's current state. The convention throughout this
 * backend (Buffer, Kernel::dispatch, kernels.hpp kernel sets) is:
 * - buffers rest in D3D12_RESOURCE_STATE_COMMON between operations;
 * - each op transitions its inputs COMMON -> NON_PIXEL_SHADER_RESOURCE and its
 *   output COMMON -> UNORDERED_ACCESS before Dispatch, and back after;
 * - when one resource occupies two slots of a dispatch (e.g. in-place
 *   elementwise), only one transition is recorded per edge.
 *
 * Command recording model: one command allocator / one command list, reused
 * round-robin via begin() ... submit_and_wait(); there is never more than one
 * batch in flight (submit_and_wait blocks until the GPU fence is signalled).
 */
class Device {
  public:
    /** \brief Create the factory, pick an adapter, create the device, and set up
     *         the compute queue / allocator / (closed) command list / fence.
     * \param force_warp  when true, use the Microsoft WARP software adapter
     *                    directly instead of enumerating hardware.
     *
     * Adapter selection: enumerate adapters, take the first whose
     * DXGI_ADAPTER_FLAG_SOFTWARE flag is clear; if none exists, fall back to
     * init(true) (WARP). Device creation tries feature level 12.0 and falls
     * back to 11.0. The command list is created in the recording state and
     * immediately Closed, ready for begin(). Throws std::runtime_error on any
     * failed D3D call (via check()).
     */
    void init(bool force_warp = false);

    /// \brief Non-owning pointer to the underlying ID3D12Device (null before init()).
    ID3D12Device* dev();
    /// \brief True when the selected adapter is WARP (software rendering).
    bool warp() const;
    /// \brief Description string of the selected adapter (from DXGI_ADAPTER_DESC1).
    std::string adapter_name() const;

    /// \brief Reset the command allocator and command list to start recording a
    ///        new batch. The previous batch must already have been submitted.
    void begin();
    /** \brief Close the command list, execute it on the compute queue and block
     *         until completion.
     *
     * Signals the fence, waits on the fence event (hard 120 s timeout, beyond
     * any single dispatch's ~2 s TDR budget), then checks
     * GetDeviceRemovedReason() and throws if the device was removed mid-batch.
     */
    void submit_and_wait();
    /** \brief Record a transition resource barrier over all subresources.
     *
     * \param r       resource to transition (must not be null).
     * \param before  state the resource is currently in (caller-tracked).
     * \param after   state to move it into.
     * \note No validation: passing a `before` that does not match reality makes
     *       the debug layer complain and can produce undefined behavior.
     */
    void transition(ID3D12Resource* r, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
    /// \brief Record a UAV barrier on `r` (nullptr barriers flush everything);
    ///        required between two dispatches when one writes what the next reads
    ///        via the UAV.
    void uav_barrier(ID3D12Resource* r);
    /// \brief The open/closed command list used by begin()/submit_and_wait(); also
    ///        exposed for callers recording extra commands (e.g. CopyBufferRegion).
    ID3D12GraphicsCommandList* list();

  private:
    friend class Buffer;
    ComPtr<IDXGIFactory6> factory_;
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<ID3D12Device> dev_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> alloc_;
    ComPtr<ID3D12GraphicsCommandList> list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_ = nullptr;
    uint64_t fence_value_ = 0;
    DXGI_ADAPTER_DESC1 adesc_{};
    bool warp_ = false;
};

/// \brief Heap/state profile of a Buffer.
enum class BufferKind {
    SrcRead,   ///< default heap, bound as SRV (t*) only; rests in COMMON
    Uav,       ///< default heap with D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; bound as u*
    CpuWrite,  ///< upload heap (GENERIC_READ); mappable for direct CPU writes
    Readback   ///< readback heap (COPY_DEST); mappable for CPU reads of GPU results
};

/** \brief Committed D3D12 buffer of a fixed size, classified by BufferKind.
 *
 * \note Only `Uav` buffers carry D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, so
 *       only they can be bound at the u0 slot of a Kernel; SRV binding works for
 *       default-heap buffers regardless of the flag. Upload/download of
 *       non-CpuWrite/Readback buffers goes through a temporary staging buffer
 *       and a synchronous device copy (see upload()/download()).
 */
class Buffer {
  public:
    /// \brief Create a committed resource on `d`'s device: heap type and initial
    ///        state are chosen from `kind` (see BufferKind); size in bytes.
    Buffer(Device& d, uint64_t size, BufferKind kind);
    /// \brief Buffer size in bytes as passed to the constructor.
    uint64_t size() const;
    /// \brief Underlying ID3D12Resource (for transitions / copy commands).
    ID3D12Resource* res();
    /// \brief GPU virtual address used for root-descriptor (SRV/UAV) binding.
    D3D12_GPU_VIRTUAL_ADDRESS gpu_addr();

    /** \brief Copy `bytes` from CPU memory into the buffer at `dst_offset`.
     *
     * CpuWrite buffers: mapped write, no GPU involvement. Everything else:
     * creates a temporary CpuWrite staging buffer and issues a CopyBufferRegion
     * on the device queue (COMMON -> COPY_DEST -> COMMON transitions, then a
     * synchronous submit_and_wait()).
     */
    void upload(const void* data, uint64_t bytes, uint64_t dst_offset = 0);
    /** \brief Copy `bytes` from the buffer at `src_offset` into CPU memory.
     *
     * Always goes through a temporary Readback staging buffer and a synchronous
     * device copy (COMMON -> COPY_SOURCE -> COMMON transitions).
     */
    void download(void* data, uint64_t bytes, uint64_t src_offset = 0);
    /// \brief Map this (Readback) buffer and memcpy `bytes` into `data`.
    ///        Only meaningful on Readback/CpuWrite heaps.
    void map_read(void* data, uint64_t bytes);

  private:
    Device* dev_;
    ComPtr<ID3D12Resource> res_;
    uint64_t size_;
    BufferKind kind_;
};

/** \brief Replace `&#64;NAME&#64;` tokens in a shader template from the defines map.
 *
 * \param tmpl     raw template text containing `&#64;TOKEN&#64;` placeholders.
 * \param defines  token name (without '@') -> replacement text, applied in map
 *                 (alphabetical) order.
 * \return Fully substituted source.
 * \throws std::runtime_error if any '@' remains after substitution — a hard
 *         error that catches typos in both templates and define maps.
 */
std::string substitute_tokens(const std::string& tmpl, const std::map<std::string, std::string>& defines);

/** \brief A compute shader: HLSL compiled at runtime into a root signature + PSO pair.
 *
 * The root signature is fixed for all kernels:
 * `[0]=SRV t0, [1]=SRV t1, [2]=UAV u0, [3]=32-bit constants b0 (max 8 words)`.
 * Shaders see the constants as cbuffer KC with `uint4 p` (words 0-3) and
 * `uint4 q` (words 4-7); per-kernel meanings are documented next to each
 * template in kernels.inl.
 *
 * Compilation loads d3dcompiler_47.dll dynamically and calls its D3DCompile
 * (target cs_5_1 by default); compile errors surface the compiler's own
 * diagnostic text in the thrown std::runtime_error.
 */
class Kernel {
  public:
    /// \brief Compile plain HLSL (no token substitution).
    Kernel(Device& d, const char* hlsl, const char* entry = "cs_main",
           const char* target = "cs_5_1");

    /// \brief Compile a template: &#64;TOKEN&#64; occurrences are replaced from `defines`
    ///        (via substitute_tokens) before compilation; a leftover '@' after
    ///        substitution is a hard error (catches typos in templates).
    Kernel(Device& d, const char* hlsl_template,
           const std::map<std::string, std::string>& defines, const char* entry = "cs_main",
           const char* target = "cs_5_1");

    /** \brief Record one dispatch of this kernel with the given bindings.
     *
     * Binds a at t0, b at t1, out at u0 (root descriptors, i.e. raw GPU
     * addresses) and uploads `consts` as the eight 32-bit root constants, then
     * Dispatch(gx, gy, gz) thread groups.
     *
     * \param d       device whose command list the dispatch is recorded into.
     * \param a       buffer bound as SRV t0.
     * \param b       buffer bound as SRV t1 (may alias `a`).
     * \param out     buffer bound as UAV u0 (may alias an input for in-place ops).
     * \param consts  eight 32-bit words visible in-shader as KC.p (0-3) / KC.q (4-7).
     * \param gx,gy,gz  dispatch grid in thread groups (not threads).
     *
     * \note Caller responsibilities: correct resource states (see Device's
     *       state discipline), a uav_barrier() between two dispatches where one
     *       writes what the next reads, and keeping each dispatch chunk under
     *       ~2 s so the TDR never fires.
     */
    void dispatch(Device& d, Buffer& a, Buffer& b, Buffer& out, const uint32_t consts[8],
                  uint32_t gx, uint32_t gy, uint32_t gz = 1);

  private:
    ComPtr<ID3D12RootSignature> rootsig_;
    ComPtr<ID3D12PipelineState> pso_;
};

}  // namespace sal::gpu

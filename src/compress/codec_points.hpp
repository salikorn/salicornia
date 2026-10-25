/**
 * \file codec_points.hpp
 * \brief Bake-off result parsing shared by spikes, tests, and tools.
 *
 * Declares parse_codec_points() and its CodecPoint record; the implementation
 * lives in codec_points.cpp.
 */
#pragma once

#include <string>
#include <vector>

namespace sal {

/// \brief One (codec, bits-per-weight, relative-MSE) measurement extracted from
///        a GPU codec bake-off run.
struct CodecPoint {
    std::string codec;    ///< Codec identifier, e.g. "ae_L64".
    double bpw = 0.0;     ///< Bits per weight reported by the run.
    double rel_mse = 0.0; ///< Relative MSE; may be NaN/Inf (see parse_codec_points).
};

/// \brief Parses spike_vqvae --run output lines of the form
///        `[gpu_codec_point] "codec":"ae_L64","bpw":4.117,"rel_mse":7.48e-01,...`
///
/// Scans \p out for every occurrence of the `[gpu_codec_point]` tag and pulls
/// the "codec", "bpw", and "rel_mse" fields via std::stod.
///
/// \param out Raw stdout text of a spike run (may contain any other lines).
/// \return One CodecPoint per matched tag, in order of appearance; empty when
///         no tag (or a tag missing its required fields) is found. Parsing
///         stops at the first malformed record.
///
/// \note NaN/Inf rel_mse values are preserved as non-finite doubles — callers
///       should check std::isfinite before using the value.
std::vector<CodecPoint> parse_codec_points(const std::string& out);

}  // namespace sal

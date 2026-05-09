// seeker_v2 native nvjpeg hardware JPEG encoder
// =====================================================================
// Wraps NvJpegEncoder from the Jetson multimedia API. Encodes BGR/YUV
// frames using the hardware JPEG block (NVENC/VIC pipeline) instead of
// libjpeg-turbo on the CPU.
//
// On Xavier the hardware encoder is roughly 4-6× faster than turbo for
// 2472×2064 frames at quality 75 (∼3 ms vs ∼17 ms), and crucially it
// frees a CPU core that the AGC stretch + V4L2 ioctl loop are fighting
// over.
//
// Build is gated on the Jetson multimedia API headers being present
// (see CMakeLists.txt). On x86/Windows dev hosts the module is simply
// not built; Python detects the missing import and falls back to
// cv2.imencode().
//
// Python API:
//   import seeker_v2.native.seeker_nvjpeg as nj
//   enc = nj.JpegEncoder(quality=75)
//   buf = enc.encode_bgr(bgr_uint8_array)   # bytes
//
// Phase 2.3 of the v2 rewrite.
// =====================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

// Jetson multimedia API. Header only present on JetPack hosts; the
// CMake guard prevents this TU from being compiled elsewhere.
#include "NvJpegEncoder.h"

namespace py = pybind11;

namespace {

// Convert a packed BGR buffer (H, W, 3) into a planar I420 / YUV420
// buffer in-place into `dst`, sized W*H*3/2.
//
// NvJpegEncoder::encodeFromBuffer accepts I420 directly. We do the
// conversion in plain C — it's ~2 ms at 2472×2064, dwarfed by the
// hardware encode itself. (A future optimization could push this onto
// the VIC via NvBufSurfTransform, but it's not currently the
// bottleneck.)
//
// BT.601 limited-range coefficients to match cv2.cvtColor's default.
inline void bgr_to_i420(const uint8_t* bgr, int w, int h, uint8_t* dst) {
    uint8_t* y_plane  = dst;
    uint8_t* u_plane  = dst + w * h;
    uint8_t* v_plane  = u_plane + (w * h) / 4;
    const int uv_w = w / 2;

    for (int j = 0; j < h; ++j) {
        const uint8_t* row = bgr + j * w * 3;
        uint8_t* y_row = y_plane + j * w;
        for (int i = 0; i < w; ++i) {
            const uint8_t b = row[3*i + 0];
            const uint8_t g = row[3*i + 1];
            const uint8_t r = row[3*i + 2];
            // Y = 0.299 R + 0.587 G + 0.114 B  (BT.601)
            int y = ( 66*r + 129*g +  25*b + 128) >> 8;
            y_row[i] = static_cast<uint8_t>(y + 16);
        }
    }
    // Subsample chroma 2×2: average each 2×2 block.
    for (int j = 0; j < h; j += 2) {
        const uint8_t* row0 = bgr + (j    ) * w * 3;
        const uint8_t* row1 = bgr + (j + 1) * w * 3;
        uint8_t* u_row = u_plane + (j / 2) * uv_w;
        uint8_t* v_row = v_plane + (j / 2) * uv_w;
        for (int i = 0; i < w; i += 2) {
            int b = (row0[3*i+0] + row0[3*(i+1)+0] +
                     row1[3*i+0] + row1[3*(i+1)+0]) >> 2;
            int g = (row0[3*i+1] + row0[3*(i+1)+1] +
                     row1[3*i+1] + row1[3*(i+1)+1]) >> 2;
            int r = (row0[3*i+2] + row0[3*(i+1)+2] +
                     row1[3*i+2] + row1[3*(i+1)+2]) >> 2;
            int u = ( -38*r -  74*g + 112*b + 128) >> 8;
            int v = ( 112*r -  94*g -  18*b + 128) >> 8;
            u_row[i/2] = static_cast<uint8_t>(u + 128);
            v_row[i/2] = static_cast<uint8_t>(v + 128);
        }
    }
}

class JpegEncoder {
public:
    JpegEncoder(int quality, const std::string& name)
        : quality_(quality), name_(name)
    {
        if (quality_ < 1 || quality_ > 100) {
            throw std::invalid_argument("quality must be 1..100");
        }
        encoder_ = NvJPEGEncoder::createJPEGEncoder(name_.c_str());
        if (!encoder_) {
            throw std::runtime_error("NvJPEGEncoder::createJPEGEncoder failed");
        }
    }

    ~JpegEncoder() {
        if (encoder_) {
            delete encoder_;
            encoder_ = nullptr;
        }
    }

    JpegEncoder(const JpegEncoder&)            = delete;
    JpegEncoder& operator=(const JpegEncoder&) = delete;

    // Encode BGR (H, W, 3) uint8 → JPEG bytes.
    py::bytes encode_bgr(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img) {
        if (img.ndim() != 3 || img.shape(2) != 3) {
            throw std::invalid_argument("expected (H, W, 3) BGR uint8 array");
        }
        const int h = static_cast<int>(img.shape(0));
        const int w = static_cast<int>(img.shape(1));
        if ((w & 1) || (h & 1)) {
            throw std::invalid_argument("nvjpeg requires even W and H");
        }

        const size_t i420_bytes = static_cast<size_t>(w) * h * 3 / 2;
        if (i420_buf_.size() < i420_bytes) {
            i420_buf_.resize(i420_bytes);
        }
        // Output buffer: max possible JPEG ≤ 2 × source bytes is more
        // than enough for quality 75; nvjpeg writes the actual size back.
        const size_t max_jpeg = static_cast<size_t>(w) * h * 2 + 1024;
        if (jpeg_buf_.size() < max_jpeg) {
            jpeg_buf_.resize(max_jpeg);
        }

        const uint8_t* bgr_ptr = img.data();
        uint8_t* i420_ptr = i420_buf_.data();
        uint8_t* jpeg_ptr = jpeg_buf_.data();
        unsigned long jpeg_size = max_jpeg;

        {
            // Heavy work: release the GIL so other Python threads in
            // this process (the publisher loop, stats emit) can run.
            py::gil_scoped_release no_gil;

            bgr_to_i420(bgr_ptr, w, h, i420_ptr);

            std::lock_guard<std::mutex> lock(mu_);

            // The Jetson multimedia API exposes encodeFromBuffer that
            // takes a NvBuffer; the simpler encodeFromFd path needs a
            // dmabuf. Use the in-memory variant.
            int rc = encoder_->encodeFromBuffer(
                i420_ptr,
                JCS_YCbCr,           // input colorspace = YUV
                w, h,
                quality_,
                jpeg_ptr,
                jpeg_size
            );
            if (rc < 0) {
                // Restore GIL just to throw.
                py::gil_scoped_acquire reacquire;
                throw std::runtime_error("NvJPEGEncoder::encodeFromBuffer failed");
            }
        }
        return py::bytes(reinterpret_cast<const char*>(jpeg_ptr), jpeg_size);
    }

    // Encode a pre-built I420 buffer (H, W*3/2). Useful when the caller
    // already has YUV (e.g. straight from a UYVY camera).
    py::bytes encode_i420(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
                          int w, int h) {
        const size_t expected = static_cast<size_t>(w) * h * 3 / 2;
        if (static_cast<size_t>(img.size()) != expected) {
            throw std::invalid_argument("I420 buffer size mismatch");
        }
        const size_t max_jpeg = static_cast<size_t>(w) * h * 2 + 1024;
        if (jpeg_buf_.size() < max_jpeg) {
            jpeg_buf_.resize(max_jpeg);
        }
        unsigned long jpeg_size = max_jpeg;
        const uint8_t* i420_ptr = img.data();
        uint8_t* jpeg_ptr = jpeg_buf_.data();

        {
            py::gil_scoped_release no_gil;
            std::lock_guard<std::mutex> lock(mu_);
            int rc = encoder_->encodeFromBuffer(
                const_cast<uint8_t*>(i420_ptr),
                JCS_YCbCr,
                w, h,
                quality_,
                jpeg_ptr,
                jpeg_size
            );
            if (rc < 0) {
                py::gil_scoped_acquire reacquire;
                throw std::runtime_error("NvJPEGEncoder::encodeFromBuffer failed");
            }
        }
        return py::bytes(reinterpret_cast<const char*>(jpeg_ptr), jpeg_size);
    }

    void set_quality(int q) {
        if (q < 1 || q > 100) throw std::invalid_argument("quality 1..100");
        quality_ = q;
    }
    int get_quality() const { return quality_; }
    const std::string& name() const { return name_; }

private:
    NvJPEGEncoder* encoder_{nullptr};
    int quality_;
    std::string name_;
    std::vector<uint8_t> i420_buf_;
    std::vector<uint8_t> jpeg_buf_;
    std::mutex mu_;
};

} // anonymous namespace

PYBIND11_MODULE(seeker_nvjpeg, m) {
    m.doc() = "seeker_v2: hardware JPEG encoder (Jetson nvjpeg)";

    py::class_<JpegEncoder>(m, "JpegEncoder")
        .def(py::init<int, const std::string&>(),
             py::arg("quality") = 75,
             py::arg("name")    = "seeker_v2_jpeg")
        .def("encode_bgr",  &JpegEncoder::encode_bgr,  py::arg("bgr"))
        .def("encode_i420", &JpegEncoder::encode_i420,
             py::arg("i420"), py::arg("width"), py::arg("height"))
        .def_property("quality",
                      &JpegEncoder::get_quality,
                      &JpegEncoder::set_quality)
        .def_property_readonly("name", &JpegEncoder::name);
}

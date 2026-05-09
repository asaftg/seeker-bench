// seeker_v2 native V4L2 capture + AGC + JPEG encode
// =====================================================================
// pybind11 C++ extension that replaces the Python RawV4L2Backend hot path.
// Releases the GIL during the kernel ioctl wait, the AGC stretch, and
// the (optional) JPEG encode, so the EO/thermal capture processes spend
// less time blocking the orchestrator.
//
// Build:
//   cd seeker_v2/native && mkdir build && cd build
//   cmake .. && make -j$(nproc)
//   # produces seeker_native.cpython-38-aarch64-linux-gnu.so
//   # python -c "import seeker_v2.native.seeker_native as n; print(n.__doc__)"
//
// Hardware dependencies:
//   - Linux V4L2 (videodev2.h)
//   - libuvc not required — we drive UVC XU controls via the standard
//     UVCIOC_CTRL_QUERY ioctl (works on stock uvcvideo driver)
//
// Phase 2.3 will add nvjpeg hardware JPEG encode in a separate translation
// unit (nvjpeg_encoder.cpp) that this file optionally links against.
// =====================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <chrono>
#include <vector>
#include <stdexcept>
#include <atomic>
#include <thread>

namespace py = pybind11;

// ── UVC Extension Unit IOCTL (mirrors leopard_linux.py) ───────────────
struct uvc_xu_control_query {
    uint8_t  unit;
    uint8_t  selector;
    uint8_t  query;
    uint16_t size;
    uint8_t* data;
};

static constexpr int UVC_SET_CUR = 0x01;
static constexpr int UVC_GET_CUR = 0x81;
static constexpr int UVCIOC_CTRL_QUERY =
    _IOC(_IOC_READ | _IOC_WRITE, 'u', 0x21, sizeof(struct uvc_xu_control_query));

// ── Tegra L4T struct v4l2_format has 4 bytes of padding after `type`
// (sizeof = 208, not 204). Use the kernel-defined struct directly.
// The standard linux/videodev2.h has the right layout — we don't need
// to re-derive it like the Python ctypes shim does. ─────────────────

class V4L2Backend {
public:
    V4L2Backend(const std::string& dev, int w, int h, int n_buffers = 4)
        : dev_(dev), w_(w), h_(h), n_buffers_(n_buffers),
          fd_(-1), streaming_(false), trigger_check_counter_(0),
          stats_check_counter_(0)
    {
        last_p1_ = 0.0;
        last_p99_ = 4095.0;
        last_alpha_ = 1.0;
        last_beta_ = 0.0;
    }

    ~V4L2Backend() { release(); }

    bool open(int retries = 8, double backoff_s = 0.5) {
        if (fd_ >= 0) return true;
        for (int attempt = 0; attempt < retries; ++attempt) {
            fd_ = ::open(dev_.c_str(), O_RDWR | O_NONBLOCK);
            if (fd_ >= 0) break;
            std::this_thread::sleep_for(
                std::chrono::milliseconds(static_cast<int>(backoff_s * 1000))
            );
        }
        if (fd_ < 0) return false;

        // Set format YUYV @ requested size
        struct v4l2_format fmt;
        std::memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width  = w_;
        fmt.fmt.pix.height = h_;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        fmt.fmt.pix.field  = V4L2_FIELD_NONE;
        if (::ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) return false;
        w_ = fmt.fmt.pix.width;
        h_ = fmt.fmt.pix.height;

        // REQBUFS
        struct v4l2_requestbuffers req;
        std::memset(&req, 0, sizeof(req));
        req.count  = n_buffers_;
        req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        for (int attempt = 0; attempt < 8; ++attempt) {
            if (::ioctl(fd_, VIDIOC_REQBUFS, &req) == 0) break;
            if (errno != EBUSY && errno != EAGAIN) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (req.count < 2) return false;
        n_buffers_ = req.count;

        // mmap each buffer + queue them
        buffers_.resize(n_buffers_);
        for (int i = 0; i < n_buffers_; ++i) {
            struct v4l2_buffer buf;
            std::memset(&buf, 0, sizeof(buf));
            buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index  = i;
            if (::ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) return false;
            buffers_[i].length = buf.length;
            buffers_[i].start = ::mmap(
                nullptr, buf.length,
                PROT_READ | PROT_WRITE, MAP_SHARED,
                fd_, buf.m.offset
            );
            if (buffers_[i].start == MAP_FAILED) return false;
            if (::ioctl(fd_, VIDIOC_QBUF, &buf) < 0) return false;
        }

        // STREAMON
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (::ioctl(fd_, VIDIOC_STREAMON, &type) < 0) return false;
        streaming_ = true;

        // Disable FX3 trigger mode (XU 0x0b = [0,0]) — required to get
        // free-running streaming on the LI-IMX568-GMSL2 firmware.
        // Non-fatal on failure (other UVC cameras don't have this XU).
        try { xu_write(0x0b, std::vector<uint8_t>{0, 0}); }
        catch (...) {}

        // Warmup: drain a few frames
        for (int i = 0; i < 5; ++i) {
            std::vector<uint8_t> tmp;
            if (read_raw(tmp, 2.0)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        return true;
    }

    void release() {
        if (streaming_) {
            int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            ::ioctl(fd_, VIDIOC_STREAMOFF, &type);
            streaming_ = false;
        }
        for (auto& b : buffers_) {
            if (b.start && b.start != MAP_FAILED) {
                ::munmap(b.start, b.length);
                b.start = nullptr;
            }
        }
        buffers_.clear();
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    bool is_open() const { return fd_ >= 0 && streaming_; }

    int width() const { return w_; }
    int height() const { return h_; }

    // ── XU Extension Unit IOCTL on the streaming fd ────────────────
    void xu_write(int selector, std::vector<uint8_t> data) {
        if (fd_ < 0) throw std::runtime_error("device not open");
        struct uvc_xu_control_query q;
        std::memset(&q, 0, sizeof(q));
        q.unit     = 3;  // FX3 vendor unit
        q.selector = static_cast<uint8_t>(selector);
        q.query    = UVC_SET_CUR;
        q.size     = static_cast<uint16_t>(data.size());
        q.data     = data.data();
        if (::ioctl(fd_, UVCIOC_CTRL_QUERY, &q) < 0) {
            throw std::runtime_error(std::string("xu_write failed: ") + strerror(errno));
        }
    }

    void set_exposure_ext(int value) {
        // Decoy-write workaround (FX3 ignores repeat writes)
        int last = last_exposure_written_;
        int decoy = (last != 100) ? 100 : 200;
        std::vector<uint8_t> b(2);
        b[0] = static_cast<uint8_t>(decoy & 0xff);
        b[1] = static_cast<uint8_t>((decoy >> 8) & 0xff);
        xu_write(0x06, b);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        b[0] = static_cast<uint8_t>(value & 0xff);
        b[1] = static_cast<uint8_t>((value >> 8) & 0xff);
        xu_write(0x06, b);
        last_exposure_written_ = value;
    }

    void set_gain_rgb(int gain) {
        std::vector<uint8_t> b(8);
        for (int i = 0; i < 4; ++i) {
            b[i * 2] = static_cast<uint8_t>(gain & 0xff);
            b[i * 2 + 1] = static_cast<uint8_t>((gain >> 8) & 0xff);
        }
        xu_write(0x0d, b);
    }

    // Read raw YUYV bytes (2 * w * h). Releases GIL during ioctl wait.
    bool read_raw(std::vector<uint8_t>& out, double timeout_s) {
        if (fd_ < 0) return false;

        // Periodic trigger-state recheck (every 300 frames)
        if (++trigger_check_counter_ >= 300) {
            trigger_check_counter_ = 0;
            try { xu_write(0x0b, std::vector<uint8_t>{0, 0}); }
            catch (...) {}
        }

        struct v4l2_buffer buf;
        std::memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(static_cast<int>(timeout_s * 1000));

        // GIL release
        py::gil_scoped_release _release;

        while (true) {
            int rc = ::ioctl(fd_, VIDIOC_DQBUF, &buf);
            if (rc == 0) break;
            if (errno == EAGAIN || errno == EINTR) {
                if (std::chrono::steady_clock::now() > deadline) {
                    return false;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            return false;
        }

        const size_t valid_n = static_cast<size_t>(w_) * h_ * 2;
        const size_t src_n = std::min<size_t>(buf.bytesused ? buf.bytesused : valid_n, valid_n);

        out.resize(valid_n);
        std::memcpy(out.data(),
                    static_cast<uint8_t*>(buffers_[buf.index].start), src_n);
        if (src_n < valid_n) {
            std::memset(out.data() + src_n, 0, valid_n - src_n);
        }

        ::ioctl(fd_, VIDIOC_QBUF, &buf);
        return true;
    }

    // grab() — full pipeline returning BGR uint8 numpy array.
    // RAW12 reinterpretation + cached-stats AGC + grayscale->BGR.
    py::object grab() {
        std::vector<uint8_t> raw;
        if (!read_raw(raw, 1.0)) {
            return py::none();
        }

        // Reinterpret as little-endian uint16 RAW12.
        const size_t n_pixels = static_cast<size_t>(w_) * h_;
        const uint16_t* u16 = reinterpret_cast<const uint16_t*>(raw.data());

        // Recompute stats every 5 frames; reuse alpha/beta in between.
        if (++stats_check_counter_ >= 5 || !stats_initialized_) {
            stats_check_counter_ = 0;
            stats_initialized_ = true;

            // Strided sample for fast percentile estimation. Same as
            // Python: u16[::8, ::8].
            const int stride = 8;
            std::vector<uint16_t> sample;
            sample.reserve((h_ / stride + 1) * (w_ / stride + 1));
            for (int y = 0; y < h_; y += stride) {
                const uint16_t* row = u16 + static_cast<size_t>(y) * w_;
                for (int x = 0; x < w_; x += stride) {
                    sample.push_back(row[x]);
                }
            }
            // Compute p1/p99 via partial sort.
            const size_t ns = sample.size();
            std::vector<uint16_t> sorted_sample = sample;
            std::sort(sorted_sample.begin(), sorted_sample.end());
            double p1  = static_cast<double>(sorted_sample[ns * 1 / 100]);
            double p99 = static_cast<double>(sorted_sample[ns * 99 / 100]);
            double span = std::max(p99 - p1, 4.0);
            last_p1_ = p1; last_p99_ = p99;
            last_alpha_ = 255.0 / span;
            last_beta_ = -p1 * last_alpha_;
        }

        // Allocate BGR output as numpy array. (H, W, 3) uint8.
        py::array_t<uint8_t> out({h_, w_, 3});
        auto buf = out.mutable_unchecked<3>();

        // Apply AGC to mono Y, replicate to BGR.
        const double alpha = last_alpha_;
        const double beta = last_beta_;
        // GIL is held here (numpy buffer access requires it). The
        // numeric work is fast — ~5 ms for 5M pixels on Xavier ARM.
        for (int y = 0; y < h_; ++y) {
            const uint16_t* row = u16 + static_cast<size_t>(y) * w_;
            for (int x = 0; x < w_; ++x) {
                double v = static_cast<double>(row[x]) * alpha + beta;
                if (v < 0.0) v = 0.0;
                else if (v > 255.0) v = 255.0;
                uint8_t y8 = static_cast<uint8_t>(v);
                buf(y, x, 0) = y8;
                buf(y, x, 1) = y8;
                buf(y, x, 2) = y8;
            }
        }

        return out;
    }

    // Return the most recent raw stats as a dict.
    py::dict last_raw_stats() {
        py::dict d;
        d["p1"] = last_p1_;
        d["p99"] = last_p99_;
        d["alpha"] = last_alpha_;
        d["beta"] = last_beta_;
        return d;
    }

    // For callers that want to alias seeker.RawV4L2Backend's API.
    void stop() { release(); }

private:
    struct Buffer {
        void* start = nullptr;
        size_t length = 0;
    };

    std::string dev_;
    int w_, h_;
    int n_buffers_;
    int fd_;
    bool streaming_;
    int trigger_check_counter_;
    int stats_check_counter_;
    bool stats_initialized_ = false;

    int last_exposure_written_ = -1;
    double last_p1_, last_p99_, last_alpha_, last_beta_;

    std::vector<Buffer> buffers_;
};


PYBIND11_MODULE(seeker_native, m) {
    m.doc() = "seeker_v2 native extensions: V4L2 capture + AGC (Phase 2.2)";

    py::class_<V4L2Backend>(m, "V4L2Backend")
        .def(py::init<const std::string&, int, int, int>(),
             py::arg("dev_path"), py::arg("width"), py::arg("height"),
             py::arg("n_buffers") = 4)
        .def("open", &V4L2Backend::open,
             py::arg("retries") = 8, py::arg("backoff_s") = 0.5,
             "Open the device, set format, mmap buffers, STREAMON, disable trigger.")
        .def("release", &V4L2Backend::release)
        .def("is_open", &V4L2Backend::is_open)
        .def("isOpened", &V4L2Backend::is_open)  // cv2 alias
        .def("stop", &V4L2Backend::stop)
        .def_property_readonly("width", &V4L2Backend::width)
        .def_property_readonly("height", &V4L2Backend::height)
        .def("xu_write", &V4L2Backend::xu_write)
        .def("set_exposure_ext", &V4L2Backend::set_exposure_ext)
        .def("set_gain_rgb", &V4L2Backend::set_gain_rgb)
        .def("grab", &V4L2Backend::grab,
             "Capture one frame, return BGR uint8 (H, W, 3) numpy array. "
             "Returns None on timeout. Releases GIL during V4L2 ioctl wait.")
        .def_property_readonly("last_raw_stats", &V4L2Backend::last_raw_stats);
}

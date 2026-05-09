// seeker_v2 native TensorRT runner with DLA core selection
// =====================================================================
// Phase 2.4 — offload the *thermal* H/V YOLO classifier to DLA0 so the
// EO classifier has the iGPU to itself. The Xavier AGX has 2 DLA cores
// (DLA0 / DLA1) plus the iGPU; running both YOLO classifiers on the
// iGPU costs ~30% per-call latency due to context-switch + L2 cache
// thrash. This eliminates that.
//
// What this does NOT do: it doesn't replace ultralytics. The Python
// inference process still drives the model — preprocess, NMS,
// postprocess, bytetrack — but the forward pass is executed via this
// TRT runner against an engine that was built with `--useDLACore=0`.
// See scripts/build_dla_engine.sh.
//
// Why C++? Because the TRT Python bindings on JP5.1 (TRT 8.5) don't
// expose `setDLACore()` on the deserialization path consistently, and
// even when they do, the CUDA→DLA scheduler call overhead from Python
// (∼1 ms per inference) eats most of the benefit. C++ launches the
// kernel with sub-100 µs overhead and the GIL is released for the
// duration of the synchronous wait.
//
// API:
//   import seeker_v2.native.seeker_dla as dla
//   r = dla.TrtRunner("models/seeker_thermal_hv_dla0.engine", dla_core=0)
//   r.input_shape         # (1, 3, 640, 640)
//   r.infer(chw_float32)  # returns {"output0": np.ndarray}
//
// Build is gated on TensorRT being present on the host (CMakeLists.txt).
// =====================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using namespace nvinfer1;

namespace {

class TrtLogger : public ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) {
            std::fprintf(stderr, "[trt] %s\n", msg);
        }
    }
};

static TrtLogger g_logger;

inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL:  return 1;
        case DataType::kUINT8: return 1;
        default: return 0;
    }
}

inline std::string dtype_name(DataType dt) {
    switch (dt) {
        case DataType::kFLOAT: return "float32";
        case DataType::kHALF:  return "float16";
        case DataType::kINT8:  return "int8";
        case DataType::kINT32: return "int32";
        case DataType::kBOOL:  return "bool";
        case DataType::kUINT8: return "uint8";
        default: return "unknown";
    }
}

inline size_t volume(const Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

class TrtRunner {
public:
    TrtRunner(const std::string& engine_path, int dla_core)
        : engine_path_(engine_path), dla_core_(dla_core)
    {
        // ── Load engine bytes ─────────────────────────────────────────
        std::ifstream f(engine_path_, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open engine: " + engine_path_);
        f.seekg(0, std::ios::end);
        size_t sz = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> buf(sz);
        f.read(buf.data(), sz);

        // ── Build runtime + select DLA ───────────────────────────────
        runtime_.reset(createInferRuntime(g_logger));
        if (!runtime_) throw std::runtime_error("createInferRuntime failed");

        if (dla_core_ >= 0) {
            int n = runtime_->getNbDLACores();
            if (dla_core_ >= n) {
                throw std::runtime_error("DLA core out of range");
            }
            runtime_->setDLACore(dla_core_);
        }

        engine_.reset(runtime_->deserializeCudaEngine(buf.data(), sz));
        if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");

        ctx_.reset(engine_->createExecutionContext());
        if (!ctx_) throw std::runtime_error("createExecutionContext failed");

        cudaStreamCreate(&stream_);

        // ── Walk bindings, allocate device buffers ───────────────────
        int nb = engine_->getNbIOTensors();
        for (int i = 0; i < nb; ++i) {
            const char* name = engine_->getIOTensorName(i);
            TensorIOMode mode = engine_->getTensorIOMode(name);
            DataType dt = engine_->getTensorDataType(name);
            Dims d = engine_->getTensorShape(name);
            size_t elt = dtype_size(dt);
            size_t bytes = volume(d) * elt;

            void* dptr = nullptr;
            cudaMalloc(&dptr, bytes);

            Binding b;
            b.name = name;
            b.is_input = (mode == TensorIOMode::kINPUT);
            b.dtype = dt;
            b.dims = d;
            b.bytes = bytes;
            b.dev_ptr = dptr;
            bindings_.push_back(b);

            ctx_->setTensorAddress(name, dptr);
        }
    }

    ~TrtRunner() {
        for (auto& b : bindings_) {
            if (b.dev_ptr) cudaFree(b.dev_ptr);
        }
        if (stream_) cudaStreamDestroy(stream_);
    }

    TrtRunner(const TrtRunner&)            = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;

    // Run inference with a single FP32 input named like the first input
    // tensor. Copies host→device, executes, copies back. Returns a dict
    // {output_name: numpy array} for every output binding.
    py::dict infer(py::array_t<float, py::array::c_style | py::array::forcecast> input) {
        // First input binding
        Binding* in_b = nullptr;
        for (auto& b : bindings_) {
            if (b.is_input) { in_b = &b; break; }
        }
        if (!in_b) throw std::runtime_error("engine has no input binding");

        if (static_cast<size_t>(input.nbytes()) != in_b->bytes) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                "input size mismatch: got %zu bytes, engine expects %zu",
                static_cast<size_t>(input.nbytes()), in_b->bytes);
            throw std::invalid_argument(msg);
        }

        py::dict out;
        std::vector<std::pair<Binding*, py::array_t<float>>> out_arrays;
        for (auto& b : bindings_) {
            if (b.is_input) continue;
            // Build host array shaped to engine output. Float32 only
            // for now — that's what the YOLO export produces.
            std::vector<py::ssize_t> shape;
            shape.reserve(b.dims.nbDims);
            for (int i = 0; i < b.dims.nbDims; ++i) {
                shape.push_back(b.dims.d[i]);
            }
            py::array_t<float> arr(shape);
            out_arrays.emplace_back(&b, arr);
            out[b.name.c_str()] = arr;
        }

        const float* host_in = input.data();
        {
            std::lock_guard<std::mutex> lock(mu_);
            py::gil_scoped_release no_gil;

            cudaMemcpyAsync(in_b->dev_ptr, host_in, in_b->bytes,
                            cudaMemcpyHostToDevice, stream_);
            if (!ctx_->enqueueV3(stream_)) {
                py::gil_scoped_acquire reacquire;
                throw std::runtime_error("enqueueV3 failed");
            }
            for (auto& [b, arr] : out_arrays) {
                cudaMemcpyAsync(arr.mutable_data(), b->dev_ptr, b->bytes,
                                cudaMemcpyDeviceToHost, stream_);
            }
            cudaStreamSynchronize(stream_);
        }
        return out;
    }

    // Introspection: list bindings as a list of dicts.
    py::list bindings_info() const {
        py::list out;
        for (auto& b : bindings_) {
            py::dict d;
            d["name"]    = b.name;
            d["input"]   = b.is_input;
            d["dtype"]   = dtype_name(b.dtype);
            std::vector<int> shape;
            for (int i = 0; i < b.dims.nbDims; ++i) shape.push_back(b.dims.d[i]);
            d["shape"]   = shape;
            d["bytes"]   = b.bytes;
            out.append(d);
        }
        return out;
    }

    int dla_core() const { return dla_core_; }
    const std::string& engine_path() const { return engine_path_; }

private:
    struct Binding {
        std::string name;
        bool is_input{false};
        DataType dtype;
        Dims dims;
        size_t bytes{0};
        void* dev_ptr{nullptr};
    };

    struct RuntimeDeleter { void operator()(IRuntime* p) const { if (p) delete p; } };
    struct EngineDeleter  { void operator()(ICudaEngine* p) const { if (p) delete p; } };
    struct CtxDeleter     { void operator()(IExecutionContext* p) const { if (p) delete p; } };

    std::string engine_path_;
    int dla_core_{-1};
    std::unique_ptr<IRuntime, RuntimeDeleter> runtime_;
    std::unique_ptr<ICudaEngine, EngineDeleter> engine_;
    std::unique_ptr<IExecutionContext, CtxDeleter> ctx_;
    cudaStream_t stream_{nullptr};
    std::vector<Binding> bindings_;
    std::mutex mu_;
};

} // anonymous namespace

PYBIND11_MODULE(seeker_dla, m) {
    m.doc() = "seeker_v2: TensorRT runner with DLA core selection (Phase 2.4)";

    py::class_<TrtRunner>(m, "TrtRunner")
        .def(py::init<const std::string&, int>(),
             py::arg("engine_path"), py::arg("dla_core") = -1,
             "Load a TRT engine and bind it to dla_core (>=0) or iGPU (-1).")
        .def("infer", &TrtRunner::infer, py::arg("input"),
             "Run inference. input must be FP32 with the engine's input shape.")
        .def("bindings", &TrtRunner::bindings_info)
        .def_property_readonly("dla_core",    &TrtRunner::dla_core)
        .def_property_readonly("engine_path", &TrtRunner::engine_path);
}

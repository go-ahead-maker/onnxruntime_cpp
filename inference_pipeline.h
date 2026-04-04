/**
 * @file inference_pipeline.h
 * @brief Header file for ONNX Runtime CPU Inference Pipeline
 */

#ifndef INFERENCE_PIPELINE_H
#define INFERENCE_PIPELINE_H

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <onnxruntime_cxx_api.h>

namespace onnx_inference {

/**
 * @brief Structure to hold input/output tensor information
 */
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    ONNXTensorElementDataType type;
};

/**
 * @brief Configuration for the inference pipeline
 */
struct PipelineConfig {
    std::string model_path;
    int num_threads = 1;
    bool enable_profiling = false;
    std::string profile_file_prefix = "onnx_profile";
    OrtExecutionProvider execution_provider = OrtExecutionProvider::CPU; // Only CPU for this version
};

/**
 * @brief Main inference pipeline class using ONNX Runtime
 */
class InferencePipeline {
public:
    /**
     * @brief Construct a new Inference Pipeline object
     * @param config Pipeline configuration
     */
    explicit InferencePipeline(const PipelineConfig& config);
    
    /**
     * @brief Destroy the Inference Pipeline object
     */
    ~InferencePipeline();
    
    // Disable copy
    InferencePipeline(const InferencePipeline&) = delete;
    InferencePipeline& operator=(const InferencePipeline&) = delete;
    
    // Enable move
    InferencePipeline(InferencePipeline&&) noexcept;
    InferencePipeline& operator=(InferencePipeline&&) noexcept;
    
    /**
     * @brief Initialize the pipeline (load model, create session)
     * @return true if successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Run inference with float input data
     * @param inputs Map of input name to float data vector
     * @return Map of output name to float data vector
     */
    std::map<std::string, std::vector<float>> run(
        const std::map<std::string, std::vector<float>>& inputs);
    
    /**
     * @brief Run inference with raw input tensors
     * @param input_names Vector of input names
     * @param input_shapes Vector of input shapes
     * @param input_data Vector of input data pointers
     * @return Vector of output tensors (name, shape, data)
     */
    std::vector<std::tuple<std::string, std::vector<int64_t>, std::vector<float>>> runRaw(
        const std::vector<std::string>& input_names,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<std::vector<float>>& input_data);
    
    /**
     * @brief Get input tensor information
     * @return Vector of TensorInfo for all inputs
     */
    std::vector<TensorInfo> getInputInfo() const;
    
    /**
     * @brief Get output tensor information
     * @return Vector of TensorInfo for all outputs
     */
    std::vector<TensorInfo> getOutputInfo() const;
    
    /**
     * @brief Check if the pipeline is initialized
     * @return true if initialized, false otherwise
     */
    bool isInitialized() const { return is_initialized_; }

private:
    /**
     * @brief Create OrtValue tensor from float data
     */
    Ort::Value createTensor(Ort::MemoryInfo& memory_info, 
                           std::vector<float>& data,
                           const std::vector<int64_t>& shape);
    
    /**
     * @brief Extract data from OrtValue tensor
     */
    std::vector<float> extractTensorData(Ort::Value& tensor);
    
    PipelineConfig config_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    bool is_initialized_ = false;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<TensorInfo> input_info_;
    std::vector<TensorInfo> output_info_;
    
    mutable std::mutex mutex_;
};

} // namespace onnx_inference

#endif // INFERENCE_PIPELINE_H

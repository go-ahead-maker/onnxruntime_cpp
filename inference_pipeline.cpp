/**
 * @file inference_pipeline.cpp
 * @brief Implementation of ONNX Runtime CPU Inference Pipeline
 */

#include "inference_pipeline.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace onnx_inference {

InferencePipeline::InferencePipeline(const PipelineConfig& config)
    : config_(config),
      env_(ORT_LOGGING_LEVEL_WARNING, "InferencePipeline") {
    
    // Configure session options
    session_options_.SetIntraOpNumThreads(config_.num_threads);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    if (config_.enable_profiling) {
        session_options_.EnableProfiling(config_.profile_file_prefix.c_str());
    }
}

InferencePipeline::~InferencePipeline() = default;

InferencePipeline::InferencePipeline(InferencePipeline&&) noexcept = default;
InferencePipeline& InferencePipeline::operator=(InferencePipeline&&) noexcept = default;

bool InferencePipeline::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (is_initialized_) {
        std::cerr << "Pipeline already initialized" << std::endl;
        return false;
    }
    
    try {
        // Load model
        session_ = std::make_unique<Ort::Session>(env_, 
                                                   config_.model_path.c_str(), 
                                                   session_options_);
        
        // Get input information
        size_t num_inputs = session_->GetInputCount();
        Ort::AllocatorWithDefaultOptions allocator;
        
        for (size_t i = 0; i < num_inputs; ++i) {
            auto name_ptr = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(name_ptr.get());
            
            TensorInfo info;
            info.name = input_names_.back();
            
            auto type_info = session_->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            
            info.type = tensor_info.GetElementType();
            info.shape = tensor_info.GetShape();
            
            input_info_.push_back(info);
        }
        
        // Get output information
        size_t num_outputs = session_->GetOutputCount();
        
        for (size_t i = 0; i < num_outputs; ++i) {
            auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(name_ptr.get());
            
            TensorInfo info;
            info.name = output_names_.back();
            
            auto type_info = session_->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            
            info.type = tensor_info.GetElementType();
            info.shape = tensor_info.GetShape();
            
            output_info_.push_back(info);
        }
        
        is_initialized_ = true;
        std::cout << "Pipeline initialized successfully with " 
                  << num_inputs << " inputs and " 
                  << num_outputs << " outputs" << std::endl;
        
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Error initializing pipeline: " << e.what() << std::endl;
        return false;
    }
}

std::map<std::string, std::vector<float>> InferencePipeline::run(
    const std::map<std::string, std::vector<float>>& inputs) {
    
    if (!is_initialized_) {
        throw std::runtime_error("Pipeline not initialized");
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        // Prepare input tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_name_ptrs;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        for (const auto& [name, data] : inputs) {
            // Find the corresponding input index
            auto it = std::find(input_names_.begin(), input_names_.end(), name);
            if (it == input_names_.end()) {
                throw std::runtime_error("Input name '" + name + "' not found in model");
            }
            
            size_t idx = std::distance(input_names_.begin(), it);
            const auto& info = input_info_[idx];
            
            // Verify data size matches expected shape
            size_t expected_size = std::accumulate(info.shape.begin(), info.shape.end(), 1, std::multiplies<int64_t>());
            if (data.size() != expected_size) {
                throw std::runtime_error("Input data size mismatch for '" + name + 
                                        "'. Expected: " + std::to_string(expected_size) + 
                                        ", Got: " + std::to_string(data.size()));
            }
            
            input_tensors.push_back(createTensor(memory_info, const_cast<std::vector<float>&>(data), info.shape));
            input_name_ptrs.push_back(name.c_str());
        }
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_.data(),
            output_names_.size()
        );
        
        // Extract output data
        std::map<std::string, std::vector<float>> results;
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            results[output_info_[i].name] = extractTensorData(output_tensors[i]);
        }
        
        return results;
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime error: ") + e.what());
    }
}

std::vector<std::tuple<std::string, std::vector<int64_t>, std::vector<float>>> InferencePipeline::runRaw(
    const std::vector<std::string>& input_names,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<float>>& input_data) {
    
    if (!is_initialized_) {
        throw std::runtime_error("Pipeline not initialized");
    }
    
    if (input_names.size() != input_shapes.size() || input_names.size() != input_data.size()) {
        throw std::runtime_error("Input vectors must have the same size");
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        // Prepare input tensors
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_name_ptrs;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        
        for (size_t i = 0; i < input_names.size(); ++i) {
            input_tensors.push_back(createTensor(memory_info, 
                                                 const_cast<std::vector<float>&>(input_data[i]), 
                                                 input_shapes[i]));
            input_name_ptrs.push_back(input_names[i].c_str());
        }
        
        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_name_ptrs.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_.data(),
            output_names_.size()
        );
        
        // Extract output data
        std::vector<std::tuple<std::string, std::vector<int64_t>, std::vector<float>>> results;
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto data = extractTensorData(output_tensors[i]);
            auto shape = output_info_[i].shape;
            results.emplace_back(output_info_[i].name, shape, data);
        }
        
        return results;
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(std::string("ONNX Runtime error: ") + e.what());
    }
}

std::vector<TensorInfo> InferencePipeline::getInputInfo() const {
    return input_info_;
}

std::vector<TensorInfo> InferencePipeline::getOutputInfo() const {
    return output_info_;
}

Ort::Value InferencePipeline::createTensor(Ort::MemoryInfo& memory_info, 
                                           std::vector<float>& data,
                                           const std::vector<int64_t>& shape) {
    return Ort::Value::CreateTensor<float>(memory_info, data.data(), data.size(), 
                                           shape.data(), shape.size());
}

std::vector<float> InferencePipeline::extractTensorData(Ort::Value& tensor) {
    if (!tensor.IsTensor()) {
        throw std::runtime_error("Output is not a tensor");
    }
    
    float* float_data = tensor.GetTensorMutableData<float>();
    auto type_info = tensor.GetTensorTypeAndShapeInfo();
    size_t size = type_info.GetElementCount();
    
    return std::vector<float>(float_data, float_data + size);
}

} // namespace onnx_inference

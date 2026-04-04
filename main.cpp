/**
 * @file main.cpp
 * @brief Example usage of the ONNX Runtime CPU Inference Pipeline
 */

#include "inference_pipeline.h"
#include <iostream>
#include <random>
#include <chrono>

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model.onnx> [num_threads]" << std::endl;
    std::cout << "  model.onnx   - Path to the ONNX model file" << std::endl;
    std::cout << "  num_threads  - Number of threads for inference (default: 1)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    int num_threads = (argc >= 3) ? std::stoi(argv[2]) : 1;
    
    std::cout << "=== ONNX Runtime CPU Inference Pipeline ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Number of threads: " << num_threads << std::endl;
    std::cout << std::endl;
    
    try {
        // Configure the pipeline
        onnx_inference::PipelineConfig config;
        config.model_path = model_path;
        config.num_threads = num_threads;
        config.enable_profiling = false;
        
        // Create and initialize the pipeline
        onnx_inference::InferencePipeline pipeline(config);
        
        if (!pipeline.initialize()) {
            std::cerr << "Failed to initialize pipeline" << std::endl;
            return 1;
        }
        
        // Get model information
        auto input_info = pipeline.getInputInfo();
        auto output_info = pipeline.getOutputInfo();
        
        std::cout << "\n=== Model Information ===" << std::endl;
        std::cout << "Inputs:" << std::endl;
        for (const auto& info : input_info) {
            std::cout << "  - " << info.name << " (shape: [";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                std::cout << info.shape[i];
                if (i < info.shape.size() - 1) std::cout << ", ";
            }
            std::cout << "])" << std::endl;
        }
        
        std::cout << "\nOutputs:" << std::endl;
        for (const auto& info : output_info) {
            std::cout << "  - " << info.name << " (shape: [";
            for (size_t i = 0; i < info.shape.size(); ++i) {
                std::cout << info.shape[i];
                if (i < info.shape.size() - 1) std::cout << ", ";
            }
            std::cout << "])" << std::endl;
        }
        
        // Example: Run inference with dummy data
        // Note: Replace this with your actual input data based on your model
        std::cout << "\n=== Running Inference Example ===" << std::endl;
        
        // Prepare dummy input data (adjust based on your model's input shape)
        std::map<std::string, std::vector<float>> inputs;
        
        if (!input_info.empty()) {
            // Calculate total size from shape
            size_t input_size = 1;
            for (auto dim : input_info[0].shape) {
                if (dim > 0) {
                    input_size *= dim;
                } else {
                    // Handle dynamic dimensions (use a default value)
                    input_size *= 1;
                }
            }
            
            // Generate random input data
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            
            std::vector<float> dummy_data(input_size);
            for (auto& val : dummy_data) {
                val = dist(gen);
            }
            
            inputs[input_info[0].name] = dummy_data;
            
            std::cout << "Created dummy input for '" << input_info[0].name 
                      << "' with " << input_size << " elements" << std::endl;
        }
        
        // Warm-up run
        std::cout << "\nRunning warm-up inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        auto outputs = pipeline.run(inputs);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Warm-up completed in " << duration.count() << " ms" << std::endl;
        
        // Benchmark runs
        const int num_runs = 10;
        std::cout << "\nRunning benchmark (" << num_runs << " iterations)..." << std::endl;
        
        std::vector<double> latencies;
        for (int i = 0; i < num_runs; ++i) {
            start = std::chrono::high_resolution_clock::now();
            outputs = pipeline.run(inputs);
            end = std::chrono::high_resolution_clock::now();
            double latency_ms = std::chrono::duration<double, std::milli>(end - start).count();
            latencies.push_back(latency_ms);
        }
        
        // Calculate statistics
        double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        std::cout << "\n=== Benchmark Results ===" << std::endl;
        std::cout << "Average latency: " << avg_latency << " ms" << std::endl;
        std::cout << "Min latency: " << min_latency << " ms" << std::endl;
        std::cout << "Max latency: " << max_latency << " ms" << std::endl;
        
        // Display output results
        std::cout << "\n=== Output Results ===" << std::endl;
        for (const auto& [name, data] : outputs) {
            std::cout << "Output '" << name << "': " << data.size() << " elements" << std::endl;
            std::cout << "  First 5 values: [";
            for (size_t i = 0; i < std::min(size_t(5), data.size()); ++i) {
                std::cout << data[i];
                if (i < std::min(size_t(5), data.size()) - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        std::cout << "\n=== Inference Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

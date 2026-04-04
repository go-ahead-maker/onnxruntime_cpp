# ONNX Runtime CPU Inference Pipeline

基于 C++ 和 ONNX Runtime 的高性能 CPU 推理 Pipeline。

## 项目结构

```
.
├── CMakeLists.txt              # CMake 构建配置
├── inference_pipeline.h        # Pipeline 头文件
├── inference_pipeline.cpp      # Pipeline 实现
├── main.cpp                    # 示例主程序
└── README.md                   # 说明文档
```

## 功能特性

- ✅ 基于 ONNX Runtime C++ API
- ✅ 支持多线程推理
- ✅ 自动图优化 (Graph Optimization)
- ✅ 输入/输出张量信息查询
- ✅ 支持动态 batch size 和动态维度
- ✅ 线程安全的推理接口
- ✅ 性能基准测试功能
- ✅ 可选的性能分析 (Profiling)

## 前置要求

### 1. 编译器
- GCC 7+ 或 Clang 5+ (支持 C++17)
- CMake 3.15+

### 2. ONNX Runtime

从 [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) 下载预编译的 CPU 版本：

```bash
# 示例：下载 v1.16.0 Linux CPU 版本
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
export ONNXRUNTIME_ROOT=$(pwd)/onnxruntime-linux-x64-1.16.0
```

或者从源码编译：
```bash
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel
export ONNXRUNTIME_ROOT=$(pwd)/build/Linux/Release/dist
```

## 编译方法

```bash
# 创建构建目录
mkdir build && cd build

# 配置 (指定 ONNX Runtime 路径)
cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..

# 编译
make -j$(nproc)

# 安装 (可选)
make install
```

## 使用方法

### 基本用法

```cpp
#include "inference_pipeline.h"

// 配置 Pipeline
onnx_inference::PipelineConfig config;
config.model_path = "model.onnx";
config.num_threads = 4;
config.enable_profiling = false;

// 创建并初始化
onnx_inference::InferencePipeline pipeline(config);
pipeline.initialize();

// 准备输入数据
std::map<std::string, std::vector<float>> inputs;
inputs["input"] = std::vector<float>(/* your data */);

// 运行推理
auto outputs = pipeline.run(inputs);

// 获取结果
for (const auto& [name, data] : outputs) {
    std::cout << "Output: " << name << std::endl;
}
```

### 命令行使用

```bash
# 运行示例程序
./inference model.onnx [num_threads]

# 示例
./inference resnet50.onnx 4
./inference yolov5.onnx 8
```

### 高级用法：自定义输入形状

```cpp
// 使用 runRaw 方法指定自定义输入形状
std::vector<std::string> input_names = {"input"};
std::vector<std::vector<int64_t>> shapes = {{1, 3, 224, 224}};
std::vector<std::vector<float>> data = {/* your data */};

auto results = pipeline.runRaw(input_names, shapes, data);

for (const auto& [name, shape, data] : results) {
    std::cout << "Output: " << name 
              << ", Shape: [" << shape[0] << ", " << shape[1] << "]"
              << std::endl;
}
```

## API 参考

### PipelineConfig

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_path | std::string | - | ONNX 模型文件路径 |
| num_threads | int | 1 | 推理线程数 |
| enable_profiling | bool | false | 是否启用性能分析 |
| profile_file_prefix | std::string | "onnx_profile" | 性能分析文件前缀 |

### InferencePipeline 类

#### 主要方法

- `bool initialize()`: 加载模型并创建推理会话
- `std::map<std::string, std::vector<float>> run(inputs)`: 执行推理
- `std::vector<TensorInfo> getInputInfo()`: 获取输入张量信息
- `std::vector<TensorInfo> getOutputInfo()`: 获取输出张量信息
- `bool isInitialized()`: 检查是否已初始化

#### TensorInfo 结构

```cpp
struct TensorInfo {
    std::string name;           // 张量名称
    std::vector<int64_t> shape; // 张量形状
    ONNXTensorElementDataType type; // 数据类型
};
```

## 性能优化建议

1. **线程数设置**: 
   - 对于 CPU 密集型模型，设置为物理核心数
   - 对于 I/O 密集型，可适当增加线程数

2. **批量推理**:
   - 尽可能使用较大的 batch size 提高吞吐量

3. **图优化**:
   - 默认启用所有图优化级别
   - 可通过 `session_options_.SetGraphOptimizationLevel()` 调整

4. **内存复用**:
   - 重复使用输入 buffer 减少内存分配开销

## 示例输出

```
=== ONNX Runtime CPU Inference Pipeline ===
Model path: model.onnx
Number of threads: 4

Pipeline initialized successfully with 1 inputs and 1 outputs

=== Model Information ===
Inputs:
  - input (shape: [1, 3, 224, 224])

Outputs:
  - output (shape: [1, 1000])

=== Running Inference Example ===
Created dummy input for 'input' with 150528 elements

Running warm-up inference...
Warm-up completed in 45 ms

Running benchmark (10 iterations)...

=== Benchmark Results ===
Average latency: 42.3 ms
Min latency: 41.8 ms
Max latency: 43.1 ms

=== Output Results ===
Output 'output': 1000 elements
  First 5 values: [0.0012, 0.0034, 0.0089, 0.0156, 0.0201]

=== Inference Complete ===
```

## 故障排除

### 常见问题

1. **找不到 ONNX Runtime**
   ```
   确保设置了正确的 ONNXRUNTIME_ROOT 路径
   ```

2. **模型加载失败**
   ```
   检查模型文件路径是否正确
   验证 ONNX 模型格式是否有效 (可使用 netron 查看)
   ```

3. **输入尺寸不匹配**
   ```
   使用 getInputInfo() 查看期望的输入形状
   确保输入数据大小与形状匹配
   ```

## License

MIT License

## 参考资料

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/cxx/)
- [ONNX Format Specification](https://onnx.ai/onnx/)

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <assert.h>

int main() {
	std::string onnxpath = "./grfb_unet.onnx";
	std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());
	Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "grfb_unet-onnx");
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

	// 加载会话
	Ort::Session session_(env, modelPath.c_str(), session_options);
	std::cout << "model has been loaded" << std::endl;

	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;

	size_t numInputNodes = session_.GetInputCount();
	size_t numOutputNodes = session_.GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	input_node_names.reserve(numInputNodes);

	// 获取输入信息
	int input_w = 0;
	int input_h = 0;
	for (int i = 0; i < numInputNodes; i++) {
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_w = input_dims[3];
		input_h = input_dims[2];
		std::cout << "input format: " << input_dims[0] << "," << input_dims[1] << "," << input_dims[2] << "," << input_dims[3] << std::endl;
	}

	// 获取输出信息
	int output_h = 0;
	int output_w = 0;

	Ort::TypeInfo output_type_info = session_.GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();
	output_h = output_dims[3]; 
	output_w = output_dims[2]; 
	std::cout << "output format: " << output_dims[0] << "," << output_dims[1] << "," << output_dims[2] << "," << output_dims[3] << std::endl;
	for (int i = 0; i < numOutputNodes; i++) {
		auto out_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(out_name.get());
	}

	std::cout << "input name: " << input_node_names[0] << " output name: " << output_node_names[0] << std::endl;
	return 0;
}
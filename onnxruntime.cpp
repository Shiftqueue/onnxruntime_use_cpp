#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <assert.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

float __mean[] = { 0.709, 0.381, 0.224 };
float __std[] = { 0.127, 0.079, 0.043 };

int main()
{
    try {
        // 加载模型并创建环境空间
        const wchar_t* model_path = L"./grfb_unet.onnx";
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "grfb_unet");
        Ort::SessionOptions session_options;
        Ort::Session session(env, model_path, session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        // 获取输入输出
        size_t num_input_nodes = session.GetInputCount();
        size_t num_output_nodes = session.GetOutputCount();
        std::vector<const char*> input_node_names = { "input" };
        std::vector<const char*> output_node_names = { "output" };

        // 加载测试图像
        cv::Mat original_image = cv::imread("images/003.jpg", cv::IMREAD_COLOR);
        cv::Mat resized_image;
        cv::resize(original_image, resized_image, cv::Size(640, 640));
        cv::imshow("Resized Image", resized_image);

        // 确定输入数据维度
        std::vector<int64_t> input_node_dims = { 1,3,640,640 };
        size_t input_tensor_size = 1 * 3 * 640 * 640;

        // 填充数据输入
        std::vector<float> input_tensor_values(input_tensor_size);
        for (int h = 0; h < 640; ++h) {
            for (int w = 0; w < 640; ++w) {
                for (int c = 0; c < 3; ++c) {
                    // 均一化像素值
                    float pix = resized_image.at<cv::Vec3b>(h, w)[c];
                    pix = pix / 255.0f;
                    pix = (pix - __mean[c]) / __std[c];
                    input_tensor_values[640 * 640 * c + h * 640 + w] = pix;
                }
            }
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_size,
            input_node_dims.data(),
            input_node_dims.size()
        );

        assert(input_tensor.IsTensor());

        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));


        // 启动模型预测并获取输出张量
        auto output_tensors = session.Run(
            Ort::RunOptions{ nullptr },
            input_node_names.data(),
            ort_inputs.data(),
            ort_inputs.size(),
            output_node_names.data(),
            1
        );

        // 解析输出张量
        Ort::Value& output_tensor = output_tensors[0];
        const float* output_data = output_tensor.GetTensorData<float>();
        std::vector<int64_t> output_dims = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

        // 存储输出图像
        cv::Mat result_image(640, 640, CV_8UC1);

        // 对输出的2通道图像进行二分类预测
        for (int h = 0; h < 640; ++h) {
            for (int w = 0; w < 640; ++w) {
                int index_max = output_data[w + h * 640] > output_data[w + h * 640 + 640 * 640] ? 0 : 1; 
                result_image.at<uchar>(h, w) = 255 * index_max; 
            }
        }

        // 显示结果
        cv::imshow("Result Image", result_image);
        cv::waitKey(0); 
        cv::imwrite("result_image.png", result_image);

    }
    catch (const Ort::Exception& e) {
        // 打印异常
        std::cerr << "Caught Ort::Exception: " << std::string(e.what()) << std::endl;
        size_t pos = std::string(e.what()).find("ErrorCode: ");
        if (pos != std::string::npos) {
            std::string error_code_str = std::string(e.what()).substr(pos + 12); 
            int error_code = std::stoi(error_code_str);
            std::cerr << "Error Code: " << error_code << std::endl;
        }
        return -1;
    }

    return 0;
}


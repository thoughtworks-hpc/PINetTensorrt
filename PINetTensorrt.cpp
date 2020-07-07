#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <dirent.h>
#include <string.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace {
    const std::string gSampleName = "TensorRT.onnx_PINet";

    const int output_base_index = 3;
    const float threshold_point = 0.81f;
    const float threshold_instance = 0.22f;
    const int resize_ratio = 8;

    int64 total_inference_execute_elasped_time = 0;
    int64 total_inference_execute_times = 0;

    using LaneLine = std::vector<cv::Point2f>;
    using LaneLines = std::vector<LaneLine>;

    cv::Mat chwDataToMat(int channelNum, int height, int width, float* data, cv::Mat& mask) {
        std::vector<cv::Mat> channels(channelNum);
        int data_size = width * height;
        for (int c = 0; c < channelNum; ++c) {
            float* channel_data = data + data_size * c;
            cv::Mat channel(width, height, CV_32FC1);
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w, ++channel_data) {
                    channel.at<float>(w, h) = *channel_data * (int)mask.at<uchar>(w, h);
                }
            }
            channels[c] = channel;
        }

        cv::Mat mergedMat;
        cv::merge(channels.data(), channelNum, mergedMat);
        return mergedMat;
    }

    void getFiles(std::string root_dir, std::string ext, std::vector<std::string>& files) {
        DIR *dir;
        struct dirent *ptr;

        if ((dir = opendir(root_dir.c_str())) == NULL) {
            perror("Open dir error...");
            return;
        }
    
        while ((ptr = readdir(dir)) != NULL) {
            if (strcmp(ptr->d_name,".") == 0 || strcmp(ptr->d_name,"..") == 0) {
                continue;
            } else if(ptr->d_type == 8)  {// file
                char* dot = strchr(ptr->d_name, '.');
                if (dot && !strcasecmp(dot, ext.c_str())) {
                    std::string filename(root_dir);
                    filename.append("/").append(ptr->d_name);
                    files.push_back(filename);
                }
            } else if(ptr->d_type == 10) { // link file  
                continue;
            } else if(ptr->d_type == 4)  {// dir
                std::string dir_path(root_dir);
                dir_path.append("/").append(ptr->d_name);
                getFiles(dir_path.c_str(), ext, files);
            }  
        }

        closedir(dir);  
    }
}

//! \brief  The SampleOnnxMNIST class implements the ONNX PINet sample
//!
//! \details It creates the network using an ONNX model
//!
class PINetTensorrt
{
    template <typename T>
    using UniquePtr = std::unique_ptr<T, common::InferDeleter>;

public:
    PINetTensorrt(const common::OnnxParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

    void setImageFile(const std::string& imageFileName) {
        mImageFileName = imageFileName;
    }

private:
    common::OnnxParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    std::vector<nvinfer1::Dims> mOutputDims; //!< The dimensions of the output to the network.
    std::string mImageFileName;            //!< The number to classify
    cv::Mat mInputImage;

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder,
        UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvinfer1::IBuilderConfig>& config,
        UniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const common::BufferManager& buffers);
    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const common::BufferManager& buffers);

    LaneLines generate_result(float* confidance, float* offsets, float* instance, float thresh);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool PINetTensorrt::build()
{
    auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false;
    }

    auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), common::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    if (gLogger.getReportableSeverity() == Logger::Severity::kVERBOSE) {
        for (int i = 0; i < network->getNbInputs(); ++i) {
            nvinfer1::Dims dim = network->getInput(i)->getDimensions();
            gLogInfo << "InputDims: " << i << " " << dim.d[0] << " " << dim.d[1] << " " << dim.d[2] << std::endl;
        }

        for (int i = 0; i < network->getNbOutputs(); ++i) {
            nvinfer1::Dims dim = network->getOutput(i)->getDimensions();
            gLogInfo << "OutputDims: " << i << " " << dim.d[0] << " " << dim.d[1] << " " << dim.d[2] << std::endl;
        }
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    assert(network->getNbOutputs() == 6);
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        nvinfer1::Dims dim = network->getOutput(i)->getDimensions();
        mOutputDims.push_back(dim);
        assert(dim.nbDims == 3);
    }

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool PINetTensorrt::constructNetwork(UniquePtr<nvinfer1::IBuilder>& builder,
    UniquePtr<nvinfer1::INetworkDefinition>& network, UniquePtr<nvinfer1::IBuilderConfig>& config,
    UniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(mParams.onnxFileName.c_str(), static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1 << 30);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        common::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    common::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool PINetTensorrt::infer()
{
    // Create RAII buffer manager object
    common::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    auto inferenceBeginTime = std::chrono::high_resolution_clock::now();
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    auto inference_execute_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - inferenceBeginTime);
    total_inference_execute_elasped_time += inference_execute_elapsed_time.count();
    ++total_inference_execute_times;

    //gLogInfo << "inference elapsed time: " << inferenceElapsedTime.count() / 1000.f << " milliseconds" << std::endl;

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool PINetTensorrt::processInput(const common::BufferManager& buffers)
{
    const int inputC = mInputDims.d[0];
    const int inputW = mInputDims.d[1];
    const int inputH = mInputDims.d[2];

    cv::Mat image = cv::imread(mImageFileName, 1);
    assert(inputC == image.channels());
    cv::resize(image, image, cv::Size(inputH, inputW));

    mInputImage = image;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    uchar* imageData = image.ptr<uchar>();
    for (int c = 0; c < inputC; ++c) {
        for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j) {
            hostDataBuffer[c * volChl + j] = float(imageData[j * inputC + c]) / 255.f;
        }
    }

    return true;
}

LaneLines PINetTensorrt::generate_result(float* confidance_data, float* offsets_data, float* instance_data, float thresh)
{
    const nvinfer1::Dims& dim            = mOutputDims[output_base_index];//1 32 64
    const nvinfer1::Dims& offset_dim     = mOutputDims[output_base_index + 1];//2 32 64
    const nvinfer1::Dims& instance_dim   = mOutputDims[output_base_index + 2];//4 32 64

    cv::Mat mask = cv::Mat::zeros(dim.d[2], dim.d[1], CV_8UC1);
    float* confidance_ptr = confidance_data;
    for (int i = 0; i < dim.d[1]; ++i) {
        for (int j = 0; j < dim.d[2]; ++j, ++confidance_ptr) {
            if (*confidance_ptr > thresh) {
                mask.at<uchar>(j, i) = 1;
            }
        }
    }

    if (gLogger.getReportableSeverity() == Logger::Severity::kVERBOSE) {
        gLogInfo << "Output mask:" << std::endl;
        for (int i = 0; i < dim.d[1]; ++i) {
            for (int j = 0; j < dim.d[2]; ++j) {
                gLogInfo << (int)mask.at<uchar>(j, i);
            }
            gLogInfo << std::endl;
        }

        cv::Mat maskImage = mInputImage.clone();
        cv::Scalar color(0, 0, 255);
        for (int i = 0; i < dim.d[1]; ++i) {
            for (int j = 0; j < dim.d[2]; ++j) {
                if ((int)mask.at<uchar>(j, i)) {
                    cv::circle(maskImage, cv::Point2f(j * 8, i * 8), 3, color, -1);
                }
            }
        }
        cv::imshow("mask", maskImage);
        cv::waitKey(0);
    }

    cv::Mat offsets  = chwDataToMat(offset_dim.d[0], offset_dim.d[1], offset_dim.d[2], offsets_data, mask);
    cv::Mat features = chwDataToMat(instance_dim.d[0], instance_dim.d[1], instance_dim.d[2], instance_data, mask);    

    if (gLogger.getReportableSeverity() == Logger::Severity::kVERBOSE) {
        gLogInfo << "Output offset:" << std::endl;
        for (int i = 0; i < dim.d[1]; ++i) {
            for (int j = 0; j < dim.d[2]; ++j) {
                gLogInfo << (offsets.at<cv::Vec2f>(j, i)[0] ? 1 : 0);
            }
            gLogInfo << std::endl;
        }

        cv::Mat offsetImage = mInputImage.clone();
        cv::Scalar color(0, 0, 255);
        for (int i = 0; i < dim.d[1]; ++i) {
            for (int j = 0; j < dim.d[2]; ++j) {
                if ((int)mask.at<uchar>(j, i)) {
                    cv::Vec2f pointOffset = offsets.at<cv::Vec2f>(j, i);
                    cv::Point2f point(pointOffset[0] + j, pointOffset[1] + i);
                    cv::circle(offsetImage, point * 8, 3, color, -1);
                }
            }
        }
        cv::imshow("offset", offsetImage);
        cv::waitKey(0);

        gLogInfo << "Output instance:" << std::endl;
        for (int i = 0; i < dim.d[1]; ++i) {
            for (int j = 0; j < dim.d[2]; ++j) {
                gLogInfo << (features.at<cv::Vec4f>(j, i)[0] ? 1 : 0);
            }
            gLogInfo << std::endl;
        }
    }

    LaneLines laneLines;
    std::vector<cv::Vec4f> laneFeatures;

    auto findNearestFeature = [&laneFeatures](const cv::Vec4f& feature) -> int {
        for (int i = 0; i < laneFeatures.size(); ++i) {
            auto delta = laneFeatures[i] - feature;
            if (delta.dot(delta) <= threshold_instance) {
                return i;
            }
        }
        return -1;
    };

    for (int i = 0; i < dim.d[1]; ++i) {
        for (int j = 0; j < dim.d[2]; ++j) {
            if ((int)mask.at<uchar>(j, i) == 0) {
                continue;
            }

            const cv::Vec2f& offset = offsets.at<cv::Vec2f>(j, i);
            cv::Point2f point(offset[0] + j, offset[1] + i);
            if (point.x > dim.d[2] || point.x < 0.f) continue;
            if (point.y > dim.d[1] || point.y < 0.f) continue;

            const cv::Vec4f& feature = features.at<cv::Vec4f>(j, i);
            int lane_index = findNearestFeature(feature);
            
            if (lane_index == -1) {
                laneLines.emplace_back(LaneLine({point}));
                laneFeatures.emplace_back(feature);
            } else {
                auto& laneline = laneLines[lane_index];
                auto& lanefeature = laneFeatures[lane_index];

                auto point_size = laneline.size(); 

                lanefeature = lanefeature.mul(cv::Vec4f::all(point_size)) + feature;
                lanefeature = lanefeature.mul(cv::Vec4f::all(1.f / (point_size + 1)));
                laneline.emplace_back(point);
            }
        }
    }

    for (auto itr = laneLines.begin(); itr != laneLines.end();) {
        if ((*itr).size() < 2) {
            itr = laneLines.erase(itr);
        } else {
            ++itr;
        }
    }

    return laneLines;
}

//!
//! \brief verify result
//!
//! \return whether output matches expectations
//!
bool PINetTensorrt::verifyOutput(const common::BufferManager& buffers)
{
    float *confidance, *offset, *instance;
    confidance = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[output_base_index + 0]));    
    offset     = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[output_base_index + 1]));    
    instance   = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[output_base_index + 2]));    
 
    nvinfer1::Dims confidanceDims = mOutputDims[output_base_index + 0];
    nvinfer1::Dims offsetDims     = mOutputDims[output_base_index + 1];
    nvinfer1::Dims instanceDims   = mOutputDims[output_base_index + 2];
    
    assert(confidanceDims.d[0] == 1);
    assert(offsetDims.d[0]     == 2);
    assert(instanceDims.d[0]   == 4);

    LaneLines lanelines = generate_result(confidance, offset, instance, threshold_point);
    if (lanelines.empty())
        return false;

    cv::Scalar color[] = {{255,   0,   0}, {  0, 255,   0}, {  0,   0, 255}, 
                        {255, 255,   0}, {255,   0, 255}, {  0, 255, 255}, 
                        {255, 255, 255}, {100, 255,   0}, {100,   0, 255}, 
                        {255, 100,   0}, {  0, 100, 255}, {255,   0, 100}, 
                        {  0, 255, 100}};

    cv::Mat lanelineImage = mInputImage;
    for (int i = 0; i < lanelines.size(); ++i) {
        for (const auto& point : lanelines[i]) {
            cv::circle(lanelineImage, cv::Point2f(point * 8), 3, color[i], -1);
        }
    }

    if (gLogger.getReportableSeverity() == Logger::Severity::kVERBOSE) {
        cv::imwrite("lanelines.jpg", lanelineImage);

        cv::imshow("lanelines", lanelineImage);
        cv::waitKey(0);
    }

    return true;
}
//!
//! \brief Initializes members of the params struct using the command line args
//!
common::OnnxParams initializeSampleParams(const common::Args& args)
{
    common::OnnxParams params;
    if (args.dataDirs.empty()) {//!< Use default directories if user hasn't provided directory paths
        params.dataDirs.push_back("/home/xuwen/devel/PINetTensorrt/data");
    } else {//!< Use the data directory provided by the user
        params.dataDirs = args.dataDirs;
    }
    params.dataDirs.push_back("/home/xuwen/devel/PINetTensorrt");

    char pwd[1024] = {0};
    getcwd(pwd, sizeof(pwd));

    params.onnxFileName = "pinet.onnx";
    params.inputTensorNames.push_back("0");
    params.batchSize = 1;
    params.outputTensorNames.push_back("1431");
    params.outputTensorNames.push_back("1438");
    params.outputTensorNames.push_back("1445");
    params.outputTensorNames.push_back("1679");
    params.outputTensorNames.push_back("1686");
    params.outputTensorNames.push_back("1693");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./pinettensorrt [-h or --help] [-d or --datadir=<path to data path>] [--useDLACore=<int>]" << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data path, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    common::Args args;
    bool argsOK = common::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    setReportableSeverity(Logger::Severity::kINFO);
    auto test = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(test);

    common::OnnxParams onnx_args = initializeSampleParams(args);
    PINetTensorrt sample(onnx_args);

    gLogInfo << "Building and running a GPU inference engine for Onnx PINet" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(test);
    }

    std::vector<std::string> filenames;
    filenames.reserve(20480);
    for (size_t i = 0; i < onnx_args.dataDirs.size() - 1; i++) {
        getFiles(onnx_args.dataDirs[i], ".jpg", filenames);
    }

    auto inference_begin_time = std::chrono::high_resolution_clock::now();

    for (const auto& filename : filenames) {
        sample.setImageFile(filename);
        if (!sample.infer()) {
            gLogger.reportFail(test);
        }
    }

    auto inference_elpased_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - inference_begin_time);

    gLogger.reportPass(test);

    gLogInfo << std::endl;

    gLogInfo <<     "totally inference time      : " << inference_elpased_time.count() / 1000.f << " milliseconds" << std::endl;
    if (filenames.size()) {
        gLogInfo << "totally inference times     : " << filenames.size() << std::endl;
        gLogInfo << "average inference time      : " << inference_elpased_time.count() / filenames.size() / 1000.f << " milliseconds"<< std::endl;
    }

    if (total_inference_execute_times > 0) {
        gLogInfo << "totally execute elapsed time: " << total_inference_execute_elasped_time / 1000.f << " milliseconds" << std::endl << std::endl;
        gLogInfo << "inference execute times     : " << total_inference_execute_times << std::endl;
        gLogInfo << "average execute elapsed time: " << total_inference_execute_elasped_time / total_inference_execute_times / 1000.f << " milliseconds" << std::endl << std::endl;
    }

    return 0;
}

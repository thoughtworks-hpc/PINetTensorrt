#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const std::string gSampleName = "TensorRT.onnx_PINet";

const int output_base_index = 3;
const float threshold_point = 0.81f;
const float threshold_instance = 0.22f;
const int resize_ratio = 8;

namespace {

    using LaneLine = std::vector<cv::Point2f>;
    using LaneLines = std::vector<LaneLine>;

    float lengthSquare2(const cv::Point2f& p0, const cv::Point2f& p1) {
        cv::Point2f d = p0 - p1;
        return d.x * d.x + d.y * d.y;
    }

    void nearestInsert(LaneLine& laneline, const cv::Point2f& p, float thresh_hold) {
        float distance = 0.f;
        int index = -1;
        for (int i = 0; i < laneline.size(); ++i) {
            float length = lengthSquare2(laneline[i], p);
            if (length < distance && length < thresh_hold) {
                distance = length;
                index = i;
            }
        }

        if (index == -1) {
            return;
        }

        int pre_index = index;
        int next_index = index;
        
        if (index == 0) {
            next_index = 1;

            cv::Vec2f v_p0 = laneline[pre_index] - p;
            cv::Vec2f v_p1 = laneline[next_index] - p;
            if (v_p0.dot(v_p1) < 0) {
                laneline.insert(laneline.begin() + next_index, p);
            } else {
                laneline.insert(laneline.begin() + pre_index, p);
            }
        } else if (index == laneline.size() - 1) {
            pre_index = index - 1;
            
            cv::Vec2f v_p0 = laneline[pre_index] - p;
            cv::Vec2f v_p1 = laneline[next_index] - p;
            if (v_p0.dot(v_p1) < 0) {
                laneline.insert(laneline.begin() + next_index, p);
            } else {
                laneline.emplace_back(p);
            }   
        } else {
            pre_index = index - 1;
            next_index = index + 1;
            
            cv::Vec2f v_p0 = laneline[pre_index] - p;
            cv::Vec2f v_p1 = laneline[next_index] - p;
            if (v_p0.dot(v_p1) < 0) {
                laneline.insert(laneline.begin() + index, p);
            } else {
                laneline.insert(laneline.begin() + next_index, p);
            }
        }
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

    void setImageFile(const std::string& imageFile) {
        mImageFile = imageFile;
    }

private:
    common::OnnxParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    std::vector<nvinfer1::Dims> mOutputDims; //!< The dimensions of the output to the network.
    std::string mImageFile;            //!< The number to classify
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

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    printf("InputDims %d %d %d\n", mInputDims.d[0], mInputDims.d[1], mInputDims.d[2]);

    assert(network->getNbOutputs() == 6);

    for (int i = 0; i < network->getNbOutputs(); ++i) {
        nvinfer1::Dims dim = network->getOutput(i)->getDimensions();
        mOutputDims.push_back(dim);
        assert(dim.nbDims == 3);
        printf("OutputDims %d %d %d %d\n", i, dim.d[0], dim.d[1], dim.d[2]);
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
    auto parsed = parser->parseFromFile(
        locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()));
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

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

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

    printf("input tensor: %d %d %d\n", inputC, inputW, inputH);

    cv::Mat image = cv::imread(locateFile(mImageFile, mParams.dataDirs), 1);
    assert(inputC == image.channels());

    printf("input image: %d %d %d\n", image.channels(), image.cols, image.rows);
    cv::resize(image, image, cv::Size(inputH, inputW));

    printf("input image: %d %d %d\n", image.channels(), image.cols, image.rows);
    mInputImage = image;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int host_index = 0;
    uchar* imageData = image.ptr<uchar>();
    for (int c = 0; c < inputC; ++c) {
        // The color image to input should be in BGR order
        for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j) {
            hostDataBuffer[c * volChl + j] = float(imageData[j * inputC + c]) / 255.f;
        }
    }
    
    return true;
}

cv::Mat chwDataToMat(int numberOfChannel, int width, int height, float* data, cv::Mat& mask) {
    std::vector<cv::Mat> channels;
    int data_size = width * height * sizeof(float);
    for (int c = 0; c < numberOfChannel; ++c) {
        float* channel_data = data + data_size * c;
        cv::Mat channel(width, height, CV_32FC1);
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w, ++channel_data) {
                channel.at<float>(w, h) = *channel_data * mask.at<int>(w, h);
            }
        }
        channels.emplace_back(channel);
    }

    cv::Mat mergedMat;
    cv::merge(channels.data(), numberOfChannel, mergedMat);
    return mergedMat;
}

LaneLines PINetTensorrt::generate_result(float* confidance, float* offsets, float* instance, float thresh)
{
    const nvinfer1::Dims& dim            = mOutputDims[output_base_index];//1 32 64
    const nvinfer1::Dims& offset_dim     = mOutputDims[output_base_index + 1];//2 32 64
    const nvinfer1::Dims& instance_dim   = mOutputDims[output_base_index + 2];//4 32 64

    cv::Mat mask = cv::Mat::zeros(dim.d[1], dim.d[2], CV_8UC1);
    float* confidance_ptr = confidance;
    for (int i = 0; i < dim.d[1]; ++i) {
        for (int j = 0; j < dim.d[2]; ++j, ++confidance_ptr) {
            if (*confidance_ptr > thresh) {
                mask.at<uchar>(i, j) = 1;
            }
        }
    }

    gLogInfo << "Output mask:" << std::endl;
    for (int i = 0; i < dim.d[1]; ++i) {
        for (int j = 0; j < dim.d[2]; ++j) {
            gLogInfo << (int)mask.at<uchar>(i, j);
        }
        gLogInfo << std::endl;
    }

    cv::Mat lanesImage = mInputImage.clone();
    cv::Scalar lanesColor(0, 0, 255);
    for (int i = 0; i < dim.d[1]; ++i) {
        for (int j = 0; j < dim.d[2]; ++j) {
            if ((int)mask.at<uchar>(i, j)) {
                cv::circle(lanesImage, cv::Point2f(j * 8, i * 8), 3, lanesColor, -1);
            }
        }
    }

    cv::imshow("lanes", lanesImage);
    cv::waitKey(0);

    gLogInfo << "Construct lanelines:" << std::endl;

    cv::Mat offset = chwDataToMat(offset_dim.d[0], offset_dim.d[1], offset_dim.d[2], offsets, mask);
    cv::Mat feature = chwDataToMat(instance_dim.d[0], instance_dim.d[1], instance_dim.d[2], instance, mask);

    //gLogInfo << "offset: "<< offset << std::endl;
    //std::cout << cv::format(offset, cv::Formatter::FMT_NUMPY) << ";" << endl << endl;

    gLogInfo << "chw data to mat finish" << std::endl;
    LaneLines lanelines;
    
    for (int i = 0; i < dim.d[2]; ++i) {
        for (int j = 0; j < dim.d[1]; ++j) {
            const cv::Vec2f& feature_value = feature.at<cv::Vec2f>(i, j);
            if (feature_value[0] * feature_value[0] + feature_value[1] * feature_value[1] < 0.000001f) {
                continue;
            }

            float point_x = (feature_value[0] + j) * resize_ratio;
            float point_y = (feature_value[1] + i) * resize_ratio;

            if (point_x < 0 || point_x >= dim.d[1] || point_y < 0 || point_y >= dim.d[2]) {
                continue;
            }

            if (lanelines.empty()) {
                LaneLine line;
                line.reserve(256);
                line.emplace_back(cv::Point2f(point_x, point_y));
                lanelines.emplace_back(line);
            } else {
                for (auto& laneline : lanelines) {
                    nearestInsert(laneline, cv::Point2f(point_x, point_y), threshold_point);
                }
            }
        }
    }

    for (auto itr = lanelines.begin(); itr != lanelines.end();) {
        auto& laneline = *itr;
        if (laneline.size() > 2) {
            laneline.shrink_to_fit();
            ++itr;
        } else {
            itr = lanelines.erase(itr);
        }
    }

    return lanelines;
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

    gLogInfo << "generate_result finish!" << std::endl;

    cv::Mat result = mInputImage.clone();
    cv::Scalar colors[3] = { cv::Scalar(1, 0, 0), cv::Scalar(0, 1, 0), cv::Scalar(0, 0, 1) };
    for (int i = 0; i < lanelines.size(); ++i) {
        for (const auto& point : lanelines[i]) {
            cv::circle(result, point, 3, colors[i % 3]);
        }
    }

    gLogInfo << "draw lanelines to image finish" << std::endl;
    cv::imwrite("result.png", result);
    return true;
}
//!
//! \brief Initializes members of the params struct using the command line args
//!
common::OnnxParams initializeSampleParams(const common::Args& args)
{
    common::OnnxParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("/home/xuwen/devel/PINetTensorrt/data");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    //params.onnxFileName = "pinet.onnx";
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

    PINetTensorrt sample(initializeSampleParams(args));
    sample.setImageFile("1492638000682869180/1.jpg");

    gLogInfo << "Building and running a GPU inference engine for Onnx PINet" << std::endl;

    if (!sample.build())
    {
        return gLogger.reportFail(test);
    }
    if (!sample.infer())
    {
        return gLogger.reportFail(test);
    }

    return gLogger.reportPass(test);
}

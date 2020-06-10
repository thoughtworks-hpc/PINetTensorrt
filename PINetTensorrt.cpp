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
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    std::string mImageFile;            //!< The number to classify

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
    mOutputDims = network->getOutput(0)->getDimensions();

    assert(mOutputDims.nbDims == 3);

    printf("OutputDims %d %d %d\n", mOutputDims.d[0], mOutputDims.d[1], mOutputDims.d[2]);

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
    const int inputH = mInputDims.d[1];
    const int inputW = mInputDims.d[2];

    cv::Mat fileData = cv::imread(locateFile(mImageFile, mParams.dataDirs), 1);
    cv::resize(fileData, fileData, cv::Size(inputW, inputH));

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    int host_index = 0;
    for (int i = 0; i < inputH; ++i)
    {
        for (int j = 0; j < inputW; ++j, ++host_index) {
            hostDataBuffer[host_index] = 1.0 - float(fileData.at<uchar>(i, j) / 255.0);
        }
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool PINetTensorrt::verifyOutput(const common::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[0];
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
    }
    gLogInfo << std::endl;

    return idx;
    //return idx == mNumber && val > 0.9f;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
common::OnnxParams initializeSampleParams(const common::Args& args)
{
    common::OnnxParams params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("../data");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    //params.onnxFileName = "pinet.onnx";
    params.onnxFileName = "pinet1.0.0.onnx";
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

    setReportableSeverity(Logger::Severity::kVERBOSE);
    auto test = gLogger.defineTest(gSampleName, argc, argv);

    gLogger.reportTestStart(test);

    PINetTensorrt sample(initializeSampleParams(args));
    sample.setImageFile("1.jpg");

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

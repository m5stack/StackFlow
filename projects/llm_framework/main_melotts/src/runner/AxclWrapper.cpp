#include "AxclWrapper.h"

#include "../../../../SDK/components/utilities/include/sample_log.h"

AxclWrapper::AxclWrapper() : runner_(std::make_unique<middleware::runtime_runner>())
{
}

AxclWrapper::~AxclWrapper()
{
    finalize();
}

bool AxclWrapper::initialize(const std::string& configFile, uint32_t index, uint32_t kind, const std::string& modelPath,
                             bool input_cached, bool output_cached, uint32_t group, uint32_t batch)
{
    // runner_ = std::make_unique<middleware::runtime_runner>();
    if (!runner_) {
        SLOGE("Failed to create runner instance.");
        return false;
    }

    if (!runner_->init(configFile, index, kind)) {
        SLOGE("Init failed for runner with config {%s}, index {%u}, kind {%u}.", configFile.c_str(), index, kind);
        runner_.reset();
        return false;
    }

    if (!runner_->load(modelPath)) {
        SLOGE("Loading model {%s} failed.", modelPath.c_str());
        return false;
    }

    if (!runner_->prepare(input_cached, output_cached, group, batch)) {
        SLOGE("Prepare failed for runner with input_cached {%s}, output_cached {%s}, group {%u}, batch {%u}.",
              input_cached ? "true" : "false", output_cached ? "true" : "false", group, batch);
        runner_.reset();
        return false;
    }

    return true;
}

bool AxclWrapper::run(bool sync)
{
    if (!runner_) {
        SLOGE("Runner not initialized before running.");
        return false;
    }
    return runner_->run(sync);
}

bool AxclWrapper::set()
{
    if (!runner_) {
        SLOGE("Runner not initialized before running.");
        return false;
    }
    return runner_->set();
}

size_t AxclWrapper::getInputSize(int index) const
{
    if (!runner_) {
        SLOGE("Runner not initialized when getting input size for index %d.", index);
        return 0;
    }
    return runner_->get_input_size(index);
}

size_t AxclWrapper::getOutputSize(int index) const
{
    if (!runner_) {
        SLOGE("Runner not initialized when getting output size for index %d.", index);
        return 0;
    }
    return runner_->get_output_size(index);
}

void* AxclWrapper::getInputPointer(int index)
{
    if (!runner_) {
        SLOGE("Runner not initialized when getting input pointer for index %d.", index);
        return nullptr;
    }
    return runner_->get_input_pointer(index);
}

void* AxclWrapper::getOutputPointer(int index)
{
    if (!runner_) {
        SLOGE("Runner not initialized when getting output pointer for index %d.", index);
        return nullptr;
    }
    return runner_->get_output_pointer(index);
}

void AxclWrapper::finalize()
{
    if (runner_) {
        runner_->final();
        runner_.reset();
        SLOGI("AxclWrapper for model {%s} finalized.", model_path_.c_str());
    }
}
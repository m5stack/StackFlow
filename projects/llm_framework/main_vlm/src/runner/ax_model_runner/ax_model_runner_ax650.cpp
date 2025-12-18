#include "ax_model_runner_ax650.hpp"
#include <cstring>
#include <fstream>
#include <algorithm>
#include <memory>
#include <unordered_set>
#include <ax_sys_api.h>
#include <ax_ivps_api.h>
#include <ax_engine_api.h>
#include <fcntl.h>
#include "memory_utils.hpp"
#include "sample_log.h"

#define AX_CMM_ALIGN_SIZE 128
const char *AX_CMM_SESSION_NAME = "npu";

typedef enum {
    AX_ENGINE_ABST_DEFAULT = 0,
    AX_ENGINE_ABST_CACHED  = 1,
} AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

struct ax_runner_ax650_handle_t {
    AX_ENGINE_HANDLE handle     = nullptr;
    AX_ENGINE_CONTEXT_T context = 0;
    std::vector<AX_ENGINE_IO_INFO_T *> io_info;
    std::vector<AX_ENGINE_IO_T> io_data;
};

static int prepare_io_struct_only(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data)
{
    memset(io_data, 0, sizeof(*io_data));
    io_data->pInputs    = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    io_data->nInputSize = info->nInputSize;
    memset(io_data->pInputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nInputSize);

    io_data->pOutputs    = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    io_data->nOutputSize = info->nOutputSize;
    memset(io_data->pOutputs, 0, sizeof(AX_ENGINE_IO_BUFFER_T) * info->nOutputSize);

    for (uint i = 0; i < info->nInputSize; ++i) io_data->pInputs[i].nSize = info->pInputs[i].nSize;
    for (uint i = 0; i < info->nOutputSize; ++i) io_data->pOutputs[i].nSize = info->pOutputs[i].nSize;

    return 0;
}

static int prepare_io_with_alloc(
    AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data,
    std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T> strategy,
    std::vector<std::string> skip_alloc_names = {})
{
    int ret = prepare_io_struct_only(info, io_data);
    if (ret != 0) return ret;

    for (uint i = 0; i < info->nInputSize; ++i) {
        auto &buffer = io_data->pInputs[i];
        if (std::find(skip_alloc_names.begin(), skip_alloc_names.end(), info->pInputs[i].pName) !=
            skip_alloc_names.end()) {
            continue;
        }
        if (strategy.first == AX_ENGINE_ABST_CACHED) {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer.phyAddr), &buffer.pVirAddr, buffer.nSize, AX_CMM_ALIGN_SIZE,
                                        (const AX_S8 *)(AX_CMM_SESSION_NAME));
        } else {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer.phyAddr), &buffer.pVirAddr, buffer.nSize, AX_CMM_ALIGN_SIZE,
                                  (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        if (ret != 0) {
            ALOGE("Alloc input[%d] failed", i);
            return ret;
        }
        memset(buffer.pVirAddr, 0, buffer.nSize);
    }

    for (uint i = 0; i < info->nOutputSize; ++i) {
        auto &buffer = io_data->pOutputs[i];
        if (std::find(skip_alloc_names.begin(), skip_alloc_names.end(), info->pOutputs[i].pName) !=
            skip_alloc_names.end()) {
            continue;
        }
        if (strategy.second == AX_ENGINE_ABST_CACHED) {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer.phyAddr), &buffer.pVirAddr, buffer.nSize, AX_CMM_ALIGN_SIZE,
                                        (const AX_S8 *)(AX_CMM_SESSION_NAME));
        } else {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer.phyAddr), &buffer.pVirAddr, buffer.nSize, AX_CMM_ALIGN_SIZE,
                                  (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        if (ret != 0) {
            ALOGE("Alloc output[%d] failed", i);
            return ret;
        }
        memset(buffer.pVirAddr, 0, buffer.nSize);
    }
    return 0;
}

int ax_runner_ax650::sub_init()
{
    if (!m_handle) return -1;

    int ret = AX_ENGINE_CreateContext(m_handle->handle);
    if (ret != 0) return ret;

    ret = AX_ENGINE_CreateContextV2(m_handle->handle, &m_handle->context);
    if (ret != 0) return ret;

    AX_U32 io_count = 0;
    ret             = AX_ENGINE_GetGroupIOInfoCount(m_handle->handle, &io_count);
    if (ret != 0) return ret;

    m_handle->io_info.resize(io_count);
    m_handle->io_data.resize(io_count);
    mgroup_input_tensors.resize(io_count);
    mgroup_output_tensors.resize(io_count);

    std::vector<std::string> skip_alloc_names = {"K_cache", "V_cache"};
    for (size_t grpid = 0; grpid < io_count; grpid++) {
        AX_ENGINE_IO_INFO_T *io_info = nullptr;
        ret                          = AX_ENGINE_GetGroupIOInfo(m_handle->handle, grpid, &io_info);
        if (ret != 0) return ret;
        m_handle->io_info[grpid] = io_info;

        if (grpid == 0) {
            ret = prepare_io_with_alloc(io_info, &m_handle->io_data[grpid],
                                        {AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED});
        } else if (grpid == io_count - 1) {
            ret = prepare_io_with_alloc(io_info, &m_handle->io_data[grpid],
                                        {AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED}, skip_alloc_names);
        } else {
            ret = prepare_io_struct_only(io_info, &m_handle->io_data[grpid]);
        }
        if (ret != 0) return ret;
    }

    if (io_count > 2) {
        auto &first_io_data = m_handle->io_data[0];
        auto &first_io_info = m_handle->io_info[0];
        auto &last_io_data  = m_handle->io_data[io_count - 1];
        auto &last_io_info  = m_handle->io_info[io_count - 1];
        for (uint i = 0; i < last_io_data.nInputSize; ++i) {
            if (std::find(skip_alloc_names.begin(), skip_alloc_names.end(), last_io_info->pInputs[i].pName) !=
                skip_alloc_names.end()) {
                for (uint j = 0; j < first_io_data.nInputSize; ++j) {
                    if (first_io_info->pInputs[j].pName == last_io_info->pInputs[i].pName) {
                        last_io_data.pInputs[i].phyAddr  = first_io_data.pInputs[j].phyAddr;
                        last_io_data.pInputs[i].pVirAddr = first_io_data.pInputs[j].pVirAddr;
                    }
                }
            }
        }

        for (size_t grpid = 1; grpid < io_count - 1; grpid++) {
            auto &io_info = m_handle->io_info[grpid];
            auto &io_data = m_handle->io_data[grpid];

            size_t min_inputs = std::min(io_info->nInputSize, last_io_data.nInputSize);
            for (size_t i = 0; i < min_inputs; i++) {
                io_data.pInputs[i].phyAddr  = last_io_data.pInputs[i].phyAddr;
                io_data.pInputs[i].pVirAddr = last_io_data.pInputs[i].pVirAddr;
            }

            size_t min_outputs = std::min(io_info->nOutputSize, last_io_data.nOutputSize);
            for (size_t i = 0; i < min_outputs; i++) {
                io_data.pOutputs[i].phyAddr  = last_io_data.pOutputs[i].phyAddr;
                io_data.pOutputs[i].pVirAddr = last_io_data.pOutputs[i].pVirAddr;
            }
        }
    }

    for (size_t grpid = 0; grpid < io_count; grpid++) {
        auto &io_info = m_handle->io_info[grpid];
        auto &io_data = m_handle->io_data[grpid];

        for (size_t i = 0; i < io_info->nOutputSize; i++) {
            ax_runner_tensor_t tensor;
            tensor.nIdx     = i;
            tensor.sName    = io_info->pOutputs[i].pName ? std::string(io_info->pOutputs[i].pName) : "";
            tensor.nSize    = io_info->pOutputs[i].nSize;
            tensor.phyAddr  = io_data.pOutputs[i].phyAddr;
            tensor.pVirAddr = io_data.pOutputs[i].pVirAddr;
            for (size_t j = 0; j < io_info->pOutputs[i].nShapeSize; j++) {
                tensor.vShape.push_back(io_info->pOutputs[i].pShape[j]);
            }
            mgroup_output_tensors[grpid].push_back(tensor);
        }

        for (size_t i = 0; i < io_info->nInputSize; i++) {
            ax_runner_tensor_t tensor;
            tensor.nIdx     = i;
            tensor.sName    = io_info->pInputs[i].pName ? std::string(io_info->pInputs[i].pName) : "";
            tensor.nSize    = io_info->pInputs[i].nSize;
            tensor.phyAddr  = io_data.pInputs[i].phyAddr;
            tensor.pVirAddr = io_data.pInputs[i].pVirAddr;
            for (size_t j = 0; j < io_info->pInputs[i].nShapeSize; j++) {
                tensor.vShape.push_back(io_info->pInputs[i].pShape[j]);
            }
            mgroup_input_tensors[grpid].push_back(tensor);
        }
    }

    if (!mgroup_output_tensors.empty()) moutput_tensors = mgroup_output_tensors[0];
    if (!mgroup_input_tensors.empty()) minput_tensors = mgroup_input_tensors[0];

    build_tensor_maps();

    return 0;
}

int ax_runner_ax650::init(const char *model_file, bool use_mmap)
{
    if (use_mmap) {
        MMap model_buffer(model_file);
        if (!model_buffer.data()) return -1;
        auto ret = init((char *)model_buffer.data(), model_buffer.size());
        model_buffer.close_file();
        return ret;
    } else {
        char *model_buffer = nullptr;
        size_t len         = 0;
        if (!read_file(model_file, &model_buffer, &len)) return -1;
        auto ret = init(model_buffer, len);
        delete[] model_buffer;
        return ret;
    }
}

int ax_runner_ax650::init(char *model_buffer, size_t model_size)
{
    if (m_handle) deinit();
    m_handle = new ax_runner_ax650_handle_t;
    int ret  = AX_ENGINE_CreateHandle(&m_handle->handle, model_buffer, model_size);
    if (0 != ret) {
        ALOGE("AX_ENGINE_CreateHandle failed: 0x%x", ret);
        delete m_handle;
        m_handle = nullptr;
        return ret;
    }
    return sub_init();
}

void ax_runner_ax650::deinit()
{
    if (!m_handle) return;

    std::unordered_set<unsigned long> freed_phy_addrs;

    for (size_t g = 0; g < m_handle->io_data.size(); ++g) {
        auto &io = m_handle->io_data[g];

        if (io.pInputs) {
            for (size_t j = 0; j < io.nInputSize; ++j) {
                AX_ENGINE_IO_BUFFER_T *pBuf = io.pInputs + j;
                if (pBuf->phyAddr != 0) {
                    if (freed_phy_addrs.find(pBuf->phyAddr) == freed_phy_addrs.end()) {
                        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
                        freed_phy_addrs.insert(pBuf->phyAddr);
                    }
                }
            }
            delete[] io.pInputs;
            io.pInputs = nullptr;
        }

        if (io.pOutputs) {
            for (size_t j = 0; j < io.nOutputSize; ++j) {
                AX_ENGINE_IO_BUFFER_T *pBuf = io.pOutputs + j;
                if (pBuf->phyAddr != 0) {
                    if (freed_phy_addrs.find(pBuf->phyAddr) == freed_phy_addrs.end()) {
                        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
                        freed_phy_addrs.insert(pBuf->phyAddr);
                    }
                }
            }
            delete[] io.pOutputs;
            io.pOutputs = nullptr;
        }
    }

    if (m_handle->handle) {
        AX_ENGINE_DestroyHandle(m_handle->handle);
    }

    delete m_handle;
    m_handle = nullptr;

    moutput_tensors.clear();
    minput_tensors.clear();
    map_input_tensors.clear();
    map_output_tensors.clear();
    mgroup_output_tensors.clear();
    mgroup_input_tensors.clear();
    map_group_input_tensors.clear();
    map_group_output_tensors.clear();
}

int ax_runner_ax650::inference()
{
    if (!m_handle) return -1;
    for (size_t i = 0; i < get_num_inputs(); i++) {
        auto &tensor = get_input(i);
        AX_SYS_MflushCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }

    int ret = AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data[0]);

    for (size_t i = 0; i < get_num_outputs(); i++) {
        auto &tensor = get_output(i);
        AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }
    return ret;
}

int ax_runner_ax650::inference(int grpid)
{
    if (!m_handle) return -1;
    if (grpid < 0 || grpid >= (int)m_handle->io_data.size()) return -1;

    for (size_t i = 0; i < mgroup_input_tensors[grpid].size(); i++) {
        auto &tensor = mgroup_input_tensors[grpid][i];
        AX_SYS_MflushCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }

    int ret = AX_ENGINE_RunGroupIOSync(m_handle->handle, m_handle->context, grpid, &m_handle->io_data[grpid]);

    for (size_t i = 0; i < mgroup_output_tensors[grpid].size(); i++) {
        auto &tensor = mgroup_output_tensors[grpid][i];
        AX_SYS_MinvalidateCache(tensor.phyAddr, tensor.pVirAddr, tensor.nSize);
    }
    return ret;
}
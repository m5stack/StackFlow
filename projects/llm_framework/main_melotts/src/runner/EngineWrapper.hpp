/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/

#pragma once

#include <string>
#include <vector>
#include <cstring>
#include <array>
#include <cstdint>

#include "ax_engine_api.h"


class EngineWrapper {
public:
    EngineWrapper() :
            m_hasInit(false),
            m_handle(nullptr) {}

    ~EngineWrapper() {
        Release();
    }

    int Init(const char* strModelPath, uint32_t nNpuType = 0);

    int SetInput(void* pInput, int index);

    int Run();

    int GetOutput(void* pOutput, int index);

    int GetInputSize(int index);
    int GetOutputSize(int index);

    void* GetOutputPtr(int index);

    int Release();

protected:
    bool m_hasInit;
    AX_ENGINE_HANDLE m_handle;
    AX_ENGINE_IO_INFO_T *m_io_info{};
    AX_ENGINE_IO_T m_io{};
    int m_input_num{}, m_output_num{};
};
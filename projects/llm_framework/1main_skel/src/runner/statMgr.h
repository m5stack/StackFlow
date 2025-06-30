/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#ifndef _STAT_MGR_H_
#define _STAT_MGR_H_
#include "ax_global_type.h"
#include "ax_sys_api.h"
#include "ax_skel_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _STAT_OBJECT_NUM_T {
    AX_U32 nBodyNum;
    AX_U32 nVehicleNum;
    AX_U32 nCycleNum;
    AX_U32 nFaceNum;
    AX_U32 nPlateNum;
} STAT_OBJECT_NUM_T;

extern AX_VOID StatTrackMgr(const AX_SKEL_OBJECT_ITEM_T *pstObjectItems, STAT_OBJECT_NUM_T *pObjectNum);
extern AX_VOID StatPushMgr(const AX_SKEL_OBJECT_ITEM_T *pstObjectItems, STAT_OBJECT_NUM_T *pObjectNum);

#ifdef __cplusplus
}
#endif

#endif

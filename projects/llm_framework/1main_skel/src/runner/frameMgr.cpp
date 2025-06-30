/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <vector>
#include "frameMgr.h"
#include <iostream>

AX_BOOL g_bSkel_frame_mgr_inited = AX_FALSE;
AX_POOL g_skel_frame_poolId = AX_INVALID_POOLID;

AX_VOID FrameMgrCreate(AX_U32 nFrameSize, AX_U32 nDepth) {
    AX_POOL_CONFIG_T stPoolConfig;

    memset(&stPoolConfig, 0, sizeof(AX_POOL_CONFIG_T));
    stPoolConfig.MetaSize = 4096;
    stPoolConfig.BlkCnt = nDepth;
    stPoolConfig.BlkSize = nFrameSize;
    stPoolConfig.CacheMode = AX_POOL_CACHE_MODE_NONCACHE;
    memset(stPoolConfig.PartitionName, 0, sizeof(stPoolConfig.PartitionName));
    strcpy((AX_CHAR *)stPoolConfig.PartitionName, "anonymous");

    g_skel_frame_poolId = AX_POOL_CreatePool(&stPoolConfig);
    g_bSkel_frame_mgr_inited = AX_TRUE;
}

AX_BOOL FrameMgrGet(AX_U64 *YUVDataPhy, AX_VOID **YUVDataVir, AX_U32 nFrameSize, AX_U64 nFrameId, AX_BLK *nBlkId) {
    if (!g_bSkel_frame_mgr_inited
        || g_skel_frame_poolId == AX_INVALID_POOLID) {
        return AX_FALSE;
    }

    AX_BLK blkId = AX_POOL_GetBlock(g_skel_frame_poolId, nFrameSize, NULL);

    if (blkId != AX_INVALID_BLOCKID) {
        *YUVDataPhy = AX_POOL_Handle2PhysAddr(blkId);
        *YUVDataVir = AX_POOL_GetBlockVirAddr(blkId);
        *nBlkId = blkId;

        return AX_TRUE;
    }

    return AX_FALSE;
}

AX_VOID FrameMgrRelease(AX_BLK blkId) {
    if (blkId != AX_INVALID_BLOCKID) {
        blkId = AX_POOL_ReleaseBlock(blkId);
    }
}

AX_VOID FrameMgrDestroy(AX_VOID) {
    if (!g_bSkel_frame_mgr_inited
        || g_skel_frame_poolId == AX_INVALID_POOLID) {
        return;
    }

    AX_POOL_DestroyPool(g_skel_frame_poolId);

    g_skel_frame_poolId = AX_INVALID_POOLID;
    g_bSkel_frame_mgr_inited = AX_FALSE;
}

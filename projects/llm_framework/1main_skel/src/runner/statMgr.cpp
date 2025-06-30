/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#include <map>
#include "statMgr.h"
#include <string>

AX_VOID StatTrackMgr(const AX_SKEL_OBJECT_ITEM_T *pstObjectItems, STAT_OBJECT_NUM_T *pObjectNum) {
    if (!pstObjectItems || !pObjectNum) {
        return;
    }

    std::string strObjectCategory = pstObjectItems->pstrObjectCategory;

    if (pstObjectItems->eTrackState == AX_SKEL_TRACK_STATUS_NEW) {
        if (strObjectCategory == "body") {
            pObjectNum->nBodyNum ++;
        }
        else if (strObjectCategory == "vehicle") {
            pObjectNum->nVehicleNum ++;
        }
        else if (strObjectCategory == "cycle") {
            pObjectNum->nCycleNum ++;
        }
        else if (strObjectCategory == "face") {
            pObjectNum->nFaceNum ++;
        }
        else if (strObjectCategory == "plate") {
            pObjectNum->nPlateNum ++;
        }
    }
}

AX_VOID StatPushMgr(const AX_SKEL_OBJECT_ITEM_T *pstObjectItems, STAT_OBJECT_NUM_T *pObjectNum) {
    if (!pstObjectItems || !pObjectNum) {
        return;
    }

    std::string strObjectCategory = pstObjectItems->pstrObjectCategory;

    if (pstObjectItems->eTrackState == AX_SKEL_TRACK_STATUS_SELECT) {
        if (strObjectCategory == "body") {
            pObjectNum->nBodyNum ++;
        }
        else if (strObjectCategory == "vehicle") {
            pObjectNum->nVehicleNum ++;
        }
        else if (strObjectCategory == "cycle") {
            pObjectNum->nCycleNum ++;
        }
        else if (strObjectCategory == "face") {
            pObjectNum->nFaceNum ++;
        }
        else if (strObjectCategory == "plate") {
            pObjectNum->nPlateNum ++;
        }
    }
}

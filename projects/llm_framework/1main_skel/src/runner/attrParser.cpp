/**************************************************************************************************
 *
 * Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor Co., Ltd.
 *
 **************************************************************************************************/

#include "attrParser.h"
#include "picojson.h"

#define PICO_OBJECT get<picojson::object>()
#define PICO_OBJECT_SIZE PICO_OBJECT.size()
#define PICO_ARRAY get<picojson::array>()
#define PICO_ARRAY_SIZE PICO_ARRAY.size()
#define PICO_VALUE get<double>()
#define PICO_BOOL get<bool>()
#define PICO_STRING get<std::string>()
#define PICO_ROOT obj.PICO_OBJECT

typedef struct _AI_Face_Attr_t {
    AX_F32 fYaw;
    AX_F32 fPitch;
    AX_F32 fRoll;
    AX_F32 fMask;
    AX_U8 age;
    AX_U8 gender;
    AX_F32 fScore;
} AI_Face_Attr_t;

typedef struct _AI_Plate_Attr_t {
    AX_BOOL bExist;
    AX_BOOL bValid;
    /*
    string:
        blue
        yellow
        black
        white
        green
        small_new_energy
        large_new_energy
        absence
        unknown
    */
    std::string strPlateColor;
    /*
    string:
        one_row
        two_rows
        unknown
    */
    std::string strPlateType;
    /* string: UTF8*/
    std::string strPlateCode;

    _AI_Plate_Attr_t() {
        bExist = AX_FALSE;
        bValid = AX_FALSE;
        strPlateColor = "";
        strPlateType = "";
        strPlateCode = "";
    }
} AI_Plate_Attr_t;

static AX_VOID FaceAttrResult(const AX_SKEL_OBJECT_ITEM_T *pstObjectItems, ATTRINFO_T *pAttrInfo) {
    picojson::value obj;
    AI_Face_Attr_t face_attr;

    pAttrInfo->eType = ATTR_TYPE_FACE;

    for (size_t i = 0; i < pstObjectItems->nMetaInfoSize; i++) {
        if (!strcmp(pstObjectItems->pstMetaInfo[i].pstrType, "face_attr")) {
            std::string value = pstObjectItems->pstMetaInfo[i].pstrValue;
            std::string strParseRet = picojson::parse(obj, value);
            if (!strParseRet.empty() || !obj.is<picojson::object>()) {
                break;
            }
            // age
            face_attr.age = PICO_ROOT["age"].PICO_VALUE;
            printf("ageis %d\n",face_attr.age);
            // gender
            face_attr.gender = PICO_ROOT["gender"].PICO_VALUE;
            // yaw
            face_attr.fYaw = PICO_ROOT["yaw"].PICO_VALUE;
            // pitch
            face_attr.fPitch = PICO_ROOT["pitch"].PICO_VALUE;
            // roll
            face_attr.fRoll = PICO_ROOT["roll"].PICO_VALUE;
            // mask
            face_attr.fMask = PICO_ROOT["mask"].PICO_VALUE;
            // score
            face_attr.fScore = PICO_ROOT["score"].PICO_VALUE;
            printf("score is %f\n",face_attr.fScore);
        }
    }
}

static AX_VOID PlateAttrResult(const AX_SKEL_OBJECT_ITEM_T *pstObjectItems, ATTRINFO_T *pAttrInfo) {
    picojson::value obj;
    AI_Plate_Attr_t plat_attr;

    pAttrInfo->eType = ATTR_TYPE_PLATE;

    plat_attr.bExist = AX_FALSE;
    plat_attr.bValid = AX_FALSE;
    plat_attr.strPlateColor = "";
    plat_attr.strPlateType = "";
    plat_attr.strPlateCode = "";

    for (size_t i = 0; i < pstObjectItems->nMetaInfoSize; i++) {
        if (!strcmp(pstObjectItems->pstMetaInfo[i].pstrType, "plate_attr")) {
            std::string value = pstObjectItems->pstMetaInfo[i].pstrValue;
            std::string strParseRet = picojson::parse(obj, value);
            if (!strParseRet.empty() || !obj.is<picojson::object>()) {
                break;
            }

            plat_attr.bExist = AX_TRUE;
            // color
            plat_attr.strPlateColor = "unknown";
            if (PICO_ROOT.end() != PICO_ROOT.find("color")) {
                plat_attr.strPlateColor = PICO_ROOT["color"].PICO_OBJECT["name"].PICO_STRING;
            }

            // style
            plat_attr.strPlateType = "unknown";
            if (PICO_ROOT.end() != PICO_ROOT.find("style")) {
                plat_attr.strPlateType = PICO_ROOT["style"].PICO_OBJECT["name"].PICO_STRING;
            }

            // code
            plat_attr.strPlateCode = PICO_ROOT["code_result"].PICO_STRING;

            if (PICO_ROOT["code_killed"].PICO_BOOL) {
                plat_attr.bValid = AX_FALSE;
            } else {
                plat_attr.bValid = AX_TRUE;
            }

            pAttrInfo->bExist = AX_TRUE;

            strncpy(pAttrInfo->tPlateInfo.szColor, (const AX_CHAR *)plat_attr.strPlateColor.c_str(), sizeof(pAttrInfo->tPlateInfo.szColor) - 1);

            pAttrInfo->tPlateInfo.bValid = plat_attr.bValid;
            strncpy(pAttrInfo->tPlateInfo.szNum, (const AX_CHAR *)plat_attr.strPlateCode.c_str(), sizeof(pAttrInfo->tPlateInfo.szNum) - 1);
        }
    }
}

static AX_VOID VehicleAttrResult(const AX_SKEL_OBJECT_ITEM_T *pstObjectItems, ATTRINFO_T *pAttrInfo) {
    picojson::value obj;
    AI_Plate_Attr_t plat_attr;

    pAttrInfo->eType = ATTR_TYPE_VEHICLE;

    plat_attr.bExist = AX_FALSE;
    plat_attr.bValid = AX_FALSE;
    plat_attr.strPlateColor = "";
    plat_attr.strPlateType = "";
    plat_attr.strPlateCode = "";

    for (size_t i = 0; i < pstObjectItems->nMetaInfoSize; i++) {
        if (!strcmp(pstObjectItems->pstMetaInfo[i].pstrType, "plate_attr")) {
            std::string value = pstObjectItems->pstMetaInfo[i].pstrValue;
            std::string strParseRet = picojson::parse(obj, value);
            if (!strParseRet.empty() || !obj.is<picojson::object>()) {
                break;
            }

            plat_attr.bExist = AX_TRUE;
            // color
            plat_attr.strPlateColor = "unknown";
            if (PICO_ROOT.end() != PICO_ROOT.find("color")) {
                plat_attr.strPlateColor = PICO_ROOT["color"].PICO_OBJECT["name"].PICO_STRING;
            }

            // style
            plat_attr.strPlateType = "unknown";
            if (PICO_ROOT.end() != PICO_ROOT.find("style")) {
                plat_attr.strPlateType = PICO_ROOT["style"].PICO_OBJECT["name"].PICO_STRING;
            }

            // code
            plat_attr.strPlateCode = PICO_ROOT["code_result"].PICO_STRING;

            if (PICO_ROOT["code_killed"].PICO_BOOL) {
                plat_attr.bValid = AX_FALSE;
            } else {
                plat_attr.bValid = AX_TRUE;
            }

            pAttrInfo->bExist = AX_TRUE;

            strncpy(pAttrInfo->tPlateInfo.szColor, (const AX_CHAR *)plat_attr.strPlateColor.c_str(), sizeof(pAttrInfo->tPlateInfo.szColor) - 1);

            pAttrInfo->tPlateInfo.bValid = plat_attr.bValid;
            strncpy(pAttrInfo->tPlateInfo.szNum, (const AX_CHAR *)plat_attr.strPlateCode.c_str(), sizeof(pAttrInfo->tPlateInfo.szNum) - 1);
        }
    }
}

AX_VOID AttrParser(const AX_SKEL_OBJECT_ITEM_T *pstObjectItems, ATTRINFO_T *pAttrInfo) {
    if (!pstObjectItems || !pAttrInfo) {
        return;
    }

    pAttrInfo->bExist = AX_FALSE;
    pAttrInfo->eType = ATTR_TYPE_BUTT;

    std::string strObjectCategory = pstObjectItems->pstrObjectCategory;

    if (strObjectCategory == "face") {
        FaceAttrResult(pstObjectItems, pAttrInfo);
    }
    else if (strObjectCategory == "plate") {
        PlateAttrResult(pstObjectItems, pAttrInfo);
    }
    else if (strObjectCategory == "vehicle") {
        VehicleAttrResult(pstObjectItems, pAttrInfo);
    }
}

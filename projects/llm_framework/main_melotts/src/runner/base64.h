#pragma once

#include <string.h>
#include <stdlib.h>
#include <wchar.h>
#include <assert.h>
#include <iostream>
typedef unsigned char uint8;
typedef unsigned long base64_uint32;
// base64_uint32 base64_encode(char* input, uint8* encode);
int base64_decode(const uint8* code, base64_uint32 code_len, char* str);
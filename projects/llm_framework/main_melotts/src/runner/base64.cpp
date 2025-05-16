#include "base64.h"

static uint8 alphabet_map[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static uint8 reverse_map[]  = {
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 62,
    255, 255, 255, 63,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  255, 255, 255, 255, 255, 255, 255, 0,
    1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
    23,  24,  25,  255, 255, 255, 255, 255, 255, 26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
    39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  255, 255, 255, 255, 255};

// //GB2312到UTF-8的转换
// char* G2U(const char* gb2312)
// {
// 	int len = MultiByteToWideChar(CP_ACP, 0, gb2312, -1, NULL, 0);
// 	wchar_t* wstr = new wchar_t[len + 1];
// 	memset(wstr, 0, len + 1);
// 	MultiByteToWideChar(CP_ACP, 0, gb2312, -1, wstr, len);
// 	len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
// 	char* str = new char[len + 1];
// 	memset(str, 0, len + 1);
// 	WideCharToMultiByte(CP_UTF8, 0, wstr, -1, str, len, NULL, NULL);
// 	if (wstr) delete[] wstr;
// 	return str;
// }
// //UTF-8到GB2312的转换
// char* U2G(const char* utf8)
// {
// 	int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
// 	wchar_t* wstr = new wchar_t[len + 1];
// 	memset(wstr, 0, len + 1);
// 	MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wstr, len);
// 	len = WideCharToMultiByte(CP_ACP, 0, wstr, -1, NULL, 0, NULL, NULL);
// 	char* str = new char[len + 1];
// 	memset(str, 0, len + 1);
// 	WideCharToMultiByte(CP_ACP, 0, wstr, -1, str, len, NULL, NULL);
// 	if (wstr) delete[] wstr;
// 	return str;
// }

// base64_uint32 base64_encode(char* input, uint8* encode)
// {
// 	//1、包含中文的字符串 字符编码（windows默认是gbk）转换成unicode

// 	//2、字符编码方式是utf-8的二进制
// 	// uint8* text = (uint8*)G2U(input);
// 	base64_uint32 text_len = (base64_uint32)strlen((char*)input);

// 	base64_uint32 i, j;
// 	for (i = 0, j = 0; i + 3 <= text_len; i += 3)
// 	{
// 		encode[j++] = alphabet_map[text[i] >> 2]; //取出第一个字符的前6位并找出对应的结果字符 		encode[j++] =
// alphabet_map[((text[i] << 4) & 0x30) | (text[i + 1] >> 4)];
// //将第一个字符的后2位与第二个字符的前4位进行组合并找到对应的结果字符 		encode[j++] = alphabet_map[((text[i + 1] << 2) &
// 0x3c) | (text[i + 2] >> 6)];   //将第二个字符的后4位与第三个字符的前2位组合并找出对应的结果字符 		encode[j++] =
// alphabet_map[text[i + 2] & 0x3f];                         //取出第三个字符的后6位并找出结果字符
// 	}

// 	if (i < text_len)
// 	{
// 		base64_uint32 tail = text_len - i;
// 		if (tail == 1)
// 		{
// 			encode[j++] = alphabet_map[text[i] >> 2];
// 			encode[j++] = alphabet_map[(text[i] << 4) & 0x30];
// 			encode[j++] = '=';
// 			encode[j++] = '=';
// 		}
// 		else //tail==2
// 		{
// 			encode[j++] = alphabet_map[text[i] >> 2];
// 			encode[j++] = alphabet_map[((text[i] << 4) & 0x30) | (text[i + 1] >> 4)];
// 			encode[j++] = alphabet_map[(text[i + 1] << 2) & 0x3c];
// 			encode[j++] = '=';
// 		}
// 	}
// 	encode[j] = 0;
// 	return j;
// }

int base64_decode(const uint8* code, base64_uint32 code_len, char* str)
{
    uint8 plain[1024];
    assert((code_len & 0x03) == 0);  // 如果它的条件返回错误，则终止程序执行。4的倍数。

    base64_uint32 i, j = 0;
    uint8 quad[4];
    for (i = 0; i < code_len; i += 4) {
        for (base64_uint32 k = 0; k < 4; k++) {
            quad[k] = reverse_map[code[i + k]];  // 分组，每组四个分别依次转换为base64表内的十进制数
        }

        assert(quad[0] < 64 && quad[1] < 64);

        plain[j++] =
            (quad[0] << 2) |
            (quad[1] >>
             4);  // 取出第一个字符对应base64表的十进制数的前6位与第二个字符对应base64表的十进制数的前2位进行组合

        if (quad[2] >= 64)
            break;
        else if (quad[3] >= 64) {
            plain[j++] =
                (quad[1] << 4) |
                (quad[2] >>
                 2);  // 取出第二个字符对应base64表的十进制数的后4位与第三个字符对应base64表的十进制数的前4位进行组合
            break;
        } else {
            plain[j++] = (quad[1] << 4) | (quad[2] >> 2);
            plain[j++] = (quad[2] << 6) | quad[3];  // 取出第三个字符对应base64表的十进制数的后2位与第4个字符进行组合
        }
    }
    plain[j] = 0;
    // char str[1024] = "";
    strcpy(str, (char*)plain);
    // strcpy_s(str, sizeof(plain), U2G(str));
    return j;
}
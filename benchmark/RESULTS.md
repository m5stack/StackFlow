# Results

## ModuleLLM (AX630C)

### LLM
| model                           | ttft (ms)  | avg-token/s | model version | llm version |
|---------------------------------|------------|-------------|---------------|-------------|
| qwen2.5-0.5B-prefill-20e        | 359.8      | 10.32       | v0.2          | v1.8        |
| qwen2.5-0.5B-p256-ax630c        | 1126.19    | 10.30       | v0.4          | v1.8        |
| qwen2.5-0.5B-Int4-ax630c        | 442.95     | 12.52       | v0.4          | v1.8        |
| qwen2.5-coder-0.5B-ax630c       | 361.81     | 10.28       | v0.2          | v1.8        |
| qwen2.5-1.5B-ax630c             | 1029.41    | 3.59        | v0.3          | v1.8        |
| qwen2.5-1.5B-p256-ax630c        | 3056.54    | 3.57        | v0.4          | v1.8        |
| qwen2.5-1.5B-Int4-ax630c        | 1219.54    | 4.63        | v0.4          | v1.8        |
| deepseek-r1-1.5B-ax630c         | 1075.04    | 3.57        | v0.3          | v1.8        |
| deepseek-r1-1.5B-p256-ax630c    | 3056.86    | 3.57        | v0.4          | v1.8        |
| llama3.2-1B-prefill-ax630c      | 891.00     | 4.48        | v0.2          | v1.8        |
| llama3.2-1B-p256-ax630c         | 2601.11    | 4.49        | v0.4          | v1.8        |
| openbuddy-llama3.2-1B-ax630c    | 891.02     | 4.52        | v0.2          | v1.8        |

`The input text used by the llm test is "hello!“`

### VLM
| model                           | ttft (ms)  | avg-token/s | image encode (ms) | model version | vlm version |
|---------------------------------|------------|-------------|-------------------|---------------|-------------|
| internvl2.5-1B-364-ax630c       | 1117.27    | 10.56       | 1164.61           | v0.4          | v1.7        |
| smolvlm-256M-ax630c             | 185.75     | 30.16       | 799.11            | v0.4          | v1.7        |
| smolvlm-500M-ax630c             | 365.69     | 13.14       | 838.30            | v0.4          | v1.7        |

`The image encoding test uses a jpg image with a size of 810*1080`

### STT
| model              | encode (ms) | avg-decode (ms) | model version | whisper version |
|--------------------|-------------|-----------------|---------------|-----------------|
| whisper-tiny       | 248.0       | 32.54           | v0.4          | v1.7            |
| whisper-base       | 660.31      | 51.11           | v0.4          | v1.7            |
| whisper-small      | 1606.08     | 148.92          | v0.4          | v1.7            |

`The STT test uses a 30-second wav English audio`

## Pyramid (AX650C)

### LLM
| model                           | ttft (ms)  | avg-token/s | model version | llm version |
|---------------------------------|------------|-------------|---------------|-------------|
| qwen2.5-0.5B-Int4-ax650         | 136.65     | 40.01       | v0.4          | v1.8        |
| qwen2.5-1.5B-Int4-ax650         | 285.93     | 17.62       | v0.4          | v1.8        |
| qwen2.5-3B-Int4-ax650           | 546.34     | 9.79        | v0.4          | v1.8        |
| qwen2.5-7B-Int4-ax650           | -          | -           | v0.4          | v1.8        |
| deepseek-r1-1.5B-Int4-ax650     | 281.88     | 18.31       | v0.4          | v1.8        |
| deepseek-r1-7B-Int4-ax650       | -          | -           | v0.4          | v1.8        |

`The input text used by the llm test is "hello!“`

### VLM
| model                           | ttft (ms)  | avg-token/s | image encode (ms) | model version | vlm version |
|---------------------------------|------------|-------------|-------------------|---------------|-------------|
| internvl2.5-1B-448-ax650        | 430.86     | 29.76       | 370.82            | v0.4          | v1.7        |
| smolvlm-256M-ax650              | 57.79      | 74.22       | 104.16            | v0.4          | v1.7        |
| smolvlm-500M-ax650              | -          | -           | -                 | v0.4          | v1.7        |

`The image encoding test uses a jpg image with a size of 810*1080`


### STT
| model              | encode (ms) | avg-decode (ms) | model version | whisper version |
|--------------------|-------------|-----------------|---------------|-----------------|
| whisper-tiny-ax650 | 21.19       | 5.72            | v0.4          | v1.7            |
| whisper-base-ax650 | 50.1        | 9.93            | v0.4          | v1.7            |
| whisper-small-ax650| 188.98      | 29.77           | v0.4          | v1.7            |

`The STT test uses a 30-second wav English audio`

## Sipeed (AX650N)

### LLM
| model                           | ttft (ms)  | avg-token/s | model version | llm version |
|---------------------------------|------------|-------------|---------------|-------------|
| qwen2.5-0.5B-Int4-ax650         | 134.49     | 42.34       | v0.4          | v1.8        |
| qwen2.5-1.5B-Int4-ax650         | 282.07     | 18.31       | v0.4          | v1.8        |
| qwen2.5-3B-Int4-ax650           | 543.67     | 9.73        | v0.4          | v1.8        |
| qwen2.5-7B-Int4-ax650           | 1136.33    | 4.54        | v0.4          | v1.8        | 4335MB CMM
| deepseek-r1-1.5B-Int4-ax650     | 281.80     | 18.37       | v0.4          | v1.8        |
| deepseek-r1-7B-Int4-ax650       | -          | -           | v0.4          | v1.8        |

`The input text used by the llm test is "hello!“`

### VLM
| model                           | ttft (ms)  | avg-token/s | image encode (ms) | model version | vlm version |
|---------------------------------|------------|-------------|-------------------|---------------|-------------|
| internvl2.5-1B-448-ax650        | 427.35     | 30.44       | 371.7             | v0.4          | v1.7        |
| smolvlm-256M-ax650              | 57.31      | 74.16       | 106.44            | v0.4          | v1.7        |

`The image encoding test uses a jpg image with a size of 810*1080`


### STT
| model              | encode (ms) | avg-decode (ms) | model version | whisper version |
|--------------------|-------------|-----------------|---------------|-----------------|
| whisper-tiny-ax650 | 21.16       | 5.62            | v0.4          | v1.7            |
| whisper-base-ax650 | 50.21       | 9.68            | v0.4          | v1.7            |
| whisper-small-ax650| 188.69      | 28.34           | v0.4          | v1.7            |

`The STT test uses a 30-second wav English audio`
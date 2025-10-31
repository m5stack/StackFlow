# llm_cosy_voice

使用 npu 加速的文字转语音单元，用于提供文字转语音服务，可使用语音克隆，用于提供多语言转语音服务。

## setup

配置单元工作。

发送 json：

```json
cosy_voice
{
  "request_id": "2",
  "work_id": "cosy_voice",
  "action": "setup",
  "object": "cosy_voice.setup",
  "data": {
    "model": "CosyVoice2-0.5B-axcl",
    "response_format": "file",
    "input": "tts.utf-8",
    "enoutput": false
  }
}
```


- request_id：参考基本数据解释。
- work_id：配置单元时，为 `cosy_voice`。
- action：调用的方法为 `setup`。
- object：传输的数据类型为 `cosy_voice.setup`。
- model：使用的模型为 `CosyVoice2-0.5B-axcl` 模型。
- prompt_files：要克隆的音频信息文件。
- response_format：返回结果为 `sys.pcm`, 系统音频数据，并直接发送到 llm-audio 模块进行播放。返回结果为 `file`, 生成的音频写 wav 文件，可用 `prompt_data` 指定路径或文件名。
- input：输入的为 `tts.utf-8`,代表的是从用户输入。
- enoutput：是否起用用户结果输出。

响应 json：

```json
{
    "created": 1761791627,
    "data": "None",
    "error": {
        "code": 0,
        "message": ""
    },
    "object": "None",
    "request_id": "2",
    "work_id": "cosy_voice.1000"
}
```

- created：消息创建时间，unix 时间。
- work_id：返回成功创建的 work_id 单元。

## inference

### 流式输入

```json
{
    "request_id": "2",
    "work_id": "cosy_voice.1000",
    "action": "inference",
    "object": "cosy_voice.utf-8.stream",
    "data": {
        "delta": "今天天气真好！",
        "index": 0,
        "finish": true
    }
}
```
- object：传输的数据类型为 `cosy_voice.utf-8.stream` 代表的是从用户 utf-8 的流式输入
- delta：流式输入的分段数据
- index：流式输入的分段索引
- finish:流式输入是否完成的标志位

### 非流式输入

```json
{
    "request_id": "2",
    "work_id": "cosy_voice.1000",
    "action": "inference",
    "object": "cosy_voice.utf-8",
    "data": "今天天气真好！"
}
```
- object：传输的数据类型为 `cosy_voice.utf-8` 代表的是从用户 utf-8 的非流式输入
- data：非流式输入的数据

## pause

暂停单元工作。

发送 json：

```json
{
  "request_id": "5",
  "work_id": "cosy_voice.1000",
  "action": "pause"
}
```

响应 json：

```json
{
    "created": 1761791706,
    "data": "None",
    "error": {
        "code": 0,
        "message": ""
    },
    "object": "None",
    "request_id": "5",
    "work_id": "cosy_voice.1000"
}
```

error::code 为 0 表示执行成功。

## exit

单元退出。

发送 json：

```json
{
  "request_id": "7",
  "work_id": "cosy_voice.1000",
  "action": "exit"
}
```

响应 json：

```json
{
    "created": 1761791854,
    "data": "None",
    "error": {
        "code": 0,
        "message": ""
    },
    "object": "None",
    "request_id": "7",
    "work_id": "cosy_voice.1000"
}
```

error::code 为 0 表示执行成功。

## taskinfo

获取任务列表。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "cosy_voice",
  "action": "taskinfo"
}
```

响应 json：

```json
{
    "created": 1761791739,
    "data": [
        "cosy_voice.1000"
    ],
    "error": {
        "code": 0,
        "message": ""
    },
    "object": "llm.tasklist",
    "request_id": "2",
    "work_id": "cosy_voice"
}
```

获取任务运行参数。

```json
{
  "request_id": "2",
  "work_id": "cosy_voice.1000",
  "action": "taskinfo"
}
```

响应 json：

```json
{
    "created": 1761791761,
    "data": {
        "enoutput": false,
        "inputs": [
            "tts.utf-8"
        ],
        "model": "CosyVoice2-0.5B-axcl",
        "response_format": "sys.pcm"
    },
    "error": {
        "code": 0,
        "message": ""
    },
    "object": "cosy_voice.taskinfo",
    "request_id": "2",
    "work_id": "cosy_voice.1000"
}
```

> **注意：work_id 是按照单元的初始化注册顺序增加的，并不是固定的索引值。**  
> **同类型单元不能配置多个单元同时工作，否则会产生未知错误。例如 tts 和 melo tts 不能同时拍起用工作。**

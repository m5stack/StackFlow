# llm-asr

语音转文字单元，用于提供语音转文字服务，可选择中英文模型，用于提供中英文语音转文字服务。

## setup

配置单元工作。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "asr",
  "action": "setup",
  "object": "asr.setup",
  "data": {
    "model": "sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23",
    "response_format": "asr.utf-8.stream",
    "input": "sys.pcm",
    "enoutput": true,
    "endpoint_config.rule1.min_trailing_silence": 2.4,
    "endpoint_config.rule2.min_trailing_silence": 1.2,
    "endpoint_config.rule3.min_trailing_silence": 30.1,
    "endpoint_config.rule1.must_contain_nonsilence": true,
    "endpoint_config.rule2.must_contain_nonsilence": true,
    "endpoint_config.rule3.must_contain_nonsilence": true
  }
}
```

- request_id：参考基本数据解释。
- work_id：配置单元时，为 `asr`。
- action：调用的方法为 `setup`。
- object：传输的数据类型为 `asr.setup`。
- model：使用的模型为 `sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23` 中文模型。
- response_format：返回结果为 `asr.utf-8.stream`, utf-8 的流式输出。
- input：输入的为 `sys.pcm`,代表的是系统音频。
- enoutput：是否起用用户结果输出。
- endpoint_config.rule1.min_trailing_silence：唤醒后 2.4 s 后语音产生断点。
- endpoint_config.rule2.min_trailing_silence：识别语音后停顿 1.2 s 后语音产生断点。
- endpoint_config.rule3.min_trailing_silence：最长能够识别 30.1 s 后语音产生断点。

响应 json：

```json
{
  "created": 1731488402,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "2",
  "work_id": "asr.1001"
}
```

- created：消息创建时间，unix 时间。
- work_id：返回成功创建的 work_id 单元。

## link

链接上级单元的输出。

发送 json：

```json
{
  "request_id": "3",
  "work_id": "asr.1001",
  "action": "link",
  "object": "work_id",
  "data": "kws.1000"
}
```

响应 json：

```json
{
  "created": 1731488402,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "3",
  "work_id": "asr.1001"
}
```

error::code 为 0 表示执行成功。

将 asr 和 kws 单元链接起来，当 kws 发出唤醒数据时，asr 单元开始识别用户的语音，识别完成后自动暂停，直到下一次的唤醒。

> **link 时必须保证 kws 此时已经配置好进入工作状态。link 也可以在 setup 阶段进行。**

示例:

```json
{
  "request_id": "2",
  "work_id": "asr",
  "action": "setup",
  "object": "asr.setup",
  "data": {
    "model": "sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23",
    "response_format": "asr.utf-8.stream",
    "input": [
      "sys.pcm",
      "kws.1000"
    ],
    "enoutput": true,
    "endpoint_config.rule1.min_trailing_silence": 2.4,
    "endpoint_config.rule2.min_trailing_silence": 1.2,
    "endpoint_config.rule3.min_trailing_silence": 30.1,
    "endpoint_config.rule1.must_contain_nonsilence": true,
    "endpoint_config.rule2.must_contain_nonsilence": true,
    "endpoint_config.rule3.must_contain_nonsilence": true
  }
}
```

## unlink

取消链接。

发送 json：

```json
{
  "request_id": "4",
  "work_id": "asr.1001",
  "action": "unlink",
  "object": "work_id",
  "data": "kws.1000"
}
```

响应 json：

```json
{
  "created": 1731488402,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "4",
  "work_id": "asr.1001"
}
```

error::code 为 0 表示执行成功。

## pause

暂停单元工作。

发送 json：

```json
{
  "request_id": "5",
  "work_id": "asr.1001",
  "action": "pause"
}
```

响应 json：

```json
{
  "created": 1731488402,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "5",
  "work_id": "asr.1001"
}
```

error::code 为 0 表示执行成功。

## work

恢复单元工作。

发送 json：

```json
{
  "request_id": "6",
  "work_id": "asr.1001",
  "action": "work"
}
```

响应 json：

```json
{
  "created": 1731488402,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "6",
  "work_id": "asr.1001"
}
```

error::code 为 0 表示执行成功。

## exit

单元退出。

发送 json：

```json
{
  "request_id": "7",
  "work_id": "asr.1001",
  "action": "exit"
}
```

响应 json：

```json
{
  "created": 1731488402,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "7",
  "work_id": "asr.1001"
}
```

error::code 为 0 表示执行成功。

## taskinfo

获取任务列表。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "asr",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1731580350,
  "data": [
    "asr.1001"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "asr.tasklist",
  "request_id": "2",
  "work_id": "asr"
}
```

获取任务运行参数。

```json
{
  "request_id": "2",
  "work_id": "asr.1001",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1731579679,
  "data": {
    "enoutput": false,
    "inputs_": [
      "sys.pcm"
    ],
    "model": "sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23",
    "response_format": "asr.utf-8-stream"
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "asr.taskinfo",
  "request_id": "2",
  "work_id": "asr.1001"
}
```

> **注意：work_id 是按照单元的初始化注册顺序增加的，并不是固定的索引值。**
# llm-kws

语音唤醒单元，用于提供语音唤醒服务，可选择中英文模型，用于提供中英文唤醒服务。

## setup

配置单元工作。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "kws",
  "action": "setup",
  "object": "kws.setup",
  "data": {
    "model": "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
    "response_format": "kws.bool",
    "input": "sys.pcm",
    "enoutput": true,
    "kws": "你好你好",
    "enwake_audio": true
  }
}
```

- request_id：参考基本数据解释。
- work_id：配置单元时，为 `kws`。
- action：调用的方法为 `setup`。
- object：传输的数据类型为 `kws.setup`。
- model：使用的模型为 `sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01` 中文模型。
- response_format：返回结果为 `kws.bool` 型。
- input：输入的为 `sys.pcm`,代表的是系统音频。
- enoutput：是否起用用户结果输出。
- kws：中文唤醒词为 `"你好你好"`。
- enwake_audio：是否起用唤醒音频输出。默认是 true

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
  "work_id": "kws.1000"
}
```

- created：消息创建时间，unix 时间。
- work_id：返回成功创建的 work_id 单元。

## pause

暂停单元工作。

发送 json：

```json
{
  "request_id": "3",
  "work_id": "kws.1000",
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
  "request_id": "3",
  "work_id": "kws.1000"
}
```

error::code 为 0 表示执行成功。

## work

恢复单元工作。

发送 json：

```json
{
  "request_id": "4",
  "work_id": "kws.1000",
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
  "request_id": "4",
  "work_id": "kws.1000"
}
```

error::code 为 0 表示执行成功。

## exit

单元退出。

发送 json：

```json
{
  "request_id": "5",
  "work_id": "kws.1000",
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
  "request_id": "5",
  "work_id": "kws.1000"
}
```

error::code 为 0 表示执行成功。

## taskinfo

获取任务列表。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "kws",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1731580350,
  "data": [
    "kws.1000"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "kws.tasklist",
  "request_id": "2",
  "work_id": "kws"
}
```

获取任务运行参数。

```json
{
  "request_id": "2",
  "work_id": "kws.1000",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1731652086,
  "data": {
    "enoutput": true,
    "inputs_": [
      "sys.pcm"
    ],
    "model": "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01",
    "response_format": "kws.bool"
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "kws.taskinfo",
  "request_id": "2",
  "work_id": "kws.1000"
}
```

> **注意：work_id 是按照单元的初始化注册顺序增加的，并不是固定的索引值。**

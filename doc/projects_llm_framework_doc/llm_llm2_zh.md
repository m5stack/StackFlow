# llm-llm2

大模型单元，用于提供新一代大模型推理服务。

相较于 `llm-llm`，`llm-llm2` 主要面向 `main_llm2` 中的 Qwen3 / Qwen3.5 系列模型，并支持在单次请求中动态覆盖采样参数，例如 `temperature`、`top_p`、`top_k`。

## setup

配置单元工作。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "llm2",
  "action": "setup",
  "object": "llm.setup",
  "data": {
    "model": "Qwen3.5-0.8B-Int4-ax650",
    "response_format": "llm.utf-8.stream",
    "input": "llm.utf-8",
    "enoutput": true,
    "prompt": "You are a helpful assistant."
  }
}
```

- request_id：参考基本数据解释。
- work_id：配置单元时，为 `llm2`。
- action：调用的方法为 `setup`。
- object：传输的数据类型为 `llm.setup`。
- model：使用的模型名称，例如 `Qwen3.5-0.8B-Int4-ax650`。
- response_format：返回结果格式，例如 `llm.utf-8.stream`。
- input：输入类型，通常为 `llm.utf-8`、`llm.utf-8.stream`，也可以在 setup 时配置为数组。
- enoutput：是否启用用户结果输出。
- prompt：模型的系统提示词。

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
  "work_id": "llm2.1002"
}
```

- created：消息创建时间，unix 时间。
- work_id：返回成功创建的 work_id 单元。

## inference

### 流式输入

普通流式输入：

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "你可以用英语讲一个故事吗？",
    "index": 0,
    "finish": true
  }
}
```

- object：传输的数据类型为 `llm.utf-8.stream`，代表从用户输入 utf-8 流式文本。
- delta：流式输入的分段数据。
- index：流式输入的分段索引。
- finish：流式输入是否完成的标志位。

### 流式输入并动态覆盖采样参数

如果希望在单次请求中动态覆盖采样参数，需要将 `data.delta` 改为一个 JSON 字符串。

示例：覆盖 `temperature`

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "{\"prompt\":\"你可以用英语讲一个故事吗？\",\"temperature\":0.7}",
    "index": 0,
    "finish": true
  }
}
```

示例：覆盖 `top_p`

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "{\"prompt\":\"你可以用英语讲一个故事吗？\",\"top_p\":0.8}",
    "index": 0,
    "finish": true
  }
}
```

示例：覆盖 `top_k`

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "{\"prompt\":\"你可以用英语讲一个故事吗？\",\"top_k\":10}",
    "index": 0,
    "finish": true
  }
}
```

示例：同时覆盖多个参数

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "{\"prompt\":\"你可以用英语讲一个故事吗？\",\"temperature\":0.7,\"top_p\":0.8}",
    "index": 0,
    "finish": true
  }
}
```

说明：

- 不传 `temperature`、`top_p`、`top_k` 时，使用模型默认的 `post_config.json` 配置。
- 请求中只会覆盖显式传入的字段，未传入字段仍然沿用默认值。
- 若同时传入 `top_p` 和 `top_k`，当前实现优先使用 `top_p`。

### 非流式输入

普通非流式输入：

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8",
  "data": "你可以用英语讲一个故事吗？"
}
```

如果需要在非流式请求中动态覆盖采样参数，则可以将 `data` 直接写成 JSON 字符串：

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8",
  "data": "{\"prompt\":\"你可以用英语讲一个故事吗？\",\"temperature\":0.7,\"top_k\":10}"
}
```

### 响应 json

流式响应 json：

```json
{"created":1742779468,"data":{"delta":"从前","finish":false,"index":0},"error":{"code":0,"message":""},"object":"llm.utf-8.stream","request_id":"2","work_id":"llm2.1002"}
{"created":1742779469,"data":{"delta":"有一个","finish":false,"index":1},"error":{"code":0,"message":""},"object":"llm.utf-8.stream","request_id":"2","work_id":"llm2.1002"}
{"created":1742779470,"data":{"delta":"小故事。","finish":false,"index":2},"error":{"code":0,"message":""},"object":"llm.utf-8.stream","request_id":"2","work_id":"llm2.1002"}
{"created":1742779470,"data":{"delta":"","finish":true,"index":3},"error":{"code":0,"message":""},"object":"llm.utf-8.stream","request_id":"2","work_id":"llm2.1002"}
```

非流式响应 json：

```json
{
  "created": 1742780120,
  "data": "Once upon a time, there was a little story...",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "llm.utf-8",
  "request_id": "2",
  "work_id": "llm2.1002"
}
```

## link

链接上级单元的输出。

发送 json：

```json
{
  "request_id": "3",
  "work_id": "llm2.1002",
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
  "work_id": "llm2.1002"
}
```

error::code 为 0 表示执行成功。

## unlink

取消链接。

发送 json：

```json
{
  "request_id": "4",
  "work_id": "llm2.1002",
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
  "work_id": "llm2.1002"
}
```

error::code 为 0 表示执行成功。

## pause

暂停单元工作。

发送 json：

```json
{
  "request_id": "5",
  "work_id": "llm2.1002",
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
  "work_id": "llm2.1002"
}
```

error::code 为 0 表示执行成功。

## exit

退出单元工作。

发送 json：

```json
{
  "request_id": "7",
  "work_id": "llm2.1002",
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
  "work_id": "llm2.1002"
}
```

error::code 为 0 表示执行成功。

## taskinfo

获取任务列表。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "llm2",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1731652149,
  "data": [
    "llm2.1002"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "llm.tasklist",
  "request_id": "2",
  "work_id": "llm2"
}
```

获取任务运行参数。

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1731652187,
  "data": {
    "enoutput": true,
    "inputs": [
      "llm.utf-8"
    ],
    "model": "Qwen3.5-0.8B-Int4-ax650",
    "response_format": "llm.utf-8.stream"
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "llm.taskinfo",
  "request_id": "2",
  "work_id": "llm2.1002"
}
```

> 注意：`work_id` 是按照单元的初始化注册顺序增加的，并不是固定的索引值。
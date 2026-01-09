# llm-vlm

大模型单元，用于提供多模态大模型推理服务。

## setup

配置单元工作。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "vlm",
  "action": "setup",
  "object": "vlm.setup",
  "data": {
    "model": "internvl2.5-1B-364-ax630c",
    "response_format": "vlm.utf-8.stream",
    "input": "vlm.utf-8",
    "enoutput": true,
    "max_token_len": 256,
    "prompt": "You are a knowledgeable assistant capable of answering various questions and providing information."
  }
}
```

- request_id：参考基本数据解释。
- work_id：配置单元时，为 `vlm`。
- action：调用的方法为 `setup`。
- object：传输的数据类型为 `vlm.setup`。
- model：使用的模型为 `internvl2.5-1B-364-ax630c` 多模态模型。
- response_format：返回结果为 `vlm.utf-8.stream`, utf-8 的流式输出。
- input：输入的为 `vlm.utf-8`,代表的是从用户输入。
- enoutput：是否起用用户结果输出。
- max_token_len：最大输出 token,该值的最大值受到模型的最大限制。
- prompt：模型的系统提示词。

响应 json：

```json
{
  "created": 1737599810,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "2",
  "work_id": "vlm.1003"
}
```

- created：消息创建时间，unix 时间。
- work_id：返回成功创建的 work_id 单元。

## inference

推理用户数据

发送 json：

```json
{
  "request_id": "4",
  "work_id": "vlm.1003",
  "action": "inference",
  "object": "vlm.utf-8.stream",
  "data": {
    "delta": "May i know your name?",
    "index": 0,
    "finish": true
  }
}
```

非流式响应 json：

```json
{
  "created": 1737600915,
  "data": "I am an AI assistant whose name is LittleAI. How can I help you today?",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "vlm.utf-8",
  "request_id": "4",
  "work_id": "vlm.1003"
}
```

流式响应

```json
{
  "created": 1737600539,
  "data": {
    "delta": "I am an",
    "finish": false,
    "index": 0
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "vlm.utf-8.stream",
  "request_id": "4",
  "work_id": "vlm.1003"
}
```

```json
{
  "created": 1737600539,
  "data": {
    "delta": " AI assistant whose",
    "finish": false,
    "index": 1
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "vlm.utf-8.stream",
  "request_id": "4",
  "work_id": "vlm.1003"
}
```

```json
{
  "created": 1737600540,
  "data": {
    "delta": " name is Intern",
    "finish": false,
    "index": 2
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "vlm.utf-8.stream",
  "request_id": "4",
  "work_id": "vlm.1003"
}
```

```json
{
  "created": 1737600540,
  "data": {
    "delta": "A.",
    "finish": false,
    "index": 3
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "vlm.utf-8.stream",
  "request_id": "4",
  "work_id": "vlm.1003"
}
```

```json
{
  "created": 1737600540,
  "data": {
    "delta": "",
    "finish": true,
    "index": 4
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "vlm.utf-8.stream",
  "request_id": "4",
  "work_id": "vlm.1003"
}
```

发送 json：

```json
{
  "request_id": "4",
  "work_id": "vlm.1003",
  "action": "inference",
  "object": "vlm.jpeg.stream.base64",
  "data": {
    "delta": "U29tZSByYW5kb20gYmFzZTY0IGRhdGEgZm9yIHlvdSB0byBjb25zdWx0IGFwcGxpY2F0aW9ucy4=",
    "index": 0,
    "finish": true
  }
}
```

> **由于图片数据较多，因此上述 json 仅展示片段，实际使用时，需要发送完整的图片数据**

## link

链接上级单元的输出。

发送 json：

```json
{
  "request_id": "3",
  "work_id": "vlm.1003",
  "action": "link",
  "object": "work_id",
  "data": "kws.1000"
}
```

响应 json：

```json
{
  "created": 1737599866,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "3",
  "work_id": "vlm.1003"
}
```

error::code 为 0 表示执行成功。

将 vlm 和 kws 单元链接起来，当 kws 发出唤醒数据时，vlm 单元停止上次未完成的推理，用于重复唤醒功能。

> **link 时必须保证 kws 此时已经配置好进入工作状态。link 也可以在 setup 阶段进行。**

示例:

```json
{
  "request_id": "2",
  "work_id": "vlm",
  "action": "setup",
  "object": "vlm.setup",
  "data": {
    "model": "internvl2.5-1B-364-ax630c",
    "response_format": "vlm.utf-8.stream",
    "input": [
      "vlm.utf-8",
      "kws.1000",
      "whisper.1001"
    ],
    "enoutput": true,
    "max_token_len": 256,
    "prompt": "You are a knowledgeable assistant capable of answering various questions and providing information."
  }
}
```

链接 llm-camera 单元的输出。

发送 json：

```json
{
  "request_id": "3",
  "work_id": "vlm.1003",
  "action": "link",
  "object": "work_id",
  "data": "camera.1000"
}
```

响应 json：

```json
{
    "created": 1750992545,
    "data": "None",
    "error": {
        "code": 0,
        "message": ""
    },
    "object": "None",
    "request_id": "3",
    "work_id": "vlm.1003"
}
```

> **link 时必须保证 camera 此时已经配置好进入工作状态。当同时使用 AX630C MIPI 摄像头，需要在 llm-camera 初始化的时候配置为 AI-ISP 关闭模式。**

## unlink

取消链接。

发送 json：

```json
{
  "request_id": "4",
  "work_id": "vlm.1003",
  "action": "unlink",
  "object": "work_id",
  "data": "kws.1000"
}
```

响应 json：

```json
{
  "created": 1737600178,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "4",
  "work_id": "vlm.1003"
}
```

error::code 为 0 表示执行成功。

## pause

暂停单元工作。

发送 json：

```json
{
  "request_id": "5",
  "work_id": "vlm.1003",
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
  "work_id": "vlm.1003"
}
```

error::code 为 0 表示执行成功。

## work

恢复单元工作。

发送 json：

```json
{
  "request_id": "6",
  "work_id": "vlm.1003",
  "action": "work"
}
```

响应 json：

```json
{
  "created": 1737600236,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "6",
  "work_id": "vlm.1003"
}
```

error::code 为 0 表示执行成功。

## exit

退出单元工作。

发送 json：

```json
{
  "request_id": "7",
  "work_id": "vlm.1003",
  "action": "exit"
}
```

响应 json：

```json
{
  "created": 1737600704,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "7",
  "work_id": "vlm.1003"
}
```

error::code 为 0 表示执行成功。

## taskinfo

获取任务列表。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "vlm",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1737600076,
  "data": [
    "vlm.1003"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "vlm.tasklist",
  "request_id": "2",
  "work_id": "vlm"
}
```

获取任务运行参数。

```json
{
  "request_id": "2",
  "work_id": "vlm.1003",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1737600098,
  "data": {
    "enoutput": true,
    "inputs": [
      "vlm.utf-8",
      "kws.1000"
    ],
    "model": "internvl2.5-1B-364-ax630c",
    "response_format": "vlm.utf-8.stream"
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "vlm.taskinfo",
  "request_id": "2",
  "work_id": "vlm.1003"
}
```

> **注意：work_id 是按照单元的初始化注册顺序增加的，并不是固定的索引值。**
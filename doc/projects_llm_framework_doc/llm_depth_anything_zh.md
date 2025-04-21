# llm-depth_anything

depth_anything 视觉单元，用于提供图片深度信息。

## setup

配置单元工作。

发送 json：

```json
{
  "request_id": "4",
  "work_id": "depth_anything",
  "action": "setup",
  "object": "depth_anything.setup",
  "data": {
    "model": "depth-anything-ax630c",
    "response_format": "jpeg.base64.stream",
    "input": "camera.1001",
    "enoutput": true
  }
}
```

- request_id：参考基本数据解释。
- work_id：配置单元时，为 `depth_anything`。
- action：调用的方法为 `setup`。
- object：传输的数据类型为 `depth_anything.setup`。
- model：使用的模型为 `depth-anything-ax630c` 模型。
- response_format：返回结果为 `jpeg.base64.stream`。
- input：输入的为 `camera.1001`，代表的是从 camera 单元内部输入，详见 camera 单位文档。
- enoutput：是否启用用户结果输出。

响应 json：

```json
{
  "created": 1737601952,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "4",
  "work_id": "depth_anything.1007"
}
```

- created：消息创建时间，unix 时间。
- work_id：返回成功创建的 work_id 单元。

> **输入为 `camera.1001` 时必须保证 camera 单元此时已经配置好进入工作状态。camera 单元配置的 json 参考如下**

```json
{
  "request_id": "4",
  "work_id": "camera",
  "action": "setup",
  "object": "camera.setup",
  "data": {
    "response_format": "camera.raw",
    "input": "/dev/video0",
    "enoutput": false,
    "frame_width": 384,
    "frame_height": 256
  }
}
```

## exit

单元退出。

发送 json：

```json
{
  "request_id": "7",
  "work_id": "depth_anything.1007",
  "action": "exit"
}
```

响应 json：

```json
{
  "created": 1737603622,
  "data": "None",
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "None",
  "request_id": "7",
  "work_id": "depth_anything.1007"
}
```

error::code 为 0 表示执行成功。

## taskinfo

获取任务列表。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "depth_anything",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1737601986,
  "data": [
    "depth_anything.1007"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "depth_anything.tasklist",
  "request_id": "2",
  "work_id": "depth_anything"
}
```

获取任务运行参数。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "depth_anything.1007",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1737602022,
  "data": {
    "enoutput": true,
    "inputs": [
      "camera.1001"
    ],
    "model": "depth-anything-ax630c",
    "response_format": "jpeg.base64.stream"
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "depth_anything.taskinfo",
  "request_id": "2",
  "work_id": "depth_anything.1007"
}
```

> **注意：work_id 是按照单元的初始化注册顺序增加的，并不是固定的索引值。**  
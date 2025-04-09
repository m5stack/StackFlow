# llm-camera

视频源单元，用于从 USB V4L2 视频设备中获取视频流到内部通道。

## setup

配置单元工作。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "camera",
  "action": "setup",
  "object": "camera.setup",
  "data": {
    "response_format": "image.yuyv422.base64",
    "input": "/dev/video0",
    "enoutput": false,
    "frame_width": 320,
    "frame_height": 320,
    "enable_webstream":false,
    "rtsp":"rtsp.1280x720.h265"
  }
}
```

- request_id：参考基本数据解释。
- work_id：配置单元时，为 `camera`。
- action：调用的方法为 `setup`。
- object：传输的数据类型为 `camera.setup`。
- response_format：返回结果为 `image.yuyv422.base64`，是 yuyv422 格式。可选 image.jpeg.base64 格式输出。
- input：读取的设备名。示例："/dev/video0", "axera_single_sc850sl"
- frame_width：输出的视频帧宽。
- frame_height：输出的视频帧高。
- enoutput：是否起用用户结果输出。如果不需要获取摄像头图片，请不要开启该参数，视频流会增加信道的通信压力。
- enable_webstream：是否启用 webstream 流输出，webstream 会监听 tcp:8989 端口，一但收到客户端连接，将会以 HTTP 协议 multipart/x-mixed-replace 类型推送 jpeg 图片。
- rtsp：是否启用 rtsp 流输出，rtsp 会建立一个 rtsp://{DevIp}:8554/axstream0 RTSP TCP 服务端，可使用RTSP 协议向该端口拉取视频流。视频流的格式为 1280x720 H265。注意，该视频流只在 AX630C MIPI 摄像头上有效，UVC 摄像头无法使用 RTSP。

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
  "work_id": "camera.1003"
}
```

- created：消息创建时间，unix 时间。
- work_id：返回成功创建的 work_id 单元。

## exit

单元退出。

发送 json：

```json
{
  "request_id": "7",
  "work_id": "camera.1003",
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
  "work_id": "camera.1003"
}
```

error::code 为 0 表示执行成功。

## taskinfo

获取任务列表。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "camera",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1731652311,
  "data": [
    "camera.1003"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "camera.tasklist",
  "request_id": "2",
  "work_id": "camera"
}
```

获取任务运行参数。

发送 json：

```json
{
  "request_id": "2",
  "work_id": "camera.1003",
  "action": "taskinfo"
}
```

响应 json：

```json
{
  "created": 1731652344,
  "data": {
    "enoutput": false,
    "response_format": "image.yuyv422.base64",
    "input": "/dev/video0",
    "frame_width": 320,
    "frame_height": 320
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "camera.taskinfo",
  "request_id": "2",
  "work_id": "camera.1003"
}
```

> **注意：work_id 是按照单元的初始化注册顺序增加的，并不是固定的索引值。**  
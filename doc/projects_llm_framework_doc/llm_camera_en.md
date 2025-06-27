# llm-camera

Video source unit for obtaining video streams from USB V4L2 video devices to internal channels.

## setup

Configuration unit operation.

Send JSON:

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

- request_id: Refer to the basic data explanation.
- work_id: When configuring the unit, it is `camera`.
- action: The method called is `setup`.
- object: The type of data transmitted is `camera.setup`.
- response_format: The output format is `image.yuyv422.base64`, which is in yuyv422 format. An optional format is image.jpeg.base64.
- input: The device name to be read. Example: "/dev/video0", "axera_single_sc850sl"
- frame_width: The width of the video frame output.
- frame_height: The height of the video frame output.
- enoutput: Whether to enable user result output. If you do not need to obtain camera images, do not enable this parameter, as the video stream will increase the communication pressure on the channel.
- enable_webstream: Whether to enable webstream output, webstream will listen on tcp:8989 port, and once a client connection is received, it will push jpeg images in HTTP protocol multipart/x-mixed-replace type.
- rtsp: Whether to enable rtsp stream output, rtsp will establish an RTSP TCP server at rtsp://{DevIp}:8554/axstream0, and you can pull the video stream from this port using the RTSP protocol. The video stream format is 1280x720 H265. Note that this video stream is only valid on the AX630C MIPI camera, and the UVC camera cannot use RTSP.
- VinParam.bAiispEnable: Whether to enable AI-ISP, enabled by default. Set to 0 to disable, only valid when using AX630C MIPI camera.

Response JSON:

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

- created: Message creation time, in Unix time.
- work_id: The work_id unit successfully created.

## exit

Exit the unit.

Send JSON:

```json
{
  "request_id": "7",
  "work_id": "camera.1003",
  "action": "exit"
}
```

Response JSON:

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

error::code of 0 indicates successful execution.

## taskinfo

Get the task list.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "camera",
  "action": "taskinfo"
}
```

Response JSON:

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

Get task running parameters.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "camera.1003",
  "action": "taskinfo"
}
```

Response JSON:

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

> **Note: The work_id increases according to the order of unit initialization registration, not as a fixed index value.
**
# llm-depth_anything

depth_anything visual unit, used to provide image depth information.

## setup

Configuring the unit to work.

Send JSON:

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

- request_id: Reference basic data explanation.
- work_id: When configuring the unit, it is `depth_anything`.
- action: The method called is `setup`.
- object: The data type being transmitted is `depth_anything.setup`.
- model: The model used is the `depth-anything-ax630c` model.
- response_format: The return result is `jpeg.base64.stream`.
- input: The input is `camera.1001`, which refers to the input from the camera unit, as detailed in the camera unit
  documentation.
- enoutput: Whether to enable the user result output.

Response JSON:

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

- created: Message creation time, in Unix time.
- work_id: The work_id of the successfully created unit.

> **When the input is `camera.1001`, the camera unit must already be configured and in working status. The JSON
configuration for the camera unit is as follows:**

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

Unit exit.

Send JSON:

```json
{
  "request_id": "7",
  "work_id": "depth_anything.1007",
  "action": "exit"
}
```

Response JSON:

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

error::code 0 indicates successful execution.

## taskinfo

Get task list.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "depth_anything",
  "action": "taskinfo"
}
```

Response JSON:

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

Get task running parameters.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "depth_anything.1007",
  "action": "taskinfo"
}
```

Response JSON:

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

> **Note: The work_id increases according to the order of unit initialization and is not a fixed index value.**
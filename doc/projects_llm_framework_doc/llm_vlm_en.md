# llm-vlm

A large model unit designed to provide multimodal inference services.

## setup

Configures the unit's operation.

Send the following JSON:

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

- request_id: Reference for basic data explanation.
- work_id: Set to `vlm` when configuring the unit.
- action: The method being called is `setup`.
- object: Data type being transferred is `vlm.setup`.
- model: The model used is `internvl2.5-1B-364-ax630c`, a multimodal model.
- response_format: The output is in `vlm.utf-8.stream`, a UTF-8 stream format.
- input: The input is `vlm.utf-8`, representing user input.
- enoutput: Specifies whether to enable user output.
- max_token_len: Maximum output tokens, subject to the model's maximum limit.
- prompt: System prompt for the model.

Response JSON:

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

- created: Message creation time in Unix timestamp.
- work_id: Returned work_id after successful creation.

## inference

Infer user data.

Send the following JSON:

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

Non-streaming response JSON:

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

Streaming response:

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

Send the following JSON:

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

> **Due to large image data, the above JSON only shows a segment. When used in practice, you must send the complete
image data.**

## link

Link to the upper unit's output.

Send the following JSON:

```json
{
  "request_id": "3",
  "work_id": "vlm.1003",
  "action": "link",
  "object": "work_id",
  "data": "kws.1000"
}
```

Response JSON:

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

Error code `0` indicates success.

This links the `vlm` and `kws` units. When `kws` sends wake-up data, `vlm` stops the previous unfinished inference to
repeat the wake-up function.

> **Linking must ensure that `kws` is configured and ready to work. Linking can also be done during the setup phase.**

Example:

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

Linking the Output of the llm-camera Unit.

Sending JSON:

```json
{
  "request_id": "3",
  "work_id": "vlm.1003",
  "action": "link",
  "object": "work_id",
  "data": "camera.1000"
}
```

Response JSON:

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

> **Ensure that the camera is properly configured and ready for operation when performing the link action. If using the AX630C MIPI camera, configure it in AI-ISP disabled mode during the initialization of llm-camera.**

## unlink

Unlink units.

Send the following JSON:

```json
{
  "request_id": "4",
  "work_id": "vlm.1003",
  "action": "unlink",
  "object": "work_id",
  "data": "kws.1000"
}
```

Response JSON:

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

Error code `0` indicates success.

## pause

Pause unit work.

Send the following JSON:

```json
{
  "request_id": "5",
  "work_id": "vlm.1003",
  "action": "pause"
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
  "request_id": "5",
  "work_id": "vlm.1003"
}
```

Error code `0` indicates success.

## work

Resume unit work.

Send the following JSON:

```json
{
  "request_id": "6",
  "work_id": "vlm.1003",
  "action": "work"
}
```

Response JSON:

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

Error code `0` indicates success.

## exit

Exit unit work.

Send the following JSON:

```json
{
  "request_id": "7",
  "work_id": "vlm.1003",
  "action": "exit"
}
```

Response JSON:

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

Error code `0` indicates success.

## taskinfo

Get the task list.

Send the following JSON:

```json
{
  "request_id": "2",
  "work_id": "vlm",
  "action": "taskinfo"
}
```

Response JSON:

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

Get task running parameters.

```json
{
  "request_id": "2",
  "work_id": "vlm.1003",
  "action": "taskinfo"
}
```

Response JSON:

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

> **Note: `work_id` increases sequentially as units are initialized, it is not a fixed index value.**
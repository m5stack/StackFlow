# llm-llm2

Large Model Unit, used to provide next-generation large model inference services.

Compared with `llm-llm`, `llm-llm2` mainly targets the Qwen3 / Qwen3.5 model series in `main_llm2`, and supports
overriding sampling parameters such as `temperature`, `top_p`, and `top_k` on a per-request basis.

## setup

Configure the unit.

Send JSON:

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

- request_id: Refer to the basic data explanation.
- work_id: For configuring the unit, it is `llm2`.
- action: The method to call is `setup`.
- object: The type of data being transmitted is `llm.setup`.
- model: The model name to use, for example `Qwen3.5-0.8B-Int4-ax650`.
- response_format: The returned result format, for example `llm.utf-8.stream`.
- input: Input type, usually `llm.utf-8` or `llm.utf-8.stream`. It can also be configured as an array during setup.
- enoutput: Whether to enable user result output.
- prompt: The system prompt for the model.

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
  "work_id": "llm2.1002"
}
```

- created: Message creation time, in Unix time.
- work_id: The successfully created work_id unit.

## inference

### streaming input

Regular streaming input:

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "Can you tell a story in English?",
    "index": 0,
    "finish": true
  }
}
```

- object: The data type transmitted is `llm.utf-8.stream`, indicating UTF-8 streaming input from the user.
- delta: Segment data of the streaming input.
- index: Segment index of the streaming input.
- finish: A flag indicating whether the streaming input has completed.

### streaming input with per-request sampling overrides

If you want to override sampling parameters for a single request, `data.delta` must be a JSON string.

Example: override `temperature`

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "{\"prompt\":\"Can you tell a story in English?\",\"temperature\":0.7}",
    "index": 0,
    "finish": true
  }
}
```

Example: override `top_p`

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "{\"prompt\":\"Can you tell a story in English?\",\"top_p\":0.8}",
    "index": 0,
    "finish": true
  }
}
```

Example: override `top_k`

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "{\"prompt\":\"Can you tell a story in English?\",\"top_k\":10}",
    "index": 0,
    "finish": true
  }
}
```

Example: override multiple parameters

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8.stream",
  "data": {
    "delta": "{\"prompt\":\"Can you tell a story in English?\",\"temperature\":0.7,\"top_p\":0.8}",
    "index": 0,
    "finish": true
  }
}
```

Notes:

- If `temperature`, `top_p`, and `top_k` are not provided, the model uses the default `post_config.json` settings.
- Only explicitly provided fields are overridden for the current request. Any omitted fields continue to use the
  default values.
- If both `top_p` and `top_k` are provided, the current implementation gives priority to `top_p`.

### non-streaming input

Regular non-streaming input:

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8",
  "data": "Can you tell a story in English?"
}
```

If you need to override sampling parameters in a non-streaming request, `data` can be sent as a JSON string:

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "inference",
  "object": "llm.utf-8",
  "data": "{\"prompt\":\"Can you tell a story in English?\",\"temperature\":0.7,\"top_k\":10}"
}
```

### response JSON

Streaming response JSON:

```json
{"created":1742779468,"data":{"delta":"Once","finish":false,"index":0},"error":{"code":0,"message":""},"object":"llm.utf-8.stream","request_id":"2","work_id":"llm2.1002"}
{"created":1742779469,"data":{"delta":" upon a time","finish":false,"index":1},"error":{"code":0,"message":""},"object":"llm.utf-8.stream","request_id":"2","work_id":"llm2.1002"}
{"created":1742779470,"data":{"delta":" there was a story.","finish":false,"index":2},"error":{"code":0,"message":""},"object":"llm.utf-8.stream","request_id":"2","work_id":"llm2.1002"}
{"created":1742779470,"data":{"delta":"","finish":true,"index":3},"error":{"code":0,"message":""},"object":"llm.utf-8.stream","request_id":"2","work_id":"llm2.1002"}
```

Non-streaming response JSON:

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

Link the output of the upper unit.

Send JSON:

```json
{
  "request_id": "3",
  "work_id": "llm2.1002",
  "action": "link",
  "object": "work_id",
  "data": "kws.1000"
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
  "request_id": "3",
  "work_id": "llm2.1002"
}
```

error::code of 0 indicates successful execution.

## unlink

Unlink.

Send JSON:

```json
{
  "request_id": "4",
  "work_id": "llm2.1002",
  "action": "unlink",
  "object": "work_id",
  "data": "kws.1000"
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
  "request_id": "4",
  "work_id": "llm2.1002"
}
```

error::code of 0 indicates successful execution.

## pause

Pause the unit.

Send JSON:

```json
{
  "request_id": "5",
  "work_id": "llm2.1002",
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
  "work_id": "llm2.1002"
}
```

error::code of 0 indicates successful execution.

## exit

Exit the unit.

Send JSON:

```json
{
  "request_id": "7",
  "work_id": "llm2.1002",
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
  "work_id": "llm2.1002"
}
```

error::code of 0 indicates successful execution.

## taskinfo

Get the task list.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "llm2",
  "action": "taskinfo"
}
```

Response JSON:

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

Get the running parameters of a task.

```json
{
  "request_id": "2",
  "work_id": "llm2.1002",
  "action": "taskinfo"
}
```

Response JSON:

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

> Note: `work_id` increases according to the order in which units are initialized and registered. It is not a fixed
> index value.
# llm-melotts

Text-to-speech unit accelerated by NPU, used to provide text-to-speech services. It supports both Chinese and English
models for text-to-speech conversion.

## setup

Configure the unit.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "melotts",
  "action": "setup",
  "object": "melotts.setup",
  "data": {
    "model": "melotts-zh-cn",
    "response_format": "sys.pcm",
    "input": "tts.utf-8",
    "enoutput": false
  }
}
```

- request_id: Refer to the basic data explanation.
- work_id: For configuration, it is `melotts`.
- action: The method to be called is `setup`.
- object: The data type being transmitted is `melotts.setup`.
- model: The model being used is the Chinese model `melotts-zh-cn`.
- response_format: The result is returned as `sys.pcm`, system audio data, which is directly sent to the llm-audio
  module for playback.
- input: The input is `tts.utf-8`, representing user input.
- enoutput: Whether to enable user result output.

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
  "work_id": "melotts.1003"
}
```

- created: Message creation time, UNIX time.
- work_id: Successfully created work_id unit.

## inference

### streaming input

```json
{
    "request_id": "2",
    "work_id": "melotts.1003",
    "action": "inference",
    "object": "melotts.utf-8.stream",
    "data": {
        "delta": "What's ur name?",
        "index": 0,
        "finish": true
    }
}
```
- object: The data type transmitted is melotts.utf-8.stream, indicating a streaming input from the user's UTF-8.
- delta: Segment data of the streaming input.
- index: Index of the segment in the streaming input.
- finish: A flag indicating whether the streaming input has completed.

### non-streaming input

```json
{
    "request_id": "2",
    "work_id": "melotts.1003",
    "action": "inference",
    "object": "melotts.utf-8",
    "data": "What's ur name?"
}
```

- object: The data type transmitted is melotts.utf-8, indicating a non-streaming input from the user's UTF-8.
- data: Data for non-streaming input.

## link

Link the output of the upper-level unit.

Send JSON:

```json
{
  "request_id": "3",
  "work_id": "melotts.1003",
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
  "work_id": "melotts.1003"
}
```

error::code being 0 indicates success.

Link the llm and melotts units. When the kws melotts unit stops the unfinished inference from the last time, it is used
for repeated wake-up functionality.

> **Ensure that kws is already configured and in working status during link. Link can also be performed during the setup
stage.**

Example:

```json
{
  "request_id": "2",
  "work_id": "melotts",
  "action": "setup",
  "object": "melotts.setup",
  "data": {
    "model": "melotts-zh-cn",
    "response_format": "sys.pcm",
    "input": [
      "tts.utf-8",
      "llm.1002",
      "kws.1000"
    ],
    "enoutput": false
  }
}
```

## unlink

Unlink.

Send JSON:

```json
{
  "request_id": "4",
  "work_id": "melotts.1003",
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
  "work_id": "melotts.1003"
}
```

error::code being 0 indicates success.

## pause

Pause the unit.

Send JSON:

```json
{
  "request_id": "5",
  "work_id": "llm.1003",
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
  "work_id": "llm.1003"
}
```

error::code being 0 indicates success.

## work

Resume the unit.

Send JSON:

```json
{
  "request_id": "6",
  "work_id": "llm.1003",
  "action": "work"
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
  "request_id": "6",
  "work_id": "llm.1003"
}
```

error::code being 0 indicates success.

## exit

Exit the unit.

Send JSON:

```json
{
  "request_id": "7",
  "work_id": "llm.1003",
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
  "work_id": "llm.1003"
}
```

error::code being 0 indicates success.

## Task Information

Get task list.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "melotts",
  "action": "taskinfo"
}
```

Response JSON:

```json
{
  "created": 1731652311,
  "data": [
    "melotts.1003"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "melotts.tasklist",
  "request_id": "2",
  "work_id": "melotts"
}
```

Get task runtime parameters.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "melotts.1003",
  "action": "taskinfo"
}
```

Response JSON:

```json
{
  "created": 1731652344,
  "data": {
    "enoutput": false,
    "inputs_": [
      "tts.utf-8"
    ],
    "model": "melotts-zh-cn",
    "response_format": "sys.pcm"
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "melotts.taskinfo",
  "request_id": "2",
  "work_id": "melotts.1003"
}
```

> **Note: work_id increases in the order of the unit's initialization registration and is not a fixed index value.**  
> **The same type of unit cannot configure multiple units to work simultaneously, or unknown errors may occur. For
example, tts and melo tts cannot be activated to work at the same time.**
# llm-tts

Text-to-Speech unit, used to provide text-to-speech services, with options for Chinese and English models to provide
text-to-speech services in both languages.

## setup

Configure the unit.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "tts",
  "action": "setup",
  "object": "tts.setup",
  "data": {
    "model": "single-speaker-english-fast",
    "response_format": "sys.pcm",
    "input": "tts.utf-8",
    "enoutput": false
  }
}
```

- request_id: Refer to the basic data explanation.
- work_id: For configuring the unit, it is `tts`.
- action: The method to call is `setup`.
- object: The type of data being transmitted is `tts.setup`.
- model: The model used is the `single-speaker-english-fast` English model.
- response_format: The returned result is `sys.pcm`, system audio data, which is directly sent to the llm-audio module
  for playback.
- input: Input is `tts.utf-8`, representing user input.
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
  "work_id": "tts.1003"
}
```

- created: Message creation time, in Unix time.
- work_id: The successfully created work_id unit.

## inference

### streaming input

```json
{
    "request_id": "2",
    "work_id": "tts.1003",
    "action": "inference",
    "object": "tts.utf-8.stream",
    "data": {
        "delta": "What's ur name?",
        "index": 0,
        "finish": true
    }
}
```
- object: The data type transmitted is tts.utf-8.stream, indicating a streaming input from the user's UTF-8.
- delta: Segment data of the streaming input.
- index: Index of the segment in the streaming input.
- finish: A flag indicating whether the streaming input has completed.

### non-streaming input

```json
{
    "request_id": "2",
    "work_id": "tts.1003",
    "action": "inference",
    "object": "tts.utf-8",
    "data": "What's ur name?"
}
```

- object: The data type transmitted is tts.utf-8, indicating a non-streaming input from the user's UTF-8.
- data: Data for non-streaming input.

## link

Link the output of the upper unit.

Send JSON:

```json
{
  "request_id": "3",
  "work_id": "tts.1003",
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
  "work_id": "tts.1003"
}
```

error::code 0 indicates successful execution.

Link the llm and tts units, so that when kws wakes up, the tts unit stops the previous unfinished inference, used for
repeat wake-up functionality.

> **When linking, ensure that kws is already configured and in working state. Linking can also be done during the setup
phase.**

Example:

```json
{
  "request_id": "2",
  "work_id": "tts",
  "action": "setup",
  "object": "tts.setup",
  "data": {
    "model": "single-speaker-fast",
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
  "work_id": "tts.1003",
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
  "work_id": "tts.1003"
}
```

error::code 0 indicates successful execution.

## pause

Pause unit work.

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

error::code 0 indicates successful execution.

## work

Resume unit work.

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

error::code 0 indicates successful execution.

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

error::code 0 indicates successful execution.

## Task Information

Get task list.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "tts",
  "action": "taskinfo"
}
```

Response JSON:

```json
{
  "created": 1731652311,
  "data": [
    "tts.1003"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "tts.tasklist",
  "request_id": "2",
  "work_id": "tts"
}
```

Get task runtime parameters.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "tts.1003",
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
    "model": "single-speaker-fast",
    "response_format": "sys.pcm"
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "tts.taskinfo",
  "request_id": "2",
  "work_id": "tts.1003"
}
```

> **Note: work_id increases according to the order of unit initialization registration and is not a fixed index value.
**  
> **The same type of unit cannot have multiple units working simultaneously, as it may cause unknown errors. For
example, tts and melo tts cannot be enabled to work at the same time.**
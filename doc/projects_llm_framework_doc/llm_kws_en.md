# llm-kws

Voice wake-up unit, used to provide voice wake-up services. You can choose between Chinese and English models to provide
wake-up services in either language.

## setup

Configure the unit to work.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "kws",
  "action": "setup",
  "object": "kws.setup",
  "data": {
    "model": "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
    "response_format": "kws.bool",
    "input": "sys.pcm",
    "enoutput": true,
    "kws": "HELLO"
  }
}
```

- request_id: Refer to basic data explanation.
- work_id: When configuring the unit, it is `kws`.
- action: The method called is `setup`.
- object: The type of data being transmitted is `kws.setup`.
- model: The model used is the Chinese model `sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`.
- response_format: The result returned is in `kws.bool` format.
- input: The input is `sys.pcm`, representing system audio.
- enoutput: Whether to enable user result output.
- kws: The Chinese wake-up word is `"你好你好"`.

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
  "work_id": "kws.1000"
}
```

- created: Message creation time in Unix time.
- work_id: The successfully created work_id unit returned.

## pause

Pause the unit's work.

Send JSON:

```json
{
  "request_id": "3",
  "work_id": "kws.1000",
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
  "request_id": "3",
  "work_id": "kws.1000"
}
```

error::code of 0 indicates successful execution.

## work

Resume the unit's work.

Send JSON:

```json
{
  "request_id": "4",
  "work_id": "kws.1000",
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
  "request_id": "4",
  "work_id": "kws.1000"
}
```

error::code of 0 indicates successful execution.

## exit

Exit the unit.

Send JSON:

```json
{
  "request_id": "5",
  "work_id": "kws.1000",
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
  "request_id": "5",
  "work_id": "kws.1000"
}
```

error::code of 0 indicates successful execution.

## Task Information

Get task list.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "kws",
  "action": "taskinfo"
}
```

Response JSON:

```json
{
  "created": 1731580350,
  "data": [
    "kws.1000"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "kws.tasklist",
  "request_id": "2",
  "work_id": "kws"
}
```

Get task runtime parameters.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "kws.1000",
  "action": "taskinfo"
}
```

Response JSON:

```json
{
  "created": 1731652086,
  "data": {
    "enoutput": true,
    "inputs_": [
      "sys.pcm"
    ],
    "model": "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01",
    "response_format": "kws.bool"
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "kws.taskinfo",
  "request_id": "2",
  "work_id": "kws.1000"
}
```

> **Note: work_id increases according to the initialization registration order of the unit and is not a fixed index
value.**
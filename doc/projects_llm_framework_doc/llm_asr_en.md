# llm-asr

A speech-to-text module that provides speech-to-text services. It supports both Chinese and English models for
converting speech to text in these languages.

## setup

Configure the unit.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "asr",
  "action": "setup",
  "object": "asr.setup",
  "data": {
    "model": "sherpa-ncnn-streaming-zipformer-20M-2023-02-17",
    "response_format": "asr.utf-8.stream",
    "input": "sys.pcm",
    "enoutput": true,
    "endpoint_config.rule1.min_trailing_silence": 2.4,
    "endpoint_config.rule2.min_trailing_silence": 1.2,
    "endpoint_config.rule3.min_trailing_silence": 30.1,
    "endpoint_config.rule1.must_contain_nonsilence": true,
    "endpoint_config.rule2.must_contain_nonsilence": true,
    "endpoint_config.rule3.must_contain_nonsilence": true
  }
}
```

- request_id: Reference basic data explanation.
- work_id: For configuration units, it is `asr`.
- action: The method to be called is `setup`.
- object: The type of data being transmitted is `asr.setup`.
- model: The model used is the Chinese model `sherpa-ncnn-streaming-zipformer-20M-2023-02-17`.
- response_format: The result format is `asr.utf-8.stream`, a UTF-8 stream output.
- input: The input is `sys.pcm`, representing system audio.
- enoutput: Whether to enable user result output.
- endpoint_config.rule1.min_trailing_silence: A speech break occurs 2.4 seconds after wake-up.
- endpoint_config.rule2.min_trailing_silence: A speech break occurs 1.2 seconds after a speech is recognized.
- endpoint_config.rule3.min_trailing_silence: A speech break occurs after a maximum of 30.1 seconds of recognition.

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
  "work_id": "asr.1001"
}
```

- created: Message creation time, Unix time.
- work_id: The successfully created work_id unit.

## link

Link the output of the upstream unit.

Send JSON:

```json
{
  "request_id": "3",
  "work_id": "asr.1001",
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
  "work_id": "asr.1001"
}
```

error::code of 0 indicates successful execution.

Link the asr and kws units so that when kws sends wake-up data, the asr unit starts recognizing the user's speech,
pauses automatically after recognition, and waits until the next wake-up.

> **Ensure that kws is already configured and working when linking. Linking can also be done during the setup phase.**

Example:

```json
{
  "request_id": "2",
  "work_id": "asr",
  "action": "setup",
  "object": "asr.setup",
  "data": {
    "model": "sherpa-ncnn-streaming-zipformer-20M-2023-02-17",
    "response_format": "asr.utf-8.stream",
    "input": [
      "sys.pcm",
      "kws.1000"
    ],
    "enoutput": true,
    "endpoint_config.rule1.min_trailing_silence": 2.4,
    "endpoint_config.rule2.min_trailing_silence": 1.2,
    "endpoint_config.rule3.min_trailing_silence": 30.1,
    "endpoint_config.rule1.must_contain_nonsilence": true,
    "endpoint_config.rule2.must_contain_nonsilence": true,
    "endpoint_config.rule3.must_contain_nonsilence": true
  }
}
```

## unlink

Unlink.

Send JSON:

```json
{
  "request_id": "4",
  "work_id": "asr.1001",
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
  "work_id": "asr.1001"
}
```

error::code of 0 indicates successful execution.

## pause

Pause the unit's work.

Send JSON:

```json
{
  "request_id": "5",
  "work_id": "asr.1001",
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
  "work_id": "asr.1001"
}
```

error::code of 0 indicates successful execution.

## work

Resume the unit's work.

Send JSON:

```json
{
  "request_id": "6",
  "work_id": "asr.1001",
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
  "work_id": "asr.1001"
}
```

error::code of 0 indicates successful execution.

## exit

Exit the unit.

Send JSON:

```json
{
  "request_id": "7",
  "work_id": "asr.1001",
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
  "work_id": "asr.1001"
}
```

error::code of 0 indicates successful execution.

## Task Information

Get task list.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "asr",
  "action": "taskinfo"
}
```

Response JSON:

```json
{
  "created": 1731580350,
  "data": [
    "asr.1001"
  ],
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "asr.tasklist",
  "request_id": "2",
  "work_id": "asr"
}
```

Get task runtime parameters.

Send JSON:

```json
{
  "request_id": "2",
  "work_id": "asr.1001",
  "action": "taskinfo"
}
```

Response JSON:

```json
{
  "created": 1731579679,
  "data": {
    "enoutput": false,
    "inputs_": [
      "sys.pcm"
    ],
    "model": "sherpa-ncnn-streaming-zipformer-20M-2023-02-17",
    "response_format": "asr.utf-8-stream"
  },
  "error": {
    "code": 0,
    "message": ""
  },
  "object": "asr.taskinfo",
  "request_id": "2",
  "work_id": "asr.1001"
}
```

> **Note: The work_id increases according to the initialization registration order of the unit and is not a fixed index
value.**
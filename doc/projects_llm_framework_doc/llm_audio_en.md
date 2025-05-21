# llm-audio

The `llm-audio` unit provides system audio playback and recording capabilities. It allows applications to play various audio formats, manage playback queues, and capture audio input.

## General API Call Conventions

All API calls to `llm-audio` are made via JSON messages.

### Request Structure

A typical request to an `llm-audio` API follows this structure:

```json
{
  "request_id": "unique_request_identifier_string", // Optional: Client-generated ID for tracking
  "api": "api_name_string",                        // Name of the API to call (e.g., "play", "cap")
  // ... other API-specific parameters
}
```

- `request_id`: An optional string that clients can use to correlate requests with responses or events.
- `api`: A mandatory string specifying the API to be invoked. (Note: While this field is part of the general convention, individual API examples below may omit it for brevity if the API name is clear from the section title. However, actual requests must include it.)

For APIs that involve sending audio data (like `play` and `queue_play`), the data is typically Base64 encoded and included in a `data` field.

### Response Structure

API calls will generally receive a response indicating the immediate outcome of the call.

**Synchronous Success Response:**

```json
{
  "request_id": "unique_request_identifier_string", // Mirrored from the request if provided
  "code": 0,                                       // 0 indicates success
  "message": "OK",                                 // Success message
  // ... API-specific response data (if any)
}
```

**Asynchronous Operation Initiation:**

For operations that take time (e.g., playing a long audio file, continuous recording), the initial response might just confirm that the operation has started. Subsequent updates or completion notifications will be sent via Event Notifications.

```json
{
  "request_id": "unique_request_identifier_string",
  "code": 0,
  "message": "Operation initiated successfully",
  "callback_topic": "topic_for_event_notifications" // Optional: Topic to listen for related events
}
```

### Error Response Structure

If an API call fails, the response will include a non-zero `code` and a descriptive `message`.

```json
{
  "request_id": "unique_request_identifier_string",
  "code": 1,                                       // Non-zero indicates an error
  "message": "Error description (e.g., Invalid parameter, Device busy)"
}
```
Common error codes might include:
- `1`: General error
- `2`: Invalid parameters
- `3`: Device not available or busy
- `4`: Operation not supported

## Event Notifications

The `llm-audio` unit can emit events to notify clients about asynchronous occurrences, such as the start or end of playback, or the availability of recorded data. Events are typically published on a specific topic if a `callback_topic` was provided in the initial request or if a global event topic is configured.

### General Event Structure

```json
{
  "event_type": "event_name_string", // e.g., "playback_started", "playback_finished", "recording_data_ready"
  "timestamp": "ISO_8601_datetime_string",
  "request_id": "original_request_identifier_string", // Optional: If the event is related to a specific request
  "data": {
    // Event-specific payload
  }
}
```

### Example Event Types

- **`playback_started`**: Indicates that an audio playback has commenced.
  ```json
  {
    "event_type": "playback_started",
    "timestamp": "2024-03-15T10:00:05Z",
    "request_id": "req_play_123",
    "data": {
      "source": "tts_engine_output.wav" // Or other identifier for the audio source
    }
  }
  ```
- **`playback_finished`**: Indicates that an audio playback has completed.
  ```json
  {
    "event_type": "playback_finished",
    "timestamp": "2024-03-15T10:00:35Z",
    "request_id": "req_play_123",
    "data": {
      "source": "tts_engine_output.wav",
      "reason": "completed" // "completed", "stopped_by_user", "error"
    }
  }
  ```
- **`recording_chunk_ready`**: Indicates a chunk of recorded audio data is available (for streaming scenarios).
  ```json
  {
    "event_type": "recording_chunk_ready",
    "timestamp": "2024-03-15T11:05:00Z",
    "request_id": "req_cap_456",
    "data": {
      "format": "audio/pcm",
      "encoding": "base64",
      "chunk_sequence_id": 1,
      "audio_data": "...." // Base64 encoded PCM data
    }
  }
  ```
- **`recording_finished`**: Indicates a recording session has ended.
  ```json
  {
    "event_type": "recording_finished",
    "timestamp": "2024-03-15T11:10:00Z",
    "request_id": "req_cap_456",
    "data": {
      "reason": "duration_limit_reached" // "duration_limit_reached", "stopped_by_user", "error"
    }
  }
  ```

## API

(Note: For brevity, the `api` field is omitted from the JSON request examples in this section, but it is required in actual API calls as described in "General API Call Conventions".)

### **play**

Immediately plays the provided audio data. This call will interrupt any currently playing audio or queued playback. Useful for urgent announcements or immediate feedback.

**Request:**

```json
{
  "input_type": "rpc.audio.wav.base64", // Or "rpc.audio.pcm.base64"
  "data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAAABkYXRhAAAAAA...==" // Base64 encoded audio data
}
```

| Parameter    | Type   | Required | Default | Description                                                                          |
|--------------|--------|----------|---------|--------------------------------------------------------------------------------------|
| `input_type` | string | Yes      | N/A     | Format of the audio data. Supported: `rpc.audio.wav.base64`, `rpc.audio.pcm.base64`.  |
| `data`       | string | Yes      | N/A     | Base64 encoded audio data. Add `...` to indicate continuation if data is large.      |

**Response (Success - Initiated):**
```json
{
  "code": 0,
  "message": "Playback initiated"
}
```

### **queue_play**

Adds the provided audio data to the end of the playback queue. Audio will play sequentially after any currently playing audio and previously queued items are finished.

**Request:**

```json
{
  "input_type": "rpc.audio.pcm.base64", // Or "rpc.audio.wav.base64"
  "data": "/////v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+...==" // Base64 encoded audio data
}
```

| Parameter    | Type   | Required | Default | Description                                                                          |
|--------------|--------|----------|---------|--------------------------------------------------------------------------------------|
| `input_type` | string | Yes      | N/A     | Format of the audio data. Supported: `rpc.audio.wav.base64`, `rpc.audio.pcm.base64`.  |
| `data`       | string | Yes      | N/A     | Base64 encoded audio data. Add `...` to indicate continuation if data is large.      |

**Response (Success - Queued):**
```json
{
  "code": 0,
  "message": "Audio queued for playback"
}
```

### **play_stop**

Stops the currently playing audio and clears the entire playback queue.

**Request:**
```json
{
  // No specific parameters needed
}
```
**Response (Success):**
```json
{
  "code": 0,
  "message": "Playback stopped and queue cleared"
}
```

### **queue_play_stop**

Clears all audio items from the playback queue. Does not affect the currently playing audio.

**Request:**
```json
{
  // No specific parameters needed
}
```
**Response (Success):**
```json
{
  "code": 0,
  "message": "Playback queue cleared"
}
```

### **audio_status**

Retrieves the current audio playback status, including queue size and information about the playing track.

**Request:**
```json
{
  // No specific parameters needed
}
```
**Response (Success):**
```json
{
  "code": 0,
  "message": "OK",
  "status": "playing", // "playing", "stopped", "idle"
  "queue_size": 1,
  "current_track_info": { // Present if status is "playing"
    "duration_ms": 30000,
    "elapsed_ms": 15000
    // Potentially other info like source, input_type
  }
}
```

### **cap**

Starts an audio recording task. Can be called repeatedly (behavior might depend on system capability, e.g., starting new independent recordings or reconfiguring an existing one). Recorded audio data (PCM stream) is output to the specified `output_channel`.

**Request:**
```json
{
  "output_channel": "ipc:///tmp/llm/my_capture.socket", // IPC socket path for PCM stream output
  "duration_ms": 10000,                               // Optional: desired duration for the recording in milliseconds
  "params": {                                         // Optional: override default capture parameters from audio.json
    "card": 0,
    "device": 0,
    "volume": 1.0,
    "channel": 1,
    "rate": 16000,
    "aistAttr.enSamplerate": 16000,
    "aistAttr.u32ChnCnt": 2,
    "aistVqeAttr.stNsCfg.bNsEnable": 1,
    "aistVqeAttr.stNsCfg.enAggressivenessLevel": 2
  }
}
```

| Parameter        | Type   | Required | Default                                     | Description                                                                                                 |
|------------------|--------|----------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| `output_channel` | string | Yes      | N/A                                         | IPC socket path (e.g., `ipc:///tmp/llm/my_capture.socket`) where the PCM audio stream will be sent.         |
| `duration_ms`    | int    | No       | Continuous until `cap_stop` or `cap_stop_all` | Requested duration for the recording in milliseconds. If 0 or not provided, recording is continuous.     |
| `params`         | object | No       | See `cap_param` in `audio.json`             | Object containing specific capture parameters to override defaults. See "Relation to `audio.json`" section.  |

**Response (Success - Initiated):**
```json
{
  "code": 0,
  "message": "Capture initiated"
}
```

### **cap_stop**

Stops a specific, ongoing recording task (if identifiable, e.g., by `output_channel` or an internal task ID not yet exposed in this doc). The last call to `cap_stop` for a channel will stop data output.

**Request:**
```json
{
  "output_channel": "ipc:///tmp/llm/my_capture.socket" // Identifies which capture task to stop
}
```

| Parameter        | Type   | Required | Default | Description                                  |
|------------------|--------|----------|---------|----------------------------------------------|
| `output_channel` | string | Yes      | N/A     | The output channel of the recording to stop. |

**Response (Success):**
```json
{
  "code": 0,
  "message": "Capture stopped for the specified channel"
}
```

### **cap_stop_all**

Forcibly stops all ongoing recording tasks.

**Request:**
```json
{
  // No specific parameters needed
}
```
**Response (Success):**
```json
{
  "code": 0,
  "message": "All capture tasks stopped"
}
```

### **setup**

Configures working parameters for the audio playback or recording units. Use `target_unit` to specify whether to configure `playback` or `capture` settings.

**Request:**
```json
{
  "target_unit": "playback", // "playback" or "capture"
  "params": {
    "volume": 0.75,        // General volume for the target unit (if applicable)
    "play_param": {        // Parameters from audio.json's play_param section
        "card": 0,
        "device": 1,
        "stAttr.enSamplerate": 44100,
        "stAttr.u32ChnCnt": 1,
        "stVqeAttr.stAgcCfg.bAgcEnable": 0 // Example: Disable Automatic Gain Control for playback
    },
    "cap_param": {         // Parameters from audio.json's cap_param section
        "card": 0,
        "device": 0,
        "aistAttr.enSamplerate": 48000,
        "aistVqeAttr.stNsCfg.bNsEnable": 1,
        "aistVqeAttr.stNsCfg.enAggressivenessLevel": 3, // Higher Noise Suppression
        "aistVqeAttr.stAgcCfg.bAgcEnable": 1,
        "aistVqeAttr.stAgcCfg.s16TargetLevel": -3
    }
  }
}
```

| Parameter           | Type   | Required | Default                               | Description                                                                                                                                |
|---------------------|--------|----------|---------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `target_unit`       | string | Yes      | N/A                                   | Specifies which unit to configure: "playback" or "capture".                                                                                |
| `params`            | object | Yes      | N/A                                   | An object containing parameters to set. This object can include a general `volume` and/or specific `play_param` or `cap_param` objects.  |
| `params.volume`     | float  | No       | Current setting                       | Sets the master volume for the specified `target_unit` (0.0 to 1.0).                                                                       |
| `params.play_param` | object | No       | See `play_param` in `audio.json`      | Object with parameters to update for the playback unit, mirroring structure in `audio.json`. Only applied if `target_unit` is "playback".  |
| `params.cap_param`  | object | No       | See `cap_param` in `audio.json`       | Object with parameters to update for the capture unit, mirroring structure in `audio.json`. Only applied if `target_unit` is "capture".    |

**Response (Success):**
```json
{
  "code": 0,
  "message": "Setup configuration applied successfully"
}
```

## Relation to `audio.json`

The file `projects/llm_framework/main_audio/audio.json` plays a crucial role in defining the default operational parameters for the `llm-audio` unit. This configuration file contains two main sections, `play_param` and `cap_param`, which outline the default settings for audio playback and capture, respectively.

- **Default Values**: When an API like `cap` or `setup` is called without specifying all possible parameters in the request, or when the `llm-audio` unit initializes, it refers to `audio.json` for default values (e.g., `card`, `device`, `rate`, `channel`, various VQE - Voice Quality Enhancement - settings like Noise Suppression `stNsCfg`, Automatic Gain Control `stAgcCfg`, etc.).
- **Parameter Structure**: The nested structure and key names within `play_param` and `cap_param` in `audio.json` (e.g., `stAttr.enSamplerate`, `aistVqeAttr.stNsCfg.bNsEnable`) are directly referenced or expected by the `setup` API and the parameter override mechanism in the `cap` API.
- **Capabilities and Formats**: `audio.json` also lists supported `capabilities` (e.g., "play", "cap"), `input_type` (e.g., "rpc.audio.wav.base64"), and `output_type` (e.g., "audio.pcm.stream"), which define the fundamental functionalities of the audio unit.

Users and developers can inspect `audio.json` to understand the default audio processing chain configurations and to know which parameters can be tuned via the `setup` API or overridden in the `cap` API. Modifying `audio.json` directly would change the system-wide default behavior of the `llm-audio` unit (requires a service restart).
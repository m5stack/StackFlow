# llm-audio

システムオーディオの再生と録音を提供するシステムオーディオユニットです。
オーディオデータはBase64エンコードされた文字列としてAPIに渡されます。

## API

- **play**: 指定されたオーディオデータを再生します。この呼び出しは、現在再生中のオーディオやキューに入っているオーディオ再生を中断し、即座に新しいオーディオの再生を開始します。例えば、緊急のアナウンスやユーザーの操作に対する即時フィードバックとして利用できます。

  リクエスト例:
  ```json
  {
    "input_type": "rpc.audio.wav.base64",
    "data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAAABkYXRhAAAAAA...=="
  }
  ```

- **queue_play**: 指定されたオーディオデータを再生キューの末尾に追加します。現在オーディオが再生中の場合は、その再生が完了した後にキュー内のオーディオが順次再生されます。複数のオーディオトラックを連続して再生したい場合などに使用します。

  リクエスト例:
  ```json
  {
    "input_type": "rpc.audio.pcm.base64",
    "data": "/////v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+...=="
  }
  ```

- **play_stop**: 現在再生中のオーディオおよびキューに入っているすべてのオーディオ再生を停止します。

- **queue_play_stop**: 再生キュー内のすべてのオーディオをクリアします。現在再生中のオーディオには影響しません。

- **audio_status**: 現在のオーディオ再生状態（再生中、停止中、アイドル状態など）、キューに入っているオーディオの数、再生中のトラック情報（総再生時間、経過時間など）を取得します。

  レスポンス例:
  ```json
  {
    "status": "playing",
    "queue_size": 1,
    "current_track_info": {
      "duration_ms": 30000,
      "elapsed_ms": 15000
    }
  }
  ```

- **cap**: オーディオ録音タスクを開始します。このAPIは繰り返し呼び出すことができ、呼び出しごとに新しい録音セッションが開始されるか、既存のセッションに影響を与える可能性があります（具体的な動作は実装によります）。成功した場合、指定された `output_channel` (IPCソケットなど) に対して、PCMストリーム形式 (`audio.pcm.stream`) で録音データが出力されます。

  リクエスト例:
  ```json
  {
    "output_channel": "ipc:///tmp/llm/my_capture.socket",
    "duration_ms": 10000,
    "params": {
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

- **cap_stop**: 現在進行中の録音タスクを停止します。このAPIは繰り返し呼び出すことができます。最後の `cap_stop` 呼び出しによって、対応する録音チャネルのデータ出力が完全に停止します。

- **cap_stop_all**: 進行中のすべての録音タスクを強制的に停止します。

- **setup**: オーディオユニットの動作パラメータ（再生ユニットまたは録音ユニット）を設定します。`target_unit` フィールドで設定対象（`playback` または `capture`）を指定します。これにより、音量、サンプリングレート、ノイズ抑制レベルなど、詳細な設定をカスタマイズできます。

  リクエスト例:
  ```json
  {
    "target_unit": "playback",
    "params": {
      "volume": 0.75,
      "play_param": {
          "card": 0,
          "device": 1,
          "stAttr.enSamplerate": 44100,
          "stAttr.u32ChnCnt": 1,
          "stVqeAttr.stAgcCfg.bAgcEnable": 0
      },
      "cap_param": {
          "card": 0,
          "device": 0,
          "aistAttr.enSamplerate": 48000,
          "aistVqeAttr.stNsCfg.bNsEnable": 1,
          "aistVqeAttr.stNsCfg.enAggressivenessLevel": 3,
          "aistVqeAttr.stAgcCfg.bAgcEnable": 1,
          "aistVqeAttr.stAgcCfg.s16TargetLevel": -3
      }
    }
  }
  ```

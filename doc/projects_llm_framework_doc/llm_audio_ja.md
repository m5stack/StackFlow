# llm-audio

`llm-audio` ユニットは、システムの音声再生および録音機能を提供します。これにより、アプリケーションは様々な音声フォーマットの再生、再生キューの管理、音声入力のキャプチャを行うことができます。

## API呼び出しの共通規約

`llm-audio` への全てのAPI呼び出しは、JSONメッセージを介して行われます。

### リクエスト構造

`llm-audio` APIへの典型的なリクエストは以下の構造に従います：

```json
{
  "request_id": "一意のリクエスト識別子文字列", // オプション：クライアント生成の追跡用ID
  "api": "API名文字列",                        // 呼び出すAPIの名前（例："play", "cap"）
  // ... その他API固有のパラメータ
}
```

- `request_id`: クライアントがリクエストとレスポンスやイベントを関連付けるために使用できるオプションの文字列です。
- `api`: 呼び出されるAPIを指定する必須の文字列です。（注意：このフィールドは共通規約の一部ですが、以下の個別のAPI例では、セクションタイトルからAPI名が明らかな場合、簡潔さのために省略されることがあります。ただし、実際のリクエストにはこれを含める必要があります。）

音声データ送信を伴うAPI（`play`や`queue_play`など）では、データは通常Base64エンコードされ、`data`フィールドに含まれます。

### レスポンス構造

API呼び出しは通常、呼び出しの即時結果を示すレスポンスを受け取ります。

**同期成功レスポンス：**

```json
{
  "request_id": "一意のリクエスト識別子文字列", // リクエストから提供された場合、そのまま反映
  "code": 0,                                       // 0は成功を示す
  "message": "OK",                                 // 成功メッセージ
  // ... API固有のレスポンスデータ（もしあれば）
}
```

**非同期操作開始：**

長時間の音声ファイルの再生や連続録音など、時間のかかる操作の場合、初期レスポンスは操作が開始されたことを確認するだけかもしれません。その後の更新や完了通知はイベント通知を介して送信されます。

```json
{
  "request_id": "一意のリクエスト識別子文字列",
  "code": 0,
  "message": "操作は正常に開始されました",
  "callback_topic": "イベント通知用トピック" // オプション：関連イベントをリッスンするトピック
}
```

### エラーレスポンス構造

API呼び出しが失敗した場合、レスポンスには0以外の`code`と説明的な`message`が含まれます。

```json
{
  "request_id": "一意のリクエスト識別子文字列",
  "code": 1,                                       // 0以外はエラーを示す
  "message": "エラー詳細（例：無効なパラメータ、デバイスビジー）"
}
```
一般的なエラーコードには以下が含まれる場合があります：
- `1`: 一般的なエラー
- `2`: 無効なパラメータ
- `3`: デバイスが利用不可またはビジー状態
- `4`: サポートされていない操作

## イベント通知

`llm-audio`ユニットは、再生の開始や終了、録音データの利用可能性など、非同期の発生をクライアントに通知するためにイベントを発行できます。イベントは通常、初期リクエストで`callback_topic`が提供された場合、またはグローバルイベントトピックが設定されている場合に、特定のトピックで公開されます。

### 一般的なイベント構造

```json
{
  "event_type": "イベント名文字列", // 例："playback_started", "playback_finished", "recording_data_ready"
  "timestamp": "ISO_8601日時文字列",
  "request_id": "元のリクエスト識別子文字列", // オプション：イベントが特定のリクエストに関連する場合
  "data": {
    // イベント固有のペイロード
  }
}
```

### イベントタイプの例

- **`playback_started`**: 音声再生が開始されたことを示します。
  ```json
  {
    "event_type": "playback_started",
    "timestamp": "2024-03-15T10:00:05Z",
    "request_id": "req_play_123",
    "data": {
      "source": "tts_engine_output.wav" // または他の音声ソース識別子
    }
  }
  ```
- **`playback_finished`**: 音声再生が完了したことを示します。
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
- **`recording_chunk_ready`**: 録音された音声データのチャンクが利用可能であることを示します（ストリーミングシナリオ用）。
  ```json
  {
    "event_type": "recording_chunk_ready",
    "timestamp": "2024-03-15T11:05:00Z",
    "request_id": "req_cap_456",
    "data": {
      "format": "audio/pcm",
      "encoding": "base64",
      "chunk_sequence_id": 1,
      "audio_data": "...." // Base64エンコードされたPCMデータ
    }
  }
  ```
- **`recording_finished`**: 録音セッションが終了したことを示します。
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

（注意：簡潔さのため、このセクションのJSONリクエスト例では`api`フィールドは省略されていますが、「API呼び出しの共通規約」で説明されているように、実際のAPI呼び出しには必須です。）

### **play**

提供された音声データを即座に再生します。この呼び出しは、現在再生中の音声またはキューに入れられた再生を中断します。緊急のアナウンスや即時フィードバックに役立ちます。

**リクエスト：**

```json
{
  "input_type": "rpc.audio.wav.base64", // または "rpc.audio.pcm.base64"
  "data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAAABkYXRhAAAAAA...==" // Base64エンコードされた音声データ
}
```

| パラメータ   | 型     | 必須 | デフォルト | 説明                                                                           |
|--------------|--------|------|------------|--------------------------------------------------------------------------------|
| `input_type` | string | はい | N/A        | 音声データのフォーマット。サポート：`rpc.audio.wav.base64`, `rpc.audio.pcm.base64`。 |
| `data`       | string | はい | N/A        | Base64エンコードされた音声データ。データが大きい場合は `...` を追加して継続を示します。   |

**レスポンス（成功 - 開始済み）：**
```json
{
  "code": 0,
  "message": "再生が開始されました"
}
```

### **queue_play**

提供された音声データを再生キューの末尾に追加します。現在再生中の音声および以前にキューに入れられたアイテムが終了した後、音声は順次再生されます。

**リクエスト：**

```json
{
  "input_type": "rpc.audio.pcm.base64", // または "rpc.audio.wav.base64"
  "data": "/////v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+/v7+...==" // Base64エンコードされた音声データ
}
```

| パラメータ   | 型     | 必須 | デフォルト | 説明                                                                           |
|--------------|--------|------|------------|--------------------------------------------------------------------------------|
| `input_type` | string | はい | N/A        | 音声データのフォーマット。サポート：`rpc.audio.wav.base64`, `rpc.audio.pcm.base64`。 |
| `data`       | string | はい | N/A        | Base64エンコードされた音声データ。データが大きい場合は `...` を追加して継続を示します。   |

**レスポンス（成功 - キュー追加済み）：**
```json
{
  "code": 0,
  "message": "音声が再生キューに追加されました"
}
```

### **play_stop**

現在再生中の音声を停止し、再生キュー全体をクリアします。

**リクエスト：**
```json
{
  // 特定のパラメータは不要
}
```
**レスポンス（成功）：**
```json
{
  "code": 0,
  "message": "再生が停止され、キューがクリアされました"
}
```

### **queue_play_stop**

再生キューから全ての音声アイテムをクリアします。現在再生中の音声には影響しません。

**リクエスト：**
```json
{
  // 特定のパラメータは不要
}
```
**レスポンス（成功）：**
```json
{
  "code": 0,
  "message": "再生キューがクリアされました"
}
```

### **audio_status**

キューサイズや再生中のトラック情報など、現在の音声再生ステータスを取得します。

**リクエスト：**
```json
{
  // 特定のパラメータは不要
}
```
**レスポンス（成功）：**
```json
{
  "code": 0,
  "message": "OK",
  "status": "playing", // "playing", "stopped", "idle"
  "queue_size": 1,
  "current_track_info": { // statusが "playing" の場合に存在
    "duration_ms": 30000,
    "elapsed_ms": 15000
    // ソース、input_typeなどの他の情報も含む可能性あり
  }
}
```

### **cap**

音声録音タスクを開始します。繰り返し呼び出すことができます（動作はシステムの能力に依存する場合があります。例えば、新しい独立した録音を開始するか、既存の録音を再設定するなど）。録音された音声データ（PCMストリーム）は、指定された`output_channel`に出力されます。

**リクエスト：**
```json
{
  "output_channel": "ipc:///tmp/llm/my_capture.socket", // PCMストリーム出力用のIPCソケットパス
  "duration_ms": 10000,                               // オプション：録音の希望持続時間（ミリ秒）
  "params": {                                         // オプション：audio.jsonからデフォルトのキャプチャパラメータを上書き
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

| パラメータ       | 型     | 必須 | デフォルト                                       | 説明                                                                                                 |
|------------------|--------|------|--------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `output_channel` | string | はい | N/A                                              | PCM音声ストリームが送信されるIPCソケットパス（例：`ipc:///tmp/llm/my_capture.socket`）。          |
| `duration_ms`    | int    | いいえ | `cap_stop` または `cap_stop_all` まで継続      | 録音の希望持続時間（ミリ秒）。0または未指定の場合、録音は継続的です。     |
| `params`         | object | いいえ | `audio.json`の`cap_param`参照                   | デフォルトを上書きする特定のキャプチャパラメータを含むオブジェクト。「`audio.json`との関連性」セクション参照。 |

**レスポンス（成功 - 開始済み）：**
```json
{
  "code": 0,
  "message": "キャプチャが開始されました"
}
```

### **cap_stop**

特定の進行中の録音タスクを停止します（`output_channel`や、このドキュメントではまだ公開されていない内部タスクIDなどで識別可能な場合）。チャネルに対する最後の`cap_stop`呼び出しでデータ出力が停止します。

**リクエスト：**
```json
{
  "output_channel": "ipc:///tmp/llm/my_capture.socket" // 停止するキャプチャタスクを識別
}
```
| パラメータ       | 型     | 必須 | デフォルト | 説明                               |
|------------------|--------|------|------------|------------------------------------|
| `output_channel` | string | はい | N/A        | 停止する録音の出力チャネル。       |

**レスポンス（成功）：**
```json
{
  "code": 0,
  "message": "指定されたチャネルのキャプチャが停止されました"
}
```

### **cap_stop_all**

進行中の全ての録音タスクを強制的に停止します。

**リクエスト：**
```json
{
  // 特定のパラメータは不要
}
```
**レスポンス（成功）：**
```json
{
  "code": 0,
  "message": "全てのキャプチャタスクが停止されました"
}
```

### **setup**

音声再生または録音ユニットの動作パラメータを設定します。`target_unit`を使用して、`playback`または`capture`設定のどちらを設定するかを指定します。

**リクエスト：**
```json
{
  "target_unit": "playback", // "playback" または "capture"
  "params": {
    "volume": 0.75,        // ターゲットユニットの一般音量（該当する場合）
    "play_param": {        // audio.jsonのplay_paramセクションのパラメータ
        "card": 0,
        "device": 1,
        "stAttr.enSamplerate": 44100,
        "stAttr.u32ChnCnt": 1,
        "stVqeAttr.stAgcCfg.bAgcEnable": 0 // 例：再生の自動ゲイン制御を無効化
    },
    "cap_param": {         // audio.jsonのcap_paramセクションのパラメータ
        "card": 0,
        "device": 0,
        "aistAttr.enSamplerate": 48000,
        "aistVqeAttr.stNsCfg.bNsEnable": 1,
        "aistVqeAttr.stNsCfg.enAggressivenessLevel": 3, // より高いノイズ抑制
        "aistVqeAttr.stAgcCfg.bAgcEnable": 1,
        "aistVqeAttr.stAgcCfg.s16TargetLevel": -3
    }
  }
}
```

| パラメータ          | 型     | 必須 | デフォルト                                    | 説明                                                                                                                                |
|---------------------|--------|------|-----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| `target_unit`       | string | はい | N/A                                           | 設定するユニットを指定します："playback" または "capture"。                                                                                |
| `params`            | object | はい | N/A                                           | 設定するパラメータを含むオブジェクト。このオブジェクトには、一般的な`volume`や特定の`play_param`または`cap_param`オブジェクトを含めることができます。  |
| `params.volume`     | float  | いいえ | 現在の設定                                | 指定された`target_unit`のマスター音量を設定します（0.0から1.0）。                                                                       |
| `params.play_param` | object | いいえ | `audio.json`の`play_param`参照             | 再生ユニット用に更新するパラメータを持つオブジェクトで、`audio.json`の構造を反映します。`target_unit`が"playback"の場合にのみ適用されます。  |
| `params.cap_param`  | object | いいえ | `audio.json`の`cap_param`参照              | キャプチャユニット用に更新するパラメータを持つオブジェクトで、`audio.json`の構造を反映します。`target_unit`が"capture"の場合にのみ適用されます。    |

**レスポンス（成功）：**
```json
{
  "code": 0,
  "message": "セットアップ設定が正常に適用されました"
}
```

## `audio.json`との関連性

ファイル`projects/llm_framework/main_audio/audio.json`は、`llm-audio`ユニットのデフォルト動作パラメータを定義する上で極めて重要な役割を果たします。この設定ファイルには、音声再生とキャプチャのデフォルト設定をそれぞれ概説する`play_param`と`cap_param`という2つの主要セクションが含まれています。

- **デフォルト値**: `cap`や`setup`のようなAPIがリクエストですべての可能なパラメータを指定せずに呼び出された場合、または`llm-audio`ユニットが初期化される際、デフォルト値（例：`card`、`device`、`rate`、`channel`、様々なVQE - 音声品質向上 - 設定、例えばノイズ抑制`stNsCfg`、自動ゲイン制御`stAgcCfg`など）について`audio.json`を参照します。
- **パラメータ構造**: `audio.json`内の`play_param`および`cap_param`のネストされた構造とキー名（例：`stAttr.enSamplerate`、`aistVqeAttr.stNsCfg.bNsEnable`）は、`setup` APIおよび`cap` APIのパラメータ上書きメカニズムによって直接参照されるか、期待されます。
- **機能とフォーマット**: `audio.json`はまた、サポートされている`capabilities`（例："play"、"cap"）、`input_type`（例："rpc.audio.wav.base64"）、および`output_type`（例："audio.pcm.stream"）をリストしており、これらはオーディオユニットの基本的な機能を定義します。

ユーザーおよび開発者は`audio.json`を検査することで、デフォルトの音声処理チェーン設定を理解し、`setup` APIを介して調整できるパラメータや`cap` APIで上書きできるパラメータを知ることができます。`audio.json`を直接変更すると、`llm-audio`ユニットのシステム全体のデフォルト動作が変更されます（サービスの再起動が必要です）。

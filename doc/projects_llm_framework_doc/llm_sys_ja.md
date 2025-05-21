# llm-sys

`llm-sys`ユニットは、StackFlowフレームワーク内の基盤となるサービスです。外部通信チャネル（シリアルおよびTCP）、システム情報の取得、コマンド実行、リセットや再起動といったハードウェア制御など、不可欠なシステムレベルの機能を提供します。また、内部的にポートリソースや設定用のシンプルなインメモリデータベースを管理します。

## API呼び出しの共通規約

`llm-sys`への全てのAPI呼び出しは、JSONメッセージを介して行われます。これらの規約を理解することは、ユニットとの対話を成功させるために不可欠です。

### リクエスト構造

`llm-sys` APIへの標準的なリクエストは、以下の構造に従う必要があります：

```json
{
  "request_id": "クライアント生成のUUIDまたはカウンター",
  "work_id": "sys",
  "action": "APIアクション名",
  // "object"や"data"のようなAPI固有のフィールドがここで必要になる場合があります
}
```

-   `request_id` (string, 必須): クライアントによって生成される一意の識別子（例：UUID、増分カウンター）。このIDは、リクエストとそれに対応するレスポンスを関連付けるために使用されます。
-   `work_id` (string, 必須): ターゲットユニットを指定します。`llm-sys` APIの場合、この値は通常`"sys"`です。
-   `action` (string, 必須): `work_id`ユニット内で呼び出される特定のAPIアクション（例：`"ping"`、`"uartsetup"`）。`work_id`と`action`の組み合わせが、事実上APIエンドポイントを定義します。
-   その他のAPI固有フィールド: 一部のAPIでは、`object`や`data`ペイロードのような追加フィールドが必要となり、これらはそれぞれのセクションで詳述されています。

### レスポンス構造

`llm-sys` APIは、要求された操作の結果を示すレスポンスを提供します。

**成功レスポンス:**

成功したAPI呼び出しは、以下の構造を持つJSONオブジェクトを返します：

```json
{
  "request_id": "リクエストからミラーリング",
  "work_id": "sys",
  "action": "APIアクション名", // リクエストからミラーリング
  "created": 1678886400,       // 整数: レスポンス生成のUnixタイムスタンプ
  "code": 0,                   // 整数: 0は成功を示す
  "message": "OK",             // 文字列: 成功メッセージ（例: "OK", "コマンドが実行されました"）
  // "data": { ... }           // オプション: 呼び出しがデータを返す場合、API固有のペイロード
}
```
-   `request_id`, `work_id`, `action`: リクエストからミラーリングされます。
-   `created`: レスポンスが生成された時刻を示すタイムスタンプ。
-   `code`: `0`は操作が成功したことを示します。
-   `message`: 人間が読める形式の成功メッセージ。
-   `data` (object, オプション): API呼び出しがデータを返すことが期待される場合（例：`sys.hwinfo`、`sys.version`）、このオブジェクト内に含まれます。

**エラーレスポンス:**

API呼び出しが失敗した場合、`llm-sys`はエラーレスポンスを返します：

```json
{
  "request_id": "リクエストからミラーリング",
  "work_id": "sys",
  "action": "APIアクション名", // リクエストからミラーリング
  "created": 1678886400,       // 整数: レスポンス生成のUnixタイムスタンプ
  "error": {
    "code": -1,                // 整数: 0以外のエラーコード
    "message": "エラー詳細" // 文字列: 詳細なエラーメッセージ
  }
}
```
-   `error` (object): 失敗に関する詳細情報が含まれます。
    -   `code` (integer): エラータイプを表す0以外の数値。
    -   `message` (string): 人間が読める形式のエラー記述。

**一般的なエラーコード:**

-   `-1`: 一般的なエラー / 未特定のエラー。
-   `-2`: 無効なパラメータ（例：必須フィールドの欠落、不正なデータ型）。
-   `-3`: デバイス/リソースがビジーまたは利用不可。
-   `-4`: サポートされていない操作。
-   `-5`: タイムアウト（例：他のサービスとの通信時に内部`unit_call_timeout`に到達）。

### 非同期操作とイベント通知

-   **同期性**: ほとんどの`llm-sys` API呼び出しは、リクエスト/レスポンスの対話に関しては同期的です。クライアントはリクエストを送信し、呼び出しの即時結果を示すレスポンスを待ちます。
-   **非同期効果**: 特定の操作は、その性質上、システム状態に対して非同期的な効果をもたらします。例：
    -   `sys.reset`: LLMフレームワークアプリケーションをリセットします。システムが再起動し、進行中の接続は失われます。
    -   `sys.reboot`: デバイス全体を再起動します。
    -   `sys.bashexec`: コマンドを開始するためのAPI呼び出しは同期的かもしれませんが、コマンド自体はバックグラウンドで実行され、その完了は初期APIレスポンスでは直接追跡されません。
-   **イベント通知**: `llm-sys`は現在、`llm-audio`の`callback_topic`のような汎用イベント通知システムを備えていません。「upgrade over」や「reset over」のような特定のステータスメッセージは、主にシステム起動時や特定の操作中にシリアルコンソールに出力するために設計されており、クライアント向けの一般的なイベントAPIの一部ではありません。一般的なAPI対話の場合、クライアントは直接の成功またはエラーレスポンスに依存する必要があります。

## 外部API

このセクションでは、`llm-sys`ユニットによって提供される外部APIについて詳述します。

### **sys.ping**

-   **説明**: `llm-sys`ユニットとの接続性をテストします。ヘルスチェックやサービスが応答していることを確認するのに役立ちます。
-   **リクエストパラメータ**: 標準リクエスト構造以外のパラメータはありません。
    ```json
    {
      "request_id": "ping_001",
      "work_id": "sys",
      "action": "ping"
    }
    ```
-   **レスポンス (成功)**:
    ```json
    {
      "request_id": "ping_001",
      "work_id": "sys",
      "action": "ping",
      "created": 1678886401,
      "code": 0,
      "message": "OK"
    }
    ```
-   **レスポンス (エラー)**:
    ```json
    {
      "request_id": "ping_001",
      "work_id": "sys",
      "action": "ping",
      "created": 1678886402,
      "error": {
        "code": -1,
        "message": "pingの処理に失敗しました"
      }
    }
    ```

### **sys.lsmode**

-   **説明**: システムによって登録または認識されたモデルをリストします。「モデル」の正確なソースと範囲（例：AIモデル、システムモジュール）は、システム全体のアーキテクチャからさらなる文脈が必要な場合があります。通常、`config_lsmod_dir`で指定されたパスにあるディレクトリをリストします。
-   **リクエストパラメータ**: 標準リクエスト構造以外のパラメータはありません。
    ```json
    {
      "request_id": "lsmode_002",
      "work_id": "sys",
      "action": "lsmode"
    }
    ```
-   **レスポンス (成功)**: `data`フィールドにモデルのリストが含まれます。このデータの正確な構造（例：文字列の配列、オブジェクトの配列）は提供されたスニペットからは完全には定義されていませんが、おそらくモデル名またはモデル詳細を持つオブジェクトの配列になるでしょう。
    ```json
    {
      "request_id": "lsmode_002",
      "work_id": "sys",
      "action": "lsmode",
      "created": 1678886403,
      "code": 0,
      "message": "OK",
      "data": {
        "models": [ "model_A", "model_B" ] // 構造例
      }
    }
    ```
-   **レスポンス (エラー)**:
    ```json
    {
      "request_id": "lsmode_002",
      "work_id": "sys",
      "action": "lsmode",
      "created": 1678886404,
      "error": {
        "code": -1,
        "message": "モードのリスト表示に失敗しました"
      }
    }
    ```

### **sys.bashexec**

-   **説明**: デバイス上でシェルコマンドを実行します。このAPIは強力なシステム対話を可能にしますが、セキュリティ上の影響があるため注意して使用する必要があります。コマンドをチャンクで送信することをサポートしています。
-   **リクエストパラメータ**:

    | パラメータ       | 型      | 必須 | デフォルト          | 説明                                                                                                 |
    |------------------|---------|------|---------------------|------------------------------------------------------------------------------------------------------|
    | `object`         | string  | はい | N/A                 | データストリームのタイプを指定します。通常、プレーンテキストのコマンドと出力には`"sys.utf-8.stream"`を使用します。 |
    | `data`           | object  | はい | N/A                 | コマンド詳細を含むペイロード。                                                                         |
    | `data.index`     | integer | はい | N/A                 | コマンドチャンクのシーケンス番号で、`0`から始まります。単一の非チャンクコマンドの場合は`0`を使用します。        |
    | `data.delta`     | string  | はい | N/A                 | コマンド文字列またはコマンド文字列のチャンク。                                                              |
    | `data.finish`    | boolean | はい | N/A                 | これがコマンドの最終または唯一のチャンクである場合は`true`に設定します。中間チャンクの場合は`false`に設定します。 |

    **例 (単一コマンド):**
    ```json
    {
      "request_id": "bashexec_003",
      "work_id": "sys",
      "action": "bashexec",
      "object": "sys.utf-8.stream",
      "data": {
        "index": 0,
        "delta": "ls -l /tmp",
        "finish": true
      }
    }
    ```
    **例 (チャンクコマンド - 概念):**
    `echo "hello world"`を2つのチャンクで送信する場合：
    チャンク 1:
    ```json
    {
      "request_id": "bashexec_004a",
      "work_id": "sys",
      "action": "bashexec",
      "object": "sys.utf-8.stream",
      "data": {
        "index": 0,
        "delta": "echo \"hello ",
        "finish": false
      }
    }
    ```
    チャンク 2:
    ```json
    {
      "request_id": "bashexec_004b",
      "work_id": "sys",
      "action": "bashexec",
      "object": "sys.utf-8.stream",
      "data": {
        "index": 1,
        "delta": "world\"",
        "finish": true
      }
    }
    ```

-   **レスポンス (成功)**: レスポンスは、コマンドが実行のために受け入れられたことを示します。コマンドの実際の出力は、通常この同期レスポンスでは返されません。コマンド出力をキャプチャする必要がある場合は、通常、出力をファイルにリダイレクトするか、このAPI自体では定義されていない別のメカニズムを使用します。
    ```json
    {
      "request_id": "bashexec_003",
      "work_id": "sys",
      "action": "bashexec",
      "created": 1678886405,
      "code": 0,
      "message": "コマンド実行が開始されました" // または同様のメッセージ
    }
    ```
-   **レスポンス (エラー)**:
    ```json
    {
      "request_id": "bashexec_003",
      "work_id": "sys",
      "action": "bashexec",
      "created": 1678886406,
      "error": {
        "code": -2,
        "message": "bashexecのパラメータが無効です（例：dataフィールドの欠落）"
      }
    }
    ```

### **sys.hwinfo**

-   **説明**: CPUステータス、メモリ使用量、温度など、デバイスからハードウェア情報を取得します。返される正確なデータは異なる場合があります。
-   **リクエストパラメータ**: 標準リクエスト構造以外のパラメータはありません。
    ```json
    {
      "request_id": "hwinfo_004",
      "work_id": "sys",
      "action": "hwinfo"
    }
    ```
-   **レスポンス (成功)**: `data`オブジェクトにハードウェア情報が含まれます。以下の構造は一例です。実際のフィールドは異なる場合があります。
    ```json
    {
      "request_id": "hwinfo_004",
      "work_id": "sys",
      "action": "hwinfo",
      "created": 1678886407,
      "code": 0,
      "message": "OK",
      "data": {
        "cpu_usage": "25%", // 例
        "memory_free": "2048MB", // 例
        "temperature_celsius": 45 // 例
      }
    }
    ```
-   **レスポンス (エラー)**:
    ```json
    {
      "request_id": "hwinfo_004",
      "work_id": "sys",
      "action": "hwinfo",
      "created": 1678886408,
      "error": {
        "code": -1,
        "message": "ハードウェア情報の取得に失敗しました"
      }
    }
    ```

### **sys.uartsetup**

-   **説明**: シリアルポート（UART）のパラメータを設定します。設定は通常、現在のセッションまたは次の再起動/リセットまで有効です。デフォルトのシリアルデバイスは通常`/dev/ttyS1`です。
-   **リクエストパラメータ**:

    | パラメータ        | 型      | 必須 | デフォルト (`main.cpp`より) | 説明                                                                                                     |
    |-------------------|---------|------|---------------------------|----------------------------------------------------------------------------------------------------------|
    | `object`          | string  | はい | N/A                       | ターゲットオブジェクトを指定します。通常は`"sys.uartsetup"`です。                                              |
    | `data`            | object  | はい | N/A                       | UART設定を含むペイロード。                                                                                |
    | `data.baud`       | integer | はい | 115200                    | ボーレート（例：9600、115200）。                                                                        |
    | `data.data_bits`  | integer | はい | 8                         | データビット数（例：7、8）。                                                                              |
    | `data.stop_bits`  | integer | はい | 1                         | ストップビット数（例：1、2）。                                                                            |
    | `data.parity`     | integer | はい | 110                       | パリティ設定。`110`（'n'のASCII値）はNoneを意味します。他の値は、サポートされていればEvenまたはOddに対応する場合があります。 |

    **例:**
    ```json
    {
      "request_id": "uartsetup_005",
      "work_id": "sys",
      "action": "uartsetup",
      "object": "sys.uartsetup",
      "data": {
        "baud": 115200,
        "data_bits": 8,
        "stop_bits": 1,
        "parity": 110 // 'n' はパリティなし
      }
    }
    ```
-   **レスポンス (成功)**:
    ```json
    {
      "request_id": "uartsetup_005",
      "work_id": "sys",
      "action": "uartsetup",
      "created": 1678886409,
      "code": 0,
      "message": "UARTが正常に設定されました"
    }
    ```
-   **レスポンス (エラー)**:
    ```json
    {
      "request_id": "uartsetup_005",
      "work_id": "sys",
      "action": "uartsetup",
      "created": 1678886410,
      "error": {
        "code": -2,
        "message": "無効なUARTパラメータ（例：サポートされていないボーレート）"
      }
    }
    ```

### **sys.reset**

-   **説明**: LLMフレームワークアプリケーション全体をリセットします。これにより通常、`llm-sys`サービスおよびその他の関連サービスが再起動され、現在の操作と接続が終了します。
-   **リクエストパラメータ**: 標準リクエスト構造以外のパラメータはありません。
    ```json
    {
      "request_id": "reset_006",
      "work_id": "sys",
      "action": "reset"
    }
    ```
-   **レスポンス (成功)**: 成功レスポンスは、リセットコマンドが受け入れられたことを示します。実際のリセットは非同期効果です。
    ```json
    {
      "request_id": "reset_006",
      "work_id": "sys",
      "action": "reset",
      "created": 1678886411,
      "code": 0,
      "message": "LLMフレームワークのリセットが開始されました"
    }
    ```
    リセット完了後、シリアルコンソールに「reset over」というメッセージが送信される場合があります。
-   **レスポンス (エラー)**:
    ```json
    {
      "request_id": "reset_006",
      "work_id": "sys",
      "action": "reset",
      "created": 1678886412,
      "error": {
        "code": -1,
        "message": "リセットの開始に失敗しました"
      }
    }
    ```

### **sys.reboot**

-   **説明**: デバイス/システム全体を再起動します。これは`sys.reset`よりも抜本的な操作です。
-   **リクエストパラメータ**: 標準リクエスト構造以外のパラメータはありません。
    ```json
    {
      "request_id": "reboot_007",
      "work_id": "sys",
      "action": "reboot"
    }
    ```
-   **レスポンス (成功)**: 再起動コマンドが受け入れられたことを示します。実際の再起動は非同期効果です。
    ```json
    {
      "request_id": "reboot_007",
      "work_id": "sys",
      "action": "reboot",
      "created": 1678886413,
      "code": 0,
      "message": "システムの再起動が開始されました"
    }
    ```
-   **レスポンス (エラー)**:
    ```json
    {
      "request_id": "reboot_007",
      "work_id": "sys",
      "action": "reboot",
      "created": 1678886414,
      "error": {
        "code": -1,
        "message": "再起動の開始に失敗しました"
      }
    }
    ```

### **sys.version**

-   **説明**: LLMフレームワークプログラムまたは`llm-sys`ユニットのバージョンを取得します。
-   **リクエストパラメータ**: 標準リクエスト構造以外のパラメータはありません。
    ```json
    {
      "request_id": "version_008",
      "work_id": "sys",
      "action": "version"
    }
    ```
-   **レスポンス (成功)**: `data`オブジェクトにバージョン情報が含まれます。以下の構造は一例です。
    ```json
    {
      "request_id": "version_008",
      "work_id": "sys",
      "action": "version",
      "created": 1678886415,
      "code": 0,
      "message": "OK",
      "data": {
        "firmware_version": "1.2.3", // 例フィールド
        "build_date": "2024-03-15"  // 例フィールド
      }
    }
    ```
-   **レスポンス (エラー)**:
    ```json
    {
      "request_id": "version_008",
      "work_id": "sys",
      "action": "version",
      "created": 1678886416,
      "error": {
        "code": -1,
        "message": "バージョン情報の取得に失敗しました"
      }
    }
    ```

## 設定ファイルとの関連性

`llm-sys`ユニットの動作は、主に`projects/llm_framework/main_sys/sys_config.json`にある設定と、`projects/llm_framework/main_sys/src/main.cpp`内で定義されたデフォルト値によって影響を受けます。

### `sys_config.json`

-   `config_enable_tcp` (整数, 0 または 1):
    -   `1`に設定されている場合（ファイルが空か値がない場合、`main.cpp`に基づいてデフォルト）、`llm-sys`用のTCPサーバーインターフェースが有効になり、TCP経由でのAPI呼び出しが可能になります。
    -   `0`に設定されている場合、TCPサーバーは無効になります。API呼び出しは通常、シリアルなどの他のチャネルを介して行われます。
    -   デフォルトのTCPサーバーポートは`10001`です（`main.cpp`の`config_tcp_server`より）。

### `main.cpp`内のデフォルト設定 (`get_run_config`)

`main.cpp`ファイルは、`sys_config.json`や他の手段によって上書きされない限り、インメモリのキーバリューストア（`key_sql`）に一連のデフォルトパラメータを初期化します。主要な設定は以下の通りです：

-   **`config_unit_call_timeout`** (整数, デフォルト: 5000ミリ秒): このタイムアウトは、`llm-sys`が他のユニット/サービスを呼び出す際に`remote_action.cpp`で使用されます。呼び出されたユニットがこの期間内に応答しない場合、呼び出しはタイムアウトエラーで失敗する可能性があります。
-   **シリアルポートのデフォルト**:
    -   `config_serial_dev`: デフォルトのシリアルデバイス（例：`"/dev/ttyS1"`）。
    -   `config_serial_baud`: デフォルトのボーレート（例：`115200`）。
    -   `config_serial_data_bits`: デフォルトのデータビット（例：`8`）。
    -   `config_serial_stop_bits`: デフォルトのストップビット（例：`1`）。
    -   `config_serial_parity`: デフォルトのパリティ（例：`110`、'n' - None）。
    これらのデフォルトは、特定のセッションで`sys.uartsetup`呼び出しによって上書きされない限り使用されます。
-   **ZMQ通信**: 様々な`config_zmq_*`設定が、内部ZMQベース通信用のIPCソケットパスとポート範囲を定義します。
-   **パス**: `config_lsmod_dir` (デフォルト: `"/opt/m5stack/data/models/"`) は `sys.lsmode` で使用されます。
-   **TCPサーバー**: `config_tcp_server` (デフォルト: `10001`) は `config_enable_tcp` が有効な場合にTCPサーバーのポートを指定します。

これらの設定を理解することは、問題の診断や`llm-sys`の動作をカスタマイズするのに役立ちます。これらの永続的な設定を変更するには、`sys_config.json`の変更または`main.cpp`の再コンパイル（デフォルト値の場合）が必要になります。

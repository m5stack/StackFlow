# LLM
AX630C AI large model local voice assistant program. Provided by M5STACK STACKFLOW for basic communication architecture, coordinating multiple working units to work together.
- llm-audio: Audio unit, providing system audio support.
- llm-kws: Keyword detection unit, providing keyword wakeup service.
- llm-asr: Speech-to-text unit, providing speech-to-text service.
- llm-llm: Large model unit, providing AI large model inference service.
- llm-melotts: NPU-accelerated TTS unit, providing text-to-speech and playback service.
- llm-tts: CPU-computed TTS unit, providing text-to-speech and playback service.

## compile
```bash
wget https://m5stack.oss-cn-shenzhen.aliyuncs.com/resource/linux/llm/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.gz
sudo tar zxvf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.gz -C /opt

sudo apt install python3 python3-pip libffi-dev
pip3 install parse scons requests kconfiglib

git clone https://github.com/m5stack/StackFlow.git
cd StackFlow
git submodule update --init
cd projects/llm_framework
scons distclean
scons -j22

```


## assistant mode
llm-sys llm-audio llm-kws llm-asr llm-llm llm-melotts Collaborative work.

Chinese environment:

1、reset

send :

```json
{
    "request_id": "11212155", 
    "work_id": "sys",
    "action": "reset"
}
```
waiting reset over!

2、init all unit

send :

```json
{
    "request_id": "1",
    "work_id": "kws",
    "action": "setup",
    "object": "kws.setup",
    "data": {
        "model": "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01","response_format": "kws.bool",
        "input": "sys.pcm",
        "enoutput": true,
        "kws": "你好你好"
        }
}

{
    "request_id": "2",
    "work_id": "asr",
    "action": "setup",
    "object": "asr.setup",
    "data": {
        "model": "sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23",
        "response_format": "asr.utf-8.stream",
        "input": "sys.pcm",
        "enoutput": true,
        "enkws":true,
        "rule1":2.4,
        "rule2":1.2,
        "rule3":30.1
        }
}

{
    "request_id": "3",
    "work_id": "llm",
    "action": "setup","object": "llm.setup",
    "data": {
        "model": "qwen2.5-0.5B-prefill-20e",
        "response_format": "llm.utf-8.stream",
        "input": "llm.utf-8",
        "enoutput": true,
        "max_token_len": 256,
        "prompt": "You are a knowledgeable assistant capable of answering various questions and providing information."
        }
}

{
    "request_id": "4",
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
waiting return work_id:
```json
{"created":1731488371,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"3","work_id":"asr.1001"}
{"created":1731488377,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"4","work_id":"llm.1002"}
{"created":1731488392,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"4","work_id":"melotts.1003"}
{"created":1731488402,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"2","work_id":"kws.1000"}
```

creat assembly line：
```json
{
    "request_id": "2",
    "work_id": "asr.1001",
    "action": "link",
    "object":"work_id",
    "data":"kws.1000"
}

{
    "request_id": "3",
    "work_id": "llm.1002",
    "action": "link",
    "object":"work_id",
    "data":"asr.1001"
}

{
    "request_id": "4",
    "work_id": "melotts.1003",
    "action": "link",
    "object":"work_id",
    "data":"llm.1002"
}

{
    "request_id": "3",
    "work_id": "llm.1002",
    "action": "link",
    "object":"work_id",
    "data":"kws.1000"
}
{
    "request_id": "4",
    "work_id": "melotts.1003",
    "action": "link",
    "object":"work_id",
    "data":"kws.1000"
}

```

waiting return status:

```json
{"created":1731488403,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"3","work_id":"llm.1002"}
{"created":1731488403,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"4","work_id":"melotts.1003"}
{"created":1731488403,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"2","work_id":"asr.1001"}
{"created":1731488403,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"4","work_id":"melotts.1003"}
{"created":1731488403,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"3","work_id":"llm.1002"}
```

**Please enjoy the large model voice assistant.**


## cv mode
1、reset

send :

```json
{
    "request_id": "11212155", 
    "work_id": "sys",
    "action": "reset"
}
```
waiting reset over!

2、init camera unit

send :

```json
{
   "request_id":"4",
   "work_id":"camera",
   "action":"setup",
   "object":"camera.setup",
   "data":{
      "response_format":"camera.raw",
      "input":"/dev/video0",
      "enoutput":false,
      "frame_width":320,
      "frame_height":320
   }
}
```
waiting return work_id:
```json
{"created":1731488371,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"4","work_id":"camera.1000"}
```

2、init yolo unit

send :

```json
{
   "request_id":"5",
   "work_id":"yolo",
   "action":"setup",
   "object":"yolo.setup",
   "data":{
      "model":"yolo11n",
      "response_format":"yolo.yolobox",
      "input":"camera.1000",
      "enoutput":true
   }
}
```
waiting return work_id:
```json
{"created":1731488371,"data":"None","error":{"code":0,"message":""},"object":"None","request_id":"5","work_id":"yolo.1001"}
```

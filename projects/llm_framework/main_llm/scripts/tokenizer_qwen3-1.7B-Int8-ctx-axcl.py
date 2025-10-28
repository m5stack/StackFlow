from transformers import AutoTokenizer, PreTrainedTokenizerFast
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import json
import argparse
import uuid
from urllib.parse import urlparse, parse_qs

tokenizers = {}

class Tokenizer_Http():
    def __init__(self, model_id, system_prompt):
        self.model_id = model_id
        self.system_prompt = system_prompt

        qwen3_path  = os.path.join(model_id, 'qwen3_tokenizer')
        qwen25_path = os.path.join(model_id, 'qwen2.5_tokenizer')

        self.tokenizer    = AutoTokenizer.from_pretrained(str(qwen3_path))
        self.tokenizer_25 = AutoTokenizer.from_pretrained(str(qwen25_path))

        self.messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        self.token_ids = []
        self.token_ids_cache = []

    def remove_think(self, text:str):
        text = text.replace("<think>","")
        text = text.replace("</think>","")
        return text.strip("\n")

    def encode(self, prompt: str, last_reply: str = None):
        if last_reply is not None:
            last_reply = self.remove_think(last_reply)
            self.messages.append({"role": "assistant", "content": last_reply})
            text = self.tokenizer_25.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            print("fff生成的文本:\n============\n", text, "============\n")
            self.token_ids = self.tokenizer.encode(text)[:-3]
            print("diff:", self.decode(self.token_ids))

        if not prompt.endswith("/no_think"):
            prompt += "/no_think"
        print("prompt:", prompt)
        self.messages.append({"role": "user", "content": prompt})

        text = self.tokenizer_25.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print("生成的文本:\n============\n", text, "============\n")
        token_ids = self.tokenizer.encode(text)
        # 找出新增部分
        diff = token_ids[len(self.token_ids):]
        self.token_ids = token_ids
        print("diff:", self.decode(diff))
        return token_ids, diff

    def decode(self, token_ids):
        self.token_ids_cache += token_ids
        text = self.tokenizer.decode(self.token_ids_cache)
        if "\ufffd" in text:
            print("text 中包含非法字符")
            return ""
        else:
            self.token_ids_cache.clear()
            return text

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def bos_token(self):
        return self.tokenizer.bos_token

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    def reset(self, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt
        else:
            self.system_prompt = system_prompt

        self.messages = [
            {"role": "system", "content": system_prompt},
        ]
        text = self.tokenizer_25.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(text)[:-3]
        self.token_ids = token_ids
        print(self.decode(token_ids))
        return token_ids


class Request(BaseHTTPRequestHandler):
    timeout = 5
    server_version = 'Apache'
    args = None

    def get_query_param(self, key):
        """
        从 GET 请求的 URL 中获取查询参数的值
        例如：/bos_id?uid=xxx
        """
        query = urlparse(self.path).query
        params = parse_qs(query)
        values = params.get(key)
        return values[0] if values else None

    def do_GET(self):
        print("GET 请求路径:", self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if '/get_uid' in self.path:
            system_prompt = self.get_query_param("system_prompt") or self.args.content
            model_id = self.get_query_param("model_id") or self.args.model_id

            new_uid = str(uuid.uuid4())
            tokenizers[new_uid] = Tokenizer_Http(model_id=model_id, system_prompt=system_prompt)
            msg = json.dumps({'uid': new_uid})

        elif '/bos_id' in self.path:
            uid = self.get_query_param("uid")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                bos_id = instance.bos_id
                msg = json.dumps({'bos_id': bos_id if bos_id is not None else -1})

        elif '/eos_id' in self.path:
            uid = self.get_query_param("uid")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                eos_id = instance.eos_id
                msg = json.dumps({'eos_id': eos_id if eos_id is not None else -1})
        else:
            msg = json.dumps({'error': 'Invalid GET endpoint'})

        print("响应消息:", msg)
        self.wfile.write(msg.encode())

    def do_POST(self):
        content_length = int(self.headers.get('content-length', 0))
        data = self.rfile.read(content_length).decode()
        print("POST 请求路径:", self.path)
        print("接收到的数据:", data)
        req = json.loads(data)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if '/encode' in self.path:
            uid = req.get('uid')
            prompt = req.get('text')
            last_reply = req.get('last_reply')
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                token_ids, diff = instance.encode(prompt, last_reply)
                msg = json.dumps({'token_ids': token_ids, 'diff': diff})

        elif '/decode' in self.path:
            uid = req.get('uid')
            token_ids = req.get('token_ids')
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                text = instance.decode(token_ids)
                msg = json.dumps({'text': text})

        elif '/reset' in self.path:
            uid = req.get("uid")
            system_prompt = req.get("system_prompt")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                token_ids = instance.reset(system_prompt)
                msg = json.dumps({'token_ids': token_ids})

        else:
            msg = json.dumps({'error': 'Invalid POST endpoint'})

        print("响应消息:", msg)
        self.wfile.write(msg.encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--model_id', type=str, default='qwen3_1.7B_tokenizer')
    parser.add_argument('--content', type=str, default='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.')
    args = parser.parse_args()

    Request.args = args

    host = (args.host, args.port)
    print('Server running at http://%s:%s' % host)
    server = HTTPServer(host, Request)
    server.serve_forever()
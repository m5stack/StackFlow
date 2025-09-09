from transformers import AutoTokenizer
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import argparse
import uuid

tokenizers = {}

class Tokenizer_Http():
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.messages = [
            {"role": "system", "content": "你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。"},
        ]
        self.token_ids = []
        self.token_ids_cache = []

    def encode(self, prompt, last_reply=None):
        if last_reply is not None:
            self.messages.append({"role": "assistant", "content": last_reply})
            text = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            self.token_ids = self.tokenizer.encode(text)[:-3]
        self.messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(text)
        diff = token_ids[len(self.token_ids):]
        self.token_ids = token_ids
        return token_ids, diff

    def encode_with_image(self, question: str, num_of_images: int, imgsz: int, last_reply=None):
        if last_reply is not None:
            self.messages.append({"role": "assistant", "content": last_reply})

        # 根据图片尺寸设定 context_len
        if imgsz == 448:
            context_len = 256
        elif imgsz == 224:
            context_len = 64
        else:
            print(f"Unsupported imgsz: {imgsz}")
            return None, None

        # 拼接带图片的用户输入
        question_with_images = question
        if num_of_images > 0:
            for _ in range(num_of_images):
                question_with_images += "\n<img>" + "<IMG_CONTEXT>" * context_len + "</img>\n"

        self.messages.append({"role": "user", "content": question_with_images})

        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        token_ids = self.tokenizer.encode(text)
        diff = token_ids[len(self.token_ids):]
        self.token_ids = token_ids
        return token_ids, diff

    def decode(self, token_ids):
        self.token_ids_cache += token_ids
        text = self.tokenizer.decode(self.token_ids_cache)
        if "\ufffd" in text:
            print("Text 中包含非法字符")
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
    @property
    def img_start_token(self):
        return self.tokenizer.encode("<img>")[0]
    @property
    def img_context_token(self):
        return self.tokenizer.encode("<IMG_CONTEXT>")[0]

    def reset(self, system_prompt=None):
        if system_prompt is None:
            system_prompt = args.content
        self.messages = [
            {"role": "system", "content": system_prompt},
        ]
        text = self.tokenizer.apply_chat_template(
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

    def do_GET(self):
        print(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if '/get_uid' in self.path:
            new_uid = str(uuid.uuid4())
            print("新 uid:", new_uid)
            tokenizers[new_uid] = Tokenizer_Http(args.model_id)
            msg = json.dumps({'uid': new_uid})
        elif '/bos_id' in self.path:
            uid = self.get_query_param("uid")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                msg = json.dumps({'bos_id': instance.bos_id if instance.bos_id is not None else -1})
        elif '/eos_id' in self.path:
            uid = self.get_query_param("uid")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                msg = json.dumps({'eos_id': instance.eos_id if instance.eos_id is not None else -1})
        elif '/img_start_token' in self.path:
            uid = self.get_query_param("uid")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                msg = json.dumps({'img_start_token': instance.img_start_token})

        elif '/img_context_token' in self.path:
            uid = self.get_query_param("uid")
            instance: Tokenizer_Http = tokenizers.get(uid)
            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                msg = json.dumps({'img_context_token': instance.img_context_token})
        else:
            msg = json.dumps({'error': 'Invalid GET endpoint'})
        print(msg)
        self.wfile.write(msg.encode())

    def do_POST(self):
        content_length = int(self.headers.get('content-length', 0))
        data = self.rfile.read(content_length).decode()
        req = json.loads(data)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if '/encode' in self.path:
            uid = req.get('uid')
            prompt = req.get('text')
            last_reply = req.get('last_reply')
            b_img_prompt = False
            instance: Tokenizer_Http = tokenizers.get(uid)
            if 'img_prompt' in req:
                b_img_prompt = req['img_prompt']
            if b_img_prompt:
                num_img = req['num_img']
                imgsz = req['imgsz']

            if instance is None:
                msg = json.dumps({'error': 'Invalid uid'})
            else:
                if b_img_prompt:
                    token_ids, diff = instance.encode_with_image(prompt, num_img, imgsz, last_reply)
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
                if system_prompt is not None:
                    print("system_prompt:", system_prompt)
                    token_ids = instance.reset(system_prompt)
                else:
                    token_ids = instance.reset()
                msg = json.dumps({'token_ids': token_ids})

        else:
            msg = json.dumps({'error': 'Invalid POST endpoint'})

        self.wfile.write(msg.encode())

    def get_query_param(self, key):
        from urllib.parse import urlparse, parse_qs
        query = urlparse(self.path).query
        params = parse_qs(query)
        values = params.get(key)
        return values[0] if values else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument('--model_id', type=str, default='internvl3_tokenizer')
    parser.add_argument('--content', type=str, default='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。')
    args = parser.parse_args()

    host = (args.host, args.port)
    print("http://%s:%s" % host)
    server = HTTPServer(host, Request)
    server.serve_forever()
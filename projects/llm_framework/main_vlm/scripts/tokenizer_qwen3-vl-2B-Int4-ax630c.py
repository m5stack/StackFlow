from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import AddedToken
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import argparse

class Tokenizer_Http:
    def __init__(self, model_id, system_content="You are a helpful assistant."):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False
        )
        self.token_ids_cache = []
        self.system_content = system_content

    def encode(self, content):
        text = [
            f'<|im_start|>system\n{self.system_content}<|im_end|>\n'
            f'<|im_start|>user\n{content}<|im_end|>\n'
            f'<|im_start|>assistant\n'
        ]
        input_ids = self.tokenizer(text)
        return input_ids["input_ids"][0]

    def encode_vpm_image(self, content="Describe this image.", num_img=1, img_token_num=256):
        imgs_token = (
            '<|vision_start|>'
            + '<|image_pad|>' * img_token_num
            + '<|vision_end|>'
        )
        imgs_token *= num_img
        text = (
            f'<|im_start|>system\n{self.system_content}<|im_end|>\n'
            f'<|im_start|>user\n{imgs_token}{content}<|im_end|>\n'
            f'<|im_start|>assistant\n'
        )
        text_inputs = self.tokenizer([text])
        return text_inputs["input_ids"][0]

    def encode_vpm_video(self, content="Describe this image.", num_img=1, img_token_num=256):
        imgs_token = (
            '<|vision_start|>'
            + '<|video_pad|>' * img_token_num * num_img
            + '<|vision_end|>'
        )
        text = (
            f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
            f'<|im_start|>user\n{imgs_token}{content}<|im_end|>\n'
            f'<|im_start|>assistant\n'
        )
        text_inputs = self.tokenizer([text])
        return text_inputs["input_ids"][0]
    
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

    @property
    def img_start_token(self):
        return self.tokenizer.encode("<|vision_start|>")[0]

    @property
    def img_context_token(self):
        return self.tokenizer.encode("<|image_pad|>")[0]

class Request(BaseHTTPRequestHandler):
    timeout = 5
    server_version = 'Apache'

    def do_GET(self):
        print(self.path)
        self.send_response(200)
        self.send_header("type", "get")
        self.end_headers()
        if self.path == '/bos_id':
            bos_id = tokenizer.bos_id
            msg = json.dumps({'bos_id': -1 if bos_id is None else bos_id})
        elif self.path == '/eos_id':
            eos_id = tokenizer.eos_id
            msg = json.dumps({'eos_id': -1 if eos_id is None else eos_id})
        elif self.path == '/img_start_token':
            img_start_token = tokenizer.img_start_token
            msg = json.dumps({'img_start_token': -1 if img_start_token is None else img_start_token})
        elif self.path == '/img_context_token':
            img_context_token = tokenizer.img_context_token
            msg = json.dumps({'img_context_token': -1 if img_context_token is None else img_context_token})
        else:
            msg = 'error'
        print(msg)
        msg = str(msg).encode()
        self.wfile.write(msg)

    def do_POST(self):
        data = self.rfile.read(int(self.headers['content-length']))
        req = json.loads(data.decode())
        if self.path == "/encode":
            prompt = req['text']
            b_img_prompt = req.get('img_prompt', False)
            img_type = req.get('img_type', 'image')
            if b_img_prompt:
                if img_type == 'image':
                    token_ids = tokenizer.encode_vpm_image(
                        prompt,
                        req.get("num_img", 1),
                        req.get("img_token_num", 256)
                    )
                elif img_type == 'video':
                    token_ids = tokenizer.encode_vpm_video(
                        prompt,
                        req.get("num_img", 1),
                        req.get("img_token_num", 256)
                    )
                else:
                    token_ids = tokenizer.encode(prompt)
            else:
                token_ids = tokenizer.encode(prompt)
            msg = json.dumps({'token_ids': -1 if token_ids is None else token_ids})
        elif self.path == "/decode":
            token_ids = req['token_ids']
            text = tokenizer.decode(token_ids)
            msg = json.dumps({'text': "" if text is None else text})
        else:
            msg = 'error'
        self.send_response(200)
        self.end_headers()
        self.wfile.write(str(msg).encode())

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--host', type=str, default='localhost')
    args.add_argument('--port', type=int, default=8080)
    args.add_argument('--model_id', type=str, default='tokenizer')
    args.add_argument('--content', type=str, default='You are a helpful assistant.')
    args = args.parse_args()
    tokenizer = Tokenizer_Http(args.model_id, system_content=args.content)
    host = (args.host, args.port)
    print(f"http://{args.host}:{args.port}")
    server = HTTPServer(host, Request)
    server.serve_forever()
from transformers import AutoTokenizer
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import argparse


class Tokenizer_Http():

    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                       trust_remote_code=True,
                                                       use_fast=False)

    def encode(self, prompt, content):
        prompt = f"<|im_start|>system\n{content}<|im_end|><|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n"
        input_ids = self.tokenizer.encode(prompt)
        return input_ids

    def encode_with_image(self, prompt, num_of_images, imgsz, content="Please describe the image shortly.") -> list:
        prompt = "<|im_start|>system\n{content}}<|im_end|>\n"
     
        prompt += "<|im_start|>user\n" + prompt

        context_len = 64
        if imgsz == 448:
            context_len = 256
        elif imgsz == 224:
            context_len = 64
        else:
            print(f"imgsz is {imgsz}")
            return
        print("context_len is ", context_len)
        
        if num_of_images > 0:
            for idx in range(num_of_images):
                prompt += "\n<img>" + "<IMG_CONTEXT>" * context_len + "</img>\n"
        
        prompt += "<|im_end|>\n<|im_start|>assistant"
        print(f"prompt is {prompt}")
        token_ids = self.tokenizer.encode(prompt)
        return token_ids
    
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids,
                                     clean_up_tokenization_spaces=False, skip_special_tokens=True)

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

            if bos_id is None:
                msg = json.dumps({'bos_id': -1})
            else:
                msg = json.dumps({'bos_id': bos_id})
        elif self.path == '/eos_id':
            eos_id = tokenizer.eos_id
            if eos_id is None:
                msg = json.dumps({'eos_id': -1})
            else:
                msg = json.dumps({'eos_id': eos_id})
        elif self.path == '/img_start_token':
            img_start_token = tokenizer.img_start_token
            if img_start_token is None:
                msg = json.dumps({'img_start_token': -1})
            else:
                msg = json.dumps({'img_start_token': img_start_token})
        elif self.path == '/img_context_token':
            img_context_token = tokenizer.img_context_token
            if img_context_token is None:
                msg = json.dumps({'img_context_token': -1})
            else:
                msg = json.dumps({'img_context_token': img_context_token})
        else:
            msg = 'error'

        print(msg)
        msg = str(msg).encode()

        self.wfile.write(msg)

    def do_POST(self):

        data = self.rfile.read(int(
            self.headers['content-length']))
        data = data.decode()

        self.send_response(200)
        self.send_header("type", "post")
        self.end_headers()

        if self.path == '/encode':
            req = json.loads(data)
            print(req)
            prompt = req['text']
            b_img_prompt = False
            if 'img_prompt' in req:
                b_img_prompt = req['img_prompt']
            if b_img_prompt:
                num_img = req['num_img']
                imgsz = req['imgsz']
                token_ids = tokenizer.encode_with_image(prompt, num_img, imgsz)
            else:
                token_ids = tokenizer.encode(prompt)
            if token_ids is None:
                msg = json.dumps({'token_ids': -1})
            else:
                msg = json.dumps({'token_ids': token_ids})

        elif self.path == '/decode':
            req = json.loads(data)
            token_ids = req['token_ids']
            text = tokenizer.decode(token_ids)
            if text is None:
                msg = json.dumps({'text': ""})
            else:
                msg = json.dumps({'text': text})
        else:
            msg = 'error'
        print(msg)
        msg = str(msg).encode()
        self.wfile.write(msg)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--host', type=str, default='0.0.0.0')
    args.add_argument('--port', type=int, default=12345)
    args.add_argument('--model_id', type=str, default='internvl2_tokenizer')
    args.add_argument('--content', type=str, default='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。')
    args = args.parse_args()

    tokenizer = Tokenizer_Http(args.model_id)
    host = (args.host, args.port)
    print('http://%s:%s' % host)
    server = HTTPServer(host, Request)
    server.serve_forever()

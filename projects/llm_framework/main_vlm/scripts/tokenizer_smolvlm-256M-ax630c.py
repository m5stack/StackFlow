from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import AddedToken
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import argparse

def _prompt_split_image(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_img_token,
):
    """Prompt with expanded image tokens for when the image is split into patches."""
    text_split_images = ""
    for n_h in range(image_rows):
        for n_w in range(image_cols):
            text_split_images += (
                f"{fake_token_around_image}"
                + f"<row_{n_h + 1}_col_{n_w + 1}>"
                + f"{image_token}" * image_seq_len
            )
        text_split_images += "\n"

    text_split_images += (
        f"\n{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )
    return text_split_images


def _prompt_single_image(
    image_seq_len, fake_token_around_image, image_token, global_img_token
):
    """Prompt with expanded image tokens for a single image."""
    return (
        f"{fake_token_around_image}"
        + f"{global_img_token}"
        + f"{image_token}" * image_seq_len
        + f"{fake_token_around_image}"
    )


def get_image_prompt_string(
    image_rows,
    image_cols,
    image_seq_len,
    fake_token_around_image,
    image_token,
    global_img_token,
):
    if image_rows == 0 and image_cols == 0:
        return _prompt_single_image(
            image_seq_len,
            fake_token_around_image=fake_token_around_image,
            image_token=image_token,
            global_img_token=global_img_token,
        )
    return _prompt_split_image(
        image_seq_len,
        image_rows,
        image_cols,
        fake_token_around_image,
        image_token,
        global_img_token,
    )

class Tokenizer_Http:

    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False
        )

    def encode(self, prompt, content):
        prompt = f"<|im_start|>User:{content}<end_of_utterance>\nAssistant:"
        input_ids = self.tokenizer(prompt)
        return input_ids["input_ids"]

    def encode_vpm(self, prompt, content="Please describe the image shortly."):
        prompt = f"<|im_start|>User:<image>{prompt}<end_of_utterance>\nAssistant:"
        text = [prompt]
        image_rows = [[0]]
        image_cols = [[0]]
        image_seq_len = 64
        image_token = "<image>"
        fake_image_token = "<fake_token_around_image>"
        global_img_token = "<global-img>"
        prompt_strings = []
        for sample, sample_rows, sample_cols in zip(text, image_rows, image_cols):
            # Replace the image token with fake tokens around the expanded image token sequence of length `image_seq_len`
            image_prompt_strings = []
            for n_rows, n_cols in zip(sample_rows, sample_cols):
                image_prompt_string = get_image_prompt_string(
                    n_rows,
                    n_cols,
                    image_seq_len,
                    image_token=image_token,
                    fake_token_around_image=fake_image_token,
                    global_img_token=global_img_token,
                )
                image_prompt_strings.append(image_prompt_string)

            split_sample = sample.split(image_token)
            if len(split_sample) == 0:
                raise ValueError("The image token should be present in the text.")

            # Place in the image prompt strings where the image tokens are
            sample = split_sample[0]
            for i, image_prompt_string in enumerate(image_prompt_strings):
                sample += image_prompt_string + split_sample[i + 1]
            prompt_strings.append(sample)

        fake_image_token = AddedToken(fake_image_token, normalized=False, special=True)
        image_token = AddedToken(image_token, normalized=False, special=True)
        end_of_utterance_token = AddedToken(
            "<end_of_utterance>", normalized=False, special=True
        )
        tokens_to_add = {
            "additional_special_tokens": [
                fake_image_token,
                image_token,
                end_of_utterance_token,
            ]
        }
        self.tokenizer.add_special_tokens(tokens_to_add)

        input_ids = self.tokenizer(prompt_strings)["input_ids"][0]
        return input_ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

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

class Request(BaseHTTPRequestHandler):
    # 通过类继承，新定义类
    timeout = 5
    server_version = "Apache"

    def do_GET(self):
        print(self.path)
        # 在新类中定义get的内容（当客户端向该服务端使用get请求时，本服务端将如下运行）
        self.send_response(200)
        self.send_header("type", "get")  # 设置响应头，可省略或设置多个
        self.end_headers()

        if self.path == "/bos_id":
            bos_id = tokenizer.bos_id
            # print(bos_id)
            # to json
            if bos_id is None:
                msg = json.dumps({"bos_id": -1})
            else:
                msg = json.dumps({"bos_id": bos_id})
        elif self.path == "/eos_id":
            eos_id = tokenizer.eos_id
            if eos_id is None:
                msg = json.dumps({"eos_id": -1})
            else:
                msg = json.dumps({"eos_id": eos_id})
        else:
            msg = "error"

        print(msg)
        msg = str(msg).encode()  # 转为str再转为byte格式

        self.wfile.write(msg)  # 将byte格式的信息返回给客户端

    def do_POST(self):
        # 在新类中定义post的内容（当客户端向该服务端使用post请求时，本服务端将如下运行）
        data = self.rfile.read(
            int(self.headers["content-length"])
        )  # 获取从客户端传入的参数（byte格式）
        data = data.decode()  # 将byte格式转为str格式

        self.send_response(200)
        self.send_header("type", "post")  # 设置响应头，可省略或设置多个
        self.end_headers()

        if self.path == "/encode":
            req = json.loads(data)
            print(req)
            prompt = req["text"]
            b_img_prompt = False
            if "img_prompt" in req:
                b_img_prompt = req["img_prompt"]
            if b_img_prompt:
                token_ids = tokenizer.encode_vpm(prompt)
            else:
                token_ids = tokenizer.encode(prompt, args.content)
            if token_ids is None:
                msg = json.dumps({"token_ids": -1})
            else:
                msg = json.dumps({"token_ids": token_ids})

        elif self.path == "/decode":
            req = json.loads(data)
            token_ids = req["token_ids"]
            text = tokenizer.decode(token_ids)
            if text is None:
                msg = json.dumps({"text": ""})
            else:
                msg = json.dumps({"text": text})
        else:
            msg = "error"
        print(msg)
        msg = str(msg).encode()  # 转为str再转为byte格式

        self.wfile.write(msg)  # 将byte格式的信息返回给客户端


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--host", type=str, default="localhost")
    args.add_argument("--port", type=int, default=8080)
    args.add_argument('--model_id', type=str, default='internvl2_tokenizer')
    args.add_argument('--content', type=str, default='')
    args = args.parse_args()

    tokenizer = Tokenizer_Http(args.model_id)


    # print(tokenizer.bos_id, tokenizer.bos_token, tokenizer.eos_id, tokenizer.eos_token)
    # print(tokenizer.encode("hello world", args.content))

    host = (args.host, args.port)  # 设定地址与端口号，'localhost'等价于'127.0.0.1'
    print("http://%s:%s" % host)
    server = HTTPServer(host, Request)  # 根据地址端口号和新定义的类，创建服务器实例
    server.serve_forever()  # 开启服务

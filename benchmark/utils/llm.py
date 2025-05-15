import socket
import json
import time
import logging
import uuid
# from .token_calc import calculate_token_length

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.work_id = None
        self.response_format = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))

    def generate_request_id(self):
        return str(uuid.uuid4())

    def send_request_stream(self, request):
        self.sock.sendall(json.dumps(request).encode('utf-8'))
        response = b""
        parsed_responses = []
        output_text = ""
        token_count = 0

        start_time = time.time()
        first_packet_time = None

        while True:
            chunk = self.sock.recv(4096)
            response += chunk

            while b'\n' in response:
                line, response = response.split(b'\n', 1)
                try:
                    parsed_response = json.loads(line.decode('utf-8'))
                    parsed_responses.append(parsed_response)

                    if "data" in parsed_response and "delta" in parsed_response["data"]:
                        if first_packet_time is None:
                            first_packet_time = time.time()
                        output_text += parsed_response["data"]["delta"]
                        token_count += 3

                    if "data" in parsed_response and parsed_response["data"].get("finish", False):
                        end_time = time.time()
                        total_time = end_time - start_time
                        first_packet_latency = first_packet_time - start_time if first_packet_time else None

                        # token_count = calculate_token_length(output_text)
                        token_speed = token_count / total_time if total_time > 0 else 0

                        logging.info("Stream reception completed.")
                        logging.info("First packet latency: %.2f seconds", first_packet_latency if first_packet_latency else 0)
                        logging.info("Total reception time: %.2f seconds", total_time)
                        logging.info("Total tokens received: %d", token_count)
                        logging.info("Token reception speed: %.2f tokens/second", token_speed)
                        logging.info("Total output text length: %d characters", len(output_text))

                        return {
                            "responses": parsed_responses,
                            "output_text": output_text,
                            "token_count": token_count,
                            "first_packet_latency": first_packet_latency,
                            "total_time": total_time,
                            "token_speed": token_speed
                        }
                except json.JSONDecodeError:
                    logging.warning("Failed to decode JSON, skipping line.")
                    continue

    def send_request_non_stream(self, request):
        self.sock.sendall(json.dumps(request).encode('utf-8'))
        response = b""
        while True:
            chunk = self.sock.recv(4096)
            response += chunk
            if b'\n' in chunk:
                break
        return json.loads(response.decode('utf-8'))

    def setup(self, model):
        setup_request = {
            "request_id": self.generate_request_id(),
            "work_id": "llm",
            "action": "setup",
            "object": "llm.setup",
            "data": {
                "model": model,
                "response_format": "llm.utf-8.stream",
                "input": "llm.utf-8",
                "enoutput": True,
                "max_token_len": 256,
                "prompt": "You are a knowledgeable assistant capable of answering various questions and providing information."
            }
        }
        response = self.send_request_non_stream(setup_request)
        self.work_id = response.get("work_id")
        self.response_format = setup_request["data"]["response_format"]
        return response

    def inference(self, input_text):
        if not self.work_id:
            raise ValueError("work_id is not set. Please call setup() first.")
        
        inference_request = {
            "request_id": self.generate_request_id(),
            "work_id": self.work_id,
            "action": "inference",
            "object": self.response_format,
            "data": {
                "delta": input_text,
                "index": 0,
                "finish": True
            }
        }
        if "stream" in self.response_format:
            logging.info("Sending stream request...")
            result = self.send_request_stream(inference_request)
            return {
                "output_text": result["output_text"],
                "token_count": result["token_count"],
                "first_packet_latency": result["first_packet_latency"],
                "total_time": result["total_time"],
                "token_speed": result["token_speed"]
            }
        else:
            logging.info("Sending non-stream request...")
            response = self.send_request_non_stream(inference_request)
            return {
                "output_text": response.get("data", ""),
                "token_count": len(response.get("data", "").split())
            }

    def exit(self):
        if not self.work_id:
            raise ValueError("work_id is not set. Please call setup() first.")
        
        exit_request = {
            "request_id": self.generate_request_id(),
            "work_id": self.work_id,
            "action": "exit"
        }
        response = self.send_request_non_stream(exit_request)
        return response

    def test(self, model, input_text):
        logging.info("Setting up...")
        setup_response = self.setup(model)

        logging.info("Running inference...")
        inference_result = self.inference(input_text)

        logging.info("Exiting...")
        exit_response = self.exit()

        return {}

if __name__ == "__main__":
    host = "192.168.20.186"
    port = 10001
    client = LLMClient(host, port)
    model_name = "qwen2.5-0.5B-p256-ax630c"
    input_text = "This is a test input for the LLM."
    client.test(model_name, input_text)
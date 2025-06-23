import socket
import json
import argparse
import uuid
import time

def create_tcp_connection(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    return sock

def send_json(sock, data):
    json_data = json.dumps(data, ensure_ascii=False) + '\n'
    print(f"Sending: {json_data}")
    sock.sendall(json_data.encode('utf-8'))

def receive_response(sock, timeout=None):
    """接收响应，带可选的超时设置"""
    old_timeout = sock.gettimeout()
    try:
        if timeout is not None:
            sock.settimeout(timeout)
        response = ''
        while True:
            part = sock.recv(4096).decode('utf-8')
            if not part:  # 连接已关闭
                return response.strip()
            response += part
            if '\n' in response:
                break
        return response.strip()
    except socket.timeout:
        return None
    finally:
        sock.settimeout(old_timeout)

def close_connection(sock):
    if sock:
        sock.close()

def create_melotts_setup_data(request_id="melotts_setup"):
    return {
        "request_id": request_id,
        "work_id": "melotts",
        "action": "setup",
        "object": "melotts.setup",
        "data": {
            "model": "melotts-zh-cn",
            "response_format": "sys.pcm",
            "input": "tts.utf-8",
            "enoutput": False
        }
    }

def list_available_tasks(sock):
    """获取可用的任务列表"""
    request_id = str(uuid.uuid4())
    send_json(sock, {
        "request_id": request_id,
        "work_id": "melotts",
        "action": "taskinfo"
    })
    
    response = receive_response(sock)
    if not response:
        return {"error": "No response received"}
    try:
        return json.loads(response)
    except:
        return {"error": "Failed to parse response"}

def parse_setup_response(response_data, sent_request_id):
    error = response_data.get('error')
    request_id = response_data.get('request_id')

    if request_id != sent_request_id:
        print(f"Request ID mismatch: sent {sent_request_id}, received {request_id}")
        return None
    if error and error.get('code') != 0:
        print(f"Error Code: {error['code']}, Message: {error['message']}")
        return None
    return response_data.get('work_id')

def setup(sock, setup_data):
    sent_request_id = setup_data['request_id']
    send_json(sock, setup_data)
    response = receive_response(sock)
    if not response:
        print("No response received during setup")
        return None
    try:
        response_data = json.loads(response)
        return parse_setup_response(response_data, sent_request_id)
    except json.JSONDecodeError:
        print(f"Invalid JSON response: {response}")
        return None

def melotts_tts_inference(sock, melotts_work_id, text, use_stream=False):
    request_id = str(uuid.uuid4())
    
    # 根据文档，选择流式或非流式请求格式
    if use_stream:
        send_json(sock, {
            "request_id": request_id,
            "work_id": melotts_work_id,
            "action": "inference",
            "object": "melotts.utf-8.stream",
            "data": {
                "delta": text,
                "index": 0,
                "finish": True
            }
        })
    else:
        # 非流式请求
        send_json(sock, {
            "request_id": request_id,
            "work_id": melotts_work_id,
            "action": "inference",
            "object": "melotts.utf-8",
            "data": text
        })

    # 关键更改：不等待响应或设置更长的超时时间
    # 由于使用sys.pcm格式，音频会直接播放，可能不会立即返回响应
    print("语音合成请求已发送，正在播放...")
    
    # 可选：设置一个较短的超时来检查是否有响应，但不要因为没响应就认为失败
    response = receive_response(sock, timeout=0.5)  # 设置短超时，只是尝试看有没有响应
    if response:
        try:
            response_data = json.loads(response)
            error = response_data.get('error')
            if error and error.get('code') != 0:
                print(f"收到错误响应: Code={error['code']}, Message={error['message']}")
                return False
            print("收到成功响应")
        except:
            print(f"收到非JSON响应: {response[:100]}...")
    else:
        # 不收到响应也视为成功，因为服务器可能正忙于播放音频
        print("未收到响应，但这不一定表示失败(服务器可能正忙于处理音频)")
    
    # 这里给TTS处理一些时间
    # 根据文本长度估计播放时间
    estimated_time = len(text) * 0.1  # 假设每个字符需要0.1秒
    estimated_time = max(1.0, min(estimated_time, 10.0))  # 至少1秒，最多10秒
    print(f"等待大约 {estimated_time:.1f} 秒让音频播放完...")
    time.sleep(estimated_time)
    
    return True

def exit_session(sock, melotts_work_id):
    send_json(sock, {
        "request_id": "melotts_exit",
        "work_id": melotts_work_id,
        "action": "exit"
    })
    response = receive_response(sock, timeout=2.0)
    if not response:
        print("退出命令已发送，但未收到响应")
        return True  # 假设成功
    try:
        response_data = json.loads(response)
        print("Exit Response:", response_data)
        return response_data.get('error', {}).get('code', -1) == 0
    except:
        print("Failed to parse exit response")
        return False

def get_task_info(sock, work_id):
    """获取任务的详细信息"""
    request_id = str(uuid.uuid4())
    send_json(sock, {
        "request_id": request_id,
        "work_id": work_id,
        "action": "taskinfo"
    })
    
    response = receive_response(sock)
    if not response:
        return {"error": "No response received"}
    try:
        return json.loads(response)
    except:
        return {"error": "Failed to parse response"}

def main(host, port):
    sock = create_tcp_connection(host, port)
    try:
        print("Setting up MeloTTS...")
        setup_data = create_melotts_setup_data()
        melotts_work_id = setup(sock, setup_data)
        
        if not melotts_work_id:
            print("Setup failed. Checking available tasks...")
            task_list = list_available_tasks(sock)
            print("Available tasks:", task_list)
            return
            
        print(f"MeloTTS SETUP finished, work_id: {melotts_work_id}")
        
        # 获取并显示任务详细信息
        task_info = get_task_info(sock, melotts_work_id)
        print("Task info:", task_info)

        # 选择流式或非流式模式
        use_stream = input("是否使用流式输入? (y/n, 默认n): ").lower() == 'y'
        
        while True:
            text = input("请输入你要合成语音的中文文本（输入exit退出）：")
            if text.lower() == 'exit':
                break
                
            print("正在合成语音...", flush=True)
            success = melotts_tts_inference(sock, melotts_work_id, text, use_stream)
            
            if success:
                print("语音合成处理完成")
            else:
                print("语音合成处理失败")
                
            # 每次请求间隔
            time.sleep(1)

        # 退出会话
        if exit_session(sock, melotts_work_id):
            print("成功退出会话")
        else:
            print("退出会话可能有问题")
            
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        close_connection(sock)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TCP Client for MeloTTS Unit.')
    parser.add_argument('--host', type=str, default='localhost', help='Server hostname (default: localhost)')
    parser.add_argument('--port', type=int, default=10001, help='Server port (default: 10001)')
    args = parser.parse_args()
    main(args.host, args.port)

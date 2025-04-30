import socket
import json
import argparse
import uuid
import time
import sys

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

def create_tts_setup_data(request_id=None, link_with=None):
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    # 基本设置
    data = {
        "model": "single_speaker_fast",
        "response_format": "sys.pcm",
        "input": "tts.utf-8",
        "enoutput": False
    }
    
    # 如果需要链接其他单元
    if link_with:
        if isinstance(link_with, list):
            inputs = ["tts.utf-8"] + link_with
            data["input"] = inputs
        else:
            inputs = ["tts.utf-8", link_with]
            data["input"] = inputs
    
    return {
        "request_id": request_id,
        "work_id": "tts",
        "action": "setup",
        "object": "tts.setup",
        "data": data
    }

def list_available_tasks(sock, work_id="tts"):
    """获取可用的任务列表"""
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

def link_units(sock, tts_work_id, target_work_id):
    """链接TTS单元与其他单元"""
    request_id = str(uuid.uuid4())
    send_json(sock, {
        "request_id": request_id,
        "work_id": tts_work_id,
        "action": "link",
        "object": "work_id",
        "data": target_work_id
    })
    
    response = receive_response(sock)
    if not response:
        print("No response received for link request")
        return False
    
    try:
        response_data = json.loads(response)
        error = response_data.get('error', {})
        if error.get('code') == 0:
            print(f"成功链接 {tts_work_id} 与 {target_work_id}")
            return True
        else:
            print(f"链接失败: {error.get('message', '未知错误')}")
            return False
    except:
        print(f"Failed to parse link response: {response}")
        return False

def unlink_units(sock, tts_work_id, target_work_id):
    """取消TTS单元与其他单元的链接"""
    request_id = str(uuid.uuid4())
    send_json(sock, {
        "request_id": request_id,
        "work_id": tts_work_id,
        "action": "unlink",
        "object": "work_id",
        "data": target_work_id
    })
    
    response = receive_response(sock)
    if not response:
        print("No response received for unlink request")
        return False
    
    try:
        response_data = json.loads(response)
        error = response_data.get('error', {})
        if error.get('code') == 0:
            print(f"成功取消链接 {tts_work_id} 与 {target_work_id}")
            return True
        else:
            print(f"取消链接失败: {error.get('message', '未知错误')}")
            return False
    except:
        print(f"Failed to parse unlink response: {response}")
        return False

def pause_unit(sock, tts_work_id):
    """暂停TTS单元工作"""
    request_id = str(uuid.uuid4())
    send_json(sock, {
        "request_id": request_id,
        "work_id": tts_work_id,
        "action": "pause"
    })
    
    response = receive_response(sock)
    if not response:
        print("No response received for pause request")
        return False
    
    try:
        response_data = json.loads(response)
        error = response_data.get('error', {})
        if error.get('code') == 0:
            print(f"成功暂停 {tts_work_id}")
            return True
        else:
            print(f"暂停失败: {error.get('message', '未知错误')}")
            return False
    except:
        print(f"Failed to parse pause response: {response}")
        return False

def resume_unit(sock, tts_work_id):
    """恢复TTS单元工作"""
    request_id = str(uuid.uuid4())
    send_json(sock, {
        "request_id": request_id,
        "work_id": tts_work_id,
        "action": "work"
    })
    
    response = receive_response(sock)
    if not response:
        print("No response received for resume request")
        return False
    
    try:
        response_data = json.loads(response)
        error = response_data.get('error', {})
        if error.get('code') == 0:
            print(f"成功恢复 {tts_work_id}")
            return True
        else:
            print(f"恢复失败: {error.get('message', '未知错误')}")
            return False
    except:
        print(f"Failed to parse resume response: {response}")
        return False

def tts_inference(sock, tts_work_id, text):
    request_id = str(uuid.uuid4())
    
    # 非流式请求
    send_json(sock, {
        "request_id": request_id,
        "work_id": tts_work_id,
        "action": "inference",
        "object": "tts.utf-8",
        "data": text
    })

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

def exit_session(sock, tts_work_id):
    request_id = str(uuid.uuid4())
    send_json(sock, {
        "request_id": request_id,
        "work_id": tts_work_id,
        "action": "exit"
    })
    response = receive_response(sock, timeout=2.0)
    if not response:
        print("退出命令已发送，但未收到响应")
        return True  # 假设成功
    try:
        response_data = json.loads(response)
        error = response_data.get('error', {})
        if error.get('code') == 0:
            print(f"成功退出 {tts_work_id}")
            return True
        else:
            print(f"退出失败: {error.get('message', '未知错误')}")
            return False
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

def print_menu():
    print("\n===== TTS控制菜单 =====")
    print("1. 合成语音")
    print("2. 链接到其他单元")
    print("3. 取消链接")
    print("4. 暂停TTS单元")
    print("5. 恢复TTS单元")
    print("6. 获取任务信息")
    print("7. 退出TTS单元")
    print("0. 退出程序")
    print("======================")

def main(host, port):
    sock = create_tcp_connection(host, port)
    try:
        print("Setting up TTS...")
        setup_data = create_tts_setup_data()
        tts_work_id = setup(sock, setup_data)
        
        if not tts_work_id:
            print("Setup failed. Checking available tasks...")
            task_list = list_available_tasks(sock)
            print("Available tasks:", task_list)
            if task_list.get('data') and isinstance(task_list.get('data'), list) and len(task_list.get('data')) > 0:
                tts_work_id = task_list.get('data')[0]
                print(f"使用已存在的TTS任务: {tts_work_id}")
            else:
                print("找不到可用的TTS任务，程序退出")
                return
        
        print(f"TTS SETUP finished, work_id: {tts_work_id}")
        
        # 获取并显示任务详细信息
        task_info = get_task_info(sock, tts_work_id)
        print("Task info:", task_info)

        while True:
            print_menu()
            choice = input("请选择操作 (0-7): ")
            
            if choice == '0':
                print("程序退出")
                break
                
            elif choice == '1':
                text = input("请输入要合成语音的文本: ")
                if text:
                    print("正在合成语音...", flush=True)
                    success = tts_inference(sock, tts_work_id, text)
                    if success:
                        print("语音合成处理完成")
                    else:
                        print("语音合成处理失败")
                else:
                    print("文本为空，取消合成")
                
            elif choice == '2':
                target_id = input("请输入要链接的单元ID (例如 kws.1000): ")
                if target_id:
                    link_units(sock, tts_work_id, target_id)
                else:
                    print("单元ID为空，取消链接操作")
                    
            elif choice == '3':
                target_id = input("请输入要取消链接的单元ID (例如 kws.1000): ")
                if target_id:
                    unlink_units(sock, tts_work_id, target_id)
                else:
                    print("单元ID为空，取消操作")
                    
            elif choice == '4':
                pause_unit(sock, tts_work_id)
                
            elif choice == '5':
                resume_unit(sock, tts_work_id)
                
            elif choice == '6':
                task_info = get_task_info(sock, tts_work_id)
                print("Task info:", json.dumps(task_info, indent=2, ensure_ascii=False))
                
            elif choice == '7':
                if exit_session(sock, tts_work_id):
                    print("TTS单元已退出")
                    # 重新检查可用任务
                    task_list = list_available_tasks(sock)
                    print("Available tasks:", task_list)
                else:
                    print("TTS单元退出失败")
                
            else:
                print("无效的选择，请重试")
                
            # 每次操作间隔
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序异常: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TCP Client for MeloTTS Unit.')
    parser.add_argument('--host', type=str, default='localhost', help='Server hostname (default: localhost)')
    parser.add_argument('--port', type=int, default=10001, help='Server port (default: 10001)')
    args = parser.parse_args()
    main(args.host, args.port)
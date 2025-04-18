#!/bin/env python3
import os
import sys
import shutil
import subprocess
import requests
import tarfile
import shutil
import concurrent.futures
import json
import glob
from datetime import datetime
'''
{package_name}_{version}-{revision}_{architecture}.deb
lib-llm_1.0-m5stack1_arm64.deb
llm-sys_1.0-m5stack1_arm64.deb
llm-audio_1.0-m5stack1_arm64.deb
llm-kws_1.0-m5stack1_arm64.deb
llm-asr_1.0-m5stack1_arm64.deb
llm-llm_1.0-m5stack1_arm64.deb
llm-tts_1.0-m5stack1_arm64.deb
'''


def create_lib_deb(package_name, version, src_folder, revision = 'm5stack1'):
    deb_file = f"{package_name}_{version}-{revision}_arm64.deb"
    deb_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debian-{}'.format(package_name))
    if os.path.exists(deb_folder):
        shutil.rmtree(deb_folder)
    os.makedirs(deb_folder, exist_ok = True)
    fileignore = {'ignore':[]}
    try:
        with open(os.path.join(src_folder, 'fileignore'), 'r') as f:
            fileignore = json.load(f)
    except:
        pass
    for item in os.listdir(src_folder):
        if item.startswith('llm_') or item.startswith('tokenizer_') or item.startswith('llm-kws_') or item.startswith('mode_') or item in fileignore['ignore']:
            continue
        elif item.startswith('lib'):
            os.makedirs(os.path.join(deb_folder, 'opt/m5stack/lib'), exist_ok = True)
            shutil.copy2(os.path.join(src_folder, item), os.path.join(deb_folder, 'opt/m5stack/lib', item))
        else:
            os.makedirs(os.path.join(deb_folder, 'opt/m5stack/share'), exist_ok = True)
            source_path = os.path.join(src_folder, item)
            target_path = os.path.join(deb_folder, 'opt/m5stack/share', item)
            if os.path.isfile(source_path):
                shutil.copy2(source_path, target_path)
            elif os.path.isdir(source_path):
                shutil.copytree(source_path, target_path)

    # os.makedirs(os.path.join(deb_folder, 'opt/m5stack/data'), exist_ok = True)

    # zip_file = 'm5stack_scripts.tar.gz'
    # down_url = 'https://m5stack.oss-cn-shenzhen.aliyuncs.com/resource/linux/llm/m5stack_scripts.tar.gz'
    # zip_file_extrpath = 'm5stack_scripts'
    # if not os.path.exists(zip_file_extrpath):
    #     # Downloading via HTTP (more common)
    #     if not os.path.exists(zip_file):
    #         response = requests.get(down_url)
    #         if response.status_code == 200:
    #             with open(zip_file, 'wb') as file:
    #                 file.write(response.content)
    #         else:
    #             print("{} down failed".format(down_url))
    #     with tarfile.open(zip_file, 'r:gz') as tar:
    #         tar.extractall(path=zip_file_extrpath)
    #     print("The {} download successful.".format(down_url))
    # if os.path.exists(zip_file_extrpath):
    #     shutil.copytree(zip_file_extrpath, os.path.join(deb_folder, 'opt/m5stack/scripts'))

    # zip_file = 'm5stack_dist-packages.tar.gz'
    # down_url = 'https://m5stack.oss-cn-shenzhen.aliyuncs.com/resource/linux/llm/m5stack_dist-packages.tar.gz'
    # zip_file_extrpath = 'm5stack_dist-packages'
    # if not os.path.exists(zip_file_extrpath):
    #     # Downloading via HTTP (more common)
    #     if not os.path.exists(zip_file):
    #         response = requests.get(down_url)
    #         if response.status_code == 200:
    #             with open(zip_file, 'wb') as file:
    #                 file.write(response.content)
    #         else:
    #             print("{} down failed".format(down_url))
    #     with tarfile.open(zip_file, 'r:gz') as tar:
    #         tar.extractall(path=zip_file_extrpath)
    #     print("The {} download successful.".format(down_url))
    # if os.path.exists(zip_file_extrpath):
    #     shutil.copytree(zip_file_extrpath, os.path.join(deb_folder, 'usr/local/lib/python3.10/dist-packages'))

    os.makedirs(os.path.join(deb_folder, 'DEBIAN'), exist_ok = True)
    with open(os.path.join(deb_folder, 'DEBIAN/control'),'w') as f:
        f.write(f'Package: {package_name}\n')
        f.write(f'Version: {version}\n')
        f.write(f'Architecture: arm64\n')
        f.write(f'Maintainer: dianjixz <dianjixz@m5stack.com>\n')
        f.write(f'Original-Maintainer: m5stack <m5stack@m5stack.com>\n')
        f.write(f'Section: llm-module\n')
        f.write(f'Priority: optional\n')
        f.write(f'Homepage: https://www.m5stack.com\n')
        f.write(f'Packaged-Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Description: llm-module\n')
        f.write(f' bsp.\n')
    with open(os.path.join(deb_folder, 'DEBIAN/postinst'),'w') as f:
        f.write(f'#!/bin/sh\n')
        f.write('''sed -i 's/dpkg -i/apt install -y/g' /usr/local/m5stack/update_check.sh\n''')
        f.write(f'[ -f "/lib/systemd/system/llm-sys.service" ] && systemctl enable llm-sys.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-sys.service" ] && systemctl start llm-sys.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-asr.service" ] && systemctl enable llm-asr.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-asr.service" ] && systemctl start llm-asr.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-audio.service" ] && systemctl enable llm-audio.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-audio.service" ] && systemctl start llm-audio.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-kws.service" ] && systemctl enable llm-kws.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-kws.service" ] && systemctl start llm-kws.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-llm.service" ] && systemctl enable llm-llm.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-llm.service" ] && systemctl start llm-llm.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-tts.service" ] && systemctl enable llm-tts.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-tts.service" ] && systemctl start llm-tts.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-camera.service" ] && systemctl enable llm-camera.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-camera.service" ] && systemctl start llm-camera.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-yolo.service" ] && systemctl enable llm-yolo.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-yolo.service" ] && systemctl start llm-yolo.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-melotts.service" ] && systemctl enable llm-melotts.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-melotts.service" ] && systemctl start llm-melotts.service\n')
        f.write(f'exit 0\n')
    with open(os.path.join(deb_folder, 'DEBIAN/prerm'),'w') as f:
        f.write(f'#!/bin/sh\n')
        f.write(f'[ -f "/lib/systemd/system/llm-tts.service" ] && systemctl stop llm-tts.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-tts.service" ] && systemctl disable llm-tts.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-llm.service" ] && systemctl stop llm-llm.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-llm.service" ] && systemctl disable llm-llm.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-kws.service" ] && systemctl stop llm-kws.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-kws.service" ] && systemctl disable llm-kws.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-audio.service" ] && systemctl stop llm-audio.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-audio.service" ] && systemctl disable llm-audio.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-asr.service" ] && systemctl stop llm-asr.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-asr.service" ] && systemctl disable llm-asr.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-sys.service" ] && systemctl stop llm-sys.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-sys.service" ] && systemctl disable llm-sys.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-camera.service" ] && systemctl stop llm-camera.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-camera.service" ] && systemctl disable llm-camera.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-yolo.service" ] && systemctl stop llm-yolo.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-yolo.service" ] && systemctl disable llm-yolo.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-melotts.service" ] && systemctl stop llm-melotts.service\n')
        f.write(f'[ -f "/lib/systemd/system/llm-melotts.service" ] && systemctl disable llm-melotts.service\n')
        f.write(f'exit 0\n')
    os.chmod(os.path.join(deb_folder, 'DEBIAN/postinst'), 0o755)
    os.chmod(os.path.join(deb_folder, 'DEBIAN/prerm'), 0o755)
    subprocess.run(["dpkg-deb", "-b", deb_folder, deb_file], check=True)
    print(f"Debian package created: {deb_file}")
    shutil.rmtree(deb_folder)
    return package_name + " creat success!"

def create_data_deb(package_name, version, src_folder, revision = 'm5stack1', depends = 'lib-llm (>= 1.6)'):
    deb_file = f"{package_name}_{version}-{revision}_arm64.deb"
    deb_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debian-{}'.format(package_name))
    if os.path.exists(deb_folder):
        shutil.rmtree(deb_folder)
    os.makedirs(deb_folder, exist_ok = True)

    zip_file = f"m5stack_{package_name[0:4]+package_name[10:]}_{version}_data.tar.gz"
    down_url = f"https://m5stack.oss-cn-shenzhen.aliyuncs.com/resource/linux/llm/m5stack_{package_name[0:4]+package_name[10:]}_{version}_data.tar.gz"
    zip_file_extrpath = f"m5stack_{package_name}_{version}_data"
    if not os.path.exists(zip_file_extrpath):
        # Downloading via HTTP (more common)
        if not os.path.exists(zip_file):
            response = requests.get(down_url)
            if response.status_code == 200:
                with open(zip_file, 'wb') as file:
                    file.write(response.content)
            else:
                print("{} down failed".format(down_url))
        with tarfile.open(zip_file, 'r:gz') as tar:
            tar.extractall(path=zip_file_extrpath)
        print("The {} download successful.".format(down_url))
    if os.path.exists(zip_file_extrpath):
        shutil.copytree(zip_file_extrpath, os.path.join(deb_folder, 'opt/m5stack/data'))

    RED = "\033[31m"
    RESET = "\033[0m"
    os.makedirs(os.path.join(deb_folder, 'opt/m5stack/data/models'), exist_ok = True)
    mode_config_file = os.path.join(src_folder,'mode_{}.json'.format(package_name[10:]))
    if os.path.exists(mode_config_file):
        shutil.copy2(mode_config_file, os.path.join(deb_folder, 'opt/m5stack/data/models', 'mode_{}.json'.format(package_name[10:])))
        try:
            with open(mode_config_file, 'r', encoding='utf-8') as file:
                data = json.load(file)
            for scripts_file in data['mode_param']['ext_scripts']:
                tokenizer_py_file = os.path.join(src_folder, scripts_file)
                if os.path.exists(tokenizer_py_file):
                    os.makedirs(os.path.join(deb_folder, 'opt/m5stack/scripts'), exist_ok = True)
                    shutil.copy2(tokenizer_py_file, os.path.join(deb_folder, 'opt/m5stack/scripts', scripts_file))
        except:
            pass
    else:
        print(RED, mode_config_file, " miss", RESET)

    os.makedirs(os.path.join(deb_folder, 'DEBIAN'), exist_ok = True)
    with open(os.path.join(deb_folder, 'DEBIAN/control'),'w') as f:
        f.write(f'Package: {package_name}\n')
        f.write(f'Version: {version}\n')
        f.write(f'Architecture: arm64\n')
        f.write(f'Maintainer: dianjixz <dianjixz@m5stack.com>\n')
        f.write(f'Original-Maintainer: m5stack <m5stack@m5stack.com>\n')
        f.write(f'Section: llm-module\n')
        f.write(f'Priority: optional\n')
        f.write(f'Depends: {depends}\n')
        f.write(f'Homepage: https://www.m5stack.com\n')
        if deb_file.startswith('llm-model-'):
            deb_name = deb_file[:deb_file.find('_')]
            old_deb_name = deb_name.replace('model-','').lower()
            f.write(f'Conflicts: {old_deb_name}\n')   
        f.write(f'Packaged-Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')     
        f.write(f'Description: llm-module\n')
        f.write(f' bsp.\n')
    with open(os.path.join(deb_folder, 'DEBIAN/postinst'),'w') as f:
        f.write(f'#!/bin/sh\n')
        f.write(f'exit 0\n')
    with open(os.path.join(deb_folder, 'DEBIAN/prerm'),'w') as f:
        f.write(f'#!/bin/sh\n')
        f.write(f'exit 0\n')
    os.chmod(os.path.join(deb_folder, 'DEBIAN/postinst'), 0o755)
    os.chmod(os.path.join(deb_folder, 'DEBIAN/prerm'), 0o755)
    subprocess.run(["dpkg-deb", "-b", deb_folder, deb_file], check=True)
    print(f"Debian package created: {deb_file}")
    shutil.rmtree(deb_folder)
    return package_name + " creat success!"

def create_bin_deb(package_name, version, src_folder, revision = 'm5stack1', depends = 'lib-llm (>= 1.7)'):
    bin_files = glob.glob(os.path.join(src_folder, package_name.replace("-", "_") + "-*"))
    version_info = 0.0
    print(os.path.join(src_folder, package_name + "-*"))
    if bin_files:
        for bin_file in bin_files:
            version_info = float(bin_file.split('-')[-1])
            if float(bin_file.split('-')[-1]) > version_info:
                version_info = float(bin_file.split('-')[-1])
        version = str(version_info)

    deb_file = f"{package_name}_{version}-{revision}_arm64.deb"
    deb_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debian-{}'.format(package_name))
    # os.makedirs(deb_folder, exist_ok=True)
    if os.path.exists(deb_folder):
        shutil.rmtree(deb_folder)
    os.makedirs(os.path.join(deb_folder, 'opt/m5stack/bin'), exist_ok = True)
    os.makedirs(os.path.join(deb_folder, 'DEBIAN'), exist_ok = True)
    # shutil.copytree(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deb_overlay'), deb_folder)
    if package_name == 'llm-openai-api':
        m5module_dir = os.path.join(src_folder, 'ModuleLLM-OpenAI-Plugin')
        if os.path.exists(m5module_dir):
            shutil.copytree(m5module_dir, os.path.join(deb_folder, 'opt/m5stack/bin/ModuleLLM-OpenAI-Plugin'))
        openai_api_dir = os.path.join(src_folder, 'openai-api')
        if os.path.exists(openai_api_dir):
            shutil.copytree(openai_api_dir, os.path.join(deb_folder, 'opt/m5stack/lib/openai-api'))
    if package_name == 'llm-kws':
        sherpa_dir = os.path.join(src_folder, 'sherpa-onnx')
        if os.path.exists(sherpa_dir):
            shutil.copytree(sherpa_dir, os.path.join(deb_folder, 'opt/m5stack/lib/sherpa-onnx'))
    if package_name == 'llm-llm':
        llm_dir = os.path.join(src_folder, 'llm')
        if os.path.exists(llm_dir):
            shutil.copytree(llm_dir, os.path.join(deb_folder, 'opt/m5stack/lib/llm'))
    if package_name == 'llm-vlm':
        vlm_dir = os.path.join(src_folder, 'vlm')
        if os.path.exists(vlm_dir):
            shutil.copytree(vlm_dir, os.path.join(deb_folder, 'opt/m5stack/lib/vlm'))
    
    bin_file_name = package_name.replace("-", "_")
    if version_info != 0.0:
        bin_file_name = package_name.replace("-", "_") + f'-{version}'
    shutil.copy2(os.path.join(src_folder, bin_file_name), os.path.join(deb_folder, 'opt/m5stack/bin', bin_file_name))
    ext_scripts_files = glob.glob(os.path.join(src_folder, package_name + "_*"))
    if ext_scripts_files:
        os.makedirs(os.path.join(deb_folder, 'opt/m5stack/scripts'), exist_ok = True)
        for ext_script_file in ext_scripts_files:
            shutil.copy2(ext_script_file, os.path.join(deb_folder, 'opt/m5stack/scripts'))

    with open(os.path.join(deb_folder, 'DEBIAN/control'),'w') as f:
        f.write(f'Package: {package_name}\n')
        f.write(f'Version: {version}\n')
        f.write(f'Architecture: arm64\n')
        f.write(f'Maintainer: dianjixz <dianjixz@m5stack.com>\n')
        f.write(f'Original-Maintainer: m5stack <m5stack@m5stack.com>\n')
        f.write(f'Section: llm-module\n')
        f.write(f'Priority: optional\n')
        # f.write(f'Depends: lib-llm\n')
        f.write(f'Depends: {depends}\n')
        f.write(f'Homepage: https://www.m5stack.com\n')
        f.write(f'Packaged-Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Description: llm-module\n')
        f.write(f' bsp.\n')
    with open(os.path.join(deb_folder, 'DEBIAN/postinst'),'w') as f:
        f.write(f'#!/bin/sh\n')
        f.write(f'[ -f "/lib/systemd/system/{package_name}.service" ] && systemctl enable {package_name}.service\n')
        f.write(f'[ -f "/lib/systemd/system/{package_name}.service" ] && systemctl start {package_name}.service\n')
        f.write(f'exit 0\n')
    with open(os.path.join(deb_folder, 'DEBIAN/prerm'),'w') as f:
        f.write(f'#!/bin/sh\n')
        f.write(f'[ -f "/lib/systemd/system/{package_name}.service" ] && systemctl stop {package_name}.service\n')
        f.write(f'[ -f "/lib/systemd/system/{package_name}.service" ] && systemctl disable {package_name}.service\n')
        f.write(f'exit 0\n')
    os.makedirs(os.path.join(deb_folder, 'lib/systemd/system'), exist_ok = True)
    with open(os.path.join(deb_folder, f'lib/systemd/system/{package_name}.service'),'w') as f:
        f.write(f'[Unit]\n')
        f.write(f'Description={package_name} Service\n')
        if package_name != 'llm-sys':
            f.write(f'After=llm-sys.service\n')
            f.write(f'Requires=llm-sys.service\n')
        f.write(f'\n')
        f.write(f'[Service]\n')
        f.write(f'ExecStart=/opt/m5stack/bin/{bin_file_name}\n')
        f.write(f'WorkingDirectory=/opt/m5stack\n')
        f.write(f'Restart=always\n')
        f.write(f'RestartSec=1\n')
        f.write(f'StartLimitInterval=0\n')
        f.write(f'\n')
        f.write(f'[Install]\n')
        f.write(f'WantedBy=multi-user.target\n')
        f.write(f'\n')

    os.chmod(os.path.join(deb_folder, 'DEBIAN/postinst'), 0o755)
    os.chmod(os.path.join(deb_folder, 'DEBIAN/prerm'), 0o755)

    subprocess.run(["dpkg-deb", "-b", deb_folder, deb_file], check=True)
    print(f"Debian package created: {deb_file}")
    shutil.rmtree(deb_folder)
    return package_name + " creat success!"

if __name__ == "__main__":

    if "clean" in sys.argv:
        os.system('rm ./*.deb')
        os.system('find . -maxdepth 1 -type d ! -name "." -exec rm -rf {} +')
        exit(0)
    if "distclean" in sys.argv:
        os.system('rm ./*.deb m5stack_* -rf')
        exit(0)

    version = '1.5'
    data_version = '0.2'
    src_folder = '../dist'
    revision = 'm5stack1'
    create_lib = True
    create_bin = True
    create_data = True
    create_llm_data = True
    # if len(sys.argv) > 1:
    #     src_folder = sys.argv[1]
    cpu_count = os.cpu_count()
    if cpu_count - 2 < 1:
        cpu_count = 2
    else:
        cpu_count = cpu_count - 2
    # cpu_count = 50
    Tasks = {
        'lib-llm':[create_lib_deb,'lib-llm', '1.7', src_folder, revision],
        'llm-sys':[create_bin_deb,'llm-sys', version, src_folder, revision],
        'llm-audio':[create_bin_deb,'llm-audio', version, src_folder, revision],
        'llm-kws':[create_bin_deb,'llm-kws', '1.6', src_folder, revision],
        'llm-asr':[create_bin_deb,'llm-asr', version, src_folder, revision],
        'llm-llm':[create_bin_deb,'llm-llm', '1.7', src_folder, revision],
        'llm-tts':[create_bin_deb,'llm-tts', version, src_folder, revision],
        'llm-melotts':[create_bin_deb,'llm-melotts', '1.6', src_folder, revision],
        'llm-camera':[create_bin_deb,'llm-camera', '1.7', src_folder, revision, 'lib-llm'],
        'llm-vlm':[create_bin_deb,'llm-vlm', '1.6', src_folder, revision],
        'llm-yolo':[create_bin_deb,'llm-yolo', '1.6', src_folder, revision],
        'llm-skel':[create_bin_deb,'llm-skel', version, src_folder, revision],
        'llm-depth-anything':[create_bin_deb,'llm-depth-anything', version, src_folder, revision],
        'llm-vad':[create_bin_deb,'llm-vad', version, src_folder, revision],
        'llm-whisper':[create_bin_deb,'llm-whisper', '1.6', src_folder, revision],
        'llm-openai-api':[create_bin_deb,'llm-openai-api', '1.6', src_folder, revision],
        'llm-model-audio-en-us':[create_data_deb,'llm-model-audio-en-us', data_version, src_folder, revision],
        'llm-model-audio-zh-cn':[create_data_deb,'llm-model-audio-zh-cn', data_version, src_folder, revision],
        'llm-model-sherpa-ncnn-streaming-zipformer-20M-2023-02-17':[create_data_deb,'llm-model-sherpa-ncnn-streaming-zipformer-20M-2023-02-17', data_version, src_folder, revision],
        'llm-model-sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23':[create_data_deb,'llm-model-sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23', data_version, src_folder, revision],
        'llm-model-sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01':[create_data_deb,'llm-model-sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01', '0.3', src_folder, revision],
        'llm-model-sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01':[create_data_deb,'llm-model-sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01', '0.3', src_folder, revision],
        'llm-model-single-speaker-english-fast':[create_data_deb,'llm-model-single-speaker-english-fast', data_version, src_folder, revision],
        'llm-model-single-speaker-fast':[create_data_deb,'llm-model-single-speaker-fast', data_version, src_folder, revision],
        'llm-model-melotts-zh-cn':[create_data_deb,'llm-model-melotts-zh-cn', '0.4', src_folder, revision],
        'llm-model-yolo11n':[create_data_deb,'llm-model-yolo11n', data_version, src_folder, revision],
        'llm-model-yolo11n-pose':[create_data_deb,'llm-model-yolo11n-pose', '0.3', src_folder, revision],
        'llm-model-yolo11n-hand-pose':[create_data_deb,'llm-model-yolo11n-hand-pose', '0.3', src_folder, revision],
        'llm-model-yolo11n-seg':[create_data_deb,'llm-model-yolo11n-seg', '0.3', src_folder, revision],
        'llm-model-depth-anything-ax630c':[create_data_deb,'llm-model-depth-anything-ax630c', '0.3', src_folder, revision],
        'llm-model-whisper-tiny':[create_data_deb,'llm-model-whisper-tiny', '0.3', src_folder, revision],
        'llm-model-whisper-base':[create_data_deb,'llm-model-whisper-base', '0.3', src_folder, revision],
        'llm-model-whisper-small':[create_data_deb,'llm-model-whisper-small', '0.3', src_folder, revision],
        'llm-model-silero-vad':[create_data_deb,'llm-model-silero-vad', '0.3', src_folder, revision],
        'llm-model-qwen2.5-0.5B-prefill-20e':[create_data_deb,'llm-model-qwen2.5-0.5B-prefill-20e', data_version, src_folder, revision],
        'llm-model-qwen2.5-0.5B-p256-ax630c':[create_data_deb,'llm-model-qwen2.5-0.5B-p256-ax630c', '0.4', src_folder, revision],
        'llm-model-qwen2.5-0.5B-Int4-ax630c':[create_data_deb,'llm-model-qwen2.5-0.5B-Int4-ax630c', '0.4', src_folder, revision],
        'llm-model-qwen2.5-1.5B-ax630c':[create_data_deb,'llm-model-qwen2.5-1.5B-ax630c', '0.3', src_folder, revision],
        'llm-model-qwen2.5-1.5B-p256-ax630c':[create_data_deb,'llm-model-qwen2.5-1.5B-p256-ax630c', '0.4', src_folder, revision],
        'llm-model-qwen2.5-1.5B-Int4-ax630c':[create_data_deb,'llm-model-qwen2.5-1.5B-Int4-ax630c', '0.4', src_folder, revision],
        'llm-model-qwen2.5-coder-0.5B-ax630c':[create_data_deb,'llm-model-qwen2.5-coder-0.5B-ax630c', data_version, src_folder, revision],
        'llm-model-llama3.2-1B-prefill-ax630c':[create_data_deb,'llm-model-llama3.2-1B-prefill-ax630c', data_version, src_folder, revision],
        'llm-model-llama3.2-1B-p256-ax630c':[create_data_deb,'llm-model-llama3.2-1B-p256-ax630c', '0.4', src_folder, revision],
        'llm-model-openbuddy-llama3.2-1B-ax630c':[create_data_deb,'llm-model-openbuddy-llama3.2-1B-ax630c', data_version, src_folder, revision],
        'llm-model-internvl2.5-1B-ax630c':[create_data_deb,'llm-model-internvl2.5-1B-ax630c', '0.4', src_folder, revision],
        'llm-model-internvl2.5-1B-364-ax630c':[create_data_deb,'llm-model-internvl2.5-1B-364-ax630c', '0.4', src_folder, revision],
        'llm-model-deepseek-r1-1.5B-ax630c':[create_data_deb,'llm-model-deepseek-r1-1.5B-ax630c', '0.3', src_folder, revision],
        'llm-model-deepseek-r1-1.5B-p256-ax630c':[create_data_deb,'llm-model-deepseek-r1-1.5B-p256-ax630c', '0.4', src_folder, revision],
        'llm-model-smolvlm-256M-ax630c':[create_data_deb,'llm-model-smolvlm-256M-ax630c', '0.4', src_folder, revision],
        'llm-model-smolvlm-500M-ax630c':[create_data_deb,'llm-model-smolvlm-500M-ax630c', '0.4', src_folder, revision],
        # 'llm-model-qwen2-0.5B-prefill-20e':[create_data_deb,'llm-model-qwen2-0.5B-prefill-20e', data_version, src_folder, revision],
        # 'llm-model-qwen2-1.5B-prefill-20e':[create_data_deb,'llm-model-qwen2-1.5B-prefill-20e', data_version, src_folder, revision]
    }
    exit_flage = 0
    for taskname in Tasks:
        if taskname in sys.argv:
            argv = Tasks[taskname][1:]
            Tasks[taskname][0](*argv)
            exit_flage = 1
    if exit_flage :
        exit(0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = []
        if (create_lib):
            for task in Tasks:
                if (Tasks[task][0] == create_lib_deb):
                    futures.append(executor.submit(*Tasks[task]))
        if (create_bin):
            for task in Tasks:
                if (Tasks[task][0] == create_bin_deb):
                    futures.append(executor.submit(*Tasks[task]))
        if (create_data):
            for task in Tasks:
                if (Tasks[task][0] == create_data_deb):
                    futures.append(executor.submit(*Tasks[task]))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(result)


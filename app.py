from flask import Flask, request, jsonify
import subprocess
import os
import boto3

app = Flask(__name__)

# 初始化S3客户端
s3 = boto3.client('s3')

@app.route('/txt2wav', methods=['POST'])
def inference():
    # 解析HTTP请求体数据
    data = request.get_json()
    wav_filename = data.get('wav_filename')
    model_path = data.get('model_path')
    speaker = data.get('speaker')
    trans = data.get('trans')
    cluster_infer_ratio = data.get('cluster_infer_ratio')
    config_path = data.get('config_path')
    kmeans_path = data.get('kmeans_path')
    slice_db = data.get('slice_db')
    wav_format = data.get('wav_format')
    apf = data.get('apf', '')
    f0_predictor = data.get('f0_predictor')
    ehc = data.get('ehc', '')

    # 从S3下载wav文件
    wav_file_path = f'/tmp/{wav_filename}'
    s3.download_file('your-bucket-name', f'path/to/{wav_filename}', wav_file_path)

    # 构建命令行参数
    cmd = [
        'python', 'inference_main.py',
        '-n', wav_file_path,
        '-m', model_path,
        '-s', speaker,
        '-t', trans,
        '-cr', str(cluster_infer_ratio),
        '-c', config_path,
        '-cm', kmeans_path,
        '-sd', slice_db,
        '-wf', wav_format,
        apf,
        '--f0_predictor', f0_predictor
    ]
    if ehc:
        cmd.append(ehc)

    # 执行inference_main.py
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # 删除临时文件
    os.remove(wav_file_path)

    # 返回执行结果
    return jsonify({
        'stdout': stdout.decode('utf-8'),
        'stderr': stderr.decode('utf-8')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
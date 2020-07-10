import requests
import argparse

PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'

'''发送图片,获取返回结果'''


def predict_result(image_path):
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    r = requests.post(PyTorch_REST_API_URL, files=payload).json()

    if r['success']:
        for (i, result) in enumerate(r['predictions']):
            print('{}. {}: {:.4f}'.format(i + 1, result['label'], result['probability']))
    else:
        print('Request failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('--file', type=str, help='test')  # 获取终端file=后面的字符串
    args = parser.parse_args()
    predict_result(args.file)

    # python torch_request.py --file=./dog.jpg    #路径不加引号

import io
import flask
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50

'''初始化一个flask'''
app = flask.Flask(__name__)
model = None
use_gpu = False  # 是否使用GPU训练模型

with open('class_map.txt', 'r') as f:
    label_map = eval(f.read())  # 转化成字典

'''加载模型'''


def load_model():
    global model
    model = resnet50(pretrained=True)
    model.eval()  # 不启用 BatchNormalization 和 Dropout
    if use_gpu:
        model.cuda()  # 将模型加载到GPU上


'''处理接收到的图片'''


def prepare_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert('RGB')  # 使用'RGB'模式读取图片

    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)

    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    image = image[None]
    if use_gpu:
        image = image.cuda()
    return torch.autograd.Variable(image, volatile=True)  # 自动微分变量


'''定义路由'''


@app.route('/predict', methods=['POST'])
def predict():
    data = {'success': False}

    if flask.request.method == 'POST':
        if flask.request.files.get('image'):
            image = flask.request.files['image'].read()
            image = Image.open(io.BytesIO(image))  # 将字节对象转为Byte字节流数据

            image = prepare_image(image, target_size=(224, 224))

            preds = F.softmax(model(image), dim=1)
            results = torch.topk(preds.cpu().data, k=3, dim=1)  # 返回Tensor中的前k个元素以及元素对应的索引值
            results = (results[0].cpu().numpy(), results[1].cpu().numpy())  # 把tensor转换成numpy的格式

            data['predictions'] = list()

            for prob, label in zip(results[0][0], results[1][0]):
                label_name = label_map[label]
                r = {'label': label_name, 'probability': float(prob)}
                data['predictions'].append(r)

            data['success'] = True

    return flask.jsonify(data)  # 将字典转成json字符串


if __name__ == '__main__':
    load_model()
    app.run()

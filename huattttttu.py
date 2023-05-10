#!/usr/bin/env python
# coding: utf-8

from __future__ import division
import os
import PIL.Image
import cv2
from loader import *
import numpy as np
import yaml
from model.TransMUNet import TransMUNet
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, flash, session
import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

## Hyper parameters
config = yaml.load(open('./config_skin.yml'), Loader=yaml.FullLoader)
number_classes = int(config['number_classes'])
input_channels = 3
best_val_loss = np.inf
patience = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = config['path_to_data']

Net = TransMUNet(n_classes=number_classes)
Net = Net.to(device)
Net.load_state_dict(torch.load(config['saved_model'], map_location='cpu')['model_weights'])
Net.eval()


def save_sample(imgfile, th=0.3, outputname=''):
    # 1、提取文件名和扩展名
    # name,ext = os.path.splitext(imgfile)
    # 2.读取图像文件，保存图像高和宽
    # img = cv2.imread(imgfile,cv2.IMREAD_UNCHANGED)
    img = cv2.imread(imgfile)
    height, width = img.shape[:2]
    # cv2.imwrite(os.path.join(name + "maskpred" + ext), img)
    # 3.将img图像转换为256 * 256得到batchimg输入网络，得到预测结果

    batchimg = np.array(Image.fromarray(img).resize([256, 256], resample=PIL.Image.BILINEAR))
    batchimg = np.expand_dims(batchimg, 0)
    batchimg = torch.from_numpy(batchimg)
    batchimg = batchimg.permute(0, 3, 1, 2).to(torch.float32).to(device)

    pred = Net(batchimg)
    # 4.预测结果像素值恢复到[0:255],并做边缘检测得到cannymskp，恢复为原来图像宽高
    mskp = pred.detach().cpu().numpy()[0, 0]
    mskp = np.where(mskp >= th, 1., 0)
    mskp = mskp * 255
    cannymskp = cv2.Canny(mskp.astype(np.uint8), 100, 200)
    # cannymskp=cannymskp.astype(np.uint8)

    mskp = np.array(Image.fromarray(mskp).resize([width, height], resample=PIL.Image.BILINEAR))
    cannymskp = np.array(Image.fromarray(cannymskp).resize([width, height], resample=PIL.Image.BILINEAR))
    # 5. 保存图像
    cv2.imwrite(os.path.join("./tmp/maskpred.png"),mskp)
    cv2.imwrite(os.path.join("./tmp/canny.png"), cannymskp)
    cannymskp = cv2.imread(os.path.join("./tmp/canny.png"))
    x2 = cv2.add(img, cannymskp)
    cv2.imwrite(os.path.join(outputname), x2, [cv2.IMWRITE_PNG_COMPRESSION, 0])


app = Flask(__name__)
app.secret_key = "asdasdasdasd"
app.config["JSON_AS_ASCII"] = False
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = './static/images1/'



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


os.makedirs('./static', exist_ok=True)


# @app.route('/fengetj')
# def fengetj():
#     if session['benign'] == [] and session['malignancy'] == []:
#         b = '0'
#         m = '0'
#         bp = '0'
#         mp = '0'
#         return render_template('fengetj.html', B=b, M=m, BP=bp, MP=mp)
#     else:
#         b = session['benign']
#         m = session['malignancy']
#         bp = session['benign_p']
#         mp = session['malignancy_p']
#         return render_template('fengetj.html', B=b, M=m, BP=bp, MP=mp)
#

@app.route('/image', methods=['GET', 'POST'])
def image():
    #
    if request.method == 'GET':
        return render_template('image.html')

    file_names = []
    mask_names = []
    # 获取上传的PNG文件
    uploaded_files = request.files.getlist("file[]")
    # path_file_number = glob.glob('./static/images1/*.png')
    j = 0
    for uploaded_file in uploaded_files:
        if uploaded_file and allowed_file(uploaded_file.filename):
            # 保存上传的文件
            # file_name = os.path.join(app.config['UPLOAD_FOLDER'], str(j) + '.png')
            uploaded_file.save('./static/' + str(j) + '.png')

            # 使用save_sample函数处理上传的文件
            save_sample('./static/' + str(j) + '.png', th=0.3, outputname='./static/' + str(j) + '_pred.png')

            # 记录文件和掩模路径
            file_names.append('../static/' + str(j) + '.png')
            mask_names.append('../static/' + str(j) + '_pred.png')
            j = j + 1
            # 使用上传的文件和掩模路径渲染模板
        else:
            return render_template('image.html', msg='非png图片文件，请重新选择！')
    # 使用上传的文件和掩模路径渲染模板
    return render_template('image.html', msg='上传成功', FILE_NAMES=file_names, MASK_NAMES=mask_names)


# 定义上传PNG图像的路由
@app.route('/image1', methods=['GET', 'POST'])
def image1():
    if request.method == 'GET':
        return render_template('image1.html')
    file_names = []
    mask_names = []
    # 获取上传的PNG文件
    uploaded_files = request.files.getlist("file[]")
    # path_file_number = glob.glob('./static/images1/*.png')
    j = 0
    for uploaded_file in uploaded_files:
        if uploaded_file and allowed_file(uploaded_file.filename):
            # 保存上传的文件
            # file_name = os.path.join(app.config['UPLOAD_FOLDER'], str(j) + '.png')
            uploaded_file.save('./static/' + str(j) + '.png')

            # 使用save_sample函数处理上传的文件
            save_sample('./static/' + str(j) + '.png', th=0.3, outputname='./static/' + str(j) + '_pred.png')

            # 记录文件和掩模路径
            file_names.append('../static/' + str(j) + '.png')
            mask_names.append('../static/' + str(j) + '_pred.png')
            j = j + 1
            # 使用上传的文件和掩模路径渲染模板
        else:
            return render_template('image1.html', msg='非png图片文件，请重新选择！')
    return render_template('image1.html', msg='上传成功', FILE_NAMES=file_names, MASK_NAMES=mask_names)


if __name__ == "__main__":
    app.run(debug=True)
#     imgfile = "./image/image1.png"
#
#     save_sample(imgfile, th=0.5)
#
#
# imgfile = "./image/image1.png"
# img = cv2.imread(imgfile,cv2.IMREAD_UNCHANGED)
# height, width = img.shape[:2]
# # saveFile = r"F:\\pythonproject\\TD-NHG/imgSave.png"
# # img_new = cv2.imwrite(saveFile, img)
#
# output_file = "./pictures/outputimage1.png"
# img_new =cv2.imwrite(output_file, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
# print(img_new)

# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:48:40 2023

@author: WSH
"""

# 导入所需的库
import sklearn.cluster as skc
import sklearn.feature_extraction.text as skt
from sklearn.metrics import silhouette_score
import os 
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

IMAGE_EXTENSION = ('.apng',
                    '.blp',
                    '.bmp',
                    '.bufr',
                    '.bw',
                    '.cur',
                    '.dcx',
                    '.dds',
                    '.dib',
                    '.emf',
                    '.eps',
                    '.fit',
                    '.fits',
                    '.flc',
                    '.fli',
                    '.fpx',
                    '.ftc',
                    '.ftu',
                    '.gbr',
                    '.gif',
                    '.grib',
                    '.h5',
                    '.hdf',
                    '.icb',
                    '.icns',
                    '.ico',
                    '.iim',
                    '.im',
                    '.j2c',
                    '.j2k',
                    '.jfif',
                    '.jp2',
                    '.jpc',
                    '.jpe',
                    '.jpeg',
                    '.jpf',
                    '.jpg',
                    '.jpx',
                    '.mic',
                    '.mpeg',
                    '.mpg',
                    '.mpo',
                    '.msp',
                    '.palm',
                    '.pbm',
                    '.pcd',
                    '.pcx',
                    '.pdf',
                    '.pgm',
                    '.png',
                    '.pnm',
                    '.ppm',
                    '.ps',
                    '.psd',
                    '.pxr',
                    '.ras',
                    '.rgb',
                    '.rgba',
                    '.sgi',
                    '.tga',
                    '.tif',
                    '.tiff',
                    '.vda',
                    '.vst',
                    '.webp',
                    '.wmf',
                    '.xbm',
                    '.xpm'
)

images_dir = r"C:\Users\WSH\Desktop\BA画风训练集\_6_blue_archive style"

# 获取该目录下所有的图片文件名
images_files_list = [ f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) and f.endswith(IMAGE_EXTENSION) ]

def ext_to_txt(ori_list: list) -> list:
    #将一个列表内扩展名全部换成.txt
    fin_list = []
    for f in ori_list:
        name, ext = os.path.splitext(f)
        new = name + ".txt"
        fin_list.append(new)
    return fin_list
       
#  与图片文件同名的.txt列表
tags_files_list = ext_to_txt(images_files_list)


def read_tags_list(tags_dir: str, tags_files_list: str) ->list:
    tags_list = []
    for name in tags_files_list:
        try:
            with open( os.path.join(images_dir, name) ) as content:
                tags_list.append( content.read() )
        except Exception as e:
            print(f"读取 {name} 发送错误, error: {e}")
            tags_list.append( "" )
    return tags_list

tags_list = read_tags_list(images_dir, tags_files_list)

# 去逗号
# tags_list = [ t.replace(" ", "_").replace(",", " " ) for t in tags_list ] 

# 提取标签特征
#tfvec = skt.CountVectorizer()
tfvec = skt.TfidfVectorizer()
X = tfvec.fit_transform(tags_list).toarray()
tf_tags_list = tfvec.get_feature_names()

# 使用肘部法则确定最优的聚类数
wss = [] # 存储每个聚类数对应的簇内平方和
# 定义一个空列表，用于存储不同k值对应的轮廓系数
silhouette_scores = []

k_range = range(1, 11) # 聚类数的范围，可以根据需要调整
for k in k_range:
    kmeans_model = skc.KMeans(n_clusters=k) # 创建K-Means模型
    y_pred = kmeans_model.fit_predict(X) # 训练模型
    wss.append(kmeans_model.inertia_) # 计算并存储簇内平方和
    score = silhouette_score(X, y_pred) if k != 1 else 0 # 计算轮廓系数
    silhouette_scores.append(score) # 将轮廓系数添加到列表中

# 绘制肘部曲线
plt.plot(k_range, wss, "o-")
plt.xlabel("Number of clusters")
plt.ylabel("Within-cluster sum of squares")
plt.title("Elbow Method")
plt.show()

# 绘制轮廓系数曲线
plt.plot(k_range, silhouette_scores, "o-")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.title("Silhouette Method")
plt.show()

# 根据肘部曲线选择一个合适的聚类数，例如3
optimal_k = 2

# 使用最优的聚类数进行聚类
kmeans_model = skc.KMeans(n_clusters=optimal_k) # 创建K-Means模型
y_pred = kmeans_model.fit_predict(X) # 训练模型并得到聚类结果
centers = kmeans_model.cluster_centers_

centers_list = centers.tolist()
a = []
for i,p in enumerate(centers[0]):
    a.append( abs(centers[0][i] - centers[1][i]) )
print(tf_tags_list[a.index( max(a) )])

"""
# 层次聚类
agg_model = skc.AgglomerativeClustering(n_clusters=2, linkage="ward")
y_pred = agg_model.fit_predict(X)
"""



with gr.Blocks() as demo:
    
    images_tuple_list = []
    for i, name in enumerate( [ (images_files_list[i]) for i in np.where(y_pred == 1)[0] ] ):
        images_tuple_list.append( (os.path.join(images_dir, name), f"{name}") )
    gr.Gallery( label="聚类图片", value=images_tuple_list ).style(grid=[6], height="auto", preview=False)
    images_tuple_list = []
    for i, name in enumerate( [ (images_files_list[i]) for i in np.where(y_pred == 0)[0] ] ):
        images_tuple_list.append( (os.path.join(images_dir, name), f"{name}") )
    gr.Gallery( label="聚类图片", value=images_tuple_list ).style(grid=[6], height="auto", preview=False)

demo.launch(debug=True, inbrowser=True)

"""
for name in type1:

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    img = mpimg.imread(os.path.join(images_dir, name))
    plt.imshow(img)
    plt.show()
"""



# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:38:31 2023

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
from scipy.cluster.hierarchy import linkage, dendrogram
from cluster_images_by_tags import read_tags
from tqdm import tqdm
from PIL import Image
from typing import Callable, List


MAX_GALLERY_NUMBER = 100  # 画廊里展示的最大聚类数量为100


# 逗号分词器
def comma_tokenizer(text: str) -> List[str]:
    """
    定义一个以逗号为分隔符的分词器函数，并对每个标签进行去空格操作
    如输入"xixi, haha"
    返回["xixi", "haha"]
    """
    return [tag.strip() for tag in text.split(',')]


def create_Vectorizer(tokenizer: Callable[ [str], List[str] ]=None,
                      use_CountVectorizer: bool=False
                      ):
    
    """
    返回指定的特征提取器
    默认返回 默认的TfidfVectorizer
    """
    if use_CountVectorizer:
        tfvec = skt.CountVectorize(tokenizer=tokenizer) 
    else:
        tfvec = skt.TfidfVectorizer(tokenizer=tokenizer)
        
    return tfvec


def cluster_images(images_dir: str, confirmed_cluster_number: int, use_cache: bool) -> list:
    """
    对指定目录下的图片进行聚类，将会使用该目录下与图片同名的txt中的tag做为特征
    confirmed_cluster_number为指定的聚类数
    如果use_bool，则会把图片拷贝到指定目录下的cache文件夹，并显示缩略图
    """
    
    images_and_tags_tuple = read_tags(images_dir)
    
    # 获取文件名列表和tags列表
    images_files_list = [image for image, _ in images_and_tags_tuple]
    tags_list = [tags for _, tags in images_and_tags_tuple]
    
    # tags转为向量特征
    tfvec = create_Vectorizer()
    X = tfvec.fit_transform(tags_list).toarray()  # 向量特征
    tf_tags_list = tfvec.get_feature_names()  # 向量每列对应的tag
    
    # 聚类，最大聚类数不能超过样本数
    kmeans_model = skc.KMeans( n_clusters=min( confirmed_cluster_number, len(tags_list) ) ) # 创建K-Means模型
    y_pred = kmeans_model.fit_predict(X) # 训练模型并得到聚类结果
    centers = kmeans_model.cluster_centers_
    
    #分类的ID
    clusters_ID = np.unique(y_pred)
    clustered_images_list = [np.compress(np.equal(y_pred, i), images_files_list).tolist() for i in clusters_ID]
    
    def cache_image(input_dir: str, cluster_list: list, resolution: int=512 ):
        """ 如果使用缓存，就调用pillow，将重复的图片缓存到同路径下的一个cache文件夹中，分辨率最大为resolution,与前面图片名字一一对应 """
        
        # 建一个文件夹
        cache_dir = os.path.join(input_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        print("缓存缩略图中，caching...")
        for cluster in tqdm(cluster_list):
            for image_name in cluster:
                # 已经存在同名缓存文件，就不缓存了
                if not os.path.join(cache_dir, image_name):
                    try:
                        with Image.open( os.path.join( input_dir, image_name ) ) as im:
                            im.thumbnail( (resolution, resolution) )
                            im.save( os.path.join(cache_dir, image_name) )
                    except Exception as e:
                        print(f"缓存 {image_name} 失败, error: {e}")
        print(f"缓存完成: {cache_dir}\nDone!")
        
    if use_cache:
        cache_image(images_dir, clustered_images_list, resolution=512)
     
    # 如果用缓存就展示缓存图
    gallery_images_dir = images_dir if not use_cache else os.path.join( images_dir, "cache" )
    
    # 注意，返回的列表长度为显示gallery组件数的两倍
    # 因为列表里偶数位为Accordion组件，奇数位为Gallery组件
    visible_gr_gallery_list = []
    for i in range( len(clustered_images_list) ):
        gallery_images_tuple_list = [ (os.path.join(gallery_images_dir,name), name) for name in clustered_images_list[i] ]
        visible_gr_gallery_list.extend( [ gr.update(visible=True), gr.update( value=gallery_images_tuple_list, visible=True ) ] )
        
    unvisible_gr_gallery_list = [ gr.update( visible=False ) for i in range( 2*( MAX_GALLERY_NUMBER-len(clustered_images_list) ) ) ]
    
    return visible_gr_gallery_list + unvisible_gr_gallery_list


def cluster_analyse(images_dir: str, max_cluster_number: int):
    """
    读取指定路径下的图片，并依据与图片同名的txt内的tags进行聚类
    将评估从聚类数从 2~max_cluster_number 的效果
    返回matplotlib类型的肘部曲线和轮廓系数
    """
    
    images_and_tags_tuple = read_tags(images_dir)
    
    # 提取标签特征
    # tfvec = skt.CountVectorizer()
    # tfvec = skt.TfidfVectorizer(tokenizer=comma_tokenizer, binary=True) # 创建TfidfVectorizer对象，并指定分词器
    
    tags_list = [tags for _, tags in images_and_tags_tuple]
    
    tfvec = create_Vectorizer()
    X = tfvec.fit_transform(tags_list).toarray()
    
    # 使用肘部法则和轮廓系数确定最优的聚类数
    wss = [] # 存储每个聚类数对应的簇内平方和
    silhouette_scores = []  # 用于存储不同k值对应的轮廓系数
    
    # 最大聚类数不能超过样本
    k_range = range( 1, min(max_cluster_number+1, len(tags_list)+1) ) # 聚类数的范围(左闭右开)
    for k in k_range:
        kmeans_model = skc.KMeans(n_clusters=k) # 创建K-Means模型
        y_pred = kmeans_model.fit_predict(X) # 训练模型
        wss.append(kmeans_model.inertia_) # 计算并存储簇内平方和
        score = silhouette_score(X, y_pred) if k != 1 else 0 # 计算轮廓系数,聚类数为1时无法计算
        silhouette_scores.append(score) # 储存轮廓系数
    
    # 绘制肘部曲线
    Elbow_gr_Plot = plt.figure()
    plt.plot(k_range, wss, "o-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Within-cluster sum of squares")
    plt.title("Elbow Method")

    # 绘制轮廓系数曲线
    Silhouette_gr_Plot = plt.figure()
    plt.plot(k_range, silhouette_scores, "o-")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Method")

    return Elbow_gr_Plot, Silhouette_gr_Plot


def create_gr_gallery(max_gallery_number: int) -> list:
    """
    根据指定的最大数，创建相应数量的带Accordion的Gallery组件
    返回一个列表，长度为2*max_gallery_number
    偶数位为Accordion组件，奇数位为Gallery组件
    """
    gr_Accordion_and_Gallery_list = []
    for i in range(max_gallery_number):
        with gr.Accordion(f"聚类{i}", open=True, visible=False) as Gallery_Accordion:
            gr_Accordion_and_Gallery_list.extend( [ Gallery_Accordion, gr.Gallery(value=[]).style(grid=[6], height="auto") ] )
    return gr_Accordion_and_Gallery_list

with gr.Blocks() as demo:
    with gr.Box():
        with gr.Row():
            with gr.Column(scale=10):
                images_dir = gr.Textbox(label="图片目录")
            """
            with gr.Column(scale=10):
                tags_dir = gr.Textbox(label="tags目录（留空则采用图片目录）")
            """
            with gr.Column(scale=1):
                use_cache = gr.Checkbox(label="使用缓存",info="注意，如果图片目录下的cache文件夹内存在同名图片，则不会重新缓存(可能会照成图片显示不一致)")
        with gr.Row():
            confirmed_cluster_number = gr.Slider(2, 100, step=1, value=2, label="聚类数")
            cluster_images_button = gr.Button("开始聚类")
    with gr.Row():
        with gr.Accordion("聚类效果分析", open=False):
            with gr.Row():
                max_cluster_number = gr.Slider(2, 100, step=1, value=10, label="最大聚类数")
                cluster_analyse_button = gr.Button("开始分析")
            with gr.Row():
                Elbow_gr_Plot = gr.Plot(label="肘部曲线（选择拐点）")
                Silhouette_gr_Plot = gr.Plot(label="轮廓系数（越大越好）")
    with gr.Row():
        with gr.Accordion("聚类图片展示", open=True):
            gr_Accordion_and_Gallery_list = create_gr_gallery(MAX_GALLERY_NUMBER)
            
    cluster_images_button.click(fn=cluster_images,
                                inputs=[images_dir, confirmed_cluster_number, use_cache],
                                outputs=gr_Accordion_and_Gallery_list
    )
            
    cluster_analyse_button.click(fn=cluster_analyse,
                                 inputs=[images_dir, max_cluster_number],
                                 outputs=[Elbow_gr_Plot, Silhouette_gr_Plot]
    )
    
demo.launch(inbrowser=True,debug=True)
    
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
import pandas as pd
from datetime import datetime
import shutil
import math
# from kneed import KneeLocator


MAX_GALLERY_NUMBER = 100  # 画廊里展示的最大聚类数量为100
CACHE_RESOLUTION = 256  # 缓存图片时最大分辨率


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


def cluster_images(images_dir: str, confirmed_cluster_number: int, use_cache: bool, global_dict_State: dict) -> list:
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
    stop_tags = tfvec.stop_words_
    
    # 分类的ID
    clusters_ID = np.unique(y_pred)
    clustered_images_list = [np.compress(np.equal(y_pred, i), images_files_list).tolist() for i in clusters_ID]
    
    # pandas处理数据
    all_center = pd.Series( np.mean(X, axis=0), tf_tags_list )

    def _create_pred_df():
        vec_df = pd.DataFrame(X,columns=tf_tags_list)  # 向量
        cluster_df = pd.DataFrame( {"images_file":images_files_list, "y_pred":y_pred} )  #聚类结果
        pred_df = pd.concat([cluster_df, vec_df ], axis=1)
        return pred_df
    
    pred_df = _create_pred_df()
    tags_df = pred_df.iloc[:,2:]  # 取出与tags有关部分
    

    def find_duplicate_tags(tags_df):
        """ 找出一个dataframe内每一行都不为0的列，返回一个pandas.Index对象 """
        try:
            tags_df.astype(float)
        except Exception as e:
            print(f"需要找出共有标签的tags_df内包含了非法值 error: {e}")
        tags_columns_index = tags_df.columns  # 取出列标签
        duplicate_tags_index = tags_columns_index[ tags_df.all(axis=0) ] # 找出每一行都为真,即不为0的列名，即共有的tags
        return duplicate_tags_index  # 输出pandas.Index对象
    
    common_duplicate_tags_set = set( find_duplicate_tags(tags_df) )
    
    def _fine_cluster_duplicate_tags(pred_df):
        cluster_duplicate_tags_list = []
        # 找到每个聚类各自的重复标签
        for i in clusters_ID:
            cluster_df = pred_df[pred_df["y_pred"] == i]
            cluster_tags_df = cluster_df.iloc[:,2:]  # 取出与tags有关部分
            cluster_duplicate_tags_list.append( set( find_duplicate_tags(cluster_tags_df) ) )
        return cluster_duplicate_tags_list
        """
        # 找到每个聚类各自的重复标签
        cluster_duplicate_tags_set_list = pred_df.groupby('y_pred',key=None).apply(lambda x: find_duplicate_tags(x.iloc[:, 2:]))
        return cluster_duplicate_tags_set_list
        """
    
    cluster_duplicate_tags_list = _fine_cluster_duplicate_tags(pred_df)
            
    
    # 赋值到全局组件中，将会传递至confirm_cluster_button.click
    global_dict_State["clustered_images_list"] = clustered_images_list
    global_dict_State["images_dir"] = images_dir
    
    def cache_image(input_dir: str, cluster_list: list, resolution: int=512 ):
        """ 如果使用缓存，就调用pillow，将重复的图片缓存到同路径下的一个cache文件夹中，分辨率最大为resolution,与前面图片名字一一对应 """
        
        # 建一个文件夹
        cache_dir = os.path.join(input_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        print("缓存缩略图中，caching...")
        for cluster in tqdm(cluster_list):
            for image_name in cluster:
                # 已经存在同名缓存文件，就不缓存了
                if not os.path.exists( os.path.join(cache_dir, image_name) ):
                    try:
                        with Image.open( os.path.join( input_dir, image_name ) ) as im:
                            im.thumbnail( (resolution, resolution) )
                            im.save( os.path.join(cache_dir, image_name) )
                    except Exception as e:
                        print(f"缓存 {image_name} 失败, error: {e}")
        print(f"缓存完成: {cache_dir}\nDone!")
        
    if use_cache:
        cache_image(images_dir, clustered_images_list, resolution=CACHE_RESOLUTION)
     
    # 如果用缓存就展示缓存图
    gallery_images_dir = images_dir if not use_cache else os.path.join( images_dir, "cache" )
    
    # 注意，返回的列表长度为显示gallery组件数的两倍
    # 因为列表里偶数位为Accordion组件，奇数位为Gallery组件
    visible_gr_gallery_list = []
    for i in range( len(clustered_images_list) ):
        gallery_images_tuple_list = [ (os.path.join(gallery_images_dir,name), name) for name in clustered_images_list[i] ]
        visible_gr_gallery_list.extend( [ gr.update(visible=True), gr.update( value=gallery_images_tuple_list, visible=True ) ] )
        
    unvisible_gr_gallery_list = [ gr.update( visible=False ) for i in range( 2*( MAX_GALLERY_NUMBER-len(clustered_images_list) ) ) ]
    
    return visible_gr_gallery_list + unvisible_gr_gallery_list + [gr.update(visible=True)] + [global_dict_State]


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
    
    print("聚类分析开始")
    for k in tqdm(k_range):
        kmeans_model = skc.KMeans(n_clusters=k) # 创建K-Means模型
        y_pred = kmeans_model.fit_predict(X) # 训练模型
        wss.append(kmeans_model.inertia_) # 计算并存储簇内平方和
        score = silhouette_score(X, y_pred) if k != 1 else None  # 计算轮廓系数,聚类数为1时无法计算
        silhouette_scores.append(score) # 储存轮廓系数
    
    Silhouette_DataFrame = pd.DataFrame( {"x":k_range, "y":silhouette_scores} )
    Elbow_DataFrame = pd.DataFrame( {"x":k_range, "y":wss} )
    
    final_clusters_number = len( np.unique(y_pred) )  # 实际最大聚类数
    head_number = max( 1, min( 10, round( math.log2(final_clusters_number) ) ) )  # 展示log2(实际聚类数)个，最少要展示1个，最多展示10个
    # 对轮廓系数从大到小排序，展示前head_number个
    bset_cluster_number_DataFrame = Silhouette_DataFrame.sort_values(by='y', ascending=False).head(head_number)
    
    """
    自动找拐点，在聚类数大了后效果不好
    kl = KneeLocator(k_range, wss, curve="convex", direction="decreasing")
    kl.plot_knee()
    print( round(kl.elbow, 3) )
    """
    
    print("聚类分析结束")
    
    # 绘制肘部曲线
    return Silhouette_DataFrame, Elbow_DataFrame, gr.update(value=bset_cluster_number_DataFrame,visible=True)


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


def confirm_cluster(process_clusters_method:int, global_dict_State: dict):
    """
    根据选择的图片处理方式，对global_dict_State中聚类后的图片列表，以及路径进行相关操作
    
    
    process_clusters_method = gr.Radio(label="图片处理方式",
                                       choices=["重命名原图片","在Cluster文件夹下生成聚类副本","移动原图至Cluster文件夹"],
                                       type="index",
    )
    
    global_dict_State["clustered_images_list"] = clustered_images_list
    global_dict_State["images_dir"] = images_dir
    
    """
    
    images_dir = global_dict_State.get("images_dir", "")
    clustered_images_list = global_dict_State.get("clustered_images_list", [] )
    
    
    def change_ext_to_txt(path: str) -> str:
        # 将一个路径或者文件的扩展名改成.txt
        path_and_name, ext = os.path.splitext(path)
        new_path = path_and_name + ".txt"
        return new_path
    def change_name_with_ext(path: str, new_name: str) -> str:
        # 保留一个路径或者文件的扩展名，更改其文件名
        path_and_name, ext = os.path.splitext(path)
        new_path = os.path.join( os.path.dirname(path_and_name), new_name+ext )
        return new_path
    
    # 获取当前时间
    time_now = datetime.now().strftime('%Y%m%d%H%M%S')
    
    
    def rename_images(images_dir: str, clustered_images_list: list):
        """ 依据clustered_images_list中聚类情况，对images_dir下图片以及同名txt文件重命名 """

        print("重命名原图中，renaming...")
        for cluster_index, cluster in tqdm( enumerate(clustered_images_list) ):
            # 重命名
            for image_index, image_name in enumerate(cluster):
                # 重命名图片
                new_image_name = change_name_with_ext(image_name, f"cluster{cluster_index}-{image_index:06d}-{time_now}")
                try:
                    os.rename( os.path.join(images_dir, image_name), os.path.join(images_dir, new_image_name) )
                except Exception as e:
                    print(f"重命名 {image_name} 失败, error: {e}")
                # 重命名txt
                txt_name = change_ext_to_txt(image_name)
                new_txt_name = change_name_with_ext(txt_name, f"cluster{cluster_index}-{image_index:06d}-{time_now}")
                try:
                    os.rename( os.path.join(images_dir, txt_name), os.path.join(images_dir, new_txt_name) )
                except Exception as e:
                    print(f"重命名 {txt_name} 失败, error: {e}")
        print("重命名完成  Done!")
        
    if process_clusters_method == 0:
        rename_images(images_dir, clustered_images_list)
    
    def copy_or_move_images(images_dir: str, clustered_images_list: list, move=False):
        """
        依据clustered_images_list中聚类情况，将images_dir下图片以及同名txt拷贝或移动至Cluster文件夹
        move=True时为移动
        """
        
        Cluster_folder_dir = os.path.join(images_dir, f"Cluster-{time_now}")
        
        process_func = shutil.move if move else shutil.copy2
        
        # 清空聚类文件夹
        if os.path.exists(Cluster_folder_dir):
            shutil.rmtree(Cluster_folder_dir)
        os.makedirs(Cluster_folder_dir, exist_ok=True)
        
        print("拷贝聚类中，coping...")
        for cluster_index, cluster in tqdm( enumerate(clustered_images_list) ):
            Cluster_son_folder_dir = os.path.join(Cluster_folder_dir, f"cluster-{cluster_index}")
            os.makedirs(Cluster_son_folder_dir, exist_ok=True)
            
            # 拷贝
            for image_name in cluster:
                # 拷贝或移动图片
                try:
                    process_func( os.path.join(images_dir, image_name), os.path.join(Cluster_son_folder_dir, image_name) )
                except Exception as e:
                    print(f"拷贝或移动 {image_name} 失败, error: {e}")
                # 拷贝或移动txtx
                txt_name = change_ext_to_txt(image_name)
                try:
                    process_func( os.path.join(images_dir, txt_name), os.path.join(Cluster_son_folder_dir, txt_name) )
                except Exception as e:
                    print(f"拷贝或移动 {txt_name} 失败, error: {e}")
        print(f"拷贝或移动完成: {Cluster_folder_dir}\nDone!")
        
    if process_clusters_method == 1:
        copy_or_move_images(images_dir, clustered_images_list, move=False)
    if process_clusters_method == 2:
        copy_or_move_images(images_dir, clustered_images_list, move=True)
    
    return gr.update(visible=False)
    



##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################



with gr.Blocks() as demo:
    
    global_dict_State = gr.State(value={})  # 这个将会起到全局变量的作用，类似于globals()
    """
    全局列表
    global_dict_State["clustered_images_list"] = clustered_images_list
    global_dict_State["images_dir"] = images_dir
    """
    
    with gr.Box():
        with gr.Row():
            with gr.Column(scale=10):
                images_dir = gr.Textbox(label="图片目录")
            """
            with gr.Column(scale=10):
                tags_dir = gr.Textbox(label="tags目录（留空则采用图片目录）")
            """
            with gr.Column(scale=1):
                use_cache = gr.Checkbox(label="使用缓存",info="如果cache目录内存在同名图片，则不会重新缓存(可能会造成图片显示不一致)")
        with gr.Row():
            with gr.Accordion("聚类效果分析", open=True):
                with gr.Row():
                    max_cluster_number = gr.Slider(2, MAX_GALLERY_NUMBER, step=1, value=10, label="分析时最大聚类数")
                    cluster_analyse_button = gr.Button("开始分析")
                with gr.Row():
                    Silhouette_gr_Plot = gr.LinePlot(label="轮廓系数",
                                                     x="x",
                                                     y="y",
                                                     tooltip=["x", "y"],
                                                     x_title="Number of clusters",
                                                     y_title="Silhouette score",
                                                     title="Silhouette Method",
                                                     overlay_point=True,
                                                     width=400,
                    )
                    Elbow_gr_Plot = gr.LinePlot(label="肘部曲线",
                                                x="x",
                                                y="y",
                                                tooltip=["x", "y"],
                                                x_title="Number of clusters",
                                                y_title="Within-cluster sum of squares",
                                                title="Elbow Method",
                                                overlay_point=True,
                                                width=400,
                    )
                with gr.Row():
                    bset_cluster_number_DataFrame = gr.DataFrame(value=[],
                                                                 label="根据轮廓曲线推荐的聚类数（y越大越好）",
                                                                 visible=False
                    )
    with gr.Box():
        with gr.Row():
            confirmed_cluster_number = gr.Slider(2, MAX_GALLERY_NUMBER, step=1, value=2, label="聚类数")
            cluster_images_button = gr.Button("开始聚类并展示结果")
    with gr.Row():
        with gr.Accordion("聚类图片展示", open=True):
            with gr.Row(visible=False) as confirm_cluster_Row:
                process_clusters_method_choices = ["重命名原图片(不推荐)","在Cluster文件夹下生成聚类副本(推荐)","移动原图至Cluster文件夹(大数据集推荐)"]
                process_clusters_method = gr.Radio(label="图片处理方式",
                                                   value=process_clusters_method_choices[1],
                                                   choices=process_clusters_method_choices,
                                                   type="index",
                )
                confirm_cluster_button = gr.Button(value="确认聚类", elem_classes="attention", variant="primary")
            gr_Accordion_and_Gallery_list = create_gr_gallery(MAX_GALLERY_NUMBER)
            
    cluster_images_button.click(fn=cluster_images,
                                inputs=[images_dir, confirmed_cluster_number, use_cache, global_dict_State],
                                outputs=gr_Accordion_and_Gallery_list + [confirm_cluster_Row] + [global_dict_State]
    )

    cluster_analyse_button.click(fn=cluster_analyse,
                                 inputs=[images_dir, max_cluster_number],
                                 outputs=[Silhouette_gr_Plot, Elbow_gr_Plot, bset_cluster_number_DataFrame]
    )
    
    confirm_cluster_button.click(fn=confirm_cluster,
                                 inputs=[process_clusters_method, global_dict_State],
                                 outputs=[confirm_cluster_Row],
    )
    
demo.launch(inbrowser=True,debug=True)
    
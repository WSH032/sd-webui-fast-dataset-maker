# -*- coding: utf-8 -*-
"""
Created on Fri May  5 17:38:31 2023

@author: WSH
"""


import gradio as gr
import os

choose_image_index = ""
images_tuple_list = []


def find_duplicates_images(images_dir: str):

    global images_tuple_list

    from imagededup.methods import PHash
    import os
    import toml
    # 载入模型
    phasher = PHash()
    # 检测重复
    duplicates = phasher.find_duplicates(images_dir)
    # 只保留确实有重复的图片
    indeed_duplicates_set = [set(v).union({k})
                             for k, v in duplicates.items() if v]
    # 将重复的图片聚类
    cluster = []
    for s in indeed_duplicates_set:
        for m in cluster:
            if s & m:
                m.union(s)
                break
        else:
            cluster.append(s)

    # 将图片的绝对路径和索引合成一个元组，最后把全部元组放入一个列表，向gradio.gallery传递
    images_tuple_list.clear()
    parent_index = 0
    for parent in cluster:
        son_index = 0
        for son in parent:
            images_tuple_list.append(
                (os.path.join(images_dir, son), f"{parent_index}:{son_index}"))
            son_index += 1
        parent_index += 1

    # 根据聚类数,生成如下字典
    """
    "1":[]
    "2":[]
    ...
    """
    delet_images_dict = {f"{i}": [] for i in range(parent_index)}
    delet_images_str = toml.dumps(delet_images_dict)

    return images_tuple_list, ""
    # return images_tuple_list, delet_images_str


def confirm(delet_images_str: str) -> str:

    import bisect
    import toml
    # 将字符串载入成字典
    delet_images_dict = toml.loads(delet_images_str)
    # 把删除标志如 "0:1" 分成 "0" 和 "1"
    [parent_index_str, son_index_str] = choose_image_index.split(":")[0:2]

    # 如果字典里没有这个键，就给他赋值一个列表
    if delet_images_dict.get(parent_index_str) is None:
        delet_images_dict[parent_index_str] = [int(son_index_str)]
    # 如果有这个键，并且该键对应的列表值中不含有这个子索引，就按二分法把子索引插入到该键对应的列表值中
    else:
        if int(son_index_str) not in delet_images_dict[parent_index_str]:
            bisect.insort(
                delet_images_dict[parent_index_str], int(son_index_str))

    # 按键名排序
    delet_images_dict = dict(
        sorted(delet_images_dict.items(), key=lambda x: int(x[0])))

    return toml.dumps(delet_images_dict)


def cancel(delet_images_str: str) -> str:
    import toml
    # 将字符串载入成字典
    delet_images_dict = toml.loads(delet_images_str)
    # 把删除标志如 "0:1" 分成 "0" 和 "1"
    [parent_index_str, son_index_str] = choose_image_index.split(":")[0:2]

    if delet_images_dict.get(parent_index_str) is not None:
        if int(son_index_str) in delet_images_dict[parent_index_str]:
            delet_images_dict[parent_index_str].remove(int(son_index_str))
    return toml.dumps(delet_images_dict)


# image_dir = r"C:\Users\WSH\Desktop\B站文件\冷门参数"
# image_list = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

def get_choose_image_index(evt: gr.SelectData):
    # evt.value 为标签 ；evt.index 为图片序号； evt.target 为调用组件名
    global choose_image_index
    choose_image_index = evt.value
    return f"选择 {evt.value}", f"取消 {evt.value}"


with gr.Blocks(css="#delet_button {color: red}") as demo:
    with gr.Row():
        with gr.Column(scale=10):
            images_dir = gr.Textbox(label="图片目录")
        with gr.Column(scale=1):
            find_duplicates_images_button = gr.Button("扫描重复图片")
    with gr.Row():
        with gr.Column(scale=10):
            with gr.Row():
                duplicates_images_gallery = gr.Gallery(label="重复图片", value=[]).style(grid=[6], height="auto", preview=False)
            with gr.Row():
                confirm_button = gr.Button("选择图片")
                cancel_button = gr.Button("取消图片")
        with gr.Column(scale=1):
            delet_button = gr.Button("删除（不可逆）", elem_id="delet_button")
            delet_images_str = gr.Textbox(label="待删除列表")

    # 按下后，在指定的目录搜索重复图像，并返回带标签的重复图像路径
    find_duplicates_images_button.click(fn=find_duplicates_images,
                                        inputs=[images_dir],
                                        outputs=[duplicates_images_gallery, delet_images_str]
                                        )

    # 点击一个图片后，记录该图片标签于全局变量choose_image_index，并且把按钮更名为该标签
    duplicates_images_gallery.select(fn=get_choose_image_index,
                                     inputs=[],
                                     outputs=[confirm_button, cancel_button]
                                     )

    # 按下后，将全局变量choose_image_index中的值加到列表中
    confirm_button.click(fn=confirm,
                         inputs=[delet_images_str],
                         outputs=[delet_images_str]
                         )

    # 按下后，将全局变量choose_image_index中的值加到列表中
    cancel_button.click(fn=cancel,
                        inputs=[delet_images_str],
                        outputs=[delet_images_str]
                        )


demo.launch(debug=False, inbrowser=True)

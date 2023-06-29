# sd-webui-fast-dataset-maker
一个有趣的扩展，整合了 `图库`，`下载`，`去重`，`聚类` ，可以快速搜集、分类、处理你的图片。

A funny extension that integrates `image-browsing` , `downloader` , `deduplicate` , `cluster` , can quickly collect, classify and process your images.

### Powered by

- [zanllp/sd-webui-infinite-image-browsing](https://github.com/zanllp/sd-webui-infinite-image-browsing)
- [WSH032/Gelbooru-API-Downloader](https://github.com/WSH032/Gelbooru-API-Downloader)
- [WSH032/image-deduplicate-cluster-webui](https://github.com/WSH032/image-deduplicate-cluster-webui)
- [toshiaki1729/dataset-tag-editor-standalone](https://github.com/toshiaki1729/dataset-tag-editor-standalone)

> 我只是把它们整合起来，并独立运行

### Credit
WebUI部分借鉴了[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)



## 现在我们有什么？
### 1. 强大的图片浏览器
来自[zanllp/sd-webui-infinite-image-browsing](https://github.com/zanllp/sd-webui-infinite-image-browsing)
![sd-webui-infinite-image-browsing](https://github.com/WSH032/sd-webui-fast-dataset-maker/assets/126865849/b98a293f-a3a0-4d59-a997-cae86e7f25b4)

### 2. Gelbooru的图片下载器
来自[WSH032/Gelbooru-API-Downloader](https://github.com/WSH032/Gelbooru-API-Downloader)
![Gelbooru-API-Downloader](https://github.com/WSH032/sd-webui-fast-dataset-maker/assets/126865849/03dbd069-336c-40a1-9e14-5143eb915339)

### 3. 基于imagededup，sklearn，WD14模型的图片去重与聚类
来自[WSH032/image-deduplicate-cluster-webui](https://github.com/WSH032/image-deduplicate-cluster-webui)
![deduplicate-webui](https://github.com/WSH032/sd-webui-fast-dataset-maker/assets/126865849/dccd5928-cdfc-4d58-b806-edc0ed2df9c9)

**[cluster-webui 展示](https://github.com/WSH032/sd-webui-fast-dataset-maker/assets/126865849/c38200fa-78dc-4b91-a006-1784fb7059bb)**

### 4. 强大的booru风格的tag编辑插件
来自[toshiaki1729/dataset-tag-editor-standalone](https://github.com/toshiaki1729/dataset-tag-editor-standalone)
![dataset-tag-editor-standalone](https://github.com/WSH032/sd-webui-fast-dataset-maker/assets/126865849/6761bbb6-b9bb-4463-a41c-0b0eceb1baab)

**[dataset-tag-editor-standalone 完整展示](https://github.com/WSH032/sd-webui-fast-dataset-maker/assets/126865849/919a2d87-d399-41a9-86bf-b8c4e50973b0)**



## 安装 Install

### （一）Colab使用

### （二）部署使用
**请尽量使用python >= 3.10**

运行以下git命令克隆仓库并初始化子模块
```shell
git clone https://github.com/WSH032/sd-webui-fast-dataset-maker.git --recurse-submodules
```
安装依赖
- Window用户可以运行 `install.ps1` 安装依赖
- Linux用户请自行通过 `pip install -r requirements.txt` 安装依赖

```
对于通过 `requirements.txt` 安装的用户请注意
如果你在使用WD14生成tag的时候想使用GPU加速
请打开 `requirements.txt` 修改安装 torch=2.0.0 + cuda118

# If you want to install with CUDA, please use the following command:

# --extra-index-url https://download.pytorch.org/whl/cu118
# torch==2.0.0+cu118
# torchvision==0.15.1+cu118
```
#### 注意
> 此WebUI使用了Gradio非官方文档的操作方法，并在 `gradio <=3.35.2, >=3.29.0` 进行了测试，但不保证未来Gradio官方不会修改相应接口
>
> `requirements.txt` 内只要求了 `gradio >= 3.31.0` ，如果出现问题请尝试降级至 `gradio <=3.35.2, >=3.31.0`
>
> 或者使用 `requirements_versions.txt` 安装依赖


### （三）做为[AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)的扩展使用
请参阅原作者各自的仓库，几乎所有扩展均可做为 SD-WebUI 的扩展使用
- [zanllp/sd-webui-infinite-image-browsing](https://github.com/zanllp/sd-webui-infinite-image-browsing)
- [WSH032/Gelbooru-API-Downloader](https://github.com/WSH032/Gelbooru-API-Downloader)
- [WSH032/image-deduplicate-cluster-webui](https://github.com/WSH032/image-deduplicate-cluster-webui)
- [toshiaki1729/dataset-tag-editor-standalone](https://github.com/toshiaki1729/dataset-tag-editor-standalone)

![A1111-SD-WebUI](https://github.com/WSH032/sd-webui-fast-dataset-maker/assets/126865849/63caea91-42ce-4be5-b3c7-90ddaff17487)


## 部署使用 - 使用方法

Please refer to the original author's repository for the specific usage of each extension

各插件具体的使用方法请查看原作者仓库自述文件，这里不再重复

### （一） 新手 请运行 `run_webui.ps1`

### （二） 进阶 Shell中运行并设置参数
```shell
python webui.py # 使用 `--help` 查看可选参数

# 例如
# python webui.py --sd_webui_config="config.json" --update_image_index --inbrowser

```

### （三） Python API 接口

请参阅 Colab 示例 或者 `run_webui.ipynb`


```python
from webui import WebuiUtils

# params for extension
# 为各扩展设置参数，具体参数请参阅 `extensions/extensions_preload.py`
# 或者运行 `WebuiUtils(help=True)` 查看帮助信息
# Pyhton 接口参数 与 命令行参数一致
webui_utils = WebuiUtils(

    # for image_browsing setting
    sd_webui_config = "config.json",
    extra_paths = ["path"],
    update_image_index = True,
    
    # disable extension
    disable_image_browsing = False,
    disable_deduplicate_cluster = False,
    disable_tag_editor = False,
)

"""
The args of `queue()` and `launch()` are the same as `gradio.Blocks.queue()` and `gradio.Blocks.launch()`

接下来把它当成 `gradio.Blocks` 使用就好，所有接口与 `gradio.Blocks` 一致
"""
webui_utils.queue(concurrency_count=2)  # equal to `gradio.Blocks.queue(concurrency_count=2)`

webui_utils.launch(debug=True, server_port=7860)  # equal to `gradio.Blocks.launch(debug=True, server_port=7860)`

```

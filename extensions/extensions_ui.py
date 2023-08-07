"""
此模块负责各扩展的载入

TODO:
为了保证载入各子扩展时导入的包和子扩展内部导入的包指向同样的内存地址
载入子扩展时是以子扩展为顶级包进行import的
而不是以webui为顶级包(import extension.extension_name.)进行import
    这样做，一个已知的可能问题（目前没观察到）是
    如果前一个子扩展导入了某个包，后一个子扩展也需要导入同样名字的包
    这时候因为sys.modules中已经存在了前一个子扩展的同名包
    所以后一个扩展不会再次进行import，而是错误地使用了前一个子扩展的包
        可能的解决方法是每载入完一个子模块，就恢复sys.modules
        (！但是这会导致官方和第三方模块被重新导入，而导致在子扩展中内存地址不同！)

        推荐的解决方法是各个扩展使用包含其项目特殊名字的包名，以此避免冲突
"""


import os
import sys
import logging
import textwrap
from pathlib import Path
from typing import Tuple, Union, Callable, List, Any

import gradio as gr
from fastapi import FastAPI

from extensions.extensions_tools import (
    EXTENSIONS_DIR,
    javascript_html,
    css_html,
    dir_path2html,
)
from modules import shared


#################### 常量 ####################

UiTabsCallbackReturnAlias = List[tuple[gr.Blocks, str, str]]  # on_ui_tabs函数 返回值 类型注解
UiTabsCallbackAlias = Callable[[], UiTabsCallbackReturnAlias]  # on_ui_tabs函数 类型注解

AppStartedCallbackAlias = Callable[[gr.Blocks, FastAPI], Any]  # on_app_started函数 类型注解

UiCallbackReturnAlias = Tuple[Union[None, UiTabsCallbackAlias], Union[None, AppStartedCallbackAlias], str, str]  # 扩展UI回调函数 返回值 类型注解
UiCallbackAlias = Callable[[str], UiCallbackReturnAlias]  # 扩展UI回调函数 类型注解


#################### 工具 ####################

def check_if_path_exists(path: str, name: str="", path_name: str="") -> bool:
    """检查路径是否存在，不存在则打印错误并返回False，存在则返回True"""
    if not os.path.exists(path):
        error_str = textwrap.dedent(f"""\
            {path_name}路径不存在:
            path: {path}"""
        )
        if name:
            error_str = f"{name}的" + error_str
        logging.error(RuntimeError(error_str))
        return False
    else:
        return True


#################### UI ####################


# 把与各扩展有关的 import 都放在各自的函数里面
# 因为有可能会有某个扩展缺失的清空，所以不需要导入该扩展的模块
# 否则会应某个扩展失效而照成整个程序无法启动

def ui_image_deduplicate_cluster_webui(extension_name: str) -> UiCallbackReturnAlias:
    """ extension_name: 扩展名字，即extensions文件夹中的文件夹名字 """

    print("构建查重聚类")

    """
    注意，在执行此函数时候，extensions_name文件夹会被添加sys.path
    所以可以以子扩展为包进行import
    请尽量以子扩展为顶级包进行import，而不是以webui为顶级包(import extension.extension_name.)进行import
    否则这里导入的包，可能会和子扩展内部导入的包指向不同内存地址
    """
    import cluster_images
    import deduplicate_images
    from img_dedup_clust.tools.js import BaseJS

    # 起到全局修改效果，用A1111_WebUI提供的gradioApp()代替documnet
    BaseJS.set_cls_attr(is_a1111_webui=True)

    def deduplicate_images_ui_tab():
        return (deduplicate_images.create_ui(), deduplicate_images.title, deduplicate_images.title)

    def cluster_images_ui_tab():
        return (cluster_images.create_ui(), cluster_images.title, cluster_images.title)

    def on_ui_tabs() -> UiTabsCallbackReturnAlias:
        """注意，此函数要求能在 sys.path 已经被还原的情况下正常调用"""
        return [cluster_images_ui_tab(), deduplicate_images_ui_tab()]

    js_str = ""

    css_str = ""
    css_path = os.path.join(EXTENSIONS_DIR, extension_name, "style.css")  # css文件应该在的位置
    if check_if_path_exists(css_path, name=extension_name, path_name="css"):
        css_str += css_html(css_path) 

    return on_ui_tabs, None, js_str, css_str


def ui_sd_webui_infinite_image_browsing(extension_name: str) -> UiCallbackReturnAlias:
    """ extension_name: 扩展名字，即extensions文件夹中的文件夹名字 """

    print("构建图库")

    """
    注意，在执行此函数时候，extensions_name文件夹会被添加sys.path
    所以可以以子扩展为包进行import
    请尽量以子扩展为顶级包进行import，而不是以webui为顶级包(import extension.extension_name.)进行import
    否则这里导入的包，可能会和子扩展内部导入的包指向不同内存地址
    """
    from PIL import Image

    from scripts.iib.api import send_img_path
    from scripts.iib.tool import locale, read_info_from_image
    from scripts.iib.logger import logger
    from app import AppUtils
    

    # 请保证这两者与原作者的一致，iib会需要这些id来做某些操作
    title = "无边图像浏览" if locale == "zh" else "Infinite image browsing"  # 显示在SD-WebUI中的名字
    elem_id = "infinite-image-browsing"  # htlm id

    ########## 后端函数函数 ##########

    def on_img_change():
        send_img_path["value"] = ""  # 真正收到图片改变才允许放行
    
    # 修改文本和图像，等待修改完成后前端触发粘贴按钮
    # 有时在触发后收不到回调，可能是在解析params。txt时除了问题删除掉就行了
    def img_update_func():
        try:
            path = send_img_path.get("value")
            logger.info("img_update_func %s", path)
            if not path:
                raise ValueError("path is None or empty")
            img = Image.open(path)
            info = read_info_from_image(img)
            return img, info
        except Exception as e:
            logger.exception("img_update_func %s",e)
            return gr.update(), gr.update()  # 不更新
    
    def not_implemented_error():
        info_str = "NotImplementedError: 独立于SD-WebUI运行时不支持"
        logger.info(info_str)


    ####################

    def create_demo():
        """ ！！！注意，所有的elem_id都不要改，js依靠这些id来操作！！！ """
        with gr.Blocks(analytics_enabled=False) as demo:
            gr.HTML("error", elem_id="infinite_image_browsing_container_wrapper")
            # 以下是使用2个组件模拟粘贴过程
            img = gr.Image(
                type="pil",
                elem_id="iib_hidden_img",
            )
            img_update_trigger = gr.Button(
                "button",
                elem_id="iib_hidden_img_update_trigger",
            )
            img_file_info = gr.Textbox(elem_id="iib_hidden_img_file_info")


            for tab in ["txt2img", "img2img", "inpaint", "extras"]:
                btn = gr.Button(f"Send to {tab}", elem_id=f"iib_hidden_tab_{tab}")
                # 独立运行时后不起作用，logger.info一个未实现错误
                btn.click(fn=not_implemented_error)
            
            img.change(on_img_change)
            img_update_trigger.click(img_update_func, outputs=[img, img_file_info])
        
        return demo

    js_str = ""
    js_dir = os.path.join(EXTENSIONS_DIR, extension_name, "javascript")  # js应该所在的文件夹
    if check_if_path_exists(js_dir, name=extension_name, path_name="javascript"):
        # 该文件夹内所有js文件的绝对路径的html js引用
        js_str += dir_path2html(
            dir = js_dir,
            ext = ".js",
            html_func = javascript_html
        )

    css_str = ""
    css_path = os.path.join(EXTENSIONS_DIR, extension_name, "style.css")  # css文件应该在的位置
    if check_if_path_exists(css_path, name=extension_name, path_name="css"):  
        css_str += css_html(css_path)

    def on_ui_tabs() -> UiTabsCallbackReturnAlias:
        """注意， 此函数要求能在 sys.path 已经被还原的情况下正常调用"""
        return [(create_demo(), title, elem_id)]

    # 一定要注意原作者是否修改这个接口！！！
    def on_app_start(_: gr.Blocks, app: FastAPI) -> None:
        app_utils = AppUtils(
            sd_webui_config = shared.cmd_opts.sd_webui_config,
            update_image_index = shared.cmd_opts.update_image_index,
            extra_paths = shared.cmd_opts.extra_paths,
        )
        app_utils.wrap_app(app)

    return on_ui_tabs, on_app_start, js_str, css_str


def ui_dataset_tag_editor_standalone(extension_name: str) -> UiCallbackReturnAlias:
    """ extension_name: 扩展名字，即extensions文件夹中的文件夹名字 """

    print("构建tag editor")

    """
    注意，在执行此函数时候，extensions_name文件夹会被添加sys.path
    所以可以以子扩展为包进行import
    请尽量以子扩展为顶级包进行import，而不是以webui为顶级包(import extension.extension_name.)进行import
    否则这里导入的包，会和子扩展内部导入的包指向不同内存地址
    """
    # 注意！Dataset Tag Editor的入口是以子模块的scripts为顶级包
    # 同时，scripts中的脚本以相对引用的方式import，所以这里要添加它们所在的路径
    sys_path_copy = sys.path.copy()  # 备份
    sys.path = [os.path.join(EXTENSIONS_DIR, extension_name, "scripts")] + sys.path

    from interface import (
        tab_main,
        tab_settings,
        versions_html,
        state,
        settings,
        paths,
        utilities,
        cleanup_tmpdr,
    )

    # 尽量保持和原作者一致
    title = "Dataset Tag Editor"  # 插件版和独立版都是这个title
    elem_id = "dataset_tag_editor_interface"  # 这个是插件版的elem_id，独立版的未指定

    ########## 启动前准备 ##########

    state.begin()  # 设置成初始值，用于等候中断重启
    settings.load()  # 载入设置
    paths.initialize()  # 创建路径

    # 设置临时文件夹
    state.temp_dir = (utilities.base_dir_path() / "temp").absolute()
    if settings.current.use_temp_files and settings.current.temp_directory != "":
        state.temp_dir = Path(settings.current.temp_directory)

    if settings.current.cleanup_tmpdir:
        cleanup_tmpdr()  # 清理上一次的临时文件

    ####################

    def create_demo():
        """ ！！！注意，所有的elem_id都不要改，保持和原作者一致！！！ """
        with gr.Blocks(title=title) as demo:
            with gr.Tab("Main"):
                tab_main.on_ui_tabs()
            with gr.Tab("Settings"):
                tab_settings.on_ui_tabs()

            gr.Textbox(elem_id="ui_created", value="", visible=False)

            # 其实 versions_html() 中有一定的异常处理，但是这里再加一层
            try:
                footer = f'<div class="versions">{versions_html()}</div>'
            except Exception as e:
                footer = f'<div class="versions">Error: {e}</div>'
            
            gr.HTML(footer, elem_id="footer")
        return demo
    
    js_str = ""
    js_dir = os.path.join(EXTENSIONS_DIR, extension_name, "javascript")  # js应该所在的文件夹
    if check_if_path_exists(js_dir, name=extension_name, path_name="javascript"):
        # 该文件夹内所有js文件的绝对路径的html js引用
        js_str += dir_path2html(
            dir = js_dir,
            ext = ".js",
            html_func = javascript_html
        )

    css_str = ""
    css_path = os.path.join(EXTENSIONS_DIR, extension_name, "css", "style.css")  # css应该所在的路径
    if check_if_path_exists(css_path, name=extension_name, path_name="css"):
        css_str += css_html(css_path)

    def on_ui_tabs() -> UiTabsCallbackReturnAlias:
        """注意， 此函数要求能在 sys.path 已经被还原的情况下正常调用"""
        return [(create_demo(), title, elem_id)]
    
    # 还原sys.path
    sys.path = sys_path_copy.copy()

    return on_ui_tabs, None, js_str, css_str


def ui_Gelbooru_API_Downloader(extension_name: str) -> UiCallbackReturnAlias:

    def not_implemented_error():
        extension_name = "Gelbooru_API_Downloader"
        info = f"NotImplemented: {extension_name}的WebUI界面尚未实现"
        download_ps1_path = os.path.join(EXTENSIONS_DIR, extension_name, "run_download_images_coroutine.ps1")
        if os.path.exists(download_ps1_path):
            info = info + f"，如果要使用下载功能，请使用: {download_ps1_path}"
        print(info)
    not_implemented_error()

    return None, None, "", ""

import os
import sys
from typing import Callable, List, Tuple

import gradio as gr
from gradio import routes

from modules import modules_tools
from modules import shared
from extensions.extensions import registered_extensions, disable_extensions, check_extensions
from extensions.extensions_tools import javascript_html


GradioTemplateResponseOriginal = routes.templates.TemplateResponse  # 备份一下，因为一会要修改


def reload_javascript(GradioTemplateResponseOriginal: Callable, js: str, css: str):

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res
    
    return template_response

def create_ui() -> Tuple[gr.Blocks, List[Callable]]:

    callbacks_ui_tabs = []
    callbacks_app_started = []
    js_str = ""
    css_str = ""

    # global registered_extensions
    # 拷贝下，避免修改了原字典
    create_ui_extensions_dict_copy = registered_extensions.copy()
    # 禁用扩展，请在  check_extensions() 之前调用，因为后者会删除字典中的键
    create_ui_extensions_dict_copy = disable_extensions( create_ui_extensions_dict_copy, vars(shared.cmd_opts) )
    # 检查扩展是否存在于文件夹中
    create_ui_extensions_dict_copy =   check_extensions(create_ui_extensions_dict_copy)
    if not create_ui_extensions_dict_copy:
        raise Exception("没有扩展被载入")

    for extension_name, extension_ui in create_ui_extensions_dict_copy.items():

        on_ui_tabs, on_app_started, extension_js_str, extension_css_str = extension_ui(extension_name)
        if on_ui_tabs:
            callbacks_ui_tabs.append(on_ui_tabs)
        if on_app_started:
            callbacks_app_started.append(on_app_started)
        if extension_js_str:
            js_str += extension_js_str
        if extension_css_str:
            css_str += extension_css_str

    parent_dir = os.path.dirname( modules_tools.modules_dir )  # 上级目录
    script_js = os.path.join(parent_dir, "script.js")  # webui的js
    js_str = javascript_html(script_js) + js_str  #  webui的js要放在前面

    # 先修改gradio，再创建demo，像SD-WebUI那样
    routes.templates.TemplateResponse  = reload_javascript(
        routes.templates.TemplateResponse,
        js = js_str,
        css = css_str
    )
    
    # 调用创建各自的WebUI组件，on_ui_tabs()返回Tuple[gr.Blocks, str, str]
    # 这个过程必须再with gr.Blocks() as demo外面完成，不然会有再创建的时候就会直接绑上demo
    interfaces_tuple_list = [ on_ui_tabs() for on_ui_tabs in callbacks_ui_tabs ]

    with gr.Blocks() as demo:
        print("开始渲染UI")
        for interface, label, ifid in interfaces_tuple_list:
            with gr.Tab(label, id=ifid, elem_id=f"tab_{ifid}"):
                interface.render()
    
    return demo, callbacks_app_started

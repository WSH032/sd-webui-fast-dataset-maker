import os
import sys
from typing import Callable, List, Tuple
from contextlib import AbstractContextManager, nullcontext

import gradio as gr
from gradio import routes

from modules import modules_tools
from modules import shared
from extensions.extensions_ui import (
    UiTabsCallbackAlias,
    UiTabsCallbackReturnAlias,
    AppStartedCallbackAlias,
)
from extensions.extensions import (
    registered_extensions,
    disable_extensions,
    check_extensions,
)
from extensions.extensions_tools import (
    EXTENSIONS_DIR,
    javascript_html,
)


#################### 全局变量 ####################

recover_sys_path = True  # 是否恢复sys.path

GradioTemplateResponseOriginal = routes.templates.TemplateResponse  # 备份一下，因为一会要修改


#################### 工具 ####################

class TempSysPath(AbstractContextManager):
    """用于临时添加路径到sys.path"""
    def __init__(self, path: str, totally_recover: bool=True):
        """临时添加路径到sys.path，非线程安全

        Args:
            path (str): 需要添加的路径
            totally_recover (bool, optional):
                如果为真，退出时会将 sys.path 恢复为进入时的状态.
                如果为假，退出时只会尝试删去先前添加的 path, 这意味着在语句块内所做的其他更改不会被还原.
                Defaults to True.
        """        
        self._path = path
        self.totally_recover = totally_recover

    def __enter__(self):
        """备份 sys.path ，然后把 path 加入 sys.path[0]"""
        self._sys_path = sys.path.copy()  # 备份 # 请一定要使用copy
        sys.path.insert(0, self._path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """尝试从 sys.path 中删去先前添加的 path"""
        if self.totally_recover:
            sys.path = self._sys_path.copy()  # 完全恢复
        else:
            if self._path in sys.path:
                sys.path.remove(self._path)  # 只是尝试删除


def reload_javascript(GradioTemplateResponseOriginal: Callable, js: str, css: str):

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res
    
    return template_response


#################### UI ####################

def create_ui() -> Tuple[gr.Blocks, List[AppStartedCallbackAlias]]:

    callbacks_ui_tabs: List[UiTabsCallbackAlias] = []
    callbacks_app_started: List[AppStartedCallbackAlias] = []
    js_str = ""
    css_str = ""

    # 拷贝下，避免修改了原字典
    create_ui_extensions_dict_copy = registered_extensions.copy()
    # 禁用扩展，请在  check_extensions() 之前调用，因为后者会删除字典中的键
    create_ui_extensions_dict_copy = disable_extensions( create_ui_extensions_dict_copy, vars(shared.cmd_opts) )
    # 检查扩展是否存在于文件夹中
    create_ui_extensions_dict_copy = check_extensions(create_ui_extensions_dict_copy)
    if not create_ui_extensions_dict_copy:
        raise RuntimeError("没有扩展能被载入")

    # 开始载入扩展
    for extension_name, extension_ui in create_ui_extensions_dict_copy.items():

        if recover_sys_path:
            # 如果要求回复sys.path，就使用TempSysPath来临时添加路径到sys.path
            load_extension_context = TempSysPath( os.path.join(EXTENSIONS_DIR, extension_name) )
        else:
            # 否则就使用nullcontext，什么都不做
            load_extension_context = nullcontext()

        with load_extension_context:
            on_ui_tabs, on_app_started, extension_js_str, extension_css_str = extension_ui(extension_name)
            if on_ui_tabs:
                callbacks_ui_tabs.append(on_ui_tabs)
            if on_app_started:
                callbacks_app_started.append(on_app_started)
            if extension_js_str:
                js_str += extension_js_str
            if extension_css_str:
                css_str += extension_css_str

    parent_dir = os.path.dirname( modules_tools.MODULES_DIR )  # WebUI的根目录CWD
    script_js = os.path.join(parent_dir, "script.js")  # webui的js
    js_str = javascript_html(script_js) + js_str  #  webui的js要放在前面，javascript_html返回值的末尾自带换行符

    # 先修改gradio，再创建demo，像SD-WebUI那样
    routes.templates.TemplateResponse  = reload_javascript(
        routes.templates.TemplateResponse,
        js = js_str,
        css = css_str
    )
    
    # 调用创建各自的WebUI组件，on_ui_tabs()返回List[Tuple[gr.Blocks, str, str]]
    # 这个过程必须再with gr.Blocks() as demo外面完成，不然会有再创建的时候就会直接绑上demo
    interfaces_tuple_list: UiTabsCallbackReturnAlias = []
    for on_ui_tabs in callbacks_ui_tabs:
        # on_ui_tabs 在设计上已经被要求为可以在sys.path被还原时正常工作，所以不用担心
        interfaces_tuple_list += on_ui_tabs() or []

    with gr.Blocks() as demo:
        print("开始渲染UI")
        for interface, label, ifid in interfaces_tuple_list:
            # 请保证id, elem_id, 和A1111-WebUI的生成一样
            with gr.Tab(label, id=ifid, elem_id=f"tab_{ifid}"):
                interface.render()

    return demo, callbacks_app_started

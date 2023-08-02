import os
import sys
from typing import Tuple, List, Callable

import gradio as gr
from fastapi import FastAPI
from gradio.utils import  TupleNoPrint

sys_path = sys.path.copy()  # 备份


class WebuiUtils(object):

    def __init__(
            self,
            cmd: bool=False,
            help: bool = False,
            **kwargs
        ) -> None:
        """初始化modules.shared.cmd_opts，用于设置各插件参数

        Args:
            cmd (bool, optional): cmd = True时: 先从命令行获取参数，否则先获取默认参数. Defaults to False.
            help (bool, optional): help = True时: 传入--help参数，用于获取帮助信息，但是会中断程序. Defaults to False.

            **kwargs:
                传入的**kwargs请参阅extensions.extensions_preload，将会覆盖modules.shared.cmd_opts中相应参数，将会用于配置插件
                虽然也可以修改modules.cmd_args的参数，但是不建议这么做，因为其是关于demo的启动参数，这些启动参数只会在__main__中被使用
               
        Example:
            from webui import WebuiUtils
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
        
        Tips:
            在后续的create_ui()中，会将各扩展的文件夹加入sys.path，以便各扩展能正常工作
            一但create_ui()创建完成，就会恢复sys.path
            如果不需要恢复sys.path，可以在实例化后， modules.ui.recover_sys_path = False
        """

        # 必须在 modules.ui 之前，将会修改cmd_opts
        import modules.shared

        self.demo = None
        self.callbacks_app_started = None
        self.shared = modules.shared

        if help:
            self.shared.set_opts(["--help"])

        if cmd:
            # 直接从命令行获取参数
            self.shared.set_opts(None, quiet=False)
        else:
            # 获取以注册的默认参数
            self.shared.set_opts([], quiet=True)
        
        assert self.shared.cmd_opts is not None, "在完成set_opts([])后，cmd_opts应该不为None"
        # 根据传入的参数来修改cmd_opts
        for name in kwargs:
            setattr(self.shared.cmd_opts, name, kwargs[name])

    @staticmethod
    def create_ui() -> Tuple[gr.Blocks, List[Callable]]:
        """
        返回Tuple[gr.Blocks, List[Callable]]

        其中第一个元素为当前渲染好的demo: gr.Blocks
        第二个元素为callbacks_app_started: List[Callable]
            需要在app, _, _ = demo.launch()之后
            通过for on_app_started in callbacks_app_started: on_app_started(demo, app)挂载fastapi
        """

        # 在modules.shared.set_opts()设置完参数之后，再import modules.ui，否则相应参数无法获取
        from modules import ui
        demo, callbacks_app_started =  ui.create_ui()
        return demo, callbacks_app_started
    
    def check_self_demo(self):
        """如果没创建过ui，就创建一个"""
        if self.demo is None or self.callbacks_app_started is None:
            self.demo, self.callbacks_app_started = self.create_ui()
        return self

    def queue(self, **kwargs):
        """为self.demo启动queue

        Args:
            **kwargs: 所传递的参数都会被传给 self.demo.queue: gr.Blocks.queue 

        Example:
            webui_utils.queue(concurrency_count=2)
        
        Returns:
            WebuiUtils: 自身实例
        """

        self.check_self_demo()

        self.demo = self.demo.queue(**kwargs) # type: ignore
        return self
    
    def launch(self, **kwargs) -> tuple[FastAPI, str, str]:
        """启动self.demo

        Args:
            **kwargs: 所传递的参数都会被传给 self.demo.launch: gr.Blocks.launch

        Example:
            webui_utils.launch(debug=True, server_port=7860)
        
        Returns:
            tuple[FastAPI, str, str]: 与 gr.Blocks.launch 返回值一致
        """

        self.check_self_demo()
        
        # 记录原先的参数
        kwargs_prevent_thread_lock = kwargs.get("prevent_thread_lock", None)
        kwargs_debug = kwargs.get("debug", None)

        # 先不要阻塞主线程，使得on_app_started可以运行
        kwargs["prevent_thread_lock"] = True
        kwargs["debug"] = False

        app, local_url, share_url = self.demo.launch(**kwargs) # type: ignore

        for on_app_started in self.callbacks_app_started: # type: ignore
            on_app_started(self.demo, app)
        
        # 目前dataset_tag_editor的重启不起作用
        # 因为其是依靠设置state的值，并通过循环来判断是否需要新的webui创建
        # 要实现的话需要在这里import dataset_tag_editor，这样会导致耦合严重

        def block_thread():
            
            # Block main thread if running in a script to stop script from exiting
            is_in_interactive_mode = bool(getattr(sys, "ps1", sys.flags.interactive))

            # 判断是否要阻塞主线程
            if kwargs_debug or int(os.getenv("GRADIO_DEBUG", 0)) == 1:
                need_block_thread = True
            elif not kwargs_prevent_thread_lock and not is_in_interactive_mode:
                need_block_thread = True
            else:
                need_block_thread = False

            # 阻塞主线程
            if need_block_thread:
                self.demo.block_thread() # type: ignore
                # 阻塞完成后关闭demo
                self.demo.close() # type: ignore

        block_thread()
        
        # 避免在jupyter中返回时，出现多余的输出，就像gradio官方做的那样
        return TupleNoPrint( (app, local_url, share_url) )

    def close(self):
        """用于在非阻塞模式下关闭demo"""
        if self.demo is not None:
            self.demo.close()



if __name__ == "__main__":
    """
    以 .py 脚本形式运行查重WebUI时候，必须要通过 __main__ 来启动
    因为查重WebUI会启用多进程，如果不用 __main__ 来启动，会无限重启WebUI
    但是以jupyter notebook形式运行，就不用；不过建议此时用api的方式
    """

    webui_utils = WebuiUtils(cmd=True)

    cmd_opts = webui_utils.shared.cmd_opts
    cmd_opts_dict = vars(cmd_opts)

    # 请使用[]而不是get，以检查是否cmd_args发生变化
    # TODO: 修改这里和modules.cmd_args成软编码
    def demo_queue():
        concurrency_count = cmd_opts_dict["concurrency_count"]
        if cmd_opts_dict["enable_queue"] or isinstance(concurrency_count, int):
            kwargs = {"concurrency_count": concurrency_count} if isinstance(concurrency_count, int) else {}
            webui_utils.queue(**kwargs)

    def demo_launch():
        kwargs = {}
        if cmd_opts_dict["inbrowser"]:
            kwargs["inbrowser"] = True

        if cmd_opts_dict["share"]:
            kwargs["share"] = True

        if cmd_opts_dict["debug"]:
            kwargs["debug"] = True

        if cmd_opts_dict["prevent_thread_lock"]:
            kwargs["prevent_thread_lock"] = True

        if isinstance( cmd_opts_dict["server_port"], int ):
            kwargs["server_port"] = cmd_opts_dict["server_port"]
            
        if isinstance( cmd_opts_dict["server_name"], str ):
            kwargs["server_name"] = cmd_opts_dict["server_name"]
        
        webui_utils.launch(**kwargs)

    demo_queue()
    demo_launch()

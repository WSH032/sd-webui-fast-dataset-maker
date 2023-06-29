import argparse

parser = argparse.ArgumentParser()

"""
在这里只注册的是demo的启动配置
在extensions/extensions_preload.py中注册扩展的ui配置
"""

# gradio内置的参数，命名请和官方保持一致，因为webui.WebuiUtils需要依靠这些名字来修改参数

# 关于launch()
parser.add_argument("--inbrowser", default=False, action="store_true", help="Auto launch gradio in browser")
parser.add_argument("--share", default=None, action="store_true", help="Launch gradio with share=True option and make accessible from internet")
parser.add_argument("--debug", default=False, action="store_true", help="if True, blocks the main thread from running. If running in Google Colab, this is needed to print the errors in the cell output.")
parser.add_argument("--prevent_thread_lock", default=False, action="store_true", help="If True, the interface will block the main thread while the server is running.")
parser.add_argument("--server_port", type=int, default=None, help="will start gradio app on this port (if available). If None, will search for an available port starting at 7860.")
parser.add_argument("--server_name", type=str, default=None, help="to make app accessible on local network, set this to \"0.0.0.0\". If None, will use \"127.0.0.1\".")

# 关于queue()
parser.add_argument("--enable_queue", default=None, action="store_true", help="if True, inference requests will be served through a queue instead of with parallel threads. Required for longer inference times (> 1min) to prevent timeout. The default option is False.")
parser.add_argument("--concurrency_count", type=int, default=None, help="Number of worker threads that will be processing requests from the queue concurrently. Increasing this number will increase the rate at which requests are processed, but will also increase the memory usage of the queue.")

"""
inline: bool | None = None,
inbrowser: bool = False,
share: bool | None = None,
debug: bool = False,
enable_queue: bool | None = None,
max_threads: int = 40,
auth: Callable | tuple[str, str] | list[tuple[str, str]] | None = None,
auth_message: str | None = None,
prevent_thread_lock: bool = False,
show_error: bool = False,
server_name: str | None = None,
server_port: int | None = None,
show_tips: bool = False,
height: int = 500,
width: int | str = "100%",
encrypt: bool | None = None,
favicon_path: str | None = None,
ssl_keyfile: str | None = None,
ssl_certfile: str | None = None,
ssl_keyfile_password: str | None = None,
ssl_verify: bool = True,
quiet: bool = False,
show_api: bool = True,
file_directories: list[str] | None = None,
allowed_paths: list[str] | None = None,
blocked_paths: list[str] | None = None,
root_path: str = "",
_frontend: bool = True,
app_kwargs: dict[str, Any] | None = None,
"""

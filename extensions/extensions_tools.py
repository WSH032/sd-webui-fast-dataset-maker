import os
from typing import Callable


EXTENSIONS_DIR = os.path.dirname( os.path.abspath(__file__) )  # extensions文件夹路径


def webpath(path: str) -> str:
    """将path转为webpath，会带上修改时间戳"""
    html_path = path.replace('\\', '/')
    return f'file={html_path}?{os.path.getmtime(path)}'


def javascript_html(js_path: str) -> str:
    """将js文件转为html字符串，末尾自带换行符"""
    head = ""
    head += f'<script type="text/javascript" src="{webpath(js_path)}"></script>\n'
    return head


def css_html(css_path: str) -> str:
    """将css文件转为html字符串，末尾不带换行符"""
    head = ""
    head += f'<link rel="stylesheet" property="stylesheet" href="{webpath(css_path)}">'
    return head


def dir_path2html(dir: str, ext: str, html_func: Callable[[str], str]) -> str:
    """
    将文件夹内的所有文件转为html字符串

    dir: str, 文件夹路径
    ext: str, 文件扩展名，如".js"
    html_func: Callable, 接受一个文件路径，返回一个html字符串
        拼接时候，如果需要换行，请在html_func内部加上换行符
    """

    # 该文件夹内所有js文件
    js_files_list = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if f.endswith(ext) and os.path.isfile(os.path.join(dir, f))
    ]
    # 转为绝对路径
    js_files_list = [ os.path.abspath(f) for f in js_files_list ]
    # 生成html
    js_html_list = [ html_func(js_file) for js_file in js_files_list ]
    # 拼接，注意每个元素内的js字符串末尾自带了换行符，所以不需要在这里加
    js_str = "".join(js_html_list)

    return js_str

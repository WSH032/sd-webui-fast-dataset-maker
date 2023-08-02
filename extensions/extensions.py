import os
import logging
from typing import Dict
from collections import OrderedDict

from extensions.extensions_tools import EXTENSIONS_DIR
from extensions.extensions_ui import (
    ui_dataset_tag_editor_standalone,
    ui_image_deduplicate_cluster_webui,
    ui_sd_webui_infinite_image_browsing,
    ui_Gelbooru_API_Downloader,
    UiCallbackAlias,
)


# 注册的扩展名字列表，键名请保证与文件夹同名
# 调整该顺序可以调整扩展tab在WebUI中的顺序
registered_extensions: OrderedDict[str, UiCallbackAlias] = OrderedDict(
    sd_webui_infinite_image_browsing = ui_sd_webui_infinite_image_browsing,
    image_deduplicate_cluster_webui = ui_image_deduplicate_cluster_webui,
    dataset_tag_editor_standalone = ui_dataset_tag_editor_standalone,
    Gelbooru_API_Downloader = ui_Gelbooru_API_Downloader,
)


def disable_extensions(registered_extensions: Dict[str, UiCallbackAlias], cmd_opts_dict: dict) -> Dict[str, UiCallbackAlias]:
    """禁用扩展"""
    # 请保证所 pop 的键名与 registered_extensions 中的键名相同
    # 用[]而不是get，以保证当 extensions_preload 或者 registered_extensions键名发生更改时候引发错误
    # TODO: 修改这里和modules.cmd_args成软编码
    if cmd_opts_dict["disable_image_browsing"]:
        registered_extensions.pop("sd_webui_infinite_image_browsing")
    if cmd_opts_dict["disable_deduplicate_cluster"]:
        registered_extensions.pop("image_deduplicate_cluster_webui")
    if cmd_opts_dict["disable_tag_editor"]:
        registered_extensions.pop("dataset_tag_editor_standalone")
    if cmd_opts_dict["disable_Gelbooru_Downloader"]:
        registered_extensions.pop("Gelbooru_API_Downloader")
    
    return registered_extensions


def check_extensions(registered_extensions: Dict[str, UiCallbackAlias]) -> Dict[str, UiCallbackAlias]:
    """检查是否扩展都存在"""
    extensions_dir_list = os.listdir(EXTENSIONS_DIR)
    not_exist_extensions_list = []

    # 获取字典的所有键名
    registered_extensions_name_list = list( registered_extensions.keys() )
    print("正在检查扩展是否存在于extensions文件夹中")
    print(f"请勿更改extensions文件夹中的文件名\n{registered_extensions_name_list}")
    
    # 检查字典中相应的键名是否存在于extensions文件夹中，不在就删除字典中相应的键
    # 遍历字典中的键
    for extension in list( registered_extensions.keys() ):
        # 检查键是否在列表中
        if extension not in extensions_dir_list:
            # 删除字典中的键
            registered_extensions.pop(extension)  # del registered_extensions[extension]
            # 记录不存在的扩展名
            not_exist_extensions_list.append(extension)
    if not_exist_extensions_list:
        logging.error(f"以下扩展不存在于extensions文件夹中，将不会被载入：{not_exist_extensions_list}")
    
    return registered_extensions

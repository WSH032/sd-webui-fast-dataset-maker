"""
插件不应依赖于modules.cmd_args中的参数
插件所需的全部参数应该在这里注册
将会在modules.shared中被调用
"""

import argparse


def preload_sd_webui_infinite_image_browsing(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--sd_webui_config", type=str, default=None, help="The path to the config file")
    parser.add_argument("--update_image_index", default=False, action="store_true", help="Update the image index")
    parser.add_argument("--extra_paths", default=[], nargs="+", help="Extra paths to use, will be added to Quick Move.")
    parser.add_argument("--disable_image_browsing", default=False, action="store_true", help="Disable sd_webui_infinite_image_browsing")
    parser.add_argument("--sd_webui_path_relative_to_config", default=False, action="store_true", help="Use the file path of the sd_webui_config file as the base for all relative paths provided within the sd_webui_config file.")
    parser.add_argument("--allow_cors", default=False, action="store_true", help="Allow Cross-Origin Resource Sharing (CORS) for the API.")
    parser.add_argument("--enable_shutdown", default=False, action="store_true", help="Enable the shutdown endpoint.",)
    parser.add_argument("--sd_webui_dir", type=str, default=None, help="The path to the sd_webui folder. When specified, the sd_webui's configuration will be used and the extension must be installed within the sd_webui. Data will be shared between the two.",)


def preload_image_deduplicate_cluster_webui(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--disable_deduplicate_cluster", default=False, action="store_true", help="Disable image_deduplicate_cluster_webui")


def preload_dataset_tag_editor_standalone(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
    "--device-id", type=int, help="CUDA Device ID to use interrogators", default=None
    )
    parser.add_argument("--disable_tag_editor", default=False, action="store_true", help="Disable dataset_tag_editor_standalone")


def preload_Gelbooru_API_Downloader(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--disable_Gelbooru_Downloader", default=False, action="store_true", help="Disable Gelbooru_API_Downloader")


# 注册的扩展名字列表
# 从逻辑功能上来说，键名可以不和文件夹同名；但是为了统一，请保证与文件夹同名
registered_extensions_preload = {
    "dataset_tag_editor_standalone": preload_dataset_tag_editor_standalone,
    "image_deduplicate_cluster_webui": preload_image_deduplicate_cluster_webui,
    "sd_webui_infinite_image_browsing": preload_sd_webui_infinite_image_browsing,
    "Gelbooru_API_Downloader": preload_Gelbooru_API_Downloader,
}

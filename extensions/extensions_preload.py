import argparse
import logging

# 插件不应依赖于modules.cmd_args中的参数
# 插件所需的全部参数应该在这里注册

def preload_sd_webui_infinite_image_browsing(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--sd_webui_config", type=str, default=None, help="The path to the config file")
    parser.add_argument("--update_image_index", action="store_true", help="Update the image index")
    parser.add_argument("--extra_paths", nargs="+", help="Extra paths to use, will be added to Quick Move.", default=[])
    parser.add_argument("--disable_image_browsing", default=False, action="store_true", help="Disable sd_webui_infinite_image_browsing")


def preload_image_deduplicate_cluster_webui(parser: argparse.ArgumentParser) -> None:

    try:
        import extensions.image_deduplicate_cluster_webui.preload
    except Exception as e:
        logging.warning(f"Failed to import extensions.image_deduplicate_cluster_webui.preload: {e}")
    parser.add_argument("--disable_deduplicate_cluster", default=False, action="store_true", help="Disable image_deduplicate_cluster_webui")


def preload_dataset_tag_editor_standalone(parser: argparse.ArgumentParser) -> None:

    parser.add_argument(
    "--device-id", type=int, help="CUDA Device ID to use interrogators", default=None
    )
    parser.add_argument("--disable_tag_editor", default=False, action="store_true", help="Disable dataset_tag_editor_standalone")


def preload_Gelbooru_API_Downloader(parser: argparse.ArgumentParser) -> None:

    parser.add_argument("--disable_Gelbooru_Downloader", default=False, action="store_true", help="Disable Gelbooru_API_Downloader")


from extensions.extensions_preload import (
    preload_dataset_tag_editor_standalone,
    preload_image_deduplicate_cluster_webui,
    preload_sd_webui_infinite_image_browsing,
    preload_Gelbooru_API_Downloader,
)

# 注册的扩展名字列表
registered_extensions_preload = {
    "dataset_tag_editor_standalone": preload_dataset_tag_editor_standalone,
    "image_deduplicate_cluster_webui": preload_image_deduplicate_cluster_webui,
    "sd_webui_infinite_image_browsing": preload_sd_webui_infinite_image_browsing,
    "Gelbooru_API_Downloader": preload_Gelbooru_API_Downloader,
}

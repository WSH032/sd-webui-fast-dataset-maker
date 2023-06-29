import os
import zipfile


def zip(need_zip_path: str, ouput_zip_path: str) -> None:
    """压缩一个文件或者文件夹

    Args:
        need_zip_path (str): 需要压缩的文件或者文件夹路径
        ouput_zip_path (str): 输出的zip文件路径，需要带有扩展名

    Raises:
        ValueError: 当need_zip_path和ouput_zip_path相同时，会报错
    """
    
    # 使用绝对路径以保证root也是绝对路径
    # 方便判断zip_asb_path是否在folder_abs_path中
    need_zip_abs_path = os.path.abspath(need_zip_path)
    ouput_zip_abs_path = os.path.abspath(ouput_zip_path)

    if need_zip_abs_path == ouput_zip_abs_path:
        # 如果需要压缩的路径与输出的zip文件路径，则报错
        raise ValueError("need_zip_path and ouput_zip_path can't be the same")

    ouput_zip_dir = os.path.dirname(ouput_zip_path)
    os.makedirs(ouput_zip_dir, exist_ok=True)

    with zipfile.ZipFile(ouput_zip_abs_path , 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        # 如果是文件直接压缩，然后退出函数
        if os.path.isfile(need_zip_abs_path):
            zip_file.write(need_zip_abs_path, os.path.basename(need_zip_abs_path))
            return

        # 不是文件就遍历文件夹来压缩
        for root, dirs, files in os.walk(need_zip_abs_path):
            for file in files:
                file_path = os.path.join(root, file)
                # 如果指定输出的zip文件在要压缩的文件夹中
                # 可能会出现递归压缩，所以需要判断跳过
                if file_path == ouput_zip_abs_path:
                    continue
                zip_file.write(file_path, os.path.relpath(file_path, need_zip_abs_path))


def unzip(need_unzip_path: str, ouput_unzip_path: str) -> None:
    """解压一个zip文件到指定的文件夹

    Args:
        need_unzip_path (str): 需要解压的zip文件路径
        ouput_unzip_path (str): 输出的文件夹路径

    Raises:
        ValueError: 当need_zip_path和ouput_zip_path相同时，会报错
    """

    # 使用绝对路径以保证root也是绝对路径
    # 方便判断zip_asb_path是否在folder_abs_path中
    need_unzip_abs_path = os.path.abspath(need_unzip_path)
    ouput_unzip_abs_path = os.path.abspath(ouput_unzip_path)

    if need_unzip_abs_path == ouput_unzip_abs_path:
        # 如果需要压缩的路径与输出的zip文件路径，则报错
        raise ValueError("zip_path should not be the same as path")

    os.makedirs(ouput_unzip_abs_path, exist_ok=True)

    with zipfile.ZipFile(need_unzip_abs_path, 'r') as zip_file:
        zip_file.extractall(ouput_unzip_abs_path)

from typing import  List, Optional
import logging

from modules.cmd_args import parser
from extensions.extensions_preload import registered_extensions_preload

def set_opts(args_list: Optional[ List[str] ] = None, quiet: bool = False):
    """
    解析参数，赋值给cmd_opts

    args_list: 传入的参数列表，如果为None，则从sys.argv中获取
    quiet = False时: print打印参数
    """
    global cmd_opts
    
    if args_list is not None:
        cmd_opts, unknown = parser.parse_known_args(args_list)
    else:
        cmd_opts, unknown = parser.parse_known_args()
    if not quiet:
        print("#"*20)

        cmd_opts_str = ""
        for key, value in vars(cmd_opts).items():
            cmd_opts_str += f"--{key}={value} "
        print(cmd_opts_str)

        if unknown:
            logging.warning(f"Unknown parameters: {unknown}")
        
        print("#"*20)

# 需要调用modules.shared.set_opts()来设置cmd_opts
cmd_opts = None

try:
    for extension_name, extension_preload in registered_extensions_preload.items():
        extension_preload(parser)
except Exception as e:
    logging.exception(f"Error when loading extension_preload: {e}")
    pass

# -*- coding: UTF-8 -*-

import os, errno
import pathlib
import time
import socket


def compute_root_dir():
    root_dir = os.path.dirname(  # os.path.dirname(__file__) ：指的是，得到当前文件的绝对路径，是去掉脚本的文件名，只返回目录。
        os.path.dirname
        (os.path.dirname
         (os.path.abspath(__file__))))  # os.path.abspath(__file__) 作用： 获取当前脚本的完整路径
    return root_dir + os.path.sep  # 路径分隔符, '/'


proj_root_dir = pathlib.Path(compute_root_dir())


def file_exists(file_path):
    return os.path.exists(file_path)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5 (except OSError, exc: for Python <2.5)
        if exc.errno == errno.EEXIST and os.path.isdir(path):  # errno.EEXIST 文件已存在
            pass  # pass 不做任何事情，一般用做占位语句
        else:
            raise  # raise 自行引发异常


def make_sure_dir(dir_path):
    if not file_exists(dir_path):
        mkdir_p(dir_path)


make_sure_dir(proj_root_dir / "logs")
default_log_file = proj_root_dir / "logs" / time.strftime("%Y%m%d_%H%M.log", time.localtime())


def print_and_write_log(msg=None, log_file=default_log_file, print_on_console=True, print_in_file=True):
    if print_on_console:
        if msg is not None:
            print(msg)
        else:
            print()

    if print_in_file:
        with open(log_file, 'a') as f:
            if msg is not None:
                f.write(str(msg) + '\n')
            else:
                f.write('\n')


def get_local_ip():
    # 获取计算机名称
    hostname = socket.gethostname()
    # 获取本机IP
    ip = socket.gethostbyname(hostname)
    return ip


def get_host_name():
    return socket.gethostname()


if __name__ == "__main__":
    # print_and_write_log(proj_root_dir)
    pass

#!/usr/bin/env python3
"""
自动生成 git commit 信息的脚本
使用方法：
1. 将此脚本放在项目根目录的 scripts 文件夹下
2. 运行: python scripts/auto_commit.py
"""

import subprocess
import datetime
import os

def get_git_status():
    """获取 git 状态信息"""
    result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
    return result.stdout

def get_changed_files():
    """获取修改的文件列表"""
    status = get_git_status()
    files = []
    for line in status.split('\n'):
        if line:
            # 提取文件名（去掉状态标记）
            file_path = line[3:].strip()
            files.append(file_path)
    return files

def generate_commit_message():
    """生成 commit 信息"""
    files = get_changed_files()
    if not files:
        print("没有需要提交的修改")
        return None
    
    # 获取当前时间
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 分析修改的文件类型
    file_types = {}
    for file in files:
        ext = os.path.splitext(file)[1]
        if ext:
            file_types[ext] = file_types.get(ext, 0) + 1
    
    # 生成文件类型统计信息
    type_info = []
    for ext, count in file_types.items():
        type_info.append(f"{ext}文件: {count}个")
    
    # 生成 commit 信息
    message = f"更新于 {time_str}\n\n"
    message += "修改文件:\n"
    for file in files:
        message += f"- {file}\n"
    message += "\n文件类型统计:\n"
    message += "\n".join(type_info)
    
    return message

def main():
    # 生成 commit 信息
    message = generate_commit_message()
    if not message:
        return
    
    # 添加所有修改
    subprocess.run(['git', 'add', '.'])
    
    # 提交
    subprocess.run(['git', 'commit', '-m', message])
    
    print("提交成功！")
    print("\n提交信息：")
    print(message)

if __name__ == "__main__":
    main() 
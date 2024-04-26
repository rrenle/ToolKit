import os
from openpyxl import Workbook

def write_filenames_to_excel(folder_path, excel_file):
    # 创建一个工作簿
    wb = Workbook()
    # 激活第一个工作表
    ws = wb.active
    # 设置表头
    ws.append(['文件名称'])

    # 遍历文件夹下的所有文件和文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 将文件名写入 Excel 表格
            ws.append([file])

    # 保存 Excel 文件
    wb.save(excel_file)

# 指定文件夹路径和 Excel 文件名
# folder_path = 'D:/Repos/downloads'
# excel_file = 'D:/Repos/downloads/filenames.xlsx'
# 接收用户输入的文件夹路径和保存文件的路径
folder_path = input("请输入要遍历的文件夹路径：")
excel_file = input("请输入要保存的 Excel 文件路径和文件名（例如：output.xlsx）：")

# 调用函数
write_filenames_to_excel(folder_path, excel_file)


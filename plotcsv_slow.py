import csv
import matplotlib.pyplot as plt
import os

csv_file_path = '/home/rbt/shared_space/legged_gym/slow.csv'
output_dir = '/home/rbt/shared_space/record/cmdofl_slow'  # 指定保存图表的目录

# 如果输出目录不存在，则创建它
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开CSV文件
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # 读取标题行
    headers = next(csv_reader)
    
    # 初始化数据存储
    data = {header: [] for header in headers[1:]}  # 跳过第一列（Step）
    
    # 读取数据
    time_values = []  # 存储所有时间值
    for i, row in enumerate(csv_reader):
        for header, value in zip(headers[1:], row[1:]):  # 跳过第一列（Step）
            data[header].append(float(value))
        
        # 计算时间t，确保t的长度与每一列的数据点数量相同
        original_time = float(row[0]) / 1000  # 原始时间值
        scaled_time = original_time * 3/4  # 缩放时间值为原来的1.5倍
        shifted_time = scaled_time - 2.5  # 将时间向左移动5秒
        time_values.append(shifted_time)  # 存储偏移后的时间值
    
    # 绘制每一列的图表
    for header, values in data.items():
        plt.figure(figsize=(10, 6))  # 设置图表大小
        plt.plot(time_values, values, label=header)  # 绘制数据
        plt.xlabel('time (s)')  # 设置x轴标签
        plt.ylabel(header)  # 设置y轴标签
        plt.title(f'Graph of {header} over time')  # 设置图表标题
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格
        
        # 设置横坐标轴的范围，从t=0s开始
        plt.xlim(0, 25)  # 设置x轴范围从0到最大时间值
        
        # 获取数据的最小值和最大值
        min_value = min(values)
        max_value = max(values)
        data_range = max_value - min_value
        
        # 根据数据的分布情况动态调整纵坐标范围
        if min_value < 0 and max_value > 0:
            # 数据既有正数又有负数
            plt.ylim(min_value - 0.4 * data_range, max_value + 0.4 * data_range)
        elif min_value >= 0:
            # 数据全为正数
            plt.ylim(min_value - 0.4 * data_range, max_value + 0.4 * data_range)
        else:
            # 数据全为负数
            plt.ylim(min_value - 0.4 * data_range, max_value + 0.4 * data_range)
        
        plt.savefig(os.path.join(output_dir, f'{header}_over_time.png'))  # 保存图表为PNG文件到指定目录
        plt.close()  # 关闭图表，以便于生成下一个图表

print("图表生成完成。")
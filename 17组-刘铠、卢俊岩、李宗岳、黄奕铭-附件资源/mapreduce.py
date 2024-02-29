# 此代码为Map-Reduce算例
from pyspark import SparkConf, SparkContext
import os
import sys
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot 

# 修改为自己的主机名
conf = SparkConf().setAppName("wordcount").setMaster("spark://master:7077")
sc = SparkContext(conf=conf)


cur_dir = sys.path[0]
# 修改相关路径
file_path0 = '/mapreduce/final_train.txt'
file_path1 = '/mapreduce/final_test.txt'

output_path = '/mapreduce/data15.txt'
path = '/mapreduce/'


# 读取文件并创建RDD
rdd0 = sc.textFile(file_path0)
rdd1 = sc.textFile(file_path1)

# 合并两个RDD
combined_rdd = rdd0.union(rdd1)

# 提取每行的第二个词
second_words = combined_rdd.map(lambda line: line.split()[1]).saveAsTextFile(output_path)

# 收集结果并写入文件
# count = 0
# with open(output_path, 'w') as out_file:
#     for word in second_words.collect():
#         print(count)
#         count = count + 1
#         out_file.write(word + '\n')

print("文件写完成")


char_set = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
         "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" ] + [" "]


def wordcount():
    """
    对所有答案进行
    :param visualize: 是否进行可视化
    :return: 将序排序结果RDD
    """
    # 读取数据，lines为RDD
    lines = sc.textFile(output_path)

    # 将每行转换为整数
    numbers = lines.map(lambda line: int(line.strip()))

    # 映射每个数字到计数值
    numberCounts = numbers.map(lambda number: (number, 1))

    # 聚合相同数字的计数值
    numberCounts = numberCounts.reduceByKey(lambda a, b: a + b)
    char_counts = numberCounts.map(lambda pair: (char_set[pair[0]], pair[1]) if pair[0] < len(char_set) else ("未知", pair[1]))
    for char, count in char_counts.collect():
        print(f"Number {char}: {count} times")
    
    # 创建条形图
    char_counts_collected = char_counts.collect()
    bar = Bar(init_opts=opts.InitOpts(bg_color="white"))  # 设置背景颜色为白色
    bar.add_xaxis([item[0] for item in char_counts_collected])
    bar.add_yaxis("频率", [item[1] for item in char_counts_collected], 
                  itemstyle_opts=opts.ItemStyleOpts(color="lightblue"))  # 设置柱子颜色为浅蓝色

    # 设置全局选项
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title="车牌频率统计", title_textstyle_opts=opts.TextStyleOpts(color="black")),  # 设置标题颜色为黑色
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=0)),  
        yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),  # 显示 y 轴分割线
        legend_opts=opts.LegendOpts(is_show=False)  # 关闭图例
    )

    # 设置条形图的间距
    bar.set_series_opts(
        bar_width="60%",  # 设置条形图的宽度为 60%  
        label_opts=opts.LabelOpts(is_show=False)
    )

    # 保存为 HTML 文件，注意修改保存路径
    savepath = '/home/hadoop/Experiment/map_reduce/map_reduce/'
    bar.render(savepath + "wordcount_bar_chart.html")
    # 使用 snapshot 将 HTML 转换为图片
    make_snapshot(snapshot, bar.render(), savepath + "wordcount_bar_chart.png")
    

if __name__ == '__main__':

    # 进行词频统计并可视化
    resRdd = wordcount()


import jieba
import numpy as np
from PIL import Image
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import WordCloud as PyeWordCloud
from pyecharts.globals import ThemeType
import os
import base64
import shutil

def load_stopwords(stopwords_file):
    """
    加载停用词表
    """
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])
    return stopwords

def process_text(text_file, stopwords):
    """
    处理文本：分词并去除停用词
    """
    # 读取文本
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 使用jieba进行分词
    words = jieba.cut(text)
    
    # 过滤停用词和单字词
    words = [word for word in words if word not in stopwords 
            and len(word) > 1 
            and not word.isspace() 
            and not word.isdecimal()]
    
    # 统计词频
    word_freq = Counter(words)
    
    return word_freq

def get_base64_image(image_path):
    """
    将图片转换为base64编码
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def create_interactive_wordcloud(word_freq, background_image, output_file):
    """
    创建交互式HTML词云图
    """
    # 转换数据格式
    words_list = [(word, freq) for word, freq in word_freq.most_common(100)]
    
    # 创建词云图
    wc = (
        PyeWordCloud(init_opts=opts.InitOpts(
            width="1200px", 
            height="800px",
            theme=ThemeType.MACARONS
        ))
        .add(
            series_name="词频",
            data_pair=words_list,
            word_size_range=[20, 100],
            textstyle_opts=opts.TextStyleOpts(
                font_family="Microsoft YaHei",
                font_weight="bold"
            ),
            tooltip_opts=opts.TooltipOpts(
                formatter="{b}: {c}次"
            ),
            shape="circle"
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="军事新闻词云图",
                subtitle="基于词频统计",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(
                    font_size=30,
                    color="#333"
                )
            ),
            tooltip_opts=opts.TooltipOpts(
                is_show=True,
                formatter="{b}: {c}次",
                textstyle_opts=opts.TextStyleOpts(
                    font_size=14
                )
            )
        )
    )
    
    # 获取背景图相对路径
    bg_relative_path = os.path.relpath(background_image, os.path.dirname(output_file))
    
    # 生成HTML并添加自定义样式
    chart_content = wc.render_embed()
    
    # 使用format方法而不是%操作符
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>军事新闻词云图</title>
        <style>
            body {{
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                background-color: #f0f0f0;
            }}
            .container {{
                position: relative;
                width: 1200px;
                height: 800px;
                margin: 0 auto;
                overflow: hidden;
            }}
            .background {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: url('{0}');
                background-size: 1200px 800px;
                background-position: center;
                background-repeat: no-repeat;
                opacity: 0.3;
            }}
            .chart-container {{
                position: relative;
                z-index: 1;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="background"></div>
            <div class="chart-container">
                {1}
            </div>
        </div>
    </body>
    </html>
    """.format(bg_relative_path, chart_content)
    
    # 保存HTML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    # 打印词频统计信息
    print("\n词频统计 (Top 20):")
    for word, freq in words_list[:20]:
        print(f"{word}: {freq}")

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 使用相对路径
    text_file = os.path.join(current_dir, 'dataset', '军事新闻文本.txt')
    stopwords_file = os.path.join(current_dir, '中文停用词表.txt')
    background_image = os.path.join(current_dir, '词云背景图.jpg')
    output_html = os.path.join(current_dir, '军事新闻词云图.html')
    
    # 确保目录存在
    os.makedirs(os.path.dirname(text_file), exist_ok=True)
    
    # 加载停用词
    print("加载停用词...")
    stopwords = load_stopwords(stopwords_file)
    
    # 处理文本
    print("处理文本...")
    word_freq = process_text(text_file, stopwords)
    
    # 创建交互式HTML词云图
    print("生成交互式HTML词云图...")
    create_interactive_wordcloud(word_freq, background_image, output_html)
    print(f"交互式词云图已保存至: {output_html}")

if __name__ == "__main__":
    main() 
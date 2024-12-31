import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import re
import matplotlib.pyplot as plt
import networkx as nx
import mlxtend
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 按顺序尝试不同字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 验证字体是否可用
def check_chinese_font():
    fonts = [f.name for f in fm.fontManager.ttflist]
    available_chinese_fonts = [f for f in ['SimHei', 'Microsoft YaHei', 'SimSun'] if f in fonts]
    if not available_chinese_fonts:
        print("警告：未找到合适的中文字体，可能会出现乱码")
        print("可用字体列表：", fonts)
    else:
        print(f"使用中文字体: {available_chinese_fonts[0]}")
    return available_chinese_fonts[0] if available_chinese_fonts else 'SimHei'

# 设置 Seaborn 样式
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

def check_mlxtend_version():
    version = mlxtend.__version__
    print(f"当前 mlxtend 版本: {version}")
    required_version = "0.20.0"
    if version < required_version:
        raise ImportError(f"请更新 mlxtend 至版本 {required_version} 或更高。当前版本: {version}")

def clean_book_titles(df):
    # 统计非字符串类型的 '题名' 数量
    non_string_count = df['题名'].apply(lambda x: not isinstance(x, str)).sum()
    if non_string_count > 0:
        print(f"\n警告：'题名' 列中有 {non_string_count} 个非字符串类型的值，将被转换为字符串。")
    
    # 将所有 '题名' 转换为字符串，并清洗空格
    df['题名'] = df['题名'].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.strip()))
    
    # 删除 '题名' 为空的记录
    initial_length = len(df)
    df = df[df['题名'] != 'nan']  # 'nan' 是字符串形式的 NaN
    df = df[df['题名'] != '']
    final_length = len(df)
    if initial_length != final_length:
        print(f"删除了 {initial_length - final_length} 行 '题名' 为空或无效的记录。")
    
    return df

def load_and_process_data():
    # 读取Excel文件，直接使用第3行作为表头
    df = pd.read_excel('F:/Project/python_visualization/book/dataset/读者当前借阅导出20241209101026825.xlsx', 
                      header=2)  # 使用第3行作为表头（索引从0开始，所以是2）
    
    print("原始数据信息：")
    print(df.info())
    print("\n列名：")
    print(df.columns.tolist())
    
    # 清洗书名
    df = clean_book_titles(df)
    
    # 删除空行和重复行
    df = df.dropna(how='all')
    df = df.drop_duplicates()
    
    print("\n处理后的数据预览：")
    print(df.head())
    
    # 创建订单数据集
    # 将相同读者的借书记录合并为一条记录
    orders = df.groupby('读者证号')['题名'].agg(lambda x: list(set(x))).reset_index()  # 使用set去重
    
    # 保存订单数据集
    orders.to_csv('F:/Project/python_visualization/book/dataset/book_orders.csv', 
                 index=False, encoding='utf-8-sig')
    print("\n订单数据已保存至 book_orders.csv")
    
    return df, orders

def filter_books(orders, top_n=500):
    # 计算每本书的借阅频率
    book_counts = orders['题名'].explode().value_counts()
    
    # 打印借阅频率最高的前10本书
    print("\n借阅频率最高的10本书：")
    print(book_counts.head(10))
    
    # 选择借阅频率最高的top_n本书
    top_books = book_counts.nlargest(top_n).index
    
    # 计算被过滤掉的书籍数量
    filtered_books = len(book_counts) - len(top_books)
    print(f"\n过滤掉了 {filtered_books} 本低频借阅书籍")
    
    # 过滤订单，只保留top_n本书
    orders['题名'] = orders['题名'].apply(lambda x: [book for book in x if book in top_books])
    
    # 删除订单中没有借阅多本书的记录
    initial_length = len(orders)
    min_books = 2  # 最少借阅数量
    orders = orders[orders['题名'].map(len) >= min_books]
    final_length = len(orders)
    
    print(f"删除了 {initial_length - final_length} 个借阅少于{min_books}本书的记录")
    print(f"过滤后共有 {len(top_books)} 本书，{final_length} 个读者")
    
    return orders

def create_book_matrix(orders):
    # 获取所有独特的书籍
    all_books = []
    for books in orders['题名']:
        all_books.extend(books)
    unique_books = sorted(list(set(all_books)))  # 排序以确保一致性
    
    print(f"\n共有 {len(unique_books)} 本不同的书籍")
    print(f"共有 {len(orders)} 个读者")
    
    # 创建二元矩阵
    book_matrix = pd.DataFrame(0, index=range(len(orders)), 
                             columns=unique_books)
    
    # 填充矩阵
    for idx, books in enumerate(orders['题名']):
        book_matrix.loc[idx, books] = 1
    
    return book_matrix

def perform_association_analysis(book_matrix):
    # 动态支持度设置
    reader_count = len(book_matrix)
    min_support = max(0.005, 3 / reader_count)  # 基础支持度
    
    # 根据数据集大小动态调整
    if reader_count > 1000:
        min_support = max(0.003, 2 / reader_count)
    elif reader_count < 100:
        min_support = max(0.01, 5 / reader_count)
    
    print(f"\n使用最小支持度: {min_support:.4f}")
    frequent_itemsets = apriori(book_matrix, 
                              min_support=min_support, 
                              use_colnames=True,
                              max_len=2)
    
    print(f"找到 {len(frequent_itemsets)} 个频繁项集")
    
    if len(frequent_itemsets) == 0:
        raise ValueError("无法生成频繁项集")
    
    # 手动生成关联规则
    rules_list = []
    for _, row in frequent_itemsets.iterrows():
        items = list(row['itemsets'])
        if len(items) == 2:  # 只处理二项集
            support = row['support']
            for i in range(2):
                antecedent = [items[i]]
                consequent = [items[1-i]]
                # 计算置信度
                confidence = support / book_matrix[antecedent[0]].mean()
                # 计算提升度
                lift = support / (book_matrix[antecedent[0]].mean() * book_matrix[consequent[0]].mean())
                
                # 计算其他指标
                consequent_support = book_matrix[consequent[0]].mean()
                antecedent_support = book_matrix[antecedent[0]].mean()
                
                # 只添加提升度大于1的规则
                if lift > 1:
                    rules_list.append({
                        'antecedents': antecedent,
                        'consequents': consequent,
                        'support': support,
                        'confidence': confidence,
                        'lift': lift,
                        'antecedent_support': antecedent_support,
                        'consequent_support': consequent_support,
                        'conviction': (1 - consequent_support) / (1 - confidence) if confidence < 1 else float('inf'),
                        'zhangs_metric': (confidence - consequent_support) / (1 - consequent_support) if consequent_support != 1 else 1,
                        'jaccard': support / (antecedent_support + consequent_support - support),
                        'kulczynski': (confidence + support/consequent_support)/2
                    })
    
    # 创建规则DataFrame
    rules = pd.DataFrame(rules_list)
    
    if len(rules) == 0:
        raise ValueError("无法生成有效的关联规则")
    
    # 过滤规则
    min_confidence = 0.15  # 最小置信度
    rules = rules[rules['confidence'] >= min_confidence]
    
    # 按多个指标排序
    rules = rules.sort_values(['lift', 'confidence', 'support'], 
                            ascending=[False, False, False])
    
    # 保存关联规则
    rules.to_csv('F:/Project/python_visualization/book/dataset/association_rules.csv', 
                index=False, encoding='utf-8-sig')
    
    print(f"生成了 {len(rules)} 条关联规则")
    
    return frequent_itemsets, rules

def print_borrow_distribution(orders):
    borrow_counts = orders['题名'].apply(len)
    print("\n每个读者借阅的书籍数量分布：")
    print(borrow_counts.value_counts().sort_index())

def plot_borrow_distribution(orders):
    borrow_counts = orders['题名'].apply(len)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=borrow_counts.value_counts().index, 
                    y=borrow_counts.value_counts().values,
                    color='skyblue')
    
    # 添加数值标签
    for i, v in enumerate(borrow_counts.value_counts().values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    plt.xlabel('借阅的书籍数量', fontsize=12)
    plt.ylabel('读者数量', fontsize=12)
    plt.title('读者借阅数量分布', fontsize=14)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def visualize_rules(rules, top_n=10):
    # 创建多个子图
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 网络图
    plt.subplot(2, 2, 1)
    top_rules = rules.nlargest(top_n, 'lift')
    G = nx.DiGraph()
    
    for _, row in top_rules.iterrows():
        antecedent = '\n'.join(row['antecedents'])
        consequent = '\n'.join(row['consequents'])
        G.add_edge(antecedent, consequent, weight=row['lift'])
    
    pos = nx.spring_layout(G, k=2)
    nx.draw_networkx_nodes(G, pos, node_size=2000, 
                          node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                          arrowsize=20, arrowstyle='->')
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title('关联规则网络图', fontsize=12)
    
    # 2. 指标分布图
    plt.subplot(2, 2, 2)
    sns.boxplot(data=rules[['support', 'confidence', 'lift']])
    plt.title('规则指标分布', fontsize=12)
    plt.xticks(rotation=45)
    
    # 3. 散点图：支持度vs提升度
    plt.subplot(2, 2, 3)
    plt.scatter(rules['support'], rules['lift'], alpha=0.5)
    plt.xlabel('支持度')
    plt.ylabel('提升度')
    plt.title('支持度vs提升度', fontsize=12)
    
    # 4. 散点图：置信度vs提升度
    plt.subplot(2, 2, 4)
    plt.scatter(rules['confidence'], rules['lift'], alpha=0.5)
    plt.xlabel('置信度')
    plt.ylabel('提升度')
    plt.title('置信度vs提升度', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def print_analysis_results(frequent_itemsets, rules):
    print("\n=== 关联规则分析结果 ===")
    print(f"共找到 {len(frequent_itemsets)} 个频繁项集")
    print(f"生成了 {len(rules)} 条关联规则")
    
    print("\n规则质量统计：")
    metrics = ['support', 'confidence', 'lift', 'conviction', 
              'zhangs_metric', 'jaccard', 'kulczynski']
    
    for metric in metrics:
        print(f"{metric}:")
        print(f"  最小值: {rules[metric].min():.4f}")
        print(f"  最大值: {rules[metric].max():.4f}")
        print(f"  平均值: {rules[metric].mean():.4f}")
        print(f"  中位数: {rules[metric].median():.4f}")
    
    # 输出最强关联规则
    print("\n最强关联规则（按多个指标排序）：")
    columns = ['antecedents', 'consequents', 'support', 'confidence', 
              'lift', 'conviction', 'zhangs_metric', 'jaccard']
    print(rules[columns].head().to_string(index=False))

def visualize_top_rules(rules, top_n=6):
    """
    创建一个热力图来展示最强关联规则的多个指标
    """
    # 选择要展示的指标
    metrics = ['support', 'confidence', 'lift', 'conviction', 'zhangs_metric', 'jaccard']
    
    # 获取前 top_n 条规则
    top_rules = rules.nlargest(top_n, 'lift')
    
    # 创建热力图数据
    heatmap_data = top_rules[metrics].copy()
    
    # 处理 inf 值
    heatmap_data = heatmap_data.replace([np.inf, -np.inf], np.nan)
    heatmap_data = heatmap_data.fillna(heatmap_data.max().max())
    
    # 创建规则标签
    rule_labels = [f"{' → '.join(map(str, row['antecedents'] + row['consequents']))}" 
                  for _, row in top_rules.iterrows()]
    
    # 设置图形大小
    plt.figure(figsize=(12, 8))
    
    # 创建热力图
    sns.heatmap(heatmap_data, 
                annot=True,          # 显示数值
                fmt='.2f',          # 数值格式
                cmap='YlOrRd',      # 色彩方案
                xticklabels=metrics,
                yticklabels=rule_labels)
    
    plt.title('最强关联规则的多维度分析', fontsize=14, pad=20)
    plt.xlabel('评估指标', fontsize=12)
    plt.ylabel('关联规则', fontsize=12)
    
    # 调整布局以防止标签被截断
    plt.tight_layout()
    plt.show()

    # 创建条形图比较不同规则的提升度
    plt.figure(figsize=(12, 6))
    colors = sns.color_palette("husl", n_colors=len(top_rules))
    
    bars = plt.bar(range(len(top_rules)), top_rules['lift'], color=colors)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('最强关联规则的提升度比较', fontsize=14)
    plt.xlabel('规则', fontsize=12)
    plt.ylabel('提升度', fontsize=12)
    plt.xticks(range(len(top_rules)), 
               [f"规则 {i+1}" for i in range(len(top_rules))],
               rotation=45)
    
    # 添加规则说明
    rule_text = "\n".join([f"规则 {i+1}: {label}" 
                          for i, label in enumerate(rule_labels)])
    plt.figtext(1.02, 0.5, rule_text, 
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def visualize_rules_table(rules, top_n=5):
    """
    创建一个可视化表格展示最强关联规则
    """
    # 选择前 top_n 条规则和需要展示的列
    top_rules = rules.nlargest(top_n, 'lift')
    columns = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # 准备表格数据
    table_data = []
    for _, row in top_rules.iterrows():
        table_data.append([
            ' → '.join(map(str, row['antecedents'])),
            ' → '.join(map(str, row['consequents'])),
            f"{row['support']:.4f}",
            f"{row['confidence']:.2f}",
            f"{row['lift']:.2f}"
        ])
    
    # 创建表格
    table = ax.table(
        cellText=table_data,
        colLabels=['前项', '后项', '支持度', '置信度', '提升度'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.3, 0.1, 0.1, 0.1]
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # 设置标题
    plt.title('最强关联规则 Top 5', pad=20, fontsize=14)
    
    # 设置单元格颜色
    # 设置表头颜色
    for j in range(len(columns)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white')
    
    # 设置数据行的交替颜色
    for i in range(len(table_data)):
        for j in range(len(columns)):
            if i % 2:
                table[(i+1, j)].set_facecolor('#D9E1F2')
    
    plt.tight_layout()
    plt.show()

def main():
    # 检查中文字体
    chinese_font = check_chinese_font()
    
    # 设置全局字体
    plt.rcParams['font.sans-serif'] = [chinese_font]
    
    # 检查 mlxtend 版本
    try:
        check_mlxtend_version()
    except ImportError as e:
        print(e)
        return
    
    # 1. 加载和处理数据
    df, orders = load_and_process_data()
    
    # 2. 过滤书籍
    orders = filter_books(orders, top_n=500)  # 调整为 500
    
    # 3. 打印借阅数量分布
    print_borrow_distribution(orders)
    
    # 4. 绘制借阅数量分布图
    plot_borrow_distribution(orders)
    
    # 5. 创建书籍借阅矩阵
    book_matrix = create_book_matrix(orders)
    
    # 6. 进行关联分析
    try:
        frequent_itemsets, rules = perform_association_analysis(book_matrix)
    except ValueError as e:
        print(f"关联规则分析失败: {str(e)}")
        return
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return
    
    # 7. 输出分析结果
    print_analysis_results(frequent_itemsets, rules)
    
    # 8. 输出一些有趣的发现
    if len(rules) > 0:
        print("\n有趣的发现：")
        # 找出提升度最高的规则
        print("提升度最高的关联规则：")
        print(rules.nlargest(5, 'lift')[['antecedents', 'consequents', 'confidence', 'lift']])
        
        # 找出置信度最高的关联规则
        print("\n置信度最高的关联规则：")
        print(rules.nlargest(5, 'confidence')[['antecedents', 'consequents', 'confidence', 'lift']])
        
        # 添加表格可视化
        visualize_rules_table(rules)
        
        # 添加新的可视化
        visualize_top_rules(rules)
        
        # 原有的可视化
        visualize_rules(rules, top_n=10)
    else:
        print("没有生成任何关联规则。")

if __name__ == "__main__":
    main()

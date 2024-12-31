import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载和基本处理
def load_and_process_data():
    # 读取Excel文件，设置多级表头
    df = pd.read_excel('F:/Project/python_visualization/Physical/dataset/_2023男学员考核成绩登记表.xlsx', header=[0, 1])
    
    # 打印表头结构，帮助调试
    print("表头结构：")
    print(df.columns)
    
    # 获取成绩列（以.1结尾的列为评定列）
    evaluation_columns = []
    project_names = {
        '单杠1练习': '单杠1练习',
        '双杠1练习': '双杠1练习',
        '仰卧起坐': '仰卧起坐',
        '3000米跑': '3000米',
        '30米×2蛇形跑': '30米×2蛇形跑',
        '100米跑': '100米',
        '单杠2练习': '单杠2练习',
        '双杠2练习': '双杠2练习',
        '徒手组合': '徒手组合'
    }
    
    # 选择评定列（.1结尾的列）
    df_clean = pd.DataFrame()
    for project in project_names.keys():
        if ('秋季学期', f'{project}.1') in df.columns:
            df_clean[project_names[project]] = df[('秋季学期', f'{project}.1')]
        elif ('春季学期', f'{project}.1') in df.columns:
            df_clean[project_names[project]] = df[('春季学期', f'{project}.1')]
    
    # 处理缺失值
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    print("\n处理后的列名：")
    print(df_clean.columns.tolist())
    
    return df_clean, df

# 2. 定义体能类型
def classify_athlete_type(row):
    # 将评定转换为分数
    score_map = {'优秀': 5, '良好': 4, '及格': 3, '不及格': 2}
    
    # 计算各类项目的平均得分
    strength_items = ['单杠1练习', '双杠1练习', '单杠2练习', '双杠2练习']
    endurance_items = ['3000米']
    speed_items = ['30米×2蛇形跑', '100米']
    coordination_items = ['徒手组合']
    core_strength_items = ['仰卧起坐']
    
    def get_avg_score(items):
        scores = [score_map.get(str(row[item]), 0) for item in items if item in row.index]
        return np.mean(scores) if scores else 0
    
    strength_score = get_avg_score(strength_items)
    endurance_score = get_avg_score(endurance_items)
    speed_score = get_avg_score(speed_items)
    coordination_score = get_avg_score(coordination_items)
    core_score = get_avg_score(core_strength_items)
    
    # 计算总体平均分
    all_scores = [strength_score, endurance_score, speed_score, coordination_score, core_score]
    avg_score = np.mean(all_scores)
    
    # 定义体能类型
    if avg_score >= 4.5:
        return '全能型'
    elif strength_score >= 4.0 and endurance_score <= 3.0:
        return '力量型'
    elif endurance_score >= 4.0 and strength_score <= 3.0:
        return '耐力型'
    elif speed_score >= 4.0 and (endurance_score <= 3.0 or strength_score <= 3.0):
        return '速度型'
    elif coordination_score >= 4.0 and core_score >= 4.0:
        return '灵活协调型'
    elif avg_score >= 3.5:
        return '均衡型'
    else:
        return '待提高型'

# 3. 主函数
def main():
    # 处理数据
    df_clean, df_original = load_and_process_data()
    
    # 将评定转换为数值
    score_map = {'优秀': 5, '良好': 4, '及格': 3, '不及格': 2}
    df_numeric = df_clean.copy()
    
    # 转换评定为数值
    for col in df_numeric.columns:
        df_numeric[col] = df_numeric[col].map(lambda x: score_map.get(str(x), 0))
    
    # 添加体能类型标签
    df_clean['体能类型'] = df_clean.apply(classify_athlete_type, axis=1)
    
    # 检查各类型的样本数量
    type_counts = df_clean['体能类型'].value_counts()
    print("\n各体能类型的样本数量：")
    print(type_counts)
    
    # 只保留样本数量大于等于3的类别
    valid_types = type_counts[type_counts >= 3].index
    mask = df_clean['体能类型'].isin(valid_types)
    df_clean_filtered = df_clean[mask]
    df_numeric_filtered = df_numeric[mask]
    
    # 将体能类型添加到原始数据集中
    df_original[('体能分析', '类型')] = df_clean['体能类型']
    
    # 保存带有体能类型的完整数据集
    output_path = 'F:/Project/python_visualization/Physical/dataset/体能分类结果.csv'
    df_original.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n分类结果已保存至：{output_path}")
    
    # 特征标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(df_numeric_filtered)  # 使用过滤后的数据
    y = df_clean_filtered['体能类型']
    
    # 打印数据集信息
    print(f"\n数据集大小: {len(y)} 样本")
    print("各类型样本数量:")
    print(y.value_counts())
    
    # 划分训练集和测试集
    # 如果某些类别样本太少，可能需要调整测试集比例
    test_size = 0.2 if len(y) >= 50 else 0.1
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:
        # 如果样本太少导致stratify失败，则不使用stratify
        print("\n警告：由于样本量较小，无法保持分层抽样，改用简单随机抽样")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    
    # 网格搜索最优参数
    param_grid = {
        'n_neighbors': [1, 3, 5],  # 添加k=1的情况
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    
    # 根据数据集大小调整交叉验证折数
    cv_folds = min(3, len(y_train) // 2)  # 确保每折至少有2个样本
    
    grid_search = GridSearchCV(knn, param_grid, cv=cv_folds, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # 输出最优参数
    print("\n最优参数:", grid_search.best_params_)
    print("最优得分:", grid_search.best_score_)
    
    # 使用最优参数的模型进行预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # 输出分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y),
                yticklabels=np.unique(y))
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.tight_layout()
    plt.show()
    
    # 绘制类型分布图
    plt.figure(figsize=(10, 6))
    type_counts = df_clean['体能类型'].value_counts()
    sns.barplot(x=type_counts.index, y=type_counts.values)
    plt.title('体能类型分布')
    plt.xticks(rotation=45)
    plt.ylabel('人数')
    plt.tight_layout()
    plt.show()
    
    # 进行交叉验证
    try:
        cv_scores = cross_val_score(best_model, X, y, cv=cv_folds)
        print("\n交叉验证得分:", cv_scores)
        print("平均交叉验证得分:", cv_scores.mean())
    except ValueError as e:
        print("\n警告：无法进行交叉验证，可能是由于某些类别的样本太少")
        print(f"错误信息: {str(e)}")

if __name__ == "__main__":
    main() 
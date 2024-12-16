import os
import pandas as pd

def parse_model_results(file_content):
    # 用于存储所有模型数据的列表
    models_data = []
    
    # 分割每个模型的数据块
    model_blocks = file_content.strip().split('\n\n')
    
    for block in model_blocks:
        if not block.strip():
            continue
            
        lines = block.strip().split('\n')
        model_name = lines[0].replace(':', '')
        
        # 提取模型大小并转换为MB
        size_line = next(line for line in lines if '模型大小' in line)
        model_size = float(size_line.split(': ')[1].replace('KB', '')) / 1024  # 转换为MB
        
        # 提取FPS
        fps_line = next(line for line in lines if 'FPS' in line)
        fps = float(fps_line.split(': ')[1])
        
        # 提取AP值
        easy_ap = float(next(line for line in lines if 'Easy' in line).split(': ')[1])
        medium_ap = float(next(line for line in lines if 'Medium' in line).split(': ')[1])
        hard_ap = float(next(line for line in lines if 'Hard' in line).split(': ')[1])
        front_ap = float(next(line for line in lines if 'Front Camera' in line).split(': ')[1])
        one_face_ap = float(next(line for line in lines if '1face' in line).split(': ')[1])
        
        # 创建模型数据字典
        model_data = {
            'name': model_name,
            'size_mb': model_size,  # 改为MB单位
            'fps': fps,
            'easy_ap': easy_ap,
            'medium_ap': medium_ap,
            'hard_ap': hard_ap,
            'front_ap': front_ap,
            'one_face_ap': one_face_ap,
            # 添加模型系列信息
            'series': model_name.split('_')[0]
        }
        
        models_data.append(model_data)
    
    return models_data

# 转换为pandas DataFrame (如果需要)
def create_dataframe(models_data):
    df = pd.DataFrame(models_data)
    return df

def create_charts():
    # 读取文件
    with open('summary.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析数据
    models_data = parse_model_results(content)

    # 转换为DataFrame
    df = create_dataframe(models_data)

    # 现在可以方便地创建各种图表了
    # 例如使用seaborn创建图表：
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 添加更多图表
    # 1. 模型大小与FPS的关系
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df,
                    x='size_mb',
                    y='fps',
                    size='hard_ap',
                    hue='series',
                    sizes=(100, 1000),
                    alpha=0.6)
    # 设置x轴刻度标签为实际值
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.title('模型大小与推理速度关系')
    plt.xlabel('模型大小 (MB)')
    plt.ylabel('每秒帧数 (FPS)')
    plt.legend(title='模型系列')
    plt.savefig('size_fps.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. 不同难度下的AP对比
    ap_data = df[['name', 'easy_ap', 'medium_ap', 'hard_ap']].melt(
        id_vars=['name'], 
        var_name='难度', 
        value_name='AP'
    )
    ap_data['难度'] = ap_data['难度'].map({
        'easy_ap': '简单',
        'medium_ap': '中等',
        'hard_ap': '困难'
    })
    plt.figure(figsize=(12, 8))
    sns.barplot(data=ap_data, x='name', y='AP', hue='难度')
    plt.title('WIDER FACE 不同难度下的检测精度对比')
    plt.xlabel('模型')
    plt.ylabel('平均精度 (AP)')
    # 获取每个模型对应的大小
    model_sizes = df.set_index('name')['size_mb']
    # 设置带模型大小的标签
    plt.xticks(range(len(df)), 
               [f'{name}\n({size:.1f}MB)' for name, size in model_sizes.items()],
               rotation=45)
    plt.tight_layout()
    plt.savefig('ap_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 3. 单人场景性能对比
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='name', y='one_face_ap')
    plt.title('单人场景检测精度')
    plt.xlabel('模型')
    plt.ylabel('平均精度 (AP)')
    # 设置带模型大小的标签
    plt.xticks(range(len(df)),
               [f'{name}\n({size:.1f}MB)' for name, size in model_sizes.items()],
               rotation=45)
    plt.tight_layout()
    plt.savefig('one_face_ap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印latex格式表格
    print("        \\begin{tabular}{l|c|c|c|c|c|c}")
    print("            \\hline")
    print("            Model & Size(MB) & FPS & Easy & Medium & Hard & 1face \\\\")
    print("            \\hline")
    
    # 找出每列的最大值
    max_size = df['size_mb'].min()  # 对于size，我们找最小值
    max_fps = df['fps'].max()
    max_easy = df['easy_ap'].max()
    max_medium = df['medium_ap'].max()
    max_hard = df['hard_ap'].max()
    max_one_face = df['one_face_ap'].max()
    
    # 打印每一行数据
    for _, row in df.iterrows():
        # 对模型名称中的下划线进行转义
        model_name = row['name'].replace('_', '\\_')
        line = f"            {model_name} & "
        
        # 添加size
        if row['size_mb'] == max_size:
            line += f"\\underline{{{row['size_mb']:.2f}}} & "
        else:
            line += f"{row['size_mb']:.2f} & "
            
        # 添加FPS
        if row['fps'] == max_fps:
            line += f"\\underline{{{row['fps']:.2f}}} & "
        else:
            line += f"{row['fps']:.2f} & "
            
        # 添加Easy AP
        if row['easy_ap'] == max_easy:
            line += f"\\underline{{{row['easy_ap']:.2f}}} & "
        else:
            line += f"{row['easy_ap']:.2f} & "
            
        # 添加Medium AP
        if row['medium_ap'] == max_medium:
            line += f"\\underline{{{row['medium_ap']:.2f}}} & "
        else:
            line += f"{row['medium_ap']:.2f} & "
            
        # 添加Hard AP
        if row['hard_ap'] == max_hard:
            line += f"\\underline{{{row['hard_ap']:.2f}}} & "
        else:
            line += f"{row['hard_ap']:.2f} & "
            
        # 添加1face AP
        if row['one_face_ap'] == max_one_face:
            line += f"\\underline{{{row['one_face_ap']:.2f}}} \\\\"
        else:
            line += f"{row['one_face_ap']:.2f} \\\\"
            
        print(line)
    
    print("            \\hline")
    print("        \\end{tabular}")

if __name__ == '__main__':
    create_charts()

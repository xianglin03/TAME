import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import os
# from ucimlrepo import fetch_ucirepo 
import copy # 用于保存最佳模型
import torch.nn.functional as F

# dataname = "shoppers"

# --- 0. 设备配置 (GPU 或 CPU) ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"--- 使用设备: {DEVICE} ---")

# --- 1. 自定义 PyTorch 模型定义 ---
class CustomTorchModel(nn.Module):
    def __init__(self, input_dim, hidden_units=[128, 64], activations=['relu', 'relu'],
                 dropout_rates=[0.2, 0.2], output_activation='sigmoid'):
        """
        自定义 PyTorch Sequential 模型.

        参数:
            input_dim (int): 输入特征的数量.
            hidden_units (list): 包含每个隐藏层神经元数量的列表.
            activations (list): 包含每个隐藏层激活函数名称的列表 (支持 'relu', 'tanh', 'leaky_relu').
            dropout_rates (list): 包含每个隐藏层之后 Dropout 比例的列表. 长度应与 hidden_units 匹配.
                                  可以包含 0 表示不使用 Dropout.
            output_activation (str): 输出层的激活函数 (支持 'sigmoid', 'none'). 'none' 表示输出 logits.
        """
        super(CustomTorchModel, self).__init__()
        if not (len(hidden_units) == len(activations) == len(dropout_rates)):
            raise ValueError("hidden_units, activations, 和 dropout_rates 列表必须有相同的长度.")

        layers = []
        current_dim = input_dim

        for i in range(len(hidden_units)):
            layers.append(nn.Linear(current_dim, hidden_units[i]))

            # 添加激活函数
            if activations[i].lower() == 'relu':
                layers.append(nn.ReLU())
            elif activations[i].lower() == 'tanh':
                layers.append(nn.Tanh())
            elif activations[i].lower() == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            else:
                print(f"警告: 不支持的隐藏层激活函数 '{activations[i]}'. 该层将不使用激活函数.")
            
            # 添加 Dropout
            if dropout_rates[i] > 0:
                layers.append(nn.Dropout(dropout_rates[i]))
            current_dim = hidden_units[i]

        # 输出层 - 假设是二分类问题
        layers.append(nn.Linear(current_dim, 1))
        if output_activation.lower() == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation.lower() != 'none': # 'none' 是有效选项，表示输出 logits
            print(f"警告: 不支持的输出层激活函数 '{output_activation}'. 输出层将不使用激活函数 (输出 logits).")

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# --- 2. 主要的数据处理和模型训练函数 ---
def train_and_save_pytorch_model(X_df, y_df, model_config, training_config, save_dir="saved_pytorch_model"):
    """
    预处理数据, 训练自定义的 PyTorch 模型, 并保存模型和预处理器.

    参数:
        X_df (pd.DataFrame): 特征 DataFrame.
        y_df (pd.DataFrame): 目标变量 DataFrame (单列).
        model_config (dict): 创建 PyTorch 模型的配置 (例如, hidden_units, activations).
        training_config (dict): 训练配置 (例如, epochs, batch_size, learning_rate).
        save_dir (str): 保存模型和预处理器的目录.
    返回:
        tuple: (保存的模型文件路径, 保存的预处理器文件路径或None, 训练历史)
    """
    print("--- 开始数据处理和模型训练 (PyTorch) ---")

    # --- 2a. 基本的目标变量预处理 ---
    if y_df.shape[1] != 1:
        raise ValueError("目标 DataFrame y_df 必须只有一列.")
    y_processed_np = y_df.iloc[:, 0].astype(int).values # 转换为 NumPy 数组
    print(f"\n目标变量 '{y_df.columns[0]}' 已处理. 分布情况:\n{pd.Series(y_processed_np).value_counts(normalize=True)}")

    # --- 2b. 特征预处理 ---
    X_features = X_df.copy() 

    numerical_cols = X_features.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_features.select_dtypes(include=['object', 'category']).columns.tolist()
    boolean_cols = X_features.select_dtypes(include='bool').columns.tolist()

    for col in boolean_cols:
        X_features[col] = X_features[col].astype(int)
    
    numerical_cols = list(set(numerical_cols + boolean_cols))
    categorical_cols = [col for col in categorical_cols if col not in numerical_cols] 

    print(f"\n识别出的数值特征: {numerical_cols}")
    print(f"识别出的类别特征: {categorical_cols}")

    # 此处的 preprocessor_for_training 是实际用于训练的预处理器实例或指示字符串
    preprocessor_for_training_obj = None 
    transformers = []
    if numerical_cols:
        transformers.append(('num', StandardScaler(), numerical_cols))
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

    if not transformers:
        print("警告: 未识别到需要标准预处理的数值或类别特征. 数据将按原样传递.")
        preprocessor_for_training_obj = 'passthrough' 
        # X_processed_np = X_features.to_numpy() # 这行在后面处理
    else:
        preprocessor_for_training_obj = ColumnTransformer(transformers=transformers, remainder='passthrough')

    # --- 2c. 数据集划分 (训练集, 验证集, 测试集) ---
    X_train_raw, X_temp_raw, y_train_np, y_temp_np = train_test_split(
        X_features, y_processed_np, test_size=0.3, random_state=42, stratify=y_processed_np
    )
    X_val_raw, X_test_raw, y_val_np, y_test_np = train_test_split(
        X_temp_raw, y_temp_np, test_size=0.5, random_state=42, stratify=y_temp_np
    )
    print(f"\n数据集划分大小: 训练集={len(y_train_np)}, 验证集={len(y_val_np)}, 测试集={len(y_test_np)}")

    if preprocessor_for_training_obj != 'passthrough':
        X_train_processed_np = preprocessor_for_training_obj.fit_transform(X_train_raw)
        X_val_processed_np = preprocessor_for_training_obj.transform(X_val_raw)
        X_test_processed_np = preprocessor_for_training_obj.transform(X_test_raw)
        try:
            # 尝试获取特征名 (如果 ColumnTransformer 支持且内部转换器也支持)
            # feature_names_out = preprocessor_for_training_obj.get_feature_names_out()
            # print(f"预处理后的特征数量 (get_feature_names_out): {len(feature_names_out)}")
            # 由于 get_feature_names_out 可能因版本或remainder设置而出错，直接用shape更稳妥
            print(f"预处理后的特征数量: {X_train_processed_np.shape[1]}")
        except Exception:
            print(f"预处理后的特征数量 (shape): {X_train_processed_np.shape[1]}")
    else:
        X_train_processed_np = X_train_raw.to_numpy()
        X_val_processed_np = X_val_raw.to_numpy()
        X_test_processed_np = X_test_raw.to_numpy()
        print(f"特征数量 (无预处理): {X_train_processed_np.shape[1]}")
        
    input_dim = X_train_processed_np.shape[1]

    X_train_tensor = torch.tensor(X_train_processed_np, dtype=torch.float32) # .to(DEVICE) 移动到 DataLoader 内部或训练循环
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val_processed_np, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_processed_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

    batch_size = training_config.get('batch_size', 32)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("\n--- 创建 PyTorch 模型 ---")
    current_model_config = model_config.copy() 
    current_model_config['input_dim'] = input_dim 
    
    pytorch_model = CustomTorchModel(**current_model_config).to(DEVICE)
    print(pytorch_model)

    # 损失函数选择: 如果输出层是 sigmoid，用 BCELoss；如果是 logits (output_activation='none')，用 BCEWithLogitsLoss
    output_act = model_config.get("output_activation", "sigmoid").lower()
    if output_act == "sigmoid":
        criterion = nn.BCELoss()
    elif output_act == "none":
        criterion = nn.BCEWithLogitsLoss() # 更稳定，直接处理 logits
    else: # 默认或未知情况，假设有 sigmoid
        print(f"警告: 输出激活函数 '{output_act}' 未明确处理损失函数选择，默认使用 BCELoss (假设输出为概率).")
        criterion = nn.BCELoss()

    optimizer = optim.Adam(pytorch_model.parameters(), lr=training_config.get('learning_rate', 0.001))

    print("\n--- 开始训练 PyTorch 模型 ---")
    epochs = training_config.get('epochs', 50)
    early_stopping_patience = training_config.get('early_stopping_patience', 10)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None 

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        pytorch_model.train() 
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()      
            outputs = pytorch_model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward()            
            optimizer.step()           

            running_train_loss += loss.item() * inputs.size(0)
            # 计算准确率: 如果是BCEWithLogitsLoss, output是logits,需要sigmoid
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            else: # BCELoss, output已经是概率
                predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        pytorch_model.eval() 
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): 
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = pytorch_model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_acc:.4f} - "
              f"验证损失: {epoch_val_loss:.4f}, 验证准确率: {epoch_val_acc:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(pytorch_model.state_dict()) 
            print(f"验证损失改善 ({best_val_loss:.4f}), 保存当前模型状态.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"\n连续 {early_stopping_patience} 个 epochs 验证损失未改善. 触发早停.")
            if best_model_state:
                pytorch_model.load_state_dict(best_model_state) 
            break
    
    if best_model_state and epochs_no_improve < early_stopping_patience: 
        pytorch_model.load_state_dict(best_model_state)
        print("训练完成，已加载验证集上表现最佳的模型状态。")

    print("\n--- 在测试集上评估最终模型 ---")
    pytorch_model.eval()
    test_inputs_final = X_test_tensor.to(DEVICE)
    test_labels_final = y_test_tensor.to(DEVICE)
    with torch.no_grad():
        test_outputs = pytorch_model(test_inputs_final)
        test_loss = criterion(test_outputs, test_labels_final)
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            test_predicted = (torch.sigmoid(test_outputs) > 0.5).float()
        else:
            test_predicted = (test_outputs > 0.5).float()
        test_accuracy = (test_predicted == test_labels_final).sum().item() / test_labels_final.size(0)
    print(f"测试损失: {test_loss.item():.4f}")
    print(f"测试准确率: {test_accuracy:.4f}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, "custom_pytorch_model.pth") 
    preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")

    torch.save(pytorch_model.state_dict(), model_path)
    print(f"\nPyTorch 模型权重已保存到: {model_path}")

    saved_preprocessor_filepath = None
    if preprocessor_for_training_obj != 'passthrough':
        joblib.dump(preprocessor_for_training_obj, preprocessor_path)
        print(f"预处理器已保存到: {preprocessor_path}")
        saved_preprocessor_filepath = preprocessor_path
    else:
        print("训练时预处理器是 'passthrough', 因此没有预处理器对象被保存.")

    print("\n--- 处理完成 ---")
    return model_path, saved_preprocessor_filepath, history, preprocessor_for_training_obj # 返回预处理器对象本身或'passthrough'


def model_output(dataname, X_input_df_main, y_condition, target_shape):
    """
    计算模型输出、损失，并将梯度列数浓缩至 target_shape。
    
    该函数将执行以下操作：
    1. 计算模型对于预处理后输入的完整梯度。
    2. 使用区域插值法 (area interpolation) 将梯度的列平滑地降维到
       指定的 `target_shape` 数量。
    
    返回:
        - 浓缩后的梯度 (torch.Tensor)
        - 模型预测概率 (torch.Tensor)
        - 损失值 (float)
    """
    
    # 1. 加载预处理器
    loaded_preprocessor_main = joblib.load(f"./model_{dataname}/preprocessor.joblib") # TODO 旧模型还是新模型
    # loaded_preprocessor_main = joblib.load(f"./model_{dataname}_new/preprocessor.joblib")

    # 2. 模型配置
    my_pytorch_model_config_main = {
        "hidden_units": [256, 128, 64],
        "activations": ["relu", "relu", "leaky_relu"],
        "dropout_rates": [0.3, 0.2, 0.1],
        "output_activation": "sigmoid"
    }

    # 3. 数据预处理
    transformed_sample = loaded_preprocessor_main.transform(X_input_df_main)
    input_dim = transformed_sample.shape[1]

    # 4. 构建模型
    model_config = my_pytorch_model_config_main.copy()
    model_config["input_dim"] = input_dim

    model = CustomTorchModel(**model_config).to(DEVICE)
    model.load_state_dict(torch.load(
        # f"./model_{dataname}/custom_pytorch_model.pth", #TODO 旧模型还是新模型
        f"./model_{dataname}_new/custom_pytorch_model.pth",
        map_location=DEVICE
    ))
    model.eval()

    # 5. 设置输入为可求导的叶子张量
    input_tensor = torch.tensor(transformed_sample, dtype=torch.float32)
    input_tensor = input_tensor.to(DEVICE)
    input_tensor.requires_grad_()

    # 6. 处理 y_condition
    if isinstance(y_condition, np.ndarray):
        y_condition = torch.tensor(y_condition, dtype=torch.float32)
    y_condition = y_condition.to(DEVICE).float()
    if y_condition.ndim == 1:
        y_condition = y_condition.unsqueeze(1)

    # 7. 前向传播
    predictions = model(input_tensor)
    if my_pytorch_model_config_main.get("output_activation", "sigmoid").lower() != "sigmoid":
        predictions = torch.sigmoid(predictions)

    # 8. 对齐尺寸和计算 Loss
    min_len = min(predictions.shape[0], y_condition.shape[0])
    loss_fn = nn.BCELoss()
    loss = loss_fn(predictions[:min_len], y_condition[:min_len])

    # 9. 反向传播
    model.zero_grad()
    loss.backward()

    # --- 梯度浓缩修改 ---

    # 10. 获取原始的完整梯度
    full_gradient = input_tensor.grad.detach()
    # full_gradient 的形状为 (N, C_in)，例如 [12000, 28]

    # 11. 准备插值
    # F.interpolate 需要一个3D输入 (N, Channels, Length)。
    # 我们需要将梯度看作是 N个样本，每个样本有1个通道，长度为 C_in。
    # 使用 unsqueeze(1) 增加一个通道维度。
    # 形状变为 (N, 1, C_in)，例如 [12000, 1, 28]
    grad_for_interpolate = full_gradient.unsqueeze(1)

    # 12. 执行插值/浓缩
    # 我们将长度从 C_in 浓缩到 target_shape。
    # mode='area' 在降采样时效果很好，可以有效避免信息丢失。
    condensed_grad = F.interpolate(
        grad_for_interpolate, 
        size=target_shape,         # 指定目标长度/列数
        mode='area'                # 使用区域平均模式
    )
    # condensed_grad 的形状为 (N, 1, target_shape)，例如 [12000, 1, 10]

    # 13. 移除之前增加的通道维度
    # 使用 squeeze(1) 恢复为2D张量。
    # 形状变为 (N, target_shape)，例如 [12000, 10]
    final_grad = condensed_grad.squeeze(1)

    # --- 修改结束 ---

    # 14. 返回浓缩后的梯度、预测和loss
    return final_grad, predictions.detach(), loss.item()


# --- 3. 示例用法 ---
# if __name__ == "__main__":
    
#     # --- A. 从 CSV 加载您的数据 ---
#     # 定义您的 CSV 文件路径
#     csv_file_path = f'data/{dataname}/train.csv'
#     # csv_file_path = "data/shoppers/online_shoppers_intention.csv"

#     # 使用 pandas 加载数据集
#     data_df = pd.read_csv(csv_file_path)

#     # 读取data_df然后将最后一列的列名作为target_column
#     target_column = data_df.columns[-1]
        
#     # 分离特征 (X) 和目标 (y)
#     # 将布尔型的 'Revenue' 目标转换为整数 (0 或 1)
#     # y_input_df_main = pd.DataFrame(data_df[target_column].astype(int)) # TODO
#     y_input_df_main = pd.DataFrame(
#         data_df[target_column].apply(lambda x: 0 if x == ' <=50K' else 1).astype(int)
#     ) # 只有 adult
    
    
#     X_input_df_main = data_df.drop(columns=[target_column])

#     print(f"已成功从 {csv_file_path} 加载数据。")
#     print(f"特征 (X) 维度: {X_input_df_main.shape}")
#     print(f"目标 (y) 维度: {y_input_df_main.shape}")

#     # 定义变量以防训练失败或跳过
#     saved_model_file_main = None
#     saved_preprocessor_file_main = None
#     preprocessor_used_in_training = None

#     # 只有在数据成功加载后才继续训练
#     if X_input_df_main is not None and y_input_df_main is not None:
#         # --- B. 定义模型和训练配置 ---
#         my_pytorch_model_config_main = {
#             "hidden_units": [256, 128, 64],          # 隐藏层单元数
#             "activations": ["relu", "relu", "leaky_relu"], # 激活函数
#             "dropout_rates": [0.3, 0.2, 0.1],      # Dropout 比率
#             "output_activation": "sigmoid" # 输出层激活函数，或 "none" 来输出 logits (BCEWithLogitsLoss 会更稳定)         
#         }

#         my_pytorch_training_config_main = {
#             "learning_rate": 0.0005,           # 学习率
#             "epochs": 100,                     # 训练轮数
#             "batch_size": 128,                 # 批次大小
#             "early_stopping_patience": 15      # 早停耐心值
#         }

#         # --- C. 运行训练和保存流程 ---
#         # 注意：train_and_save_pytorch_model 现在返回预处理器对象或字符串 'passthrough'
#         try:
#             saved_model_file_main, saved_preprocessor_file_main, training_history_main, preprocessor_used_in_training = train_and_save_pytorch_model(
#                 X_input_df_main,
#                 y_input_df_main,
#                 model_config=my_pytorch_model_config_main,
#                 training_config=my_pytorch_training_config_main,
#                 save_dir=f"model_{dataname}"  # 保存目录
#             )
#             print("训练和保存流程已完成。")
#         except NameError:
#              print("错误: `train_and_save_pytorch_model` 函数未定义。请确保它已导入或定义。")
#              print("将跳过训练和预测部分。")
#         except Exception as e:
#             print(f"训练过程中发生错误: {e}")
#             print("将跳过训练和预测部分。")

#     else:
#         print("数据未能加载，无法继续训练和预测。")

#     # --- D. 加载已保存模型和预处理器进行预测的示例 ---
#     if saved_model_file_main and os.path.exists(saved_model_file_main):
#         print("\n--- 示例: 加载已保存的模型和预处理器进行预测 ---")
        
#         loaded_preprocessor_main = None
#         # 1. 加载在训练时保存的预处理器 (如果它不是 'passthrough')
#         if saved_preprocessor_file_main and os.path.exists(saved_preprocessor_file_main):
#             try:
#                 loaded_preprocessor_main = joblib.load(saved_preprocessor_file_main)
#                 print(f"预处理器已从 {saved_preprocessor_file_main} 加载。")
#             except Exception as e:
#                 print(f"加载预处理器 {saved_preprocessor_file_main} 失败: {e}")
#         elif preprocessor_used_in_training == 'passthrough': # 检查训练时是否为 passthrough
#             print("训练时预处理器为 'passthrough'，无需加载预处理器文件。")
#         else:
#             print(f"警告: 预处理器文件路径 '{saved_preprocessor_file_main}' 未找到或未提供，并且训练时未使用 'passthrough'。")

#         # 2. 确定模型的输入维度
#         loaded_model_input_dim_main = -1
#         # 使用 X_input_df_main 进行维度推断，确保它与训练时的数据结构一致
#         if 'X_input_df_main' not in locals() or X_input_df_main is None: # 确保 X_input_df_main 存在且已加载
#             print("错误: X_input_df_main 未定义或未加载，无法推断加载模型的输入维度。")
#         elif loaded_preprocessor_main: # 如果预处理器已成功加载
#             try:
#                 # 使用原始输入数据的一个小样本来获取转换后的维度
#                 sample_X_for_dim_inference = X_input_df_main.head()
#                 transformed_sample_for_dim_inference = loaded_preprocessor_main.transform(sample_X_for_dim_inference)
#                 loaded_model_input_dim_main = transformed_sample_for_dim_inference.shape[1]
#                 print(f"从加载的预处理器推断出的模型输入维度: {loaded_model_input_dim_main}")
#             except Exception as e:
#                 print(f"使用加载的预处理器转换样本数据以获取维度时出错: {e}")
#                 loaded_model_input_dim_main = -1 # 标记为失败
#         elif preprocessor_used_in_training == 'passthrough': # 如果训练时是 passthrough
#             loaded_model_input_dim_main = X_input_df_main.shape[1]
#             print(f"训练时预处理器为 'passthrough'，使用原始特征数量作为输入维度: {loaded_model_input_dim_main}")
#             if X_input_df_main.select_dtypes(include=['object', 'category']).shape[1] > 0:
#                 print("警告: 原始数据中仍有对象/类别类型特征，且为 'passthrough'。模型可能无法直接处理这些特征。")
#         else:
#             print("错误: 无法确定模型输入维度：预处理器既未加载也非 'passthrough'。")
#             loaded_model_input_dim_main = -1 # 标记为失败

#         # 只有在成功获取维度后才继续
#         if loaded_model_input_dim_main != -1:
#             try:
#                 # 3. 实例化模型结构
#                 loaded_model_config_dict_main = my_pytorch_model_config_main.copy()
#                 loaded_model_config_dict_main["input_dim"] = loaded_model_input_dim_main
                
#                 loaded_pytorch_model_main = CustomTorchModel(**loaded_model_config_dict_main).to(DEVICE)
                
#                 # 4. 加载模型权重
#                 loaded_pytorch_model_main.load_state_dict(torch.load(saved_model_file_main, map_location=DEVICE))
#                 loaded_pytorch_model_main.eval() # 设置为评估模式
#                 print(f"PyTorch 模型权重已从 {saved_model_file_main} 加载并设置为评估模式。")

#                 # 5. 使用加载的模型进行预测
#                 # sample_X_raw_for_prediction_main = X_input_df_main.head()
#                 sample_X_raw_for_prediction_main = X_input_df_main.sample(n=20)
#                 sample_X_processed_np_for_prediction_main = None

#                 if loaded_preprocessor_main:
#                     sample_X_processed_np_for_prediction_main = loaded_preprocessor_main.transform(sample_X_raw_for_prediction_main)
#                 elif preprocessor_used_in_training == 'passthrough':
#                     # 确保在 passthrough 模式下数据是数值型的
#                     try:
#                        sample_X_processed_np_for_prediction_main = sample_X_raw_for_prediction_main.to_numpy(dtype=np.float32)
#                     except TypeError as te:
#                        print(f"错误: 'passthrough' 模式下无法将数据转为 NumPy 数组 (可能含非数值列): {te}")
#                        sample_X_processed_np_for_prediction_main = None # 标记为失败
#                 else:
#                     print("错误: 无法处理预测样本，因为预处理器状态未知。")

#                 if sample_X_processed_np_for_prediction_main is not None:
#                     sample_X_tensor_for_prediction_main = torch.tensor(sample_X_processed_np_for_prediction_main, dtype=torch.float32).to(DEVICE)
#                     with torch.no_grad(): # 预测时不需要计算梯度
#                         predictions_proba_main = loaded_pytorch_model_main(sample_X_tensor_for_prediction_main)
                        
#                         # 根据模型输出层是否有sigmoid来处理概率
#                         if my_pytorch_model_config_main.get("output_activation", "sigmoid").lower() != "sigmoid":
#                             predictions_proba_main = torch.sigmoid(predictions_proba_main)
                        
#                         predictions_binary_main = (predictions_proba_main > 0.5).float()

#                     print("\n用于预测的原始样本数据:")
#                     print(sample_X_raw_for_prediction_main)
#                     print("\n预测概率 (正类):")
#                     print(predictions_proba_main.cpu().numpy().flatten()) # 这里可以用作为矩阵
#                     print("\n预测二元类别 (0 或 1):")
#                     print(predictions_binary_main.cpu().numpy().flatten())

#             except NameError as ne:
#                 print(f"错误: 实例化或加载模型失败，可能是因为 `CustomTorchModel` 或 `DEVICE` 未定义: {ne}")
#             except RuntimeError as re:
#                 print(f"加载模型 state_dict 时发生 RuntimeError: {re}")
#                 print(f"当前模型实例期望的 input_dim (基于预处理器推断) 是: {loaded_model_input_dim_main}")
#             except Exception as ex:
#                 print(f"加载或预测过程中发生未知错误: {ex}")

#         else:
#             print("未能成功确定模型输入维度。无法加载模型进行预测。")
            
#     else:
#         # 如果训练未运行或失败，saved_model_file_main 可能为 None
#         if X_input_df_main is not None and y_input_df_main is not None: # 只有在数据加载成功但训练失败时才显示此消息
#             if not saved_model_file_main:
#                  print("\n错误：训练未成功运行或 `saved_model_file_main` 变量未设置。")
#             elif not os.path.exists(saved_model_file_main):
#                  print(f"\n错误：模型文件 {saved_model_file_main} 未找到。无法加载模型。")

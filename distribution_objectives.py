import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import traceback
import argparse
from whyshift import get_data
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, f1_score, roc_auc_score, log_loss
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from TALENT.model.utils import(
    get_deep_args, get_method,get_classical_args
)


def load_csv_as_numpy(file_path, label_column="income"):
    """
    读取 CSV 文件，并转换为 numpy 格式，分为特征矩阵 (X) 和标签向量 (y)。
    :param file_path: CSV 文件路径
    :param label_column: 目标标签列的名称
    :return: X (numpy array), y (numpy array), feature_names (list)
    """
    df = pd.read_csv(file_path)  # 读取 CSV
    feature_names = df.columns.tolist()  # 获取所有列名
    feature_names.remove(label_column)  # 去除标签列
    X = df[feature_names].to_numpy()  # 转换特征为 numpy 数组
    y = df[label_column].to_numpy()  # 转换标签为 numpy 数组
    return X, y, feature_names


# 定义不同的随机种子
seeds = [42, 2023, 789]
log_file = "error_log.txt"

# 初始化结果存储字典
metrics = ["accuracy", "auc", "f1", "balanced_accuracy"]
results = {m: {
                "modernNCA": [], 
               "RandomForest": [], 
               "realmlp": [], 
               "tabpfn": [], 
               "catboost": [], 
               "xgboost": [], 
               "mlp": []
               } for m in metrics}
results_ood = {m: {
                "modernNCA": [], 
                "RandomForest": [],
                "realmlp": [], 
                "tabpfn": [], 
                "catboost": [], 
                "xgboost": [], 
                "mlp": []
                } for m in metrics}



# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_model(X_id,y_id,X_ood,y_ood,datasetname, info):
    try:
        # 遍历随机种子
        for seed in seeds:

            size = X_id.shape[0] + X_ood.shape[0]

            if datasetname == "Setting 21" or datasetname == "Setting 22":
                X_id_tr, X_id_test, y_id_tr, y_id_test = train_test_split(X_id, y_id, train_size=0.8, random_state=seed)
            else:
                if size > 50000:
                    X_id, _, y_id, _ = train_test_split(X_id, y_id, train_size=50000/size, random_state=seed)
                    X_ood, _, y_ood, _ = train_test_split(X_ood, y_ood, train_size=50000/size, random_state=seed)
                X_id_tr, X_id_test, y_id_tr, y_id_test = train_test_split(X_id, y_id, train_size=0.8, random_state=seed)
            
            print(f"X_id_tr:{X_id_tr.shape[0]}, X_id_test:{X_id_test.shape[0]}, X_ood:{X_ood.shape[0]}")

            # 组织数据
            N = {'train': X_id_tr}  # 数值型特征
            C = None # 没有类别特征，设为 None（或使用 np.empty 代替）
            y = {'train': y_id_tr}  # 目标变量
            # 分割训练集为训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                N['train'], y['train'], test_size=0.1, random_state=seed
            )

            # 更新 N 和 y
            N = {'train': X_train, 'val': X_val}
            y = {'train': y_train, 'val': y_val}

            train_val_data = (N, C, y)
            num_classes = len(np.unique(y_train))

            args_mNCA, _, _ = get_deep_args(model_type="modernNCA", cat_policy="tabr_ohe", dataset=datasetname)
            args_RForest, _, _ = get_classical_args(model_type="RandomForest", cat_policy="ordinal", dataset=datasetname)
            args_rmlp, _, _ = get_deep_args(model_type="realmlp", cat_policy="indices", dataset=datasetname)
            args_mNCA.seed = seed
            args_RForest.seed = seed
            args_rmlp.seed = seed

            models = {
                "modernNCA":  get_method('modernNCA')(args_mNCA, info["task_type"] == "regression"),
                "RandomForest": get_method('RandomForest')(args_RForest, info["task_type"] == "regression"),
                "realmlp": get_method('realmlp')(args_rmlp, info["task_type"] == "regression"),
                "tabpfn": TabPFNClassifier(ignore_pretraining_limits=True), 
                "catboost": CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, logging_level='Silent'),
                "xgboost" : XGBClassifier(
                        objective='multi:softprob' if num_classes > 2 else 'binary:logistic',
                        num_class=num_classes if num_classes > 2 else None,
                        use_label_encoder=False,
                        eval_metric='mlogloss' if num_classes > 2 else 'logloss'
                    ),
                "mlp": MLPClassifier(random_state=1, max_iter=300)
            }

            for model_name, model in models.items():
                # 训练模型
                if model_name in ["modernNCA", "RandomForest", "realmlp"]:
                    model.fit(data=train_val_data, info=info)
                else:
                    model.fit(X_id_tr,y_id_tr)
                
                # 预测
                if model_name == "modernNCA":
                    _, _, metrixs_id, _ = model.predict(data=({'test': X_id_test}, None, {'test': y_id_test}), info=info, model_name=args_mNCA.evaluate_option)
                    _, _, metrixs_ood, _ = model.predict(data=({'test': X_ood}, None, {'test': y_ood}), info=info, model_name=args_mNCA.evaluate_option)
                elif model_name == "RandomForest":
                    _, metrixs_id, _ = model.predict(data=({'test': X_id_test}, None, {'test': y_id_test}), info=info, model_name=args_RForest.evaluate_option)
                    _, metrixs_ood, _ = model.predict(data=({'test': X_ood}, None, {'test': y_ood}), info=info, model_name=args_RForest.evaluate_option)
                elif model_name == "realmlp":
                    _, _, metrixs_id, _  = model.predict(data=({'test': X_id_test}, None, {'test': y_id_test}), info=info, model_name=args_rmlp.evaluate_option)
                    _, _, metrixs_ood, _ = model.predict(data=({'test': X_ood}, None, {'test': y_ood}), info=info, model_name=args_rmlp.evaluate_option)
                else:
                    def eval_metrics(X, y):
                        proba = model.predict_proba(X)
                        pred = np.argmax(proba, axis=1) if proba.shape[1] > 1 else (proba > 0.5).astype(int)
                        metric_dict = {
                            "Accuracy": accuracy_score(y, pred),
                            "Balanced_Acc": balanced_accuracy_score(y, pred),
                            "F1": f1_score(y, pred, average="macro"),
                            "AUC": roc_auc_score(y, proba[:, 1] if proba.shape[1] > 1 else proba),
                        }
                        return metric_dict

                    metrixs_id = eval_metrics(X_id_test, y_id_test)
                    metrixs_ood = eval_metrics(X_ood, y_ood)


                
                # 计算性能指标
                results["accuracy"][model_name].append(metrixs_id["Accuracy"])
                results["auc"][model_name].append(metrixs_id["AUC"])
                results["f1"][model_name].append(metrixs_id["F1"])
                results["balanced_accuracy"][model_name].append(metrixs_id["Balanced_Acc"])

                results_ood["accuracy"][model_name].append(metrixs_ood["Accuracy"])
                results_ood["auc"][model_name].append(metrixs_ood["AUC"])
                results_ood["f1"][model_name].append(metrixs_ood["F1"])
                results_ood["balanced_accuracy"][model_name].append(metrixs_ood["Balanced_Acc"])

                # 释放资源
                del model
                torch.cuda.empty_cache()

        with open('results.txt', 'a') as f:
            f.write(f"Processing dataset: {datasetname}\n")
            for metric in metrics:
                f.write(f"=== {metric} ===\n")
                for model_name in models.keys():
                    mean_id = np.mean(results[metric][model_name])
                    std_id = np.std(results[metric][model_name])
                    mean_ood = np.mean(results_ood[metric][model_name])
                    std_ood = np.std(results_ood[metric][model_name])
                    f.write(f"{model_name} (ID): {mean_id:.4f} ± {std_id:.4f}\n")
                    f.write(f"{model_name} (OOD): {mean_ood:.4f} ± {std_ood:.4f}\n")
    except Exception as e:
        error_message = f"Dataset: {datasetname}\nError: {str(e)}\n{traceback.format_exc()}\n{'='*80}\n"
        print(f"Error processing dataset {datasetname}:")
        print(error_message)
            





def main():
    parser = argparse.ArgumentParser(description="Run test_model with ACS dataset or settings")
    parser.add_argument('--setting', required=True, help='Setting name: e.g., income_CA_PR, pubcov_2010_2017, setting21')
    parser.add_argument('--model', required=True)
    parser.add_argument('--export_dataset', required=True)
    args = parser.parse_args()
    setting = args.setting.lower()

    info_dict = {
        'income': {"task_type": "binclass", "n_num_features": 76, "n_cat_features": 0},
        'mobility': {"task_type": "binclass", "n_num_features": 63, "n_cat_features": 0},
        'pubcov': {"task_type": "binclass", "n_num_features": 42, "n_cat_features": 0},
    }

    if setting == 'income_ca_pr':
        X_id, y_id, _ = get_data("income", "CA", True, './dataset/acs/', 2018)
        X_ood, y_ood, _ = get_data("income", "PR", True, './dataset/acs/', 2018)
        test_model(X_id, y_id, X_ood, y_ood, "income(CA-PR)", info=info_dict['income'])

    elif setting == 'mobility_ms_hi':
        X_id, y_id, _ = get_data("mobility", "MS", True, './dataset/acs/', 2018)
        X_ood, y_ood, _ = get_data("mobility", "HI", True, './dataset/acs/', 2018)
        test_model(X_id, y_id, X_ood, y_ood, "mobility(MS-HI)", info=info_dict['mobility'])

    elif setting == 'pubcov_ne_la':
        X_id, y_id, _ = get_data("pubcov", "NE", True, './dataset/acs/', 2018)
        X_ood, y_ood, _ = get_data("pubcov", "LA", True, './dataset/acs/', 2018)
        test_model(X_id, y_id, X_ood, y_ood, "pubcov (NE-LA)", info=info_dict['pubcov'])

    elif setting == 'pubcov_2010_2017':
        X_id, y_id, _ = get_data("pubcov", "CA", True, './dataset/acs/', 2010)
        X_ood, y_ood, _ = get_data("pubcov", "CA", True, './dataset/acs/', 2017)
        test_model(X_id, y_id, X_ood, y_ood, "pubcov(2010-2017)", info=info_dict['pubcov'])

    elif setting == 'setting21':
        X_src, y_src, _ = load_csv_as_numpy("acsincome/source_21.csv")
        X_tgt, y_tgt, _ = load_csv_as_numpy("acsincome/target_21.csv")
        test_model(X_src, y_src, X_tgt, y_tgt, "Setting 21", info=info_dict['income'])

    elif setting == 'setting22':
        X_src, y_src, _ = load_csv_as_numpy("/data0/jiazy/TALENT/dataset/acsincome/source_22.csv")
        X_tgt, y_tgt, _ = load_csv_as_numpy("/data0/jiazy/TALENT/dataset/acsincome/target_22.csv")
        test_model(X_src, y_src, X_tgt, y_tgt, "Setting 22", info=info_dict['income'])

    else:
        print(f"Unknown setting: {setting}")
        print("Supported options: income_ca_pr, mobility_ms_hi, pubcov_ne_la, pubcov_2010_2017, setting21, setting22")

if __name__ == "__main__":
    main()


# python distribution_shift.py --setting income_ca_pr
# python distribution_shift.py --setting mobility_ms_hi
# python distribution_shift.py --setting pubcov_ne_la
# python distribution_shift.py --setting pubcov_2010_2017
# python distribution_shift.py --setting setting21
# python distribution_shift.py --setting setting22





    

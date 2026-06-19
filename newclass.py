import pandas as pd
import numpy as np
from TALENT.model.utils import get_deep_args, get_method, get_classical_args
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from tabpfn import TabPFNClassifier
import argparse



def preprocess(train_data, y_train, test_data1, y_test, seed=42):
    N = {'train': train_data}  # 数值型特征
    C = None # 没有类别特征，设为 None（或使用 np.empty 代替）
    y = {'train': y_train}  # 目标变量
    # 分割训练集为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        N['train'], y['train'], test_size=0.1, random_state=seed
    )

    # 更新 N 和 y
    N = {'train': X_train, 'val': X_val}
    y = {'train': y_train, 'val': y_val}

    train_val_data = (N, C, y)
    test_data_structured=({'test': test_data1}, None, {'test': y_test})
    info = {
        "task_type": "multiclass",
        'n_num_features': X_train.shape[1],
        'n_cat_features': 0
    }
    return train_val_data, test_data_structured, info


def test_model(filename, info, threshold_min, threshold_max):
    seed = 42
    datasetname = filename.split('/')[-2]

    args_mNCA, _, _ = get_deep_args(model_type="modernNCA", cat_policy="tabr_ohe", dataset=datasetname)
    args_RForest, _, _ = get_classical_args(model_type="RandomForest", cat_policy="ordinal", dataset=datasetname)
    args_rmlp, _, _ = get_deep_args(model_type="realmlp", cat_policy="indices", dataset=datasetname)
    args_mNCA.seed = seed
    args_RForest.seed = seed
    args_rmlp.seed = seed


    results = {
        "tabpfn": {"aupr":0.0,"roc_auc":0.0},
        "modernNCA": {"aupr":0.0,"roc_auc":0.0},
        "RandomForest": {"aupr":0.0,"roc_auc":0.0},
        "realmlp": {"aupr":0.0,"roc_auc":0.0},
        "catboost": {"aupr":0.0,"roc_auc":0.0},
        "xgboost": {"aupr":0.0,"roc_auc":0.0},
        "mlp": {"aupr":0.0,"roc_auc":0.0},
    }

    data = pd.read_csv(filename)
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = pd.Categorical(data[column]).codes
    # data = data.astype('float32')
    data = data.astype('int')
    unique_labels = data['label'].unique()
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(data['label'])

    for test_label in unique_labels:
        # 初始划分
        test_data_original = data[data['label'] == test_label].copy()
        train_data = data[data['label'] != test_label].copy()

        # 从 train_data 中抽取与 test_data_original 等量的数据
        sample_size = len(test_data_original)
        num_classes_in_train = train_data['label'].nunique()

        if sample_size < num_classes_in_train:
            #  sample_size 小于类别数 ，无法 stratify，改为不 stratify
            train_sampled, train_remaining = train_test_split(
                train_data,
                test_size=(len(train_data) - sample_size),
                random_state=42
            )
        else:
            train_sampled, train_remaining = train_test_split(
                train_data,
                test_size=(len(train_data) - sample_size),
                random_state=42,
                stratify=train_data['label']
            )


        # 标记标签
        test_data_original['label'] = 1  # 原本属于 test_label 的样本设为 1
        train_sampled['label'] = 0       # 从 train_data 中抽出的样本设为 0

        # 构建最终的 test_data
        test_data = pd.concat([test_data_original, train_sampled], ignore_index=True)

        # 剩余部分作为训练集
        train_data = train_remaining

        # 分离特征和标签
        X_train = train_data.drop(columns=['label'])
        y_train = train_data['label']
        num_classes = len(np.unique(y_train))
        X_test = test_data.drop(columns=['label'])
        y_test = test_data['label'].values  # y_test 包含0和1
        # y_test = test_data['label']
        train_val_data, test_data_structured, info = preprocess(X_train, y_train, X_test, y_test)
        models = {
            "tabpfn": TabPFNClassifier(),
            "modernNCA":  get_method('modernNCA')(args_mNCA, info["task_type"] == "regression",label_encoder=label_encoder),
            "RandomForest": get_method('RandomForest')(args_RForest, info["task_type"] == "regression"),
            "realmlp": get_method('realmlp')(args_rmlp, info["task_type"] == "regression",label_encoder=label_encoder),
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
            if  model_name == 'modernNCA':
                model.fit(data=train_val_data, info=info)
                y_pred_proba_tensor = model.predict_proba(data=test_data_structured, info=info, model_name=args_mNCA.evaluate_option)

                # 获取最大置信度并转为 numpy
                y_pred_proba = y_pred_proba_tensor.max(dim=1).values.cpu().numpy()
            elif model_name == 'realmlp':
                model.fit(data=train_val_data, info=info)
                y_pred_proba_tensor = model.predict_proba(data=test_data_structured, info=info, model_name=args_rmlp.evaluate_option)

                # 获取最大置信度并转为 numpy
                y_pred_proba = y_pred_proba_tensor.max(dim=1).values.cpu().numpy()
            elif model_name == 'RandomForest':
                model.fit(data=train_val_data, info=info)
                y_pred_proba = model.predict_proba(X_test).max(axis=1)
            else:
                if model_name=='xgboost':
                    label_encoder_xgboost = LabelEncoder()
                    y_train = label_encoder_xgboost.fit_transform(y_train)
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test).max(axis=1)  # 多分类获取最大置信度

            # count = np.sum((y_pred_proba >= threshold_min) & (y_pred_proba <= threshold_max))
            pred_probs = ((y_pred_proba >= threshold_min) & (y_pred_proba <= threshold_max)).astype(int)
            aupr = average_precision_score(y_test, pred_probs)  # 如果是多分类则使用 average="macro"
            roc_auc = roc_auc_score(y_test, pred_probs) 
            print(f"aupr:{aupr};roc_auc:{roc_auc}")
            results[model_name]["aupr"]=aupr
            results[model_name]["roc_auc"]=roc_auc
            print(f"{model_name}:aupr :{aupr};roc_auc:{roc_auc}")
    
    with open("newclass2.txt", "a") as f:
        for model_name, model_arr in results.items():
            f.write(f"{model_name} aupr: {model_arr['aupr']} , roc_auc: {model_arr['roc_auc']}\n")




import argparse

def main():
    parser = argparse.ArgumentParser(description="Run test_model on a dataset with a threshold pair.")
    parser.add_argument('--dataset', required=True, help='Path to the dataset file')
    parser.add_argument('--model', required=True)
    parser.add_argument('--export_dataset', required=True)

    args = parser.parse_args()

    filename = args.dataset
    min_th = 0.4
    max_th = 0.6

    with open("newclass2.txt", "a") as f:
        f.write("=======================================================\n")
        f.write(f"filename: {filename}, min_threshold {min_th} --max_threshold {max_th}\n")

    info = {"task_type": "multiclass"}
    test_model(filename, info, min_th, max_th)

if __name__ == "__main__":
    main()


# python newclass2.py  --dataset /data0/jiazy/TALENT/dataset/eyemovements/eyemovements.csv --min_threshold 0.45 --max_threshold 0.55
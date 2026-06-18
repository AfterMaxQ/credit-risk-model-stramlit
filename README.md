# Lauki Finance：信用风险模型

基于机器学习的 Streamlit Web 应用，用于预测贷款申请人的信用风险。应用根据申请人的财务和人口统计数据，计算违约概率、信用评分和风险等级。

**在线体验：** [https://credit-risk-model-stramlit-git-4yntqsz8hdckaujhrecke7.streamlit.app/](https://credit-risk-model-stramlit-git-4yntqsz8hdckaujhrecke7.streamlit.app/)

## 功能

- **违约概率**：使用逻辑回归模型预测贷款违约的可能性
- **信用评分**：生成 300–900 分的信用评分
- **风险评级**：将申请人分为 Poor（差）、Average（一般）、Good（良好）、Excellent（优秀）四个等级

## 输入参数

| 参数 | 说明 |
|---|---|
| Age（年龄） | 申请人年龄（18–100） |
| Income（收入） | 年收入 |
| Loan Amount（贷款金额） | 申请贷款金额 |
| Loan Tenure（贷款期限） | 贷款期限（月） |
| Avg DPD（平均逾期天数） | 每次违约的平均逾期天数 |
| Delinquency Ratio（违约率） | 违约账户占比 |
| Credit Utilization Ratio（信用使用率） | 信用额度使用比例 |
| Open Loan Accounts（未结清账户数） | 当前未结清的账户数量 |
| Residence Type（居住类型） | 自有 / 租赁 / 按揭 |
| Loan Purpose（贷款用途） | 教育 / 购房 / 购车 / 个人 |
| Loan Type（贷款类型） | 无抵押 / 有抵押 |

## 推理流程

1. 用户在 Streamlit 界面输入申请人数据
2. 输入经过独热编码和标准化处理，匹配训练模型的特征空间
3. 逻辑回归模型计算违约概率
4. 基于非违约概率推导信用评分，映射到 300–900 区间
5. 根据评分分配风险等级（Poor / Average / Good / Excellent）

## 模型训练流程

完整的训练过程见 `credit_risk_model.ipynb`。

### 1. 数据加载与合并

加载三个数据源并按 `cust_id` 合并：

- `customers.csv` — 客户基本信息（年龄、收入、居住类型等）
- `loans.csv` — 贷款信息（贷款金额、期限、处理费等）
- `bureau_data.csv` — 征信局数据（历史逾期、信用使用率等）

```python
df = pd.merge(df_customers, df_loans, on="cust_id")
df = pd.merge(df, df_bureau)
```

### 2. 数据清洗

**缺失值处理：** `residence_type` 列存在缺失值，用众数填充。

```python
mode_residence = df_train.residence_type.mode()[0]
df_train['residence_type'].fillna(mode_residence, inplace=True)
df_test['residence_type'].fillna(mode_residence, inplace=True)
```

**异常值剔除：** `processing_fee` 与 `loan_amount` 的比值超过 3% 的记录被判定为异常，从训练集和测试集中移除。

```python
df_train = df_train[df_train.processing_fee / df_train.loan_amount < 0.03]
df_test  = df_test[df_test.processing_fee / df_test.loan_amount < 0.03]
```

**拼写错误修正：** `loan_purpose` 列中 `'Personaal'` 修正为 `'Personal'`。

```python
df_train['loan_purpose'] = df_train['loan_purpose'].replace('Personaal', 'Personal')
df_test['loan_purpose']  = df_test['loan_purpose'].replace('Personaal', 'Personal')
```

### 3. 特征工程

**构造新特征：**

| 新特征 | 计算方式 |
|---|---|
| `loan_to_income` | `loan_amount / income` |
| `delinquency_ratio` | `delinquent_months / total_loan_months * 100` |
| `avg_dpd_per_delinquency` | `total_dpd / delinquent_months`（逾期月份为 0 时取 0） |

```python
df_train["loan_to_income"] = df_train["loan_amount"] / df_train["income"]
df_train['delinquency_ratio'] = df_train['delinquent_months'] * 100 / df_train['total_loan_months']
df_train['avg_dpd_per_delinquency'] = np.where(
    df_train['delinquent_months'] != 0,
    df_train['total_dpd'] / df_train['delinquent_months'],
    0
)
```

**删除冗余列：** 移除 ID 列（`cust_id`, `loan_id`）、日期列（`disbursal_date`, `installment_start_dt`）以及已被新特征替代的原始列（`loan_amount`, `income`, `total_loan_months`, `delinquent_months`, `total_dpd`）。

### 4. 特征选择

**VIF（方差膨胀因子）检验：** 剔除高共线性特征，去除 `sanction_amount`, `processing_fee`, `gst`, `net_disbursement`, `principal_outstanding`。

**WOE/IV（证据权重/信息值）筛选：** 计算所有特征的 IV 值，保留 IV > 0.02 的特征，确保每个特征对目标变量有足够的预测能力。

```python
selected_features_iv = [feature for feature, iv in iv_values.items() if iv > 0.02]
```

### 5. 数据预处理

- **独热编码：** 对分类变量进行 One-Hot 编码（`drop_first=True` 避免虚拟变量陷阱）
- **归一化：** 使用 `MinMaxScaler` 对数值特征进行 [0, 1] 归一化

### 6. 类别不平衡处理

使用 **SMOTETomek**（SMOTE 过采样 + Tomek Links 欠采样）平衡正负样本：

```python
smt = SMOTETomek(random_state=42)
X_train_smt, y_train_smt = smt.fit_resample(X_train_encoded, y_train)
```

### 7. 模型选择与调优

对比了三种模型的基线效果：

| 模型 | 说明 |
|---|---|
| Logistic Regression | 简单可解释，最终选用 |
| Random Forest | 集成方法，效果接近 |
| XGBoost | 梯度提升，强劲基线 |

使用 **Optuna** 对逻辑回归进行 50 轮贝叶斯超参数搜索，优化宏平均 F1 分数，搜索空间包括 `C`（正则化强度）、`solver`（优化算法）、`tol`（收敛阈值）和 `class_weight`（类别权重）。

```python
param = {
    'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
    'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg']),
    'tol': trial.suggest_float('tol', 1e-6, 1e-1, log=True),
    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
}
```

### 8. 模型评估

- **ROC 曲线与 AUC** — 评估模型区分能力
- **KS 统计量** — 衡量好坏样本的区分度
- **分位数分析（Decile Analysis）** — 将预测概率分为 10 等分，观察每层的实际违约率分布，验证模型的排序能力

### 9. 模型保存

将最终模型、标准化器、特征列表等信息打包保存：

```python
model_data = {
    'model': final_model,
    'features': X_train_encoded.columns,
    'scaler': scaler,
    'cols_to_scale': cols_to_scale
}
dump(model_data, 'artifacts/model_data.joblib')
```

## 项目结构

```
credit-risk-model-stramlit/
├── app/
│   ├── main.py                  # Streamlit 界面
│   └── prediction_helper.py     # 模型推理逻辑
├── artifacts/
│   └── model_data.joblib        # 训练好的模型、标准化器、特征配置
├── dataset/
│   ├── customers.csv            # 客户数据
│   ├── loans.csv                # 贷款数据
│   └── bureau_data.csv          # 征信局数据
├── credit_risk_model.ipynb      # 模型训练 Notebook
├── requirements.txt             # Python 依赖
└── README.md
```

## 技术栈

- **Python 3.x**
- **Streamlit** — Web UI 框架
- **scikit-learn** — 逻辑回归模型、预处理、评估
- **imbalanced-learn** — SMOTETomek 采样
- **Optuna** — 超参数调优
- **pandas / numpy** — 数据处理
- **joblib** — 模型序列化

## 本地运行

```bash
# 克隆仓库
git clone https://github.com/AfterMaxQ/credit-risk-model-stramlit.git
cd credit-risk-model-stramlit

# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app/main.py
```

应用将在 `http://localhost:8501` 打开。

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

## 工作原理

1. 用户在 Streamlit 界面输入申请人数据
2. 输入经过独热编码和标准化处理，匹配训练模型的特征空间
3. 逻辑回归模型计算违约概率
4. 基于非违约概率推导信用评分，映射到 300–900 区间
5. 根据评分分配风险等级（Poor / Average / Good / Excellent）

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
- **scikit-learn** — 逻辑回归模型
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

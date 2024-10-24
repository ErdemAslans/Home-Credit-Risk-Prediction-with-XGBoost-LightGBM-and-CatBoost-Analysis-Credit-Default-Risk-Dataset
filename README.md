# Home-Credit-Risk-Prediction-with-XGBoost-LightGBM-and-CatBoost-Analysis-Credit-Default-Risk-Dataset

This project focuses on credit risk prediction using advanced machine learning models and transformation techniques on the Home Credit Default Risk dataset. The project applies XGBoost, LightGBM, and CatBoost models to predict credit default risk, utilizing SHAP values for model explainability. Key steps like handling missing data, feature engineering, and addressing class imbalance have been implemented to enhance the accuracy and robustness of the predictions.

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Models](#models)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About the Project
In this project, I tackled common challenges in credit risk prediction, such as data imbalances, missing values, and model explainability. Using the Home Credit Default Risk dataset, three different models (XGBoost, LightGBM, CatBoost) were employed to predict loan defaults. SHAP values were used to explain the model decisions, ensuring transparency and trust in the results.

This project serves as a strong foundation for data scientists and machine learning practitioners interested in financial risk analysis.

## Technologies Used
The primary technologies and tools used in this project are:
- Python
- XGBoost
- LightGBM
- CatBoost
- SHAP (SHapley Additive Explanations)
- Pandas, NumPy
- Matplotlib, Seaborn

##  Dataset
The project uses the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) dataset, which includes millions of credit applications and their repayment performance. After cleaning this complex dataset, I performed feature engineering and applied various data transformation techniques to improve model accuracy.

##  Models
Three models were utilized in this project:
1. **XGBoost**: A widely used and powerful model for credit risk prediction that efficiently captures important patterns in the data.
2. **LightGBM**: Known for its speed and effectiveness on large datasets, providing a balance between accuracy and performance.
3. **CatBoost**: Particularly effective with categorical data, this model adapted well to the complex structure of the dataset.

##  Feature Engineering
Key steps in feature engineering include:
- Handling missing data effectively.
- Creating new features such as `NEW_EXT_MEAN` and `NEW_ANNUITY_CREDIT_RATIO`.
- Addressing class imbalances using various techniques.

## Model Performance
The models were evaluated using various performance metrics, and SHAP values were employed to analyze the decision-making process. Below are some key findings:
- Features such as `NEW_EXT_MEAN` and `NEW_ANNUITY_CREDIT_RATIO` were critical in determining the model’s decisions.
- SHAP analysis provided transparency into why the models made specific predictions, a key advantage for financial institutions.



## Installation
Follow these steps to run the project locally:

### Requirements
- Python 3.7+
- Libraries: Pandas, NumPy, XGBoost, LightGBM, CatBoost, SHAP, Matplotlib, Seaborn

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/your_project_name.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Navigate to the `notebooks` folder and run the `model_training.ipynb` to train the models and analyze results.

## 🔧 Usage
1. Download the dataset from [Kaggle Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) and place it in the `data/` folder.
2. Train the model by running:
    ```bash
    python train_model.py
    ```
3. Analyze model results:
    ```bash
    python analyze_results.py
    ```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a **pull request** or open an **issue** to start a discussion.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

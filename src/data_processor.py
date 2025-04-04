import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import numpy as np


class CancerGenomicDataProcessor:
    def __init__(self, file_path, num_features=500):
        self.file_path = file_path
        self.num_features = num_features
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """ Load CSV file and drop columns with too many missing values """
        self.data = pd.read_csv(self.file_path)
        self.data = self.data.dropna(axis=1, thresh=0.8 * len(self.data))  # Keep columns with at least 80% data
        print(self.data.columns[:-1])
        print("✅ Data loaded successfully!")

    def normalize_data(self):
        """ Normalize gene expression values using MinMaxScaler """
        scaler = MinMaxScaler()
        self.data.iloc[:, :-1] = scaler.fit_transform(self.data.iloc[:, :-1])  # Normalize all columns except the last
        print("✅ Data normalized successfully!")

    def feature_selection(self):
        """ Select the most relevant genes using ANOVA F-test and Lasso Regression """
        X = self.data.iloc[:, :-1]  # Features (gene expressions)
        y = self.data.iloc[:, -1]  # Labels (0 = No Cancer, 1 = Cancer)

        # Step 1: Select top `num_features` genes using ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k=self.num_features)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)

        # Step 2: Further refine features using Lasso regression
        lasso = LogisticRegression(penalty="l1", solver="liblinear", C=0.01)
        lasso.fit(X.iloc[:, selected_indices], y)
        important_features = np.where(lasso.coef_[0] != 0)[0]

        # Keep only the important genes
        self.X_train = X.iloc[:, selected_indices].iloc[:, important_features]
        self.y_train = y

        print(f"✅ Feature selection completed! Reduced from {X.shape[1]} to {self.X_train.shape[1]} genes.")

    def process(self):
        self.load_data()
        self.normalize_data()
        self.feature_selection()


# Usage
file_path = "../data/raw/gene_expression.csv"
processor = CancerGenomicDataProcessor(file_path)
processor.process()

# Access processed data
X_train = processor.X_train
y_train = processor.y_train
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

print("--- Đang tải và tiền xử lý dữ liệu ---")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test["Id"]

cat_col_train = [
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageQual",
    "GarageCond",
]
ncat_col_train = ["LotFrontage", "GarageYrBlt", "MasVnrArea"]

for i in cat_col_train:
    train[i] = train[i].fillna(train[i].mode()[0])
for j in ncat_col_train:
    train[j] = train[j].fillna(train[j].mean())

cat_col_test = [
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "MasVnrType",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageQual",
    "GarageCond",
    "MSZoning",
    "Utilities",
    "Exterior1st",
    "Exterior2nd",
    "KitchenQual",
    "Functional",
    "SaleType",
]
ncat_col_test = [
    "LotFrontage",
    "GarageYrBlt",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "GarageCars",
    "GarageArea",
]

for i in cat_col_test:
    test[i] = test[i].fillna(test[i].mode()[0])
for j in ncat_col_test:
    test[j] = test[j].fillna(test[j].mean())

to_drop = ["Id", "Alley", "PoolQC", "Fence", "MiscFeature"]
train.drop(columns=to_drop, inplace=True)
test.drop(columns=to_drop, inplace=True)


train["TotalHouseArea"] = train["TotalBsmtSF"] + train["1stFlrSF"] + train["2ndFlrSF"]
test["TotalHouseArea"] = test["TotalBsmtSF"] + test["1stFlrSF"] + test["2ndFlrSF"]

final_df = pd.concat([train, test], axis=0)

final_df = pd.get_dummies(final_df, drop_first=True)
final_df = final_df.astype(float)

final_df = final_df.loc[:, ~final_df.columns.duplicated()]

df_train = final_df.iloc[:1460, :]
df_test = final_df.iloc[1460:, :].drop("SalePrice", axis=1)

X = df_train.drop("SalePrice", axis=1)
y = df_train["SalePrice"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(df_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\n--- Đào tạo mô hình Scikit-Learn ---")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
print(f"Random Forest RMSE trên tập Validation: {rf_rmse:.2f}")

# Tính quan trọng của đặc trưng (Feature Importance)
importances = rf_model.feature_importances_
top_features = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:10]
print("\nTop 10 đặc trưng quan trọng nhất:")
print(top_features)

print("\n--- Đào tạo mô hình MLP bằng PyTorch ---")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


class HousePriceMLP(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


input_dim = X_train.shape[1]
mlp_model = HousePriceMLP(input_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)


epochs = 200
for epoch in range(epochs):
    mlp_model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = mlp_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        mlp_model.eval()
        with torch.no_grad():
            val_outputs = mlp_model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            val_rmse = torch.sqrt(val_loss).item()
        print(
            f"Epoch [{epoch+1}/{epochs}] - Train Loss: {running_loss/len(train_loader):.2f} - Val RMSE: {val_rmse:.2f}"
        )

test_preds_rf = rf_model.predict(X_test_scaled)
sub_rf = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds_rf})
sub_rf.to_csv("submission_rf.csv", index=False)

mlp_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    test_preds_mlp = mlp_model(X_test_tensor).numpy().flatten()
sub_mlp = pd.DataFrame({"Id": test_ids, "SalePrice": test_preds_mlp})
sub_mlp.to_csv("submission_mlp.csv", index=False)

print("\nĐã tạo thành công 'submission_rf.csv' và 'submission_mlp.csv'!")

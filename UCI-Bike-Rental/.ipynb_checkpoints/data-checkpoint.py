import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime
import os

class data_generator:
    def __init__(self, path='data/hour.csv'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = pd.read_csv(path)
        self.data_wrangling()
        X, y = self.data.drop('cnt', axis=1), self.data['cnt']
        self.target_mean, self.target_std = y.mean(), y.std()
        X_transformed, y_transformed = self.feature_engineering(X, y)
        self.env_dict_train, self.env_dict_test = self.generate_enviroments(X_transformed, y_transformed)
        
    def data_wrangling(self):     
        # Feature Engineering
        # Transform 'dteday' into more useful features: day of the month, and whether it's a weekend
        self.data['day'] = self.data['dteday'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').day)
        self.data['is_weekend'] = self.data['weekday'].apply(lambda x: 1 if x in [0, 6] else 0)
        # Dropping irrelevant columns to avoid data leakage
        self.data = self.data.drop(['instant', 'dteday','casual', 'registered'], axis=1)
    
    def feature_engineering(self, X, y):
        y_transformed = (y-self.target_mean)/self.target_std
        # Normalizing and One-Hot Encoding
        # Identifying categorical and numerical features
        env_columns = ['season', 'yr']
        categorical_features = ['mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'is_weekend']
        numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'day']

        # Step 1: Data Transformation
        onehotencoder = OneHotEncoder()
        scaler = StandardScaler()
        # One-hot encode categorical features
        X_encoded = onehotencoder.fit_transform(X[categorical_features])
        # Scale numerical features
        X_scaled = scaler.fit_transform(X[numerical_features])

        # Convert the encoded features and scaled features back to DataFrame
        X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=onehotencoder.get_feature_names(categorical_features))
        X_scaled_df = pd.DataFrame(X_scaled, columns=numerical_features)

        # Reset index to ensure concatenation works correctly
        X.reset_index(drop=True, inplace=True)
        X_encoded_df.reset_index(drop=True, inplace=True)
        X_scaled_df.reset_index(drop=True, inplace=True)

        # Concatenate encoded and scaled data with other columns (if any)
        X_transformed = pd.concat([X[env_columns], X_encoded_df, X_scaled_df], axis=1)
        return X_transformed, y_transformed
    
    def generate_enviroments(self, X, y):
        # Step 2: Creating Environments
        environments = []
        # Iterate through each combination of season and year
        for season in X['season'].unique():
            for year in X['yr'].unique():
                # Create a subset of data for each environment
                X_env_subset = X[(X['season'] == season) & (X['yr'] == year)]
                y_env_subset = y[(X['season'] == season) & (X['yr'] == year)]
                # drop season and year from features 
                X_env_subset = X_env_subset.drop(['season','yr'], axis=1)
                #y_env_subset = np.random.uniform(0.1,0.2,1)*y_env_subset
                #y_env_subset = (y_env_subset-y_env_subset.mean())/y_env_subset.std()
                tX = torch.tensor(X_env_subset.to_numpy(),dtype=torch.float32).to(self.device) 
                ty = torch.tensor(y_env_subset.to_numpy(),dtype=torch.float32).view(-1, 1).to(self.device)
                environments.append([tX,ty])
        train_environments = environments[:4]
        test_environments = environments[4:]
        dict_train = {str(i):{'x':env[0],'y':env[1]} for i,env in enumerate(train_environments)}
        dict_test = {str(i):{'x': env[0],'y':env[1]} for i,env in enumerate(test_environments)}
        return dict_train, dict_test

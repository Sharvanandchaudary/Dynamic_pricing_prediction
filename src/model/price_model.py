import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import logging
import os

class PricingModel:
    def __init__(self):
        self.rf_model = None
        self.xgb_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = [
            'category', 'base_price', 'inventory_level', 'demand_last_24h',
            'competitor_price', 'season', 'day_of_week', 'time_of_day',
            'special_event', 'customer_segment', 'rating', 'historical_sales'
        ]
        
    def preprocess_data(self, df):
        """Preprocess the data for model training"""
        # Create copy of data
        processed_df = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['category', 'season', 'day_of_week', 'customer_segment']
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                processed_df[column] = self.label_encoders[column].fit_transform(processed_df[column])
            else:
                processed_df[column] = self.label_encoders[column].transform(processed_df[column])
        
        # Scale numerical features
        numerical_columns = ['base_price', 'inventory_level', 'demand_last_24h', 
                           'competitor_price', 'time_of_day', 'rating', 'historical_sales']
        processed_df[numerical_columns] = self.scaler.fit_transform(processed_df[numerical_columns])
        
        return processed_df

    def train_models(self, data_path, test_size=0.2, random_state=42):
        """Train both Random Forest and XGBoost models"""
        # Load data
        df = pd.read_csv(data_path)
        
        # Preprocess data
        processed_df = self.preprocess_data(df)
        
        # Prepare features and target
        X = processed_df[self.feature_columns]
        y = df['final_price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        )
        self.rf_model.fit(X_train, y_train)
        
        # Train XGBoost
        self.xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state
        )
        self.xgb_model.fit(X_train, y_train)
        
        # Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        return results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models and return metrics"""
        results = {}
        
        # Random Forest predictions
        rf_predictions = self.rf_model.predict(X_test)
        results['random_forest'] = {
            'mse': mean_squared_error(y_test, rf_predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_predictions)),
            'mae': mean_absolute_error(y_test, rf_predictions),
            'r2': r2_score(y_test, rf_predictions)
        }
        
        # XGBoost predictions
        xgb_predictions = self.xgb_model.predict(X_test)
        results['xgboost'] = {
            'mse': mean_squared_error(y_test, xgb_predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, xgb_predictions)),
            'mae': mean_absolute_error(y_test, xgb_predictions),
            'r2': r2_score(y_test, xgb_predictions)
        }
        
        return results
    
    def predict_price(self, features, model_type='xgboost'):
        """Make price predictions using the specified model"""
        try:
            # Preprocess features
            processed_features = self.preprocess_single_sample(features)
            
            if model_type.lower() == 'xgboost':
                prediction = self.xgb_model.predict([processed_features])[0]
            else:
                prediction = self.rf_model.predict([processed_features])[0]
            
            return round(prediction, 2)
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
    
    def preprocess_single_sample(self, features):
        """Preprocess a single sample for prediction"""
        # Create a dictionary to store processed values
        processed_dict = {}
        
        # Handle categorical features first
        for column in self.feature_columns:
            if column in self.label_encoders:
                processed_dict[column] = self.label_encoders[column].transform([features[column]])[0]
            else:
                processed_dict[column] = features[column]
        
        # Create a list of numerical features in the correct order
        numerical_columns = ['base_price', 'inventory_level', 'demand_last_24h', 
                            'competitor_price', 'time_of_day', 'rating', 'historical_sales']
        
        # Create array of numerical values for scaling
        numerical_values = [[processed_dict[col] for col in numerical_columns]]
        
        # Scale numerical features
        scaled_values = self.scaler.transform(numerical_values)[0]
        
        # Update processed dict with scaled values
        for i, col in enumerate(numerical_columns):
            processed_dict[col] = scaled_values[i]
        
        # Return values in the correct order as expected by the model
        return [processed_dict[col] for col in self.feature_columns]
    
    def save_models(self, path='models/'):
        """Save trained models and preprocessors"""
        # Create the models directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save the models
        joblib.dump(self.rf_model, f'{path}random_forest_model.joblib')
        joblib.dump(self.xgb_model, f'{path}xgboost_model.joblib')
        joblib.dump(self.label_encoders, f'{path}label_encoders.joblib')
        joblib.dump(self.scaler, f'{path}scaler.joblib')
        
    def load_models(self, path='models/'):
        """Load trained models and preprocessors"""
        self.rf_model = joblib.load(f'{path}random_forest_model.joblib')
        self.xgb_model = joblib.load(f'{path}xgboost_model.joblib')
        self.label_encoders = joblib.load(f'{path}label_encoders.joblib')
        self.scaler = joblib.load(f'{path}scaler.joblib')

# model_utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class PCBCycleTimeModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.categorical_columns = ['machine_type', 'shift']
        self.numeric_features = ['num_components', 'board_layers', 'component_density', 'operator_experience']
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the PCB dataset"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Encode categorical variables using one-hot encoding
        df_encoded = pd.get_dummies(df, columns=self.categorical_columns, drop_first=True)
        
        # Apply feature scaling to numeric features
        df_encoded[self.numeric_features] = self.scaler.fit_transform(df_encoded[self.numeric_features])
        
        # Split features and target
        X = df_encoded.drop('cycle_time', axis=1)
        y = df_encoded['cycle_time']
        
        self.feature_names = X.columns.tolist()
        return X, y
        
    def train_model(self, X, y):
        """Train Random Forest model with hyperparameter tuning"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define hyperparameter search space
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Initialize Random Forest
        rf = RandomForestRegressor(random_state=42)
        
        # Perform randomized search
        print("Performing hyperparameter tuning...")
        random_search = RandomizedSearchCV(
            rf, param_distributions, n_iter=20, cv=5, 
            scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        # Get best model
        self.model = random_search.best_estimator_
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'best_params': random_search.best_params_
        }
        
        print(f"Model Training Complete!")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R²: {metrics['r2']:.3f}")
        
        return metrics
        
    def save_model(self, model_path='pcb_model.pkl', scaler_path='scaler.pkl'):
        """Save trained model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        with open('feature_names.txt', 'w') as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
    def load_model(self, model_path='pcb_model.pkl', scaler_path='scaler.pkl'):
        """Load pre-trained model and scaler"""
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Load feature names
            if os.path.exists('feature_names.txt'):
                with open('feature_names.txt', 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            
            print("Model and scaler loaded successfully!")
            return True
        return False
        
    def predict_single(self, num_components, board_layers, component_density, 
                      machine_type, operator_experience, shift):
        """Make prediction for a single PCB assembly"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
            
        # Create input dataframe
        input_data = pd.DataFrame({
            'num_components': [num_components],
            'board_layers': [board_layers],
            'component_density': [component_density],
            'machine_type': [machine_type],
            'operator_experience': [operator_experience],
            'shift': [shift]
        })
        
        # Encode categorical variables (same as training)
        input_encoded = pd.get_dummies(input_data, columns=self.categorical_columns, drop_first=True)
        
        # Ensure all columns exist (fill missing with False)
        for feature in self.feature_names:
            if feature not in input_encoded.columns:
                if feature in self.numeric_features:
                    input_encoded[feature] = 0  # Will be scaled
                else:
                    input_encoded[feature] = False
        
        # Reorder columns to match training
        input_encoded = input_encoded[self.feature_names]
        
        # Scale numeric features
        input_encoded[self.numeric_features] = self.scaler.transform(input_encoded[self.numeric_features])
        
        # Make prediction
        prediction = self.model.predict(input_encoded)[0]
        return prediction
        
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained or loaded!")
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df

# Training script
if __name__ == "__main__":
    # Initialize model
    pcb_model = PCBCycleTimeModel()
    
    # Load and preprocess data
    csv_file = "pcb_cycle_dataset_core.csv"  # Update path as needed
    X, y = pcb_model.load_and_preprocess_data(csv_file)
    
    # Train model
    metrics = pcb_model.train_model(X, y)
    
    # Save model
    pcb_model.save_model()
    
    print("\nModel training complete! Ready for dashboard.")
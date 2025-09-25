# PCB Cycle Time Prediction Dashboard - Setup Guide

## Overview
This dashboard predicts PCB assembly cycle times using a Random Forest machine learning model. It provides an interactive web interface for manufacturing teams to optimize production planning.

## Project Structure
```
pcb-dashboard/
├── dashboard.py          # Main Streamlit application
├── model_utils.py        # ML model training and utilities
├── requirements.txt      # Python dependencies
├── pcb_cycle_dataset_core.csv  # Training data
└── README.md            # This file
```

## Setup Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv pcb_env
source pcb_env/bin/activate  # On Windows: pcb_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
- Ensure `pcb_cycle_dataset_core.csv` is in the project directory
- The dataset should contain columns: num_components, board_layers, component_density, machine_type, operator_experience, shift, cycle_time

### 3. Model Training
```bash
# Train the machine learning model
python model_utils.py
```
This will:
- Load and preprocess the dataset
- Train a Random Forest model with hyperparameter tuning
- Save the trained model (pcb_model.pkl) and scaler (scaler.pkl)
- Display training metrics

### 4. Run Dashboard
```bash
# Start the Streamlit dashboard
streamlit run dashboard.py
```

## How It Works

### Data Preprocessing
1. **Categorical Encoding**: Machine type (A/B/C) and shift (Day/Night) are one-hot encoded
2. **Feature Scaling**: Numeric features are standardized using StandardScaler
3. **Feature Engineering**: Uses 6 input features to predict cycle time

### Model Training
- **Algorithm**: Random Forest Regressor with hyperparameter tuning
- **Features**: num_components, board_layers, component_density, machine_type, operator_experience, shift
- **Performance**: MAE ~7.77, RMSE ~9.71, R² ~0.918
- **Cross-validation**: 5-fold CV for robust evaluation

### Dashboard Features
1. **Interactive Input Form**: Sidebar with all PCB specification inputs
2. **Real-time Prediction**: Instant cycle time prediction on button click
3. **Gauge Visualization**: Color-coded gauge showing efficiency levels
4. **Feature Importance**: Bar chart showing which factors most affect cycle time
5. **Data Analysis**: Overview of dataset distributions and statistics
6. **Optimization Recommendations**: Actionable suggestions based on prediction

## Usage Guide

### Making Predictions
1. **Board Specifications**:
   - Components: 50-300 (typical range)
   - Layers: 2, 4, 6, or 8
   - Density: 0.5-5.0 components per unit area

2. **Production Parameters**:
   - Machine Type: A (high-speed), B (standard), C (precision)
   - Operator Experience: 1-10 years
   - Shift: Day or Night

3. **Click Predict**: Get instant cycle time estimate with efficiency rating

### Interpreting Results
- **Green Zone (< 80s)**: Excellent efficiency
- **Yellow Zone (80-120s)**: Good efficiency, minor optimization possible
- **Red Zone (> 120s)**: High cycle time, optimization needed

### Optimization Tips
- Use Machine Type A for fastest assembly
- Assign experienced operators to complex boards
- Consider reducing component density for faster times
- Day shift typically shows better performance

## Technical Details

### Model Architecture
```python
RandomForestRegressor(
    n_estimators=200,      # Optimized through hyperparameter tuning
    max_depth=20,          # Prevents overfitting
    min_samples_split=5,   # Minimum samples to split node
    min_samples_leaf=2,    # Minimum samples in leaf
    random_state=42        # Reproducible results
)
```

### Feature Encoding
- Machine Type: One-hot encoded (machine_type_B, machine_type_C)
- Shift: Binary encoded (shift_Night)
- Numeric features: StandardScaler normalization

### Performance Metrics
- **MAE (Mean Absolute Error)**: ~7.77 seconds
- **RMSE (Root Mean Square Error)**: ~9.71 seconds  
- **R² Score**: ~0.918 (91.8% variance explained)

## Troubleshooting

### Common Issues
1. **"Model files not found"**: Run `python model_utils.py` first
2. **CSV file not found**: Ensure dataset is in correct directory
3. **Import errors**: Install all requirements via `pip install -r requirements.txt`

### Performance Tips
- Model training takes 2-3 minutes on average hardware
- Dashboard loads instantly after model is trained
- Predictions are near-instantaneous (<100ms)

## Extension Ideas
- Add batch prediction capability
- Include cost optimization features
- Implement model retraining interface
- Add historical trend analysis
- Export predictions to CSV/Excel
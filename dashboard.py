# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from model_utils import PCBCycleTimeModel
import os

# Configure Streamlit page
st.set_page_config(
    page_title="PCB Cycle Time Predictor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-result {
        font-size: 2rem;
        color: #ff6b6b;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    model = PCBCycleTimeModel()
    if model.load_model():
        return model
    else:
        st.error("❌ Model files not found! Please train the model first.")
        st.info("Run: `python model_utils.py` to train the model")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for reference"""
    if os.path.exists("pcb_cycle_dataset_core.csv"):
        return pd.read_csv("pcb_cycle_dataset_core.csv")
    return None

def create_feature_importance_plot(importance_df):
    """Create interactive feature importance plot"""
    fig = px.bar(
        importance_df.head(10), 
        x='importance', 
        y='feature',
        orientation='h',
        title="🎯 Top 10 Feature Importance",
        labels={'importance': 'Importance Score', 'feature': 'Features'},
        color='importance',
        color_continuous_scale='viridis'
    )
    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    return fig

def create_prediction_gauge(prediction):
    """Create a gauge chart for prediction"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Cycle Time (seconds)"},
        delta = {'reference': 100},
        gauge = {
            'axis': {'range': [None, 250]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 80], 'color': "lightgreen"},
                {'range': [80, 120], 'color': "yellow"},
                {'range': [120, 250], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 150
            }
        }
    ))
    fig.update_layout(height=400)
    return fig

def create_data_distribution_plots(df):
    """Create distribution plots for key features"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Components Distribution', 'Cycle Time Distribution', 
                       'Machine Type Distribution', 'Shift Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Components histogram
    fig.add_trace(
        go.Histogram(x=df['num_components'], name='Components', showlegend=False),
        row=1, col=1
    )
    
    # Cycle time histogram
    fig.add_trace(
        go.Histogram(x=df['cycle_time'], name='Cycle Time', showlegend=False),
        row=1, col=2
    )
    
    # Machine type count
    machine_counts = df['machine_type'].value_counts()
    fig.add_trace(
        go.Bar(x=machine_counts.index, y=machine_counts.values, name='Machine Type', showlegend=False),
        row=2, col=1
    )
    
    # Shift count
    shift_counts = df['shift'].value_counts()
    fig.add_trace(
        go.Bar(x=shift_counts.index, y=shift_counts.values, name='Shift', showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="📊 Dataset Overview")
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🔧 PCB Cycle Time Predictor</h1>', unsafe_allow_html=True)
    st.markdown("### Predict PCB assembly cycle times using machine learning")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Load sample data
    sample_data = load_sample_data()
    
    # Sidebar for inputs
    st.sidebar.header("🎛️ Input Parameters")
    st.sidebar.markdown("Enter the PCB assembly specifications:")
    
    # Input fields with explanations
    with st.sidebar:
        st.markdown("#### Board Specifications")
        num_components = st.number_input(
            "Number of Components", 
            min_value=50, max_value=300, value=150,
            help="Total number of components on the PCB"
        )
        
        board_layers = st.selectbox(
            "Board Layers", 
            options=[2, 4, 6, 8], index=1,
            help="Number of copper layers in the PCB"
        )
        
        component_density = st.number_input(
            "Component Density", 
            min_value=0.5, max_value=5.0, value=2.5, step=0.1,
            help="Components per square unit area"
        )
        
        st.markdown("#### Production Parameters")
        machine_type = st.selectbox(
            "Machine Type", 
            options=['A', 'B', 'C'], index=0,
            help="Type of assembly machine (A=High-speed, B=Standard, C=Precision)"
        )
        
        operator_experience = st.slider(
            "Operator Experience (years)", 
            min_value=1, max_value=10, value=5,
            help="Years of experience of the machine operator"
        )
        
        shift = st.selectbox(
            "Shift", 
            options=['Day', 'Night'], index=0,
            help="Production shift time"
        )
        
        # Predict button
        predict_button = st.button("🚀 Predict Cycle Time", type="primary")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if predict_button:
            try:
                # Make prediction
                prediction = model.predict_single(
                    num_components, board_layers, component_density,
                    machine_type, operator_experience, shift
                )
                
                # Display prediction
                st.markdown("## 🎯 Prediction Results")
                
                # Gauge chart
                gauge_fig = create_prediction_gauge(prediction)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Metrics
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric(
                        "Predicted Time", 
                        f"{prediction:.1f} sec",
                        help="Estimated cycle time for this PCB assembly"
                    )
                with col_met2:
                    efficiency = "High" if prediction < 80 else "Medium" if prediction < 120 else "Low"
                    st.metric("Efficiency Level", efficiency)
                with col_met3:
                    cost_estimate = prediction * 0.05  # Rough cost estimate
                    st.metric("Est. Cost", f"${cost_estimate:.2f}")
                
                # Recommendations
                st.markdown("### 💡 Optimization Recommendations")
                if prediction > 120:
                    st.warning("⚠️ High cycle time detected! Consider:")
                    st.write("- Using a higher-speed machine (Type A)")
                    st.write("- Reducing component density if possible")
                    st.write("- Assigning more experienced operators")
                elif prediction < 60:
                    st.success("✅ Excellent efficiency! This configuration is optimal.")
                else:
                    st.info("ℹ️ Good efficiency. Minor optimizations possible.")
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    with col2:
        # Feature importance
        st.markdown("### 📊 Feature Importance")
        try:
            importance_df = model.get_feature_importance()
            importance_fig = create_feature_importance_plot(importance_df)
            st.plotly_chart(importance_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Feature importance error: {str(e)}")
    
    # Additional analysis tabs
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📈 Data Analysis", "🔍 Model Info", "📖 User Guide"])
    
    with tab1:
        if sample_data is not None:
            st.markdown("### Dataset Overview")
            
            # Distribution plots
            dist_fig = create_data_distribution_plots(sample_data)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📋 Summary Statistics")
            st.dataframe(sample_data.describe(), use_container_width=True)
        else:
            st.warning("Sample data not available")
    
    with tab2:
        st.markdown("### 🤖 Model Information")
        st.write("**Algorithm:** Random Forest Regressor")
        st.write("**Features:** Component count, board layers, density, machine type, operator experience, shift")
        st.write("**Performance Metrics:** MAE ~7.77, RMSE ~9.71, R² ~0.918")
        
        # Model details
        try:
            importance_df = model.get_feature_importance()
            st.markdown("#### Feature Rankings")
            st.dataframe(importance_df, use_container_width=True)
        except:
            st.info("Model details not available")
    
    with tab3:
        st.markdown("### 📖 How to Use This Dashboard")
        st.markdown("""
        1. **Enter Parameters**: Use the sidebar to input your PCB specifications
        2. **Get Prediction**: Click 'Predict Cycle Time' to see estimated assembly time
        3. **Analyze Results**: Review the gauge chart and optimization recommendations
        4. **Understand Impact**: Check feature importance to see what affects cycle time most
        
        #### Tips for Better Predictions:
        - Ensure input values are within typical manufacturing ranges
        - Consider the trade-offs between speed and precision
        - Use experienced operators for complex assemblies
        - Machine Type A is fastest but may have higher setup costs
        """)

if __name__ == "__main__":
    main()
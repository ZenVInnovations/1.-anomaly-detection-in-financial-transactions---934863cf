import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from models.isolation_forest import run_isolation_forest
from models.autoencoder import run_autoencoder
import os
from datetime import datetime
import base64
from io import BytesIO
import argparse

# Page configuration
st.set_page_config(
    page_title="Anomaly Detective | Financial Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to download pandas dataframe as CSV
def download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Session state initialization
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'last_run_date' not in st.session_state:
    st.session_state.last_run_date = None

# Toggle dark mode
def toggle_dark_mode():
    st.session_state.dark_mode = not st.session_state.dark_mode

# ---------- Custom CSS with dark mode support ----------
def get_custom_css():
    if st.session_state.dark_mode:
        return """
        <style>
        .main {
            background-color: #121212;
            color: #ffffff;
        }
        .block-container {
            padding: 2rem 3rem;
        }
        .title {
            font-size: 3rem;
            font-weight: 700;
            color: #00b4d8;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #90e0ef;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.5);
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #00b4d8;
        }
        .metric-label {
            font-size: 1rem;
            color: #90e0ef;
        }
        .stButton > button {
            background-color: #0077b6 !important;
            color: white !important;
            font-size: 16px !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            border: none !important;
        }
        .report-section {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .report-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #00b4d8;
            margin-bottom: 1rem;
        }
        </style>
        """
    else:
        return """
        <style>
        .main {
            background-color: #f8f9fa;
            color: #212529;
        }
        .block-container {
            padding: 2rem 3rem;
        }
        .title {
            font-size: 3rem;
            font-weight: 700;
            color: #0077b6;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #4a4e69;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #0077b6;
        }
        .metric-label {
            font-size: 1rem;
            color: #4a4e69;
        }
        .stButton > button {
            background-color: #0077b6 !important;
            color: white !important;
            font-size: 16px !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            border: none !important;
        }
        .report-section {
            background-color: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        .report-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #0077b6;
            margin-bottom: 1rem;
        }
        </style>
        """

st.markdown(get_custom_css(), unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.image("https://via.placeholder.com/150x150.png?text=AD", width=150)
    st.markdown("# 🕵️ Anomaly Detective")
    
    # Dark mode toggle
    dark_mode_col1, dark_mode_col2 = st.columns([3, 1])
    with dark_mode_col1:
        st.write("Toggle theme:")
    with dark_mode_col2:
        mode_label = "🌙" if st.session_state.dark_mode else "☀️"
        st.button(mode_label, on_click=toggle_dark_mode)
    
    st.divider()
    
    # Model selection with enhanced UI
    st.markdown("## 🧠 Model Selection")
    model_choice = st.radio(
        "Choose your detection algorithm:",
        ["Isolation Forest", "Autoencoder (PyTorch)"],
        help="Isolation Forest works well for outlier detection while Autoencoder is better for complex patterns"
    )
    
    # Model parameters based on selection
    st.markdown("### ⚙️ Model Parameters")
    if model_choice == "Isolation Forest":
        contamination = st.slider(
            "Contamination Rate", 
            min_value=0.01, 
            max_value=0.5, 
            value=0.05, 
            step=0.01,
            help="Expected proportion of anomalies in the dataset"
        )
        n_estimators = st.slider("Number of Trees", 50, 200, 100, 10)
        model_params = {
            "contamination": contamination,
            "n_estimators": n_estimators
        }
    else:
        threshold = st.slider(
            "Anomaly Threshold", 
            min_value=0.01, 
            max_value=0.2, 
            value=0.02, 
            step=0.01,
            help="Reconstruction error threshold for anomaly classification"
        )
        epochs = st.slider("Training Epochs", 10, 100, 30, 5)
        model_params = {
            "threshold": threshold,
            "epochs": epochs
        }
    
    st.divider()
    
    # About section
    st.markdown("""
    ## ℹ️ About
    **Anomaly Detective** helps identify suspicious transactions in financial data using AI.
    
    Version 2.0 | © 2025
    """)

# ---------- Main Content ----------
# Title Section with dynamic style based on dark mode
st.markdown("<div class='title'>🔍 Anomaly Detective</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Advanced AI-powered platform for financial fraud detection</div>", unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Detection Dashboard", "📈 Analytics", "📋 Documentation"])

with tab1:
    # File uploader in a card-like container
    st.markdown("<div class='report-section'>", unsafe_allow_html=True)
    st.markdown("<div class='report-title'>📤 Upload Data</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file with transaction data", type=["csv"])
    default_file_path = 'sample.csv'
    
    use_default = st.checkbox("Use sample dataset instead", value=True if not uploaded_file else False)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Load data
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    elif use_default and os.path.exists(default_file_path):
        parser = argparse.ArgumentParser()
        parser.add_argument('--csv', type=str, default='sample.csv', help='sample.csv')
        args = parser.parse_args()

        df = pd.read_csv(args.csv)
        st.info("ℹ️ Using sample dataset for demonstration.")
    else:
        st.warning("📂 Please upload a CSV file or use the sample dataset.")

    if df is not None:
        # Data preview
        st.markdown("<div class='report-section'>", unsafe_allow_html=True)
        st.markdown("<div class='report-title'>📊 Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        
        # Data statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
        with col2:
            if 'transaction_amount' in df.columns:
                st.metric("Total Amount", f"${df['transaction_amount'].sum():,.2f}")
        with col3:
            if 'transaction_date' in df.columns:
                dates = pd.to_datetime(df['transaction_date'], errors='coerce')
                date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature selection
        st.markdown("<div class='report-section'>", unsafe_allow_html=True)
        st.markdown("<div class='report-title'>⚙️ Configure Detection</div>", unsafe_allow_html=True)
        numeric_columns = df.select_dtypes(include='number').columns.tolist()
        selected_features = st.multiselect(
            "Select features for anomaly detection",
            options=numeric_columns,
            default=numeric_columns[:min(3, len(numeric_columns))],
            help="Choose numeric columns that might indicate anomalies"
        )
        
        # Run detection button
        if selected_features:
            run_col1, run_col2 = st.columns([1, 3])
            with run_col1:
                run_button = st.button("🚀 Run Detection", use_container_width=True)
            with run_col2:
                st.markdown(f"Using **{model_choice}** with {len(selected_features)} features")
        else:
            st.warning("⚠️ Please select at least one feature for anomaly detection")
            run_button = False
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Run anomaly detection
        if run_button and selected_features:
            with st.spinner("🔍 Hunting for anomalies..."):
                true_labels = None
                if 'true_label' in df.columns:
                    true_labels = df['true_label']
                
                if model_choice == "Isolation Forest":
                    df_result, anomaly_scores, metrics = run_isolation_forest(
                        df[selected_features], 
                        true_labels,
                        contamination=model_params["contamination"]
                    )
                else:
                    df_result, anomaly_scores, metrics = run_autoencoder(
                        df[selected_features], 
                        true_labels,
                        threshold=model_params["threshold"],
                        epochs=model_params["epochs"]
                    )
                
                # Add scores back to the original dataframe
                df['anomaly_score'] = anomaly_scores
                df['status'] = df['anomaly_score'].map({0: 'Normal', 1: 'Anomaly'})
                
                # Set session state
                st.session_state.last_run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Results Dashboard
                st.markdown("<div class='report-section'>", unsafe_allow_html=True)
                st.markdown("<div class='report-title'>🎯 Detection Results</div>", unsafe_allow_html=True)
                
                # Summary metrics in cards
                anomaly_count = df['anomaly_score'].sum()
                anomaly_percentage = (anomaly_count / len(df)) * 100
                
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{anomaly_count}</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Anomalies Detected</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with metric_cols[1]:
                    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-value'>{anomaly_percentage:.1f}%</div>", unsafe_allow_html=True)
                    st.markdown("<div class='metric-label'>Anomaly Rate</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                if metrics:
                    with metric_cols[2]:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-value'>{metrics['precision']:.2f}</div>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-label'>Precision</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    with metric_cols[3]:
                        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='metric-value'>{metrics['recall']:.2f}</div>", unsafe_allow_html=True)
                        st.markdown("<div class='metric-label'>Recall</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                
                # Primary visualization with enhanced styling
                color_scheme = px.colors.sequential.Plasma if st.session_state.dark_mode else px.colors.sequential.Blues
                
                st.markdown("### 📊 Anomaly Detection Visualization")
                
                if len(selected_features) >= 2:
                    # Create 3D scatter if we have 3 or more features
                    if len(selected_features) >= 3:
                        fig = px.scatter_3d(
                            df,
                            x=selected_features[0],
                            y=selected_features[1],
                            z=selected_features[2],
                            color='status',
                            size_max=10,
                            opacity=0.7,
                            color_discrete_map={'Normal': '#0077b6', 'Anomaly': '#ef476f'},
                            title="3D Visualization of Transaction Anomalies"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 2D scatter with additional features
                    fig = px.scatter(
                        df,
                        x=selected_features[0],
                        y=selected_features[1],
                        color='status',
                        size=selected_features[0] if len(selected_features) > 2 else None,
                        hover_data=selected_features,
                        color_discrete_map={'Normal': '#0077b6', 'Anomaly': '#ef476f'},
                        title="Transaction Anomalies"
                    )
                    
                    # Update plot style based on dark mode
                    if st.session_state.dark_mode:
                        fig.update_layout(
                            plot_bgcolor='rgba(30,30,30,1)',
                            paper_bgcolor='rgba(30,30,30,1)',
                            font=dict(color='white')
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly distribution
                st.markdown("### 📊 Anomaly Distribution")
                count_df = df['status'].value_counts().reset_index()
                count_df.columns = ['Status', 'Count']
                
                fig_pie = px.pie(
                    count_df, 
                    values='Count', 
                    names='Status',
                    color='Status',
                    color_discrete_map={'Normal': '#0077b6', 'Anomaly': '#ef476f'},
                    hole=0.4,
                    title="Distribution of Normal vs Anomalous Transactions"
                )
                
                # Update plot style based on dark mode
                if st.session_state.dark_mode:
                    fig_pie.update_layout(
                        plot_bgcolor='rgba(30,30,30,1)',
                        paper_bgcolor='rgba(30,30,30,1)',
                        font=dict(color='white')
                    )
                
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Time series visualization if available
                if 'transaction_date' in df.columns:
                    try:
                        st.markdown("### 📈 Time Series Analysis")
                        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
                        time_df = df.dropna(subset=['transaction_date'])
                        
                        # Group by date and count anomalies
                        daily_anomalies = time_df.groupby([
                            pd.Grouper(key='transaction_date', freq='D'),
                            'status'
                        ]).size().unstack(fill_value=0).reset_index()
                        
                        if 'Anomaly' in daily_anomalies.columns:
                            fig_time = go.Figure()
                            fig_time.add_trace(go.Scatter(
                                x=daily_anomalies['transaction_date'],
                                y=daily_anomalies['Normal'],
                                name='Normal',
                                line=dict(color='#0077b6', width=2),
                                mode='lines'
                            ))
                            fig_time.add_trace(go.Scatter(
                                x=daily_anomalies['transaction_date'],
                                y=daily_anomalies['Anomaly'],
                                name='Anomaly',
                                line=dict(color='#ef476f', width=2),
                                mode='lines+markers',
                                marker=dict(size=8)
                            ))
                            
                            fig_time.update_layout(
                                title="Transaction Patterns Over Time",
                                xaxis_title="Date",
                                yaxis_title="Number of Transactions",
                                legend_title="Transaction Type",
                                hovermode="x unified"
                            )
                            
                            # Update plot style based on dark mode
                            if st.session_state.dark_mode:
                                fig_time.update_layout(
                                    plot_bgcolor='rgba(30,30,30,1)',
                                    paper_bgcolor='rgba(30,30,30,1)',
                                    font=dict(color='white')
                                )
                            
                            st.plotly_chart(fig_time, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in time series visualization: {e}")
                
                # Detailed anomaly table
                st.markdown("### 🧾 Detailed Anomaly Report")
                anomaly_df = df[df['anomaly_score'] == 1]
                if not anomaly_df.empty:
                    st.dataframe(anomaly_df, use_container_width=True)
                    
                    # Download report button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.markdown(
                        download_link(anomaly_df, f"anomaly_report_{timestamp}.csv", "📥 Download Anomaly Report"),
                        unsafe_allow_html=True
                    )
                else:
                    st.info("No anomalies detected in the selected features.")
                
                # Performance metrics detail
                if metrics:
                    st.markdown("### 📊 Model Performance Details")
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    with metrics_cols[1]:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                    with metrics_cols[2]:
                        st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                    with metrics_cols[3]:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Closing Summary
                st.markdown("<div class='report-section'>", unsafe_allow_html=True)
                st.markdown("<div class='report-title'>✅ Analysis Summary</div>", unsafe_allow_html=True)
                
                st.markdown(f"""
                The anomaly detection process has successfully completed at {st.session_state.last_run_date}.
                
                **Key Findings:**
                - Found {anomaly_count} potential anomalies ({anomaly_percentage:.1f}% of all transactions)
                - Used {model_choice} algorithm with {len(selected_features)} selected features
                - {', '.join(selected_features)} were the most important in detecting anomalies
                
                **Next Steps:**
                - Review the flagged transactions in detail
                - Consider adjusting model parameters for better precision
                - Try different feature combinations to improve detection
                """)
                
                # Full report download
                st.markdown(
                    download_link(df, f"full_anomaly_report_{timestamp}.csv", "📥 Download Complete Analysis"),
                    unsafe_allow_html=True
                )
                st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("## 📈 Analytics Dashboard")
    st.info("Upload data and run detection to see detailed analytics here.")
    
    if 'anomaly_score' in df.columns if df is not None else False:
        # Feature importance analysis
        st.markdown("### 🔍 Feature Analysis")
        
        # Simple correlation heatmap for selected features
        if len(selected_features) > 1:
            corr = df[selected_features].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Feature Correlation Matrix"
            )
            
            # Update plot style based on dark mode
            if st.session_state.dark_mode:
                fig_corr.update_layout(
                    plot_bgcolor='rgba(30,30,30,1)',
                    paper_bgcolor='rgba(30,30,30,1)',
                    font=dict(color='white')
                )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature distribution comparison
        st.markdown("### 📊 Feature Distribution Analysis")
        feature_to_analyze = st.selectbox("Select feature to analyze:", selected_features)
        
        if feature_to_analyze:
            fig_hist = px.histogram(
                df,
                x=feature_to_analyze,
                color='status',
                marginal="box",
                color_discrete_map={'Normal': '#0077b6', 'Anomaly': '#ef476f'},
                title=f"Distribution of {feature_to_analyze} by Transaction Status",
                opacity=0.7,
                barmode='overlay'
            )
            
            # Update plot style based on dark mode
            if st.session_state.dark_mode:
                fig_hist.update_layout(
                    plot_bgcolor='rgba(30,30,30,1)',
                    paper_bgcolor='rgba(30,30,30,1)',
                    font=dict(color='white')
                )
                
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Show statistics
            st.markdown("### 📊 Feature Statistics")
            stat_cols = st.columns(2)
            
            normal_stats = df[df['anomaly_score'] == 0][feature_to_analyze].describe()
            anomaly_stats = df[df['anomaly_score'] == 1][feature_to_analyze].describe()
            
            with stat_cols[0]:
                st.markdown("#### Normal Transactions")
                st.dataframe(normal_stats)
            
            with stat_cols[1]:
                st.markdown("#### Anomalous Transactions")
                st.dataframe(anomaly_stats)

with tab3:
    st.markdown("## 📋 Documentation")
    
    st.markdown("""
    ### 🔍 What is Anomaly Detective?
    
    Anomaly Detective is an advanced tool for detecting unusual patterns or outliers in financial transaction data. It uses machine learning algorithms to identify potentially fraudulent activities.
    
    ### 🧠 Available Models
    
    1. **Isolation Forest**
       - Works by isolating observations by randomly selecting a feature and then randomly selecting a split value
       - Effective for outlier detection in high-dimensional datasets
       - Parameters:
         - Contamination: Expected proportion of anomalies in the dataset
         - n_estimators: Number of isolation trees to build
    
    2. **Autoencoder (PyTorch)**
       - Neural network that learns to encode and decode data, with anomalies identified through reconstruction error
       - Better for capturing complex patterns and relationships
       - Parameters:
         - Threshold: Reconstruction error threshold for anomaly classification
         - Epochs: Number of training cycles
    
    ### 📊 How to Use
    
    1. Upload your transaction data (CSV format)
    2. Select features for analysis
    3. Choose a detection algorithm
    4. Configure model parameters
    5. Run detection
    6. Review results and download reports
    
    ### 📈 Understanding Results
    
    - Transactions flagged as anomalies have an anomaly score of 1
    - Higher precision indicates fewer false positives
    - Higher recall indicates fewer missed anomalies
    
    ### 🔄 Best Practices
    
    - 
    """)

    st.markdown("### 🧩 Sample Code")
    
    st.code("""
    # Example usage in Python
    import pandas as pd
    from models.isolation_forest import run_isolation_forest
    
    # Load data
    data = pd.read_csv('transactions.csv')
    
    # Select features
    features = ['transaction_amount', 'account_balance']
    
    # Run detection
    result, anomaly_scores, metrics = run_isolation_forest(
        data[features], 
        contamination=0.05
    )
    
    # Get anomalies
    anomalies = data[anomaly_scores == 1]
    print(f"Found {len(anomalies)} anomalies")
    """, language="python")
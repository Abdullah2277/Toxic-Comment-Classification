import streamlit as st
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Define label categories
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Label descriptions
LABEL_DESCRIPTIONS = {
    'toxic': 'General toxic or rude language',
    'severe_toxic': 'Extremely hateful or aggressive content',
    'obscene': 'Contains obscene or vulgar language',
    'threat': 'Contains threats or intimidation',
    'insult': 'Insulting or demeaning language',
    'identity_hate': 'Hate speech targeting identity groups'
}

# Label colors
LABEL_COLORS = {
    'toxic': '#ff6b6b',
    'severe_toxic': '#c92a2a',
    'obscene': '#ff8787',
    'threat': '#fa5252',
    'insult': '#ff922b',
    'identity_hate': '#e03131'
}

@st.cache_resource
def load_model_and_tokenizer(model_name="SyedFarhan110/toxic-comment-classifier"):
    """Load the trained model and tokenizer from Hugging Face Hub or local path"""
    try:
        # Try to load from Hugging Face Hub first (for deployment)
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            st.success(f"✅ Loaded model from Hugging Face: {model_name}")
        except:
            # Fallback to local path
            local_path = "./results/results/checkpoint-best"
            if Path(local_path).exists():
                model = AutoModelForSequenceClassification.from_pretrained(local_path)
                tokenizer = AutoTokenizer.from_pretrained(local_path)
                st.success(f"✅ Loaded model from local path: {local_path}")
            else:
                # Ultimate fallback to base model
                model = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased",
                    num_labels=len(LABEL_COLS),
                    problem_type="multi_label_classification"
                )
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                st.warning("⚠️ Using base DistilBERT model (untrained)")
        
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_text(text, model, tokenizer, device):
    """Make prediction on input text"""
    # Tokenize input
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Binary predictions (threshold 0.5)
    preds = (probs >= 0.5).astype(int)

    # Prepare results
    results = {LABEL_COLS[i]: float(probs[i]) for i in range(len(LABEL_COLS))}
    binary_results = {LABEL_COLS[i]: int(preds[i]) for i in range(len(LABEL_COLS))}

    return results, binary_results

def create_probability_chart(results):
    """Create a bar chart showing probabilities"""
    df = pd.DataFrame({
        'Category': list(results.keys()),
        'Probability': list(results.values())
    })
    df = df.sort_values('Probability', ascending=True)
    
    colors = [LABEL_COLORS[label] for label in df['Category']]
    
    fig = go.Figure(data=[
        go.Bar(
            y=df['Category'],
            x=df['Probability'],
            orientation='h',
            marker=dict(color=colors),
            text=[f'{p:.1%}' for p in df['Probability']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Toxicity Prediction Probabilities",
        xaxis_title="Probability",
        yaxis_title="Category",
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 1], tickformat='.0%')
    )
    
    return fig

def create_gauge_chart(overall_score):
    """Create a gauge chart for overall toxicity"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=overall_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Overall Toxicity Score", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def batch_predict(texts, model, tokenizer, device):
    """Predict multiple texts"""
    all_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, text in enumerate(texts):
        results, binary_results = predict_text(text, model, tokenizer, device)
        all_results.append({
            'text': text[:100] + '...' if len(text) > 100 else text,
            **results,
            'overall_toxic': max(results.values())
        })
        
        progress = (idx + 1) / len(texts)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {idx + 1}/{len(texts)}")
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(all_results)

# Main App
def main():
    # Header
    st.title("🛡️ Toxic Comment Classifier")
    st.markdown("### Multi-label classification for online toxicity detection")
    
    # Sidebar
    # with st.sidebar:
        # st.header("⚙️ Settings")
        
      
        
        # st.markdown("---")
        
        # st.header("📊 About")
        # st.markdown("""
        # This application uses a fine-tuned DistilBERT model to classify toxic comments across 6 categories:
        # """)
        
        # for label, desc in LABEL_DESCRIPTIONS.items():
        #     st.markdown(f"**{label.replace('_', ' ').title()}**: {desc}")
        
        # st.markdown("---")
        # st.markdown("**Model**: DistilBERT (base-uncased)")
        # st.markdown(f"**Device**: {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}")
    # Load model from Hugging Face Hub (change this to your username/model-name)
    model_name = "SyedFarhan110/toxic-comment-classifier"
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📋 Batch Prediction", "📈 Model Info"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Analyze a Single Comment")
        
        # Example texts
        examples = {
            "Neutral": "This is a well-written article with great insights.",
            "Mildly Toxic": "This is stupid and a waste of time.",
            "Toxic": "You are such an idiot, stop posting nonsense!",
            "Severely Toxic": "I hope you die, you worthless piece of garbage!"
        }
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("**Try examples:**")
            for ex_name, ex_text in examples.items():
                if st.button(ex_name, key=f"ex_{ex_name}"):
                    st.session_state.input_text = ex_text
        
        with col1:
            input_text = st.text_area(
                "Enter text to analyze:",
                value=st.session_state.get('input_text', ''),
                height=150,
                placeholder="Type or paste a comment here..."
            )
        
        if st.button("🔍 Analyze", type="primary", use_container_width=True):
            if input_text.strip():
                with st.spinner("Analyzing..."):
                    results, binary_results = predict_text(input_text, model, tokenizer, device)
                    
                    # Overall assessment
                    overall_score = max(results.values())
                    is_toxic = any(binary_results.values())
                    
                    st.markdown("---")
                    
                    # Display overall result
                    col1, col2, col3 = st.columns([2, 2, 3])
                    
                    with col1:
                        if is_toxic:
                            st.error("⚠️ **TOXIC CONTENT DETECTED**")
                        else:
                            st.success("✅ **CONTENT APPEARS SAFE**")
                    
                    with col2:
                        st.metric("Overall Toxicity", f"{overall_score:.1%}")
                    
                    with col3:
                        detected_labels = [label.replace('_', ' ').title() for label, val in binary_results.items() if val == 1]
                        if detected_labels:
                            st.warning(f"**Detected:** {', '.join(detected_labels)}")
                    
                    # Visualizations
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig = create_probability_chart(results)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        gauge = create_gauge_chart(overall_score)
                        st.plotly_chart(gauge, use_container_width=True)
                    
                    # Detailed results
                    st.markdown("### 📊 Detailed Results")
                    
                    for label in LABEL_COLS:
                        prob = results[label]
                        binary = binary_results[label]
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{label.replace('_', ' ').title()}**")
                            st.caption(LABEL_DESCRIPTIONS[label])
                        
                        with col2:
                            st.metric("Probability", f"{prob:.1%}")
                        
                        with col3:
                            if binary == 1:
                                st.error("DETECTED")
                            else:
                                st.success("Clear")
            else:
                st.warning("Please enter some text to analyze.")
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Analyze Multiple Comments")
        
        upload_option = st.radio(
            "Choose input method:",
            ["Upload CSV", "Paste Text"]
        )
        
        if upload_option == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload a CSV file with comments",
                type=['csv'],
                help="CSV should have a column named 'comment_text'"
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                if 'comment_text' in df.columns:
                    st.info(f"📄 Loaded {len(df)} comments")
                    
                    if st.button("🔍 Analyze All", type="primary"):
                        results_df = batch_predict(df['comment_text'].tolist(), model, tokenizer, device)
                        
                        st.success("✅ Analysis complete!")
                        
                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            toxic_count = (results_df['toxic'] > 0.5).sum()
                            st.metric("Toxic Comments", toxic_count)
                        
                        with col2:
                            avg_toxic = results_df['overall_toxic'].mean()
                            st.metric("Avg Toxicity", f"{avg_toxic:.1%}")
                        
                        with col3:
                            max_toxic = results_df['overall_toxic'].max()
                            st.metric("Max Toxicity", f"{max_toxic:.1%}")
                        
                        with col4:
                            safe_count = (results_df['overall_toxic'] < 0.5).sum()
                            st.metric("Safe Comments", safe_count)
                        
                        # Display results
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "toxicity_results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                else:
                    st.error("CSV must contain a 'comment_text' column")
        
        else:  # Paste Text
            batch_text = st.text_area(
                "Enter multiple comments (one per line):",
                height=200,
                placeholder="Comment 1\nComment 2\nComment 3\n..."
            )
            
            if st.button("🔍 Analyze All", type="primary"):
                if batch_text.strip():
                    texts = [line.strip() for line in batch_text.split('\n') if line.strip()]
                    
                    results_df = batch_predict(texts, model, tokenizer, device)
                    
                    st.success("✅ Analysis complete!")
                    
                    # Display results
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "toxicity_results.csv",
                        "text/csv",
                        key='download-csv-text'
                    )
                else:
                    st.warning("Please enter some text to analyze.")
    
    # Tab 3: Model Info
    with tab3:
        st.header("📈 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            st.markdown("""
            - **Base Model**: DistilBERT (distilbert-base-uncased)
            - **Task**: Multi-label Text Classification
            - **Number of Labels**: 6
            - **Max Sequence Length**: 128 tokens
            - **Loss Function**: Binary Cross-Entropy
            """)
            
            st.subheader("Prediction Threshold")
            st.markdown("""
            - **Default Threshold**: 0.5 (50%)
            - Probabilities ≥ 0.5 are classified as positive
            - Probabilities < 0.5 are classified as negative
            """)
        
        with col2:
            st.subheader("Label Categories")
            for label, desc in LABEL_DESCRIPTIONS.items():
                with st.expander(f"**{label.replace('_', ' ').title()}**"):
                    st.markdown(desc)
            
            st.subheader("Performance Tips")
            st.markdown("""
            - Use GPU for faster inference
            - Batch multiple predictions for efficiency
            - Consider fine-tuning on domain-specific data
            - Monitor false positives/negatives for threshold adjustment
            """)
        
        st.markdown("---")
        st.markdown("### 🔧 Technical Details")
        
        info_data = {
            "Parameter": [
                "PyTorch Version",
                "Transformers Version",
                "Device",
                "Model Parameters",
                "Precision"
            ],
            "Value": [
                torch.__version__,
                "Latest",
                str(device),
                f"{sum(p.numel() for p in model.parameters()):,}",
                "FP32" if not torch.cuda.is_available() else "FP16/FP32"
            ]
        }
        
        st.table(pd.DataFrame(info_data))

if __name__ == "__main__":
    main()

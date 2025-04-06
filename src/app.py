import streamlit as st
import pandas as pd
import os
import sys
import nltk
from io import StringIO
import altair as alt
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import numpy as np
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from src.data_loading import load_dataset, get_available_datasets, preprocess_text, normalize_sentiment_labels
from src.model_training import (
    train_model, evaluate_model, predict_sentiment, get_available_models, 
    get_feature_importance, predict_sentiment_with_probability, save_model,
    load_model, get_saved_models
)
from src.chunked_processing import (
    detect_columns, process_large_file, predict_batch, 
    save_chunked_model, load_chunked_model, get_chunked_models
)
from src.utils import (
    safe_display_dataframe, deduplicate_column_names, get_file_info,
    format_file_size, estimate_memory_usage
)

# Download NLTK dependencies
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def plot_sentiment_distribution(df, sentiment_column):
    """
    Plot the distribution of sentiment labels in the dataset.
    """
    # Check if the sentiment column exists in the dataframe
    if sentiment_column not in df.columns:
        # Create a figure with warning message
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"Error: Column '{sentiment_column}' not found in the dataset", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='red')
        ax.axis('off')
        return fig
    
    # Count sentiments
    sentiment_counts = df[sentiment_column].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    # Create a pie chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Pie chart
    ax1.pie(sentiment_counts['Count'], labels=sentiment_counts['Sentiment'], autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax1.axis('equal')
    ax1.set_title('Sentiment Distribution (Pie Chart)')
    
    # Bar chart
    ax2.bar(sentiment_counts['Sentiment'], sentiment_counts['Count'])
    ax2.set_title('Sentiment Distribution (Bar Chart)')
    ax2.set_xlabel('Sentiment')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    return fig

def generate_wordcloud(df, text_column, sentiment_column, sentiment_value):
    """
    Generate a word cloud for a specific sentiment.
    """
    # Check if the required columns exist in the dataframe
    if text_column not in df.columns or sentiment_column not in df.columns:
        # Create a figure with warning message
        fig, ax = plt.subplots(figsize=(10, 5))
        missing_columns = []
        if text_column not in df.columns:
            missing_columns.append(text_column)
        if sentiment_column not in df.columns:
            missing_columns.append(sentiment_column)
        ax.text(0.5, 0.5, f"Error: Column(s) {missing_columns} not found in the dataset", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='red')
        ax.axis('off')
        return fig
    
    # Filter the dataframe for the specific sentiment
    filtered_df = df[df[sentiment_column] == sentiment_value]
    
    # If no data for this sentiment, return a figure with a message
    if filtered_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"No data found for sentiment: {sentiment_value}", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='orange')
        ax.axis('off')
        return fig
    
    # Combine all reviews for the sentiment
    all_text = ' '.join(filtered_df[text_column].tolist())
    
    # If there's no text, return a figure with a message
    if not all_text.strip():
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, f"No text data found for sentiment: {sentiment_value}", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, color='orange')
        ax.axis('off')
        return fig
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
    
    # Create figure and plot word cloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud for {sentiment_value.capitalize()} Reviews')
    
    return fig

def plot_confusion_matrix(matrix, labels):
    """
    Plot a confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')
    
    # Set labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    
    ax.set_title("Confusion Matrix")
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.tight_layout()
    
    return fig

def plot_important_features(feature_importance, title):
    """
    Plot the most important features for a class.
    """
    # Convert to DataFrame
    features_df = pd.DataFrame(feature_importance, columns=['Feature', 'Importance'])
    features_df = features_df.sort_values('Importance', ascending=False).head(15)
    
    # Create Altair chart
    chart = alt.Chart(features_df).mark_bar().encode(
        x='Importance',
        y=alt.Y('Feature', sort='-x'),
        tooltip=['Feature', 'Importance']
    ).properties(
        title=title,
        width=600,
        height=400
    )
    
    return chart

def plot_model_comparison(metrics_dict):
    """
    Plot a comparison of model performance metrics.
    """
    # Convert the metrics dictionary to a DataFrame
    models = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model_name, metrics in metrics_dict.items():
        models.append(model_name)
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1_score'])
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    # Melt the DataFrame for easier plotting
    df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Score')
    
    # Create an Altair chart
    chart = alt.Chart(df_melted).mark_bar().encode(
        x='Model',
        y='Score',
        color='Metric',
        column='Metric'
    ).properties(
        width=150,
        height=300
    )
    
    return chart

def main():
    # Set page title and description
    st.title("Product Review Sentiment Analysis")
    st.write("Analyze sentiment in product reviews using machine learning.")
    
    # Check for large dataset warning
    if 'large_dataset_warning_shown' in st.session_state and st.session_state['large_dataset_warning_shown']:
        st.warning(f"⚠️ You have a large dataset ({st.session_state['large_dataset_name']}, {st.session_state['large_dataset_size']:.1f}MB). "
                   f"Please use the **Large File Processing** page for better performance.")
        # Automatically select the Large File Processing page
        default_page = "Large File Processing"
    else:
        default_page = "Data Analysis"
    
    # Create sidebar for dataset selection
    with st.sidebar:
        st.header("Dataset Selection")
        
        # Option to upload a dataset
        uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'])
        
        # Option to select from available datasets
        st.write("OR")
        
        # Get available datasets
        dataset_paths = get_available_datasets()
        dataset_names = [os.path.basename(path) for path in dataset_paths]
        
        if dataset_names:
            selected_dataset = st.selectbox(
                "Select a dataset from available options",
                [""] + dataset_names
            )
        else:
            st.info("No datasets found in the 'datasets/' directory.")
            selected_dataset = ""
        
        # Navigation menu
        st.header("Navigation")
        page = st.radio(
            "Go to",
            ["Data Analysis", "Model Training", "Model Comparison", "Prediction", "Model Management", "Large File Processing"],
            index=5 if default_page == "Large File Processing" else 0
        )

    # Main area for dataset preview and model training
    if uploaded_file is not None:
        # Load uploaded dataset
        dataset_content = uploaded_file.getvalue().decode('utf-8')
        df = pd.read_csv(StringIO(dataset_content))
        st.session_state['dataset_source'] = "Uploaded"
        st.session_state['df'] = df
    elif selected_dataset:
        # Load selected dataset
        dataset_path = os.path.join("datasets", selected_dataset)
        df = load_dataset(dataset_path)
        st.session_state['dataset_source'] = f"Selected: {selected_dataset}"
        st.session_state['df'] = df
    else:
        st.info("Please upload a CSV file or select an available dataset.")
        if 'df' not in st.session_state:
            return
        df = st.session_state['df']
    
    # If a dataset is loaded, proceed with analysis
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Display dataset source and preview
        st.subheader(f"Dataset: {st.session_state.get('dataset_source', 'Preview')}")
        # Use safe display function instead of direct write
        safe_display_dataframe(df, st, max_rows=5)
        
        # Column selection
        st.subheader("Column Selection")
        text_columns = df.columns.tolist()
        
        # Get default column guesses based on common column names
        default_text_col = next((col for col in ['review_text', 'review', 'text', 'comment', 'Review'] 
                                 if col in text_columns), text_columns[0])
        default_sentiment_col = next((col for col in ['sentiment', 'label', 'class', 'Sentiment'] 
                                      if col in text_columns), 
                                      text_columns[0] if len(text_columns) == 1 else text_columns[1])
        
        # Select text column
        text_column = st.selectbox(
            "Select the column containing review text",
            text_columns,
            index=text_columns.index(default_text_col)
        )
        
        # Select sentiment column
        sentiment_column = st.selectbox(
            "Select the column containing sentiment labels",
            text_columns,
            index=text_columns.index(default_sentiment_col)
        )
        
        # Validate column selections
        if text_column == sentiment_column:
            st.error("Error: Text column and sentiment column cannot be the same.")
        elif df[text_column].dtype != 'object':
            st.warning(f"Warning: The selected text column '{text_column}' may not contain text data.")
        
        # Show data quality metrics
        with st.expander("Data Quality Check"):
            # Check for missing values
            missing_text = df[text_column].isna().sum()
            missing_sentiment = df[sentiment_column].isna().sum()
            
            st.write(f"Missing values in text column: {missing_text} ({missing_text/len(df):.1%})")
            st.write(f"Missing values in sentiment column: {missing_sentiment} ({missing_sentiment/len(df):.1%})")
            
            # Check sentiment distribution
            sentiment_counts = df[sentiment_column].value_counts()
            st.write("Sentiment distribution before normalization:")
            st.write(sentiment_counts)
            
            # Check text length distribution
            df['text_length'] = df[text_column].astype(str).apply(len)
            
            # Display text length statistics
            text_length_stats = df['text_length'].describe().to_dict()
            st.write("Text length statistics:")
            st.write(f"Min: {text_length_stats['min']:.0f}, Mean: {text_length_stats['mean']:.0f}, Max: {text_length_stats['max']:.0f}")
            
            # Identify very short texts that might be problematic
            short_texts = df[df['text_length'] < 5]
            if not short_texts.empty:
                st.warning(f"Found {len(short_texts)} very short texts (less than 5 characters). These might cause issues during analysis.")
                
                # Use safe display function for short texts
                if text_column != sentiment_column:
                    # Prepare a view with just the columns we want to show
                    display_df = short_texts[[text_column, sentiment_column, 'text_length']]
                    safe_display_dataframe(display_df, st, max_rows=5)
                else:
                    safe_display_dataframe(short_texts, st, max_rows=5)
        
        # Store column selections in session state
        if 'text_column' not in st.session_state or st.session_state['text_column'] != text_column:
            st.session_state['text_column'] = text_column
        
        if 'sentiment_column' not in st.session_state or st.session_state['sentiment_column'] != sentiment_column:
            st.session_state['sentiment_column'] = sentiment_column
        
        # Text preprocessing options
        st.subheader("Text Preprocessing Options")
        
        # Create two columns for preprocessing options
        prep_col1, prep_col2 = st.columns(2)
        
        with prep_col1:
            remove_stopwords = st.checkbox("Remove stopwords", value=True)
            perform_stemming = st.checkbox("Perform stemming", value=False)
        
        with prep_col2:
            perform_lemmatization = st.checkbox("Perform lemmatization", value=True, 
                                              help="Converts words to their base form (e.g., 'running' → 'run'). Takes precedence over stemming.")
            handle_negations = st.checkbox("Handle negations", value=True,
                                         help="Preserves negations in the text (e.g., 'not good' → 'not_good')")
        
        # Display warning if both stemming and lemmatization are selected
        if perform_stemming and perform_lemmatization:
            st.info("Both stemming and lemmatization are selected. Lemmatization will be used.")
        
        # Data Analysis Page
        if page == "Data Analysis":
            st.header("Exploratory Data Analysis")
            
            # Only proceed if text and sentiment columns are selected and different
            if text_column == sentiment_column:
                st.error("Error: Text column and sentiment column cannot be the same.")
                return
            
            # Verify that the selected columns exist
            missing_columns = []
            if text_column not in df.columns:
                missing_columns.append(text_column)
            if sentiment_column not in df.columns:
                missing_columns.append(sentiment_column)
            
            if missing_columns:
                st.error(f"Error: The following selected columns do not exist in the dataset: {', '.join(missing_columns)}")
                st.info("Please select valid columns from the dropdowns above.")
                return
            
            # Show a warning if the text column might not contain text
            if df[text_column].dtype != 'object':
                st.warning(f"Warning: The selected text column '{text_column}' may not contain text data.")
            
            # Preprocess the data if not already done
            if 'df_processed' not in st.session_state:
                with st.spinner("Processing data..."):
                    # Preprocess the text data
                    df['processed_text'] = df[text_column].apply(
                        lambda x: preprocess_text(x, 
                                               remove_stopwords=remove_stopwords, 
                                               perform_stemming=perform_stemming,
                                               perform_lemmatization=perform_lemmatization,
                                               handle_negations=handle_negations)
                    )
                    
                    # Normalize sentiment labels
                    df = normalize_sentiment_labels(df, sentiment_column)
                    
                    # Check if we have enough data for each sentiment class
                    sentiment_counts = df[sentiment_column].value_counts()
                    if sentiment_counts.min() < 2:
                        st.error(f"The smallest sentiment class ({sentiment_counts.idxmin()}) has only {sentiment_counts.min()} samples. Each class needs at least 2 samples.")
                        return
                    
                    # Store in session state
                    st.session_state['df_processed'] = df
            else:
                df = st.session_state['df_processed']
            
            # Sentiment distribution
            st.subheader("Sentiment Distribution")
            fig = plot_sentiment_distribution(df, sentiment_column)
            st.pyplot(fig)
            
            # Word clouds
            st.subheader("Word Clouds by Sentiment")
            sentiment_list = df[sentiment_column].unique().tolist()
            
            if sentiment_list:
                # Create tabs for different sentiments
                sentiment_tabs = st.tabs(sentiment_list)
                
                for i, sentiment in enumerate(sentiment_list):
                    with sentiment_tabs[i]:
                        if len(df[df[sentiment_column] == sentiment]) > 0:
                            fig = generate_wordcloud(df, 'processed_text', sentiment_column, sentiment)
                            st.pyplot(fig)
                            
                            # Also show most frequent words
                            st.subheader(f"Most Frequent Words in {sentiment.capitalize()} Reviews")
                            sentiment_texts = df[df[sentiment_column] == sentiment]['processed_text']
                            all_words = ' '.join(sentiment_texts).split()
                            word_counts = Counter(all_words).most_common(15)
                            
                            # Create a DataFrame and display as a bar chart
                            word_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])
                            st.bar_chart(word_df.set_index('Word'))
                        else:
                            st.info(f"No reviews with {sentiment} sentiment found.")
            else:
                st.warning("No sentiment categories found in the data.")
        
        # Model Training Page
        elif page == "Model Training":
            st.header("Model Training")
            
            # Only proceed if text and sentiment columns are selected and different
            if text_column == sentiment_column:
                st.error("Error: Text column and sentiment column cannot be the same.")
                return
            
            # Model selection
            model_names = list(get_available_models().keys())
            model_name = st.selectbox("Select a model", model_names)
            
            # Binary classification option
            binary_sentiments = st.checkbox("Use only positive/negative sentiments", value=True)
            
            # Hyperparameter tuning option
            tune_hyperparameters = st.checkbox("Perform hyperparameter tuning", value=False)
            
            if tune_hyperparameters:
                st.info("Hyperparameter tuning may take some time to complete.")
            
            # Save model option
            save_model_option = st.checkbox("Save model after training", value=False)
            
            # Train model button
            if st.button("Train Model"):
                with st.spinner("Processing data and training model..."):
                    # Preprocess the text data
                    df['processed_text'] = df[text_column].apply(
                        lambda x: preprocess_text(x, 
                                               remove_stopwords=remove_stopwords, 
                                               perform_stemming=perform_stemming,
                                               perform_lemmatization=perform_lemmatization,
                                               handle_negations=handle_negations)
                    )
                    
                    # Normalize sentiment labels
                    df = normalize_sentiment_labels(df, sentiment_column)
                    
                    # Filter out rows with 'neutral' sentiment if we want binary classification
                    if binary_sentiments:
                        df_filtered = df[df[sentiment_column].isin(['positive', 'negative'])]
                        if len(df_filtered) == 0:
                            st.error("No rows with positive/negative sentiment found after filtering.")
                            return
                        df = df_filtered
                    
                    # Make sure we have enough samples per class
                    class_counts = df[sentiment_column].value_counts()
                    min_samples = class_counts.min()
                    if min_samples < 2:
                        st.error(f"The smallest class ({class_counts.idxmin()}) has only {min_samples} sample. Each class needs at least 2 samples.")
                        return
                    
                    # Train the model with hyperparameter tuning if selected
                    try:
                        model_result = train_model(
                            df, 'processed_text', sentiment_column, model_name, 
                            tune_hyperparameters=tune_hyperparameters
                        )
                        
                        # Unpack results (model_result is now a tuple with 6 elements)
                        model, X_train, X_test, y_train, y_test, best_params = model_result
                        
                        # Evaluate the model
                        metrics = evaluate_model(model, X_test, y_test)
                        
                        # Get feature importance
                        feature_importance = get_feature_importance(model)
                        
                        # Store model and processed data in session state
                        if 'models' not in st.session_state:
                            st.session_state['models'] = {}
                        
                        model_info = {
                            'model': model,
                            'metrics': metrics,
                            'feature_importance': feature_importance,
                            'best_params': best_params
                        }
                        
                        st.session_state['models'][model_name] = model_info
                        st.session_state['df_processed'] = df
                        
                        # Save the model if requested
                        if save_model_option:
                            model_path = save_model(model, model_name)
                            st.session_state['models'][model_name]['path'] = model_path
                            
                        st.success(f"Model {model_name} trained successfully!")
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        st.info("Try using a dataset with more samples per sentiment class, or uncheck 'Use only positive/negative sentiments' option.")
            
            # Display metrics if model is trained
            if 'models' in st.session_state and model_name in st.session_state['models']:
                model_info = st.session_state['models'][model_name]
                metrics = model_info['metrics']
                best_params = model_info.get('best_params')
                
                # Display hyperparameter tuning results if available
                if best_params:
                    st.subheader("Best Hyperparameters")
                    st.json(best_params)
                
                st.subheader("Model Evaluation Metrics")
                
                # Create two columns for metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                
                with col2:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                    st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                conf_matrix = np.array(metrics['confusion_matrix'])
                labels = list(set(y_test)) if 'y_test' in locals() else model_info.get('labels', ['positive', 'negative'])
                fig = plot_confusion_matrix(conf_matrix, labels)
                st.pyplot(fig)
                
                # Display feature importance
                st.subheader("Feature Importance")
                if 'feature_importance' in model_info:
                    feature_importance = model_info['feature_importance']
                    if feature_importance:
                        # Different handling based on model type
                        if 'all' in feature_importance:
                            # Random Forest type models
                            chart = plot_important_features(feature_importance['all'], "Important Features Overall")
                            st.altair_chart(chart)
                        else:
                            # Create tabs for each sentiment class
                            feature_tabs = st.tabs(list(feature_importance.keys()))
                            for i, sentiment in enumerate(feature_importance.keys()):
                                with feature_tabs[i]:
                                    chart = plot_important_features(
                                        feature_importance[sentiment], 
                                        f"Important Features for {sentiment.capitalize()}"
                                    )
                                    st.altair_chart(chart)
                    else:
                        st.info("Feature importance not available for this model.")
        
        # Model Comparison Page
        elif page == "Model Comparison":
            st.header("Model Comparison")
            
            # Only proceed if text and sentiment columns are selected and different
            if text_column == sentiment_column:
                st.error("Error: Text column and sentiment column cannot be the same.")
                return
            
            # Check if we already have models trained
            if 'models' in st.session_state and st.session_state['models']:
                st.success(f"You have {len(st.session_state['models'])} models already trained.")
            else:
                st.info("No models trained yet. Train models below for comparison.")
            
            # Selection box for multiple models
            model_names = list(get_available_models().keys())
            selected_models = st.multiselect(
                "Select models to train and compare",
                model_names,
                default=["Multinomial Naive Bayes", "Logistic Regression"] if model_names else []
            )
            
            # Binary classification option
            binary_sentiments = st.checkbox("Use only positive/negative sentiments", value=True, key="mc_binary")
            
            # Hyperparameter tuning option
            tune_hyperparameters = st.checkbox("Perform hyperparameter tuning", value=False, key="mc_tune")
            
            if tune_hyperparameters:
                st.info("Hyperparameter tuning may take longer, especially when training multiple models.")
            
            # Create columns for preprocessing options
            mc_prep_col1, mc_prep_col2 = st.columns(2)
            
            with mc_prep_col1:
                mc_remove_stopwords = st.checkbox("Remove stopwords", value=True, key="mc_stopwords")
                mc_perform_stemming = st.checkbox("Perform stemming", value=False, key="mc_stem")
            
            with mc_prep_col2:
                mc_perform_lemmatization = st.checkbox("Perform lemmatization", value=True, key="mc_lemma")
                mc_handle_negations = st.checkbox("Handle negations", value=True, key="mc_negation")
            
            # Train models button
            if st.button("Train Selected Models") and selected_models:
                with st.spinner("Training multiple models, please wait..."):
                    # Preprocess the data
                    df['processed_text'] = df[text_column].apply(
                        lambda x: preprocess_text(x, 
                                               remove_stopwords=mc_remove_stopwords, 
                                               perform_stemming=mc_perform_stemming,
                                               perform_lemmatization=mc_perform_lemmatization,
                                               handle_negations=mc_handle_negations)
                    )
                    
                    # Normalize sentiment labels
                    df = normalize_sentiment_labels(df, sentiment_column)
                    
                    # Filter out rows with 'neutral' sentiment if we want binary classification
                    if binary_sentiments:
                        df_filtered = df[df[sentiment_column].isin(['positive', 'negative'])]
                        if len(df_filtered) == 0:
                            st.error("No rows with positive/negative sentiment found after filtering.")
                            return
                        df = df_filtered
                    
                    # Check if we have enough data for each sentiment class
                    sentiment_counts = df[sentiment_column].value_counts()
                    if sentiment_counts.min() < 2:
                        st.error(f"The smallest sentiment class ({sentiment_counts.idxmin()}) has only {sentiment_counts.min()} samples. Each class needs at least 2 samples.")
                        return
                    
                    # Store in session state
                    st.session_state['df_processed'] = df
                    
                    # Initialize models dictionary if it doesn't exist
                    if 'models' not in st.session_state:
                        st.session_state['models'] = {}
                    
                    # Train each selected model
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, model_name in enumerate(selected_models):
                        status_text.text(f"Training {model_name}...")
                        
                        try:
                            # Train the model
                            model_result = train_model(
                                df, 'processed_text', sentiment_column, model_name, 
                                tune_hyperparameters=tune_hyperparameters
                            )
                            
                            # Unpack the results
                            model, X_train, X_test, y_train, y_test, best_params = model_result
                            
                            # Evaluate the model
                            metrics = evaluate_model(model, X_test, y_test)
                            
                            # Get feature importance
                            feature_importance = get_feature_importance(model)
                            
                            # Store the model info
                            model_info = {
                                'model': model,
                                'metrics': metrics,
                                'feature_importance': feature_importance,
                                'best_params': best_params,
                                'labels': list(set(y_test))
                            }
                            
                            st.session_state['models'][model_name] = model_info
                            
                            # Update progress
                            progress = (i + 1) / len(selected_models)
                            progress_bar.progress(progress)
                            
                        except Exception as e:
                            st.error(f"Error training {model_name}: {str(e)}")
                    
                    status_text.text("Training completed!")
                    progress_bar.progress(1.0)
                    
                    st.success(f"Successfully trained {len(selected_models)} models for comparison.")
            
            # Get metrics for all trained models
            if 'models' in st.session_state and st.session_state['models']:
                metrics_dict = {name: data['metrics'] for name, data in st.session_state['models'].items()}
                
                # Display comparison chart
                st.subheader("Model Performance Comparison")
                chart = plot_model_comparison(metrics_dict)
                st.altair_chart(chart)
                
                # Display metrics table
                st.subheader("Metrics Table")
                
                # Create a DataFrame for the table
                model_names = []
                accuracies = []
                precisions = []
                recalls = []
                f1_scores = []
                
                for model_name, metrics in metrics_dict.items():
                    model_names.append(model_name)
                    accuracies.append(f"{metrics['accuracy']:.4f}")
                    precisions.append(f"{metrics['precision']:.4f}")
                    recalls.append(f"{metrics['recall']:.4f}")
                    f1_scores.append(f"{metrics['f1_score']:.4f}")
                
                comparison_df = pd.DataFrame({
                    'Model': model_names,
                    'Accuracy': accuracies,
                    'Precision': precisions,
                    'Recall': recalls,
                    'F1 Score': f1_scores
                })
                
                st.table(comparison_df)
                
                # Allow user to select a model to view details
                st.subheader("Model Details")
                selected_model_details = st.selectbox(
                    "Select a model to view details",
                    list(st.session_state['models'].keys())
                )
                
                if selected_model_details:
                    model_info = st.session_state['models'][selected_model_details]
                    metrics = model_info['metrics']
                    best_params = model_info.get('best_params')
                    
                    # Display hyperparameter tuning results if available
                    if best_params:
                        st.subheader("Best Hyperparameters")
                        st.json(best_params)
                    
                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    conf_matrix = np.array(metrics['confusion_matrix'])
                    labels = model_info.get('labels', ['positive', 'negative'])
                    fig = plot_confusion_matrix(conf_matrix, labels)
                    st.pyplot(fig)
                    
                    # Display feature importance
                    if 'feature_importance' in model_info and model_info['feature_importance']:
                        st.subheader("Feature Importance")
                        feature_importance = model_info['feature_importance']
                        
                        # Different handling based on model type
                        if 'all' in feature_importance:
                            # Random Forest type models
                            chart = plot_important_features(feature_importance['all'], "Important Features Overall")
                            st.altair_chart(chart)
                        else:
                            # Create tabs for each sentiment class
                            feature_tabs = st.tabs(list(feature_importance.keys()))
                            for i, sentiment in enumerate(feature_importance.keys()):
                                with feature_tabs[i]:
                                    chart = plot_important_features(
                                        feature_importance[sentiment], 
                                        f"Important Features for {sentiment.capitalize()}"
                                    )
                                    st.altair_chart(chart)
            else:
                st.info("No models trained yet. Select models above and click 'Train Selected Models' to begin comparison.")
        
        # Prediction Page
        elif page == "Prediction":
            st.header("Predict Sentiment for New Review")
            
            # Check for available models
            available_models = []
            
            # Check session models
            if 'models' in st.session_state and st.session_state['models']:
                available_models.extend(list(st.session_state['models'].keys()))
            
            # Check saved models
            saved_models = get_saved_models()
            if saved_models:
                # Create readable names
                saved_model_names = [f"{os.path.splitext(m)[0].replace('_', ' ').title()} (Saved)" for m in saved_models]
                available_models.extend(saved_model_names)
            
            if not available_models:
                st.info("No models available. Train a model in the 'Model Training' section or load a saved model.")
                return
            
            # Model selection for prediction
            selected_model = st.selectbox("Select a model for prediction", available_models)
            
            # Determine if it's a saved model
            is_saved_model = "(Saved)" in selected_model if selected_model else False
            
            # Get the model
            model = None
            if is_saved_model:
                # Extract the model name
                model_name = selected_model.replace(" (Saved)", "").lower().replace(" ", "_")
                model_path = os.path.join("models", f"{model_name}.joblib")
                try:
                    model = load_model(model_path)
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    return
            else:
                # Get from session state
                if selected_model in st.session_state['models']:
                    model = st.session_state['models'][selected_model]['model']
                else:
                    st.error("Selected model not found.")
                    return
            
            # Input area for new review
            new_review = st.text_area("Enter a new product review to analyze")
            
            # Show confidence scores option
            show_confidence = st.checkbox("Show confidence scores", value=True)
            
            # Batch prediction option
            st.subheader("Batch Prediction")
            batch_input = st.text_area("Or enter multiple reviews (one per line) for batch prediction")
            
            if st.button("Predict Sentiment"):
                if not new_review and not batch_input:
                    st.error("Please enter at least one review.")
                else:
                    # Single review prediction
                    if new_review:
                        # Preprocess the new review
                        processed_review = preprocess_text(
                            new_review, 
                            remove_stopwords=remove_stopwords, 
                            perform_stemming=perform_stemming,
                            perform_lemmatization=perform_lemmatization,
                            handle_negations=handle_negations
                        )
                        
                        # Make prediction with probability if requested
                        if show_confidence and hasattr(model, 'predict_proba'):
                            prediction, probability_dict = predict_sentiment_with_probability(model, processed_review)
                            
                            # Display prediction with styling
                            if prediction == 'positive':
                                st.markdown(f"<h3 style='color:green'>Predicted Sentiment: {prediction}</h3>", unsafe_allow_html=True)
                            elif prediction == 'negative':
                                st.markdown(f"<h3 style='color:red'>Predicted Sentiment: {prediction}</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3 style='color:blue'>Predicted Sentiment: {prediction}</h3>", unsafe_allow_html=True)
                            
                            # Display confidence scores
                            st.subheader("Confidence Scores")
                            for cls, prob in probability_dict.items():
                                st.progress(min(prob, 1.0))  # In case prob > 1
                                st.write(f"{cls}: {prob:.4f}")
                        else:
                            # Make simple prediction
                            prediction = predict_sentiment(model, processed_review)
                            
                            # Display prediction with styling
                            if prediction == 'positive':
                                st.markdown(f"<h3 style='color:green'>Predicted Sentiment: {prediction}</h3>", unsafe_allow_html=True)
                            elif prediction == 'negative':
                                st.markdown(f"<h3 style='color:red'>Predicted Sentiment: {prediction}</h3>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<h3 style='color:blue'>Predicted Sentiment: {prediction}</h3>", unsafe_allow_html=True)
                    
                    # Batch prediction
                    if batch_input:
                        st.subheader("Batch Prediction Results")
                        
                        # Split input into lines
                        reviews = batch_input.strip().split('\n')
                        
                        # Process each review
                        processed_reviews = [
                            preprocess_text(
                                review, 
                                remove_stopwords=remove_stopwords, 
                                perform_stemming=perform_stemming,
                                perform_lemmatization=perform_lemmatization,
                                handle_negations=handle_negations
                            ) for review in reviews
                        ]
                        
                        # Make predictions
                        predictions = model.predict(processed_reviews)
                        
                        # Also get probabilities if model supports it and user wants confidence scores
                        probabilities = None
                        if show_confidence and hasattr(model, 'predict_proba'):
                            probabilities = model.predict_proba(processed_reviews)
                            
                            # Create a DataFrame with results including probabilities
                            results = []
                            for i, (review, pred) in enumerate(zip(reviews, predictions)):
                                probs = {cls: prob for cls, prob in zip(model.classes_, probabilities[i])}
                                result = {'Review': review, 'Sentiment': pred}
                                for cls, prob in probs.items():
                                    result[f'Confidence ({cls})'] = prob
                                results.append(result)
                            
                            results_df = pd.DataFrame(results)
                        else:
                            # Create a simple DataFrame with results
                            results_df = pd.DataFrame({
                                'Review': reviews,
                                'Sentiment': predictions
                            })
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        # Prepare results for display
                        display_cols = ['Review', 'Sentiment']
                        if 'Confidence (positive)' in results_df.columns:
                            display_cols.append('Confidence (positive)')
                        
                        # Use safe display
                        display_results = results_df[display_cols] if all(col in results_df.columns for col in display_cols) else results_df
                        safe_display_dataframe(display_results, st)
        
        # Model Management Page
        elif page == "Model Management":
            st.header("Model Management")
            
            # Get saved models
            saved_models = get_saved_models()
            
            # Get session models
            session_models = list(st.session_state.get('models', {}).keys())
            
            # Create tabs for different management functions
            tab1, tab2 = st.tabs(["Save Models", "Load Models"])
            
            with tab1:
                st.subheader("Save Trained Models")
                
                if not session_models:
                    st.info("No models in the current session. Train a model first.")
                else:
                    # Select model to save
                    model_to_save = st.selectbox(
                        "Select a model to save",
                        session_models
                    )
                    
                    # Save button
                    if st.button("Save Selected Model"):
                        if model_to_save:
                            model = st.session_state['models'][model_to_save]['model']
                            model_path = save_model(model, model_to_save)
                            st.success(f"Model saved successfully to {model_path}")
            
            with tab2:
                st.subheader("Load Saved Models")
                
                if not saved_models:
                    st.info("No saved models found. Save a model first.")
                else:
                    # Display saved models
                    st.write("Available saved models:")
                    
                    # Convert filenames to readable names
                    readable_names = {os.path.splitext(m)[0].replace('_', ' ').title(): m for m in saved_models}
                    
                    # Select model to load
                    model_to_load = st.selectbox(
                        "Select a model to load",
                        list(readable_names.keys())
                    )
                    
                    # Load button
                    if st.button("Load Selected Model"):
                        if model_to_load:
                            model_filename = readable_names[model_to_load]
                            model_path = os.path.join("models", model_filename)
                            
                            try:
                                loaded_model = load_model(model_path)
                                
                                # Add to session state
                                if 'models' not in st.session_state:
                                    st.session_state['models'] = {}
                                
                                st.session_state['models'][f"{model_to_load} (Loaded)"] = {
                                    'model': loaded_model,
                                    'path': model_path
                                }
                                
                                st.success(f"Model {model_to_load} loaded successfully!")
                            except Exception as e:
                                st.error(f"Error loading model: {str(e)}")
                
                # Option to delete saved models
                if saved_models:
                    st.subheader("Delete Saved Models")
                    
                    # Select model to delete
                    model_to_delete = st.selectbox(
                        "Select a model to delete",
                        list(readable_names.keys()),
                        key="delete_model_selectbox"
                    )
                    
                    # Delete button
                    if st.button("Delete Selected Model"):
                        if model_to_delete:
                            model_filename = readable_names[model_to_delete]
                            model_path = os.path.join("models", model_filename)
                            
                            try:
                                os.remove(model_path)
                                st.success(f"Model {model_to_delete} deleted successfully!")
                            except Exception as e:
                                st.error(f"Error deleting model: {str(e)}")

        # Large File Processing Page
        elif page == "Large File Processing":
            st.header("Large File Processing")
            st.info("This page allows you to process large CSV files (hundreds of MB) efficiently using chunk-based processing.")
            
            # Show optimization tips for very large files
            with st.expander("Tips for Processing Very Large Files (100MB+)"):
                st.markdown("""
                ### Optimization Tips for Very Large Files
                
                1. **Increase Chunk Size**: For 300MB+ files, a chunk size of 30,000-50,000 rows often works best
                2. **Reduce Features**: Use fewer features (2^17 or 2^18) to save memory
                3. **Simplify Preprocessing**: Disable lemmatization for faster processing
                4. **Sample Your Data**: Consider using a subset of your data for initial exploration
                5. **Be Patient**: Processing a 300MB file may take 10-20 minutes depending on your system
                
                For the absolute best performance with very large files (500MB+), consider:
                - Using a smaller subset of the data for initial model development
                - Upgrading your RAM if possible
                - Running the application on a more powerful machine
                """)
            
            # File upload and options section
            st.subheader("Upload or Select Large CSV File")
            
            # Option to use a previously uploaded file
            if 'large_dataset_warning_shown' in st.session_state and st.session_state['large_dataset_warning_shown']:
                use_existing = st.checkbox(
                    f"Use existing large file: {st.session_state['large_dataset_name']} ({st.session_state['large_dataset_size']:.1f}MB)",
                    value=True
                )
                
                if use_existing:
                    file_path = st.session_state['large_dataset_path']
                    large_file_name = st.session_state['large_dataset_name']
                    st.success(f"Using existing file: {large_file_name}")
                    
                    # Preview the first few rows
                    try:
                        preview_rows = pd.read_csv(file_path, nrows=5)
                        st.subheader("Data Preview")
                        safe_display_dataframe(preview_rows, st)
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                        return
                    
                    # Auto-detect columns
                    text_col, sentiment_col = detect_columns(preview_rows)
                else:
                    large_file = st.file_uploader("Upload your large CSV file", type=['csv'], key="large_csv_uploader")
                    if not large_file:
                        return
                    
                    # Save the file to a temporary location
                    file_path = os.path.join("datasets", large_file.name)
                    with open(file_path, "wb") as f:
                        f.write(large_file.getbuffer())
                    
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    large_file_name = large_file.name
                    
                    st.success(f"File uploaded successfully: {large_file_name} ({file_size_mb:.1f}MB)")
                    
                    # Preview the first few rows
                    try:
                        preview_rows = pd.read_csv(file_path, nrows=5)
                        st.subheader("Data Preview")
                        safe_display_dataframe(preview_rows, st)
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                        return
                    
                    # Auto-detect columns
                    text_col, sentiment_col = detect_columns(preview_rows)
            else:
                large_file = st.file_uploader("Upload your large CSV file", type=['csv'], key="large_csv_uploader")
                
                if not large_file:
                    # Allow selecting from existing large datasets
                    large_datasets = []
                    datasets_dir = "datasets"
                    if os.path.exists(datasets_dir):
                        for filename in os.listdir(datasets_dir):
                            file_path = os.path.join(datasets_dir, filename)
                            if os.path.isfile(file_path) and filename.endswith('.csv'):
                                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                                if file_size_mb > 10:  # Only show files larger than 10MB
                                    large_datasets.append((filename, file_size_mb, file_path))
                    
                    if large_datasets:
                        st.subheader("Or select an existing large dataset")
                        dataset_options = [f"{name} ({size:.1f}MB)" for name, size, _ in large_datasets]
                        dataset_option = st.selectbox("Select a dataset", [""] + dataset_options)
                        
                        if dataset_option:
                            # Find the selected dataset info
                            selected_idx = dataset_options.index(dataset_option)
                            large_file_name, _, file_path = large_datasets[selected_idx]
                            
                            st.success(f"Selected dataset: {large_file_name}")
                            
                            # Preview the first few rows
                            try:
                                preview_rows = pd.read_csv(file_path, nrows=5)
                                st.subheader("Data Preview")
                                safe_display_dataframe(preview_rows, st)
                            except Exception as e:
                                st.error(f"Error reading file: {str(e)}")
                                return
                            
                            # Auto-detect columns
                            text_col, sentiment_col = detect_columns(preview_rows)
                        else:
                            return
                    else:
                        st.info("Please upload a CSV file to continue.")
                        return
                else:
                    # Save the file to a temporary location
                    file_path = os.path.join("datasets", large_file.name)
                    with open(file_path, "wb") as f:
                        f.write(large_file.getbuffer())
                    
                    large_file_name = large_file.name
                    st.success(f"File uploaded successfully: {large_file_name}")
                    
                    # Preview the first few rows
                    try:
                        preview_rows = pd.read_csv(file_path, nrows=5)
                        st.subheader("Data Preview")
                        safe_display_dataframe(preview_rows, st)
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                        return
                    
                    # Auto-detect columns
                    text_col, sentiment_col = detect_columns(preview_rows)
            
            # Column selection
            st.subheader("Column Selection")
            
            # Display the detected columns
            st.info(f"Auto-detected columns - Text: {text_col}, Sentiment: {sentiment_col}")
            
            # Get all column names
            try:
                available_columns = preview_rows.columns.tolist()
            except:
                available_columns = []
            
            # Allow user to override
            selected_text_col = st.selectbox(
                "Select the column containing review text",
                available_columns,
                index=available_columns.index(text_col) if text_col in available_columns else 0
            )
            
            selected_sentiment_col = st.selectbox(
                "Select the column containing sentiment labels",
                available_columns,
                index=available_columns.index(sentiment_col) if sentiment_col in available_columns else 0
            )
            
            # Processing parameters
            st.subheader("Processing Parameters")
            
            # Get file size for recommending optimal parameters
            try:
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                
                # Set appropriate defaults based on file size
                if file_size_mb > 300:
                    default_chunk_size = 50000
                    default_features = 2**18
                    default_lemma = False
                elif file_size_mb > 100:
                    default_chunk_size = 30000
                    default_features = 2**18
                    default_lemma = True
                else:
                    default_chunk_size = 20000
                    default_features = 2**19
                    default_lemma = True
            except:
                default_chunk_size = 20000
                default_features = 2**18
                default_lemma = True
                file_size_mb = "unknown"
            
            # Display recommendation
            st.write(f"Recommended settings for {file_size_mb}MB file:")
            
            chunk_size = st.number_input(
                "Chunk size (rows per batch)",
                min_value=5000,
                max_value=100000,
                value=default_chunk_size,
                step=5000,
                help="Number of rows to process at once. Larger values use more memory but are faster."
            )
            
            test_split = st.slider(
                "Test set percentage",
                min_value=5,
                max_value=30,
                value=10,
                help="Percentage of data to use for testing the model"
            ) / 100
            
            n_features = st.select_slider(
                "Number of features",
                options=[2**16, 2**17, 2**18, 2**19, 2**20],
                value=default_features,
                format_func=lambda x: f"{x:,} ({x:,})",
                help="Number of features for the HashingVectorizer. More features can capture more patterns but use more memory."
            )
            
            # Text preprocessing options in two columns
            st.subheader("Text Preprocessing Options")
            preproc_col1, preproc_col2 = st.columns(2)
            
            with preproc_col1:
                remove_stopwords = st.checkbox("Remove stopwords", value=True, key="large_stopwords")
                perform_stemming = st.checkbox("Perform stemming", value=not default_lemma, key="large_stemming")
            
            with preproc_col2:
                perform_lemmatization = st.checkbox("Perform lemmatization", value=default_lemma, key="large_lemma")
                handle_negations = st.checkbox("Handle negations", value=True, key="large_negation")
            
            # Model name
            model_name = st.text_input(
                "Model name",
                value=f"Large_NB_{time.strftime('%Y%m%d_%H%M%S')}",
                help="Name for the saved model"
            )
            
            # Process button
            if st.button("Process and Train Model", key="process_large_file"):
                # Create a progress bar and status message
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Callback for progress updates
                def progress_callback(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                # Verify that the system has enough RAM
                try:
                    import psutil
                    available_ram = psutil.virtual_memory().available / (1024 * 1024)
                    if available_ram < file_size_mb * 2 and file_size_mb > 100:
                        st.warning(f"⚠️ Available RAM ({available_ram:.0f}MB) may be insufficient for processing "
                                f"a {file_size_mb:.0f}MB file. Consider reducing chunk size or features.")
                except ImportError:
                    pass  # Skip RAM check if psutil is not available
                
                try:
                    # Process the file in chunks
                    model, vectorizer, metrics = process_large_file(
                        file_path=file_path,
                        text_column=selected_text_col,
                        sentiment_column=selected_sentiment_col,
                        chunksize=chunk_size,
                        test_size=test_split,
                        remove_stopwords=remove_stopwords,
                        perform_stemming=perform_stemming,
                        perform_lemmatization=perform_lemmatization,
                        handle_negations=handle_negations,
                        n_features=n_features,
                        callback=progress_callback
                    )
                    
                    # Save the model
                    model_dir = save_chunked_model(model, vectorizer, metrics, model_name)
                    
                    # Store model info in session state
                    if 'chunked_models' not in st.session_state:
                        st.session_state['chunked_models'] = {}
                    
                    st.session_state['chunked_models'][model_name] = {
                        'model': model,
                        'vectorizer': vectorizer,
                        'metrics': metrics,
                        'path': model_dir
                    }
                    
                    # Show success message
                    st.success(f"Model trained and saved successfully!")
                    
                    # Show metrics
                    st.subheader("Model Performance Metrics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        st.metric("Precision", f"{metrics['precision']:.4f}")
                    
                    with col2:
                        st.metric("Recall", f"{metrics['recall']:.4f}")
                        st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                    
                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    conf_matrix = np.array(metrics['confusion_matrix'])
                    labels = metrics.get('classes', ['positive', 'negative'])
                    fig = plot_confusion_matrix(conf_matrix, labels)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
            
            # Batch prediction section
            st.subheader("Batch Prediction with Trained Models")
            
            # Check if we have chunked models available
            chunked_model_dirs = get_chunked_models()
            chunked_model_names = []
            
            for model_dir in chunked_model_dirs:
                try:
                    _, _, _, info = load_chunked_model(model_dir)
                    if info and 'name' in info:
                        chunked_model_names.append(info['name'])
                    else:
                        # Use directory name as fallback
                        chunked_model_names.append(os.path.basename(model_dir))
                except:
                    # Use directory name as fallback
                    chunked_model_names.append(os.path.basename(model_dir))
            
            # Also check session state
            if 'chunked_models' in st.session_state:
                chunked_model_names.extend(list(st.session_state['chunked_models'].keys()))
            
            if not chunked_model_names:
                st.info("No chunked models available. Train a model first using the form above.")
            else:
                # Model selection
                selected_chunked_model = st.selectbox(
                    "Select a model for prediction",
                    chunked_model_names
                )
                
                # Prediction options
                prediction_input_type = st.radio(
                    "Prediction input type",
                    ["Text Input", "File Upload"]
                )
                
                if prediction_input_type == "Text Input":
                    # Text input for prediction
                    batch_text = st.text_area(
                        "Enter multiple reviews (one per line)",
                        height=200
                    )
                    
                    if st.button("Predict Sentiments", key="predict_text_batch"):
                        if not batch_text.strip():
                            st.error("Please enter some text for prediction.")
                        else:
                            # Load the model if not in session state
                            model = None
                            vectorizer = None
                            
                            if 'chunked_models' in st.session_state and selected_chunked_model in st.session_state['chunked_models']:
                                model = st.session_state['chunked_models'][selected_chunked_model]['model']
                                vectorizer = st.session_state['chunked_models'][selected_chunked_model]['vectorizer']
                            else:
                                # Find model directory
                                for model_dir in chunked_model_dirs:
                                    try:
                                        _, _, _, info = load_chunked_model(model_dir)
                                        if info and info.get('name') == selected_chunked_model:
                                            model, vectorizer, _, _ = load_chunked_model(model_dir)
                                            break
                                    except:
                                        continue
                            
                            if model is None or vectorizer is None:
                                st.error("Error loading model. Please try again.")
                                return
                            
                            # Process the batch text
                            reviews = batch_text.strip().split('\n')
                            
                            # Create a DataFrame for the reviews
                            df = pd.DataFrame({
                                'review_text': reviews
                            })
                            
                            # Make predictions
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def predict_callback(progress, message):
                                progress_bar.progress(progress)
                                status_text.text(message)
                            
                            results = predict_batch(
                                model, vectorizer, df, 'review_text',
                                remove_stopwords=remove_stopwords,
                                perform_stemming=perform_stemming,
                                perform_lemmatization=perform_lemmatization,
                                handle_negations=handle_negations,
                                callback=predict_callback
                            )
                            
                            # Display results
                            st.subheader("Prediction Results")
                            
                            # Prepare results for display
                            display_cols = ['review_text', 'prediction']
                            if 'confidence_positive' in results.columns:
                                for col in results.columns:
                                    if col.startswith('confidence_'):
                                        display_cols.append(col)
                            
                            # Use safe display
                            display_results = results[display_cols] if all(col in results.columns for col in display_cols) else results
                            safe_display_dataframe(display_results, st)
                            
                            # Download button for results
                            csv = results[display_cols].to_csv(index=False)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                
                else:  # File Upload
                    # File upload for batch prediction
                    prediction_file = st.file_uploader(
                        "Upload a CSV file with reviews to predict",
                        type=['csv'],
                        key="prediction_csv_uploader"
                    )
                    
                    if prediction_file:
                        # Save to a temporary location
                        pred_file_path = os.path.join("datasets", f"pred_{prediction_file.name}")
                        with open(pred_file_path, "wb") as f:
                            f.write(prediction_file.getbuffer())
                        
                        # Preview file
                        try:
                            pred_preview = pd.read_csv(pred_file_path, nrows=5)
                            st.write("Preview:")
                            st.write(pred_preview)
                            
                            # Detect text column
                            pred_text_col, _ = detect_columns(pred_preview)
                            
                            # Column selection
                            pred_columns = pred_preview.columns.tolist()
                            pred_text_column = st.selectbox(
                                "Select the column containing review text",
                                pred_columns,
                                index=pred_columns.index(pred_text_col) if pred_text_col in pred_columns else 0,
                                key="pred_text_col"
                            )
                            
                            if st.button("Predict Sentiments", key="predict_file_batch"):
                                # Load the model if not in session state
                                model = None
                                vectorizer = None
                                
                                if 'chunked_models' in st.session_state and selected_chunked_model in st.session_state['chunked_models']:
                                    model = st.session_state['chunked_models'][selected_chunked_model]['model']
                                    vectorizer = st.session_state['chunked_models'][selected_chunked_model]['vectorizer']
                                else:
                                    # Find model directory
                                    for model_dir in chunked_model_dirs:
                                        try:
                                            _, _, _, info = load_chunked_model(model_dir)
                                            if info and info.get('name') == selected_chunked_model:
                                                model, vectorizer, _, _ = load_chunked_model(model_dir)
                                                break
                                        except:
                                            continue
                                
                                if model is None or vectorizer is None:
                                    st.error("Error loading model. Please try again.")
                                    return
                                
                                # Process the file in chunks
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                def predict_callback(progress, message):
                                    progress_bar.progress(progress)
                                    status_text.text(message)
                                
                                try:
                                    results = predict_batch(
                                        model, vectorizer, pred_file_path, pred_text_column,
                                        remove_stopwords=remove_stopwords,
                                        perform_stemming=perform_stemming,
                                        perform_lemmatization=perform_lemmatization,
                                        handle_negations=handle_negations,
                                        callback=predict_callback
                                    )
                                    
                                    # Display sample of results
                                    st.subheader("Prediction Results (Sample)")
                                    
                                    # Prepare results for display
                                    display_cols = [pred_text_column, 'prediction']
                                    if 'confidence_positive' in results.columns:
                                        for col in results.columns:
                                            if col.startswith('confidence_'):
                                                display_cols.append(col)
                                    
                                    # Use safe display
                                    display_results = results[display_cols] if all(col in results.columns for col in display_cols) else results
                                    safe_display_dataframe(display_results.head(20), st)
                                    
                                    # Save results to file
                                    results_path = os.path.join("datasets", f"results_{prediction_file.name}")
                                    results.to_csv(results_path, index=False)
                                    
                                    # Download button for results
                                    csv = results[display_cols].to_csv(index=False)
                                    st.download_button(
                                        label="Download All Predictions as CSV",
                                        data=csv,
                                        file_name=f"predictions_{prediction_file.name}",
                                        mime="text/csv"
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error processing file: {str(e)}")
                        
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
            
            # Managing chunked models section
            st.subheader("Manage Chunked Models")
            
            if not chunked_model_names:
                st.info("No chunked models available.")
            else:
                # Select model to delete
                model_to_delete = st.selectbox(
                    "Select a model to delete",
                    chunked_model_names,
                    key="del_chunked_model"
                )
                
                if st.button("Delete Selected Model", key="delete_chunked_btn"):
                    # Find model directory
                    model_dir = None
                    
                    if 'chunked_models' in st.session_state and model_to_delete in st.session_state['chunked_models']:
                        model_dir = st.session_state['chunked_models'][model_to_delete].get('path')
                    
                    if not model_dir:
                        for dir_path in chunked_model_dirs:
                            try:
                                _, _, _, info = load_chunked_model(dir_path)
                                if info and info.get('name') == model_to_delete:
                                    model_dir = dir_path
                                    break
                            except:
                                continue
                    
                    if model_dir and os.path.exists(model_dir):
                        try:
                            import shutil
                            shutil.rmtree(model_dir)
                            
                            # Remove from session state
                            if 'chunked_models' in st.session_state and model_to_delete in st.session_state['chunked_models']:
                                del st.session_state['chunked_models'][model_to_delete]
                            
                            st.success(f"Model '{model_to_delete}' deleted successfully.")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error deleting model: {str(e)}")
                    else:
                        st.error("Model directory not found.")

if __name__ == "__main__":
    main() 
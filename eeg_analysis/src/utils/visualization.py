import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
import plotly.express as px

class ResultVisualizer:
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        
    def plot_feature_importance(self, importances: pd.DataFrame, 
                              top_n: int = 20,
                              interactive: bool = False):
        top_features = importances.nlargest(top_n, 'importance')
        
        if interactive:
            fig = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h'
            ))
            fig.update_layout(
                title='Feature Importance',
                xaxis_title='Importance',
                yaxis_title='Feature',
                height=600
            )
            return fig
        else:
            plt.figure(figsize=(10, 8))
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            return plt.gcf()

    def plot_confusion_matrix(self, y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            normalize: bool = True,
                            interactive: bool = False):
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if interactive:
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Negative', 'Positive'],
                y=['Negative', 'Positive'],
                colorscale='Blues',
                text=np.around(cm, decimals=2),
                texttemplate='%{text}',
                textfont={"size": 16},
            ))
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual'
            )
            return fig
        else:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            plt.title('Confusion Matrix')
            return plt.gcf()

    def plot_metrics_over_time(self, metrics_history: pd.DataFrame,
                             metrics: List[str]):
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=metrics_history.index,
                y=metrics_history[metric],
                name=metric,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title='Metrics Over Time',
            xaxis_title='Time',
            yaxis_title='Value',
            height=400
        )
        return fig

    def create_feature_explorer(self, df: pd.DataFrame,
                              target_col: str,
                              exclude_cols: Optional[List[str]] = None):
        if exclude_cols is None:
            exclude_cols = []
            
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols + [target_col]]
        
        fig = px.scatter_matrix(
            df,
            dimensions=feature_cols,
            color=target_col,
            title='Feature Relationships'
        )
        
        fig.update_layout(
            height=1000,
            width=1000
        )
        return fig

def create_visualizer(style: str = 'seaborn') -> ResultVisualizer:
    return ResultVisualizer(style)
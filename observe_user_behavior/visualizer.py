import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Try to import seaborn, fallback if not available
try:
    import seaborn as sns
    HAS_SEABORN = True
    plt.style.use('seaborn-v0_8-darkgrid')
except ImportError:
    HAS_SEABORN = False
    plt.style.use('default')
    print("Warning: seaborn not available, using matplotlib defaults")

class BehaviorVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        
    def create_comprehensive_report(self, features_df, cohort_stats, output_dir='./visualizations'):
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self.plot_cohort_distribution(features_df, cohort_stats, f'{output_dir}/cohort_distribution.png')
            self.plot_resolution_analysis(features_df, f'{output_dir}/resolution_analysis.png')
            self.plot_behavioral_patterns(features_df, f'{output_dir}/behavioral_patterns.png')
            self.plot_conversation_flows(features_df, f'{output_dir}/conversation_flows.png')
            print(f"All visualizations saved to {output_dir}/")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        
    def plot_cohort_distribution(self, features_df, cohort_stats, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Cohort count distribution
        cohort_counts = features_df['cohort'].value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(cohort_counts)))
        cohort_counts.plot(kind='bar', ax=axes[0, 0], color=colors)
        axes[0, 0].set_title('Customer Cohort Distribution', fontsize=14, weight='bold')
        axes[0, 0].set_xlabel('Cohort')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Resolution rate by cohort
        resolution_by_cohort = features_df.groupby('cohort')['is_resolved'].mean() * 100
        resolution_by_cohort.plot(kind='bar', ax=axes[0, 1], color=colors)
        axes[0, 1].set_title('Resolution Rate by Cohort', fontsize=14, weight='bold')
        axes[0, 1].set_xlabel('Cohort')
        axes[0, 1].set_ylabel('Resolution Rate (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
        
        # Sentiment by cohort
        sentiment_by_cohort = features_df.groupby('cohort')['sentiment_score'].mean()
        sentiment_by_cohort.plot(kind='bar', ax=axes[1, 0], color=colors)
        axes[1, 0].set_title('Average Sentiment by Cohort', fontsize=14, weight='bold')
        axes[1, 0].set_xlabel('Cohort')
        axes[1, 0].set_ylabel('Sentiment Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Duration by cohort
        avg_duration = features_df.groupby('cohort')['conversation_duration_minutes'].mean()
        avg_duration.plot(kind='bar', ax=axes[1, 1], color=colors)
        axes[1, 1].set_title('Average Conversation Duration by Cohort', fontsize=14, weight='bold')
        axes[1, 1].set_xlabel('Cohort')
        axes[1, 1].set_ylabel('Duration (minutes)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_resolution_analysis(self, features_df, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        resolved_df = features_df[features_df['is_resolved'] == 1]
        unresolved_df = features_df[features_df['is_resolved'] == 0]
        
        # Message count distribution
        axes[0, 0].hist([resolved_df['message_count'], unresolved_df['message_count']], 
                       bins=20, label=['Resolved', 'Unresolved'], alpha=0.7, color=['green', 'red'])
        axes[0, 0].set_title('Message Count Distribution', fontsize=14, weight='bold')
        axes[0, 0].set_xlabel('Number of Messages')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Sentiment vs Urgency scatter
        scatter = axes[0, 1].scatter(features_df['sentiment_score'], features_df['urgency_score'], 
                                   c=features_df['is_resolved'], cmap='RdYlGn', alpha=0.6)
        axes[0, 1].set_title('Sentiment vs Urgency', fontsize=14, weight='bold')
        axes[0, 1].set_xlabel('Sentiment Score')
        axes[0, 1].set_ylabel('Urgency Score')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # Resolution rate by sentiment category
        try:
            resolution_by_sentiment = pd.cut(features_df['sentiment_score'], 
                                           bins=[-1, -0.5, 0, 0.5, 1], 
                                           labels=['Very Negative', 'Negative', 'Neutral', 'Positive'])
            resolution_rate = features_df.groupby(resolution_by_sentiment)['is_resolved'].mean() * 100
            resolution_rate.plot(kind='bar', ax=axes[1, 0], color=['red', 'orange', 'yellow', 'green'])
            axes[1, 0].set_title('Resolution Rate by Sentiment Category', fontsize=14, weight='bold')
            axes[1, 0].set_xlabel('Sentiment Category')
            axes[1, 0].set_ylabel('Resolution Rate (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Error plotting sentiment categories: {str(e)}', 
                           transform=axes[1, 0].transAxes, ha='center', va='center')
        
        # Conversation duration boxplot
        resolved_durations = resolved_df['conversation_duration_minutes']
        unresolved_durations = unresolved_df['conversation_duration_minutes']
        
        box_data = [resolved_durations, unresolved_durations]
        axes[1, 1].boxplot(box_data, labels=['Resolved', 'Unresolved'])
        axes[1, 1].set_title('Conversation Duration by Resolution Status', fontsize=14, weight='bold')
        axes[1, 1].set_xlabel('Resolution Status')
        axes[1, 1].set_ylabel('Duration (minutes)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_behavioral_patterns(self, features_df, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Complexity vs Urgency scatter
        scatter = axes[0, 0].scatter(features_df['complexity_score'], 
                                   features_df['urgency_score'],
                                   c=features_df['sentiment_score'], 
                                   cmap='coolwarm', 
                                   s=features_df['message_count']*10,
                                   alpha=0.6)
        axes[0, 0].set_title('Complexity vs Urgency (color=sentiment, size=messages)', fontsize=12, weight='bold')
        axes[0, 0].set_xlabel('Complexity Score')
        axes[0, 0].set_ylabel('Urgency Score')
        plt.colorbar(scatter, ax=axes[0, 0], label='Sentiment')
        
        # Response time vs message ratio
        scatter2 = axes[0, 1].scatter(features_df['avg_response_time_minutes'], 
                                    features_df['customer_agent_ratio'],
                                    c=features_df['is_resolved'],
                                    cmap='RdYlGn',
                                    alpha=0.6)
        axes[0, 1].set_title('Response Time vs Message Ratio', fontsize=14, weight='bold')
        axes[0, 1].set_xlabel('Avg Response Time (min)')
        axes[0, 1].set_ylabel('Customer/Agent Message Ratio')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # Average scores by cohort
        cohort_scores = features_df.groupby('cohort')[['urgency_score', 'complexity_score', 'sentiment_score']].mean()
        cohort_scores.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average Scores by Cohort', fontsize=14, weight='bold')
        axes[1, 0].set_xlabel('Cohort')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend(['Urgency', 'Complexity', 'Sentiment'])
        
        # Correlation matrix
        correlation_features = ['sentiment_score', 'urgency_score', 'complexity_score', 
                              'message_count', 'conversation_duration_minutes', 'is_resolved']
        correlation_matrix = features_df[correlation_features].corr()
        
        if HAS_SEABORN:
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        else:
            # Fallback correlation plot without seaborn
            im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
            axes[1, 1].set_xticks(range(len(correlation_features)))
            axes[1, 1].set_yticks(range(len(correlation_features)))
            axes[1, 1].set_xticklabels(correlation_features, rotation=45)
            axes[1, 1].set_yticklabels(correlation_features)
            plt.colorbar(im, ax=axes[1, 1])
            
        axes[1, 1].set_title('Feature Correlation Matrix', fontsize=14, weight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_conversation_flows(self, features_df, save_path=None):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cohorts = features_df['cohort'].unique()
        y_positions = {cohort: i for i, cohort in enumerate(cohorts)}
        
        for cohort in cohorts:
            cohort_data = features_df[features_df['cohort'] == cohort]
            
            resolved = cohort_data[cohort_data['is_resolved'] == 1]
            unresolved = cohort_data[cohort_data['is_resolved'] == 0]
            
            total = len(cohort_data)
            resolved_pct = len(resolved) / total * 100 if total > 0 else 0
            unresolved_pct = len(unresolved) / total * 100 if total > 0 else 0
            
            y = y_positions[cohort]
            
            # Create horizontal stacked bar chart
            ax.barh(y, resolved_pct, color='green', alpha=0.7, label='Resolved' if cohort == cohorts[0] else '')
            ax.barh(y, unresolved_pct, left=resolved_pct, color='red', alpha=0.7, 
                   label='Unresolved' if cohort == cohorts[0] else '')
            
            # Add percentage labels
            if resolved_pct > 5:  # Only show label if bar is wide enough
                ax.text(resolved_pct/2, y, f'{resolved_pct:.1f}%', ha='center', va='center', fontweight='bold')
            if unresolved_pct > 5:
                ax.text(resolved_pct + unresolved_pct/2, y, f'{unresolved_pct:.1f}%', ha='center', va='center', fontweight='bold')
        
        ax.set_yticks(range(len(cohorts)))
        ax.set_yticklabels(cohorts)
        ax.set_xlabel('Percentage')
        ax.set_title('Resolution Status by Cohort', fontsize=16, weight='bold')
        ax.legend()
        ax.set_xlim(0, 100)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
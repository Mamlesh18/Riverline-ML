import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class CohortBuilder:
    def __init__(self):
        self.cohort_definitions = self._define_cohorts()
        self.cohort_stats = {}
        
    def _define_cohorts(self):
        return {
            'quick_resolution': lambda df: (
                (df['message_count'] <= 4) & 
                (df['conversation_duration_minutes'] < 60) & 
                (df['is_resolved'] == 1)
            ),
            
            'frustrated_customer': lambda df: (
                (df['sentiment_score'] < -0.3) |
                (df['max_consecutive_customer_messages'] >= 3)
            ),
            
            'urgent_technical': lambda df: (
                (df['urgency_score'] > 0.5) & 
                (df['complexity_score'] > 0.6)
            ),
            
            'patient_detailed': lambda df: (
                (df['avg_response_time_minutes'] > 30) & 
                (df['avg_customer_message_length'] > 150) &
                (df['sentiment_score'] > -0.2)
            ),
            
            'abandoned_conversation': lambda df: (
                (df['last_message_by_customer'] == 1) & 
                (df['is_resolved'] == 0) &
                (df['message_count'] > 2)
            ),
            
            'ping_pong': lambda df: (
                (df['customer_agent_ratio'] > 0.8) & 
                (df['customer_agent_ratio'] < 1.2) &
                (df['message_count'] > 6)
            ),
            
            'escalation_needed': lambda df: (
                (df['complexity_score'] > 0.8) & 
                (df['sentiment_score'] < -0.4) &
                (df['is_resolved'] == 0)
            )
        }
    
    def assign_cohorts(self, features_df):
        features_df['cohort'] = 'standard_support'
        
        for cohort_name, condition_func in self.cohort_definitions.items():
            mask = condition_func(features_df)
            features_df.loc[mask, 'cohort'] = cohort_name
        
        return features_df
    
    def analyze_cohort_statistics(self, features_df):
        cohort_stats = {}
        
        for cohort in features_df['cohort'].unique():
            cohort_data = features_df[features_df['cohort'] == cohort]
            
            stats = {
                'count': len(cohort_data),
                'percentage': len(cohort_data) / len(features_df) * 100,
                'resolution_rate': cohort_data['is_resolved'].mean() * 100,
                'avg_sentiment': cohort_data['sentiment_score'].mean(),
                'avg_messages': cohort_data['message_count'].mean(),
                'avg_duration_minutes': cohort_data['conversation_duration_minutes'].mean(),
                'avg_response_time': cohort_data['avg_response_time_minutes'].mean()
            }
            
            cohort_stats[cohort] = stats
        
        self.cohort_stats = cohort_stats
        return pd.DataFrame(cohort_stats).T
    
    def identify_resolution_patterns(self, features_df):
        resolution_patterns = {}
        
        for cohort in features_df['cohort'].unique():
            cohort_data = features_df[features_df['cohort'] == cohort]
            
            resolved = cohort_data[cohort_data['is_resolved'] == 1]
            unresolved = cohort_data[cohort_data['is_resolved'] == 0]
            
            if len(resolved) > 0 and len(unresolved) > 0:
                patterns = {
                    'resolution_rate': len(resolved) / len(cohort_data),
                    'resolved_avg_messages': resolved['message_count'].mean(),
                    'unresolved_avg_messages': unresolved['message_count'].mean(),
                    'resolved_avg_duration': resolved['conversation_duration_minutes'].mean(),
                    'unresolved_avg_duration': unresolved['conversation_duration_minutes'].mean(),
                    'sentiment_difference': resolved['sentiment_score'].mean() - unresolved['sentiment_score'].mean()
                }
                
                resolution_patterns[cohort] = patterns
        
        return resolution_patterns
    
    def create_ml_cohorts(self, features_df, n_clusters=8):
        feature_columns = [
            'message_count', 'sentiment_score', 'urgency_score', 'complexity_score',
            'conversation_duration_minutes', 'avg_response_time_minutes',
            'customer_agent_ratio', 'message_velocity', 'max_consecutive_customer_messages'
        ]
        
        X = features_df[feature_columns].fillna(0)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        features_df['ml_cohort'] = kmeans.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, features_df['ml_cohort'])
        
        ml_cohort_profiles = {}
        for cluster in range(n_clusters):
            cluster_data = features_df[features_df['ml_cohort'] == cluster]
            profile = {
                'size': len(cluster_data),
                'resolution_rate': cluster_data['is_resolved'].mean(),
                'avg_sentiment': cluster_data['sentiment_score'].mean(),
                'avg_urgency': cluster_data['urgency_score'].mean(),
                'avg_complexity': cluster_data['complexity_score'].mean(),
                'dominant_rule_cohort': cluster_data['cohort'].mode()[0] if len(cluster_data) > 0 else 'unknown'
            }
            ml_cohort_profiles[f'ml_cluster_{cluster}'] = profile
        
        return features_df, ml_cohort_profiles, silhouette_avg
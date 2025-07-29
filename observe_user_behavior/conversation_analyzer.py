import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sqlite3

class ConversationAnalyzer:
    def __init__(self, db_path='riverline.db'):
        self.conn = sqlite3.connect(db_path)
        
    def load_conversations(self):
        query = '''
            SELECT c.*, 
                   COUNT(cm.id) as actual_message_count
            FROM conversations c
            LEFT JOIN conversation_messages cm ON c.conversation_id = cm.conversation_id
            GROUP BY c.conversation_id
        '''
        return pd.read_sql_query(query, self.conn)
    
    def extract_advanced_features(self, conversations_df):
        features_list = []
        
        for _, conv in conversations_df.iterrows():
            messages = self._get_conversation_messages(conv['conversation_id'])
            
            if messages.empty:
                continue
                
            features = self._calculate_conversation_features(messages, conv)
            features['conversation_id'] = conv['conversation_id']
            features['is_resolved'] = conv['is_resolved']
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _get_conversation_messages(self, conversation_id):
        # Fixed query to avoid duplicate column names
        query = '''
            SELECT cm.conversation_id,
                   cm.tweet_id,
                   cm.author_id,
                   cm.is_customer,
                   cm.message_text,
                   cm.sequence_number,
                   rt.created_at as tweet_created_at,
                   rt.text as tweet_text
            FROM conversation_messages cm
            JOIN raw_tweets rt ON cm.tweet_id = rt.tweet_id
            WHERE cm.conversation_id = ?
            ORDER BY cm.sequence_number
        '''
        return pd.read_sql_query(query, self.conn, params=(conversation_id,))
    
    def _calculate_conversation_features(self, messages, conv_metadata):
        # Use the renamed column
        messages['created_at'] = pd.to_datetime(messages['tweet_created_at'])
        
        features = {
            'message_count': len(messages),
            'customer_message_count': len(messages[messages['is_customer'] == 1]),
            'agent_message_count': len(messages[messages['is_customer'] == 0]),
            'sentiment_score': conv_metadata['sentiment_score'],
            'urgency_score': conv_metadata['urgency_score'],
            'complexity_score': conv_metadata['complexity_score']
        }
        
        # Calculate conversation duration
        if len(messages) > 1:
            duration = (messages['created_at'].max() - messages['created_at'].min()).total_seconds() / 60
            features['conversation_duration_minutes'] = duration
        else:
            features['conversation_duration_minutes'] = 0
        
        # Calculate response times
        response_times = []
        for i in range(1, len(messages)):
            if messages.iloc[i]['is_customer'] != messages.iloc[i-1]['is_customer']:
                time_diff = (messages.iloc[i]['created_at'] - messages.iloc[i-1]['created_at']).total_seconds() / 60
                response_times.append(time_diff)
        
        features['avg_response_time_minutes'] = np.mean(response_times) if response_times else 0
        features['max_response_time_minutes'] = np.max(response_times) if response_times else 0
        
        # Calculate average customer message length
        customer_messages = messages[messages['is_customer'] == 1]
        features['avg_customer_message_length'] = customer_messages['message_text'].str.len().mean() if len(customer_messages) > 0 else 0
        
        # Calculate customer to agent message ratio
        features['customer_agent_ratio'] = features['customer_message_count'] / max(features['agent_message_count'], 1)
        
        # Check if last message was from customer
        features['last_message_by_customer'] = 1 if messages.iloc[-1]['is_customer'] else 0
        
        # Calculate max consecutive customer messages
        consecutive_customer = 0
        max_consecutive_customer = 0
        for _, msg in messages.iterrows():
            if msg['is_customer']:
                consecutive_customer += 1
                max_consecutive_customer = max(max_consecutive_customer, consecutive_customer)
            else:
                consecutive_customer = 0
        features['max_consecutive_customer_messages'] = max_consecutive_customer
        
        # Calculate message velocity (messages per minute)
        if features['conversation_duration_minutes'] > 0:
            features['message_velocity'] = features['message_count'] / features['conversation_duration_minutes']
        else:
            features['message_velocity'] = 0
        
        return features
    
    def identify_conversation_patterns(self, features_df):
        numeric_features = [
            'message_count', 'sentiment_score', 'urgency_score', 'complexity_score',
            'conversation_duration_minutes', 'avg_response_time_minutes',
            'customer_agent_ratio', 'message_velocity'
        ]
        
        # Handle missing values
        X = features_df[numeric_features].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        # Add PCA components to dataframe
        features_df['pattern_component_1'] = X_pca[:, 0]
        features_df['pattern_component_2'] = X_pca[:, 1]
        features_df['pattern_component_3'] = X_pca[:, 2]
        
        return features_df, pca.explained_variance_ratio_
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
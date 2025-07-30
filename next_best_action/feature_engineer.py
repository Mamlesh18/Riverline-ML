import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

class FeatureEngineer:
    """
    Advanced feature engineering for NBA system
    
    Extracts and computes features from conversation data for ML models
    """
    
    def __init__(self, db_path='riverline.db'):
        self.db_path = db_path
    
    def extract_customer_features(self, conversation_id):
        """
        Extract comprehensive customer features for a specific conversation
        
        Args:
            conversation_id: ID of conversation to analyze
            
        Returns:
            dict: comprehensive feature set
        """
        # Get basic conversation data
        basic_features = self._get_basic_conversation_features(conversation_id)
        
        # Get message-level features
        message_features = self._get_message_level_features(conversation_id)
        
        # Get temporal features
        temporal_features = self._get_temporal_features(conversation_id)
        
        # Get behavioral features
        behavioral_features = self._get_behavioral_features(conversation_id)
        
        # Combine all features
        all_features = {
            **basic_features,
            **message_features,
            **temporal_features,
            **behavioral_features
        }
        
        return all_features
    
    def _get_basic_conversation_features(self, conversation_id):
        """Get basic conversation metadata"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                conversation_id,
                participant_ids,
                start_time,
                last_update_time,
                message_count,
                is_resolved,
                sentiment_score,
                urgency_score,
                complexity_score,
                COALESCE(cohort, 'standard_support') as cohort,
                COALESCE(ml_cohort, 0) as ml_cohort
            FROM conversations 
            WHERE conversation_id = ?
        '''
        
        result = pd.read_sql_query(query, conn, params=(conversation_id,))
        conn.close()
        
        if len(result) == 0:
            return {'conversation_id': conversation_id, 'error': 'conversation_not_found'}
        
        row = result.iloc[0]
        
        # Parse datetime fields
        start_time = pd.to_datetime(row['start_time'])
        last_update = pd.to_datetime(row['last_update_time'])
        
        return {
            'conversation_id': row['conversation_id'],
            'message_count': row['message_count'],
            'is_resolved': row['is_resolved'],
            'sentiment_score': row['sentiment_score'],
            'urgency_score': row['urgency_score'],
            'complexity_score': row['complexity_score'],
            'cohort': row['cohort'],
            'ml_cohort': row['ml_cohort'],
            'conversation_start': start_time,
            'last_update': last_update,
            'conversation_age_hours': (datetime.now() - start_time).total_seconds() / 3600,
            'time_since_last_update_hours': (datetime.now() - last_update).total_seconds() / 3600
        }
    
    def _get_message_level_features(self, conversation_id):
        """Extract detailed message-level features"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                cm.is_customer,
                cm.message_text,
                cm.sequence_number,
                rt.created_at,
                LENGTH(cm.message_text) as message_length
            FROM conversation_messages cm
            JOIN raw_tweets rt ON cm.tweet_id = rt.tweet_id
            WHERE cm.conversation_id = ?
            ORDER BY cm.sequence_number
        '''
        
        messages = pd.read_sql_query(query, conn, params=(conversation_id,))
        conn.close()
        
        if len(messages) == 0:
            return self._get_default_message_features()
        
        # Convert timestamps
        messages['created_at'] = pd.to_datetime(messages['created_at'])
        
        # Separate customer and agent messages
        customer_msgs = messages[messages['is_customer'] == 1]
        agent_msgs = messages[messages['is_customer'] == 0]
        
        features = {
            'total_messages': len(messages),
            'customer_message_count': len(customer_msgs),
            'agent_message_count': len(agent_msgs),
            'customer_agent_ratio': len(customer_msgs) / max(len(agent_msgs), 1),
            
            # Message length features
            'avg_message_length': messages['message_length'].mean(),
            'avg_customer_message_length': customer_msgs['message_length'].mean() if len(customer_msgs) > 0 else 0,
            'avg_agent_message_length': agent_msgs['message_length'].mean() if len(agent_msgs) > 0 else 0,
            'max_message_length': messages['message_length'].max(),
            'min_message_length': messages['message_length'].min(),
            
            # Conversation flow features
            'last_message_by_customer': 1 if messages.iloc[-1]['is_customer'] else 0,
            'first_message_by_customer': 1 if messages.iloc[0]['is_customer'] else 0,
        }
        
        # Calculate consecutive message patterns
        features.update(self._calculate_consecutive_patterns(messages))
        
        # Calculate response time features
        features.update(self._calculate_response_times(messages))
        
        return features
    
    def _calculate_consecutive_patterns(self, messages):
        """Calculate consecutive message patterns"""
        consecutive_customer = 0
        consecutive_agent = 0
        max_consecutive_customer = 0
        max_consecutive_agent = 0
        
        current_consecutive_customer = 0
        current_consecutive_agent = 0
        
        for _, msg in messages.iterrows():
            if msg['is_customer']:
                current_consecutive_customer += 1
                current_consecutive_agent = 0
                max_consecutive_customer = max(max_consecutive_customer, current_consecutive_customer)
            else:
                current_consecutive_agent += 1
                current_consecutive_customer = 0
                max_consecutive_agent = max(max_consecutive_agent, current_consecutive_agent)
        
        # Count total consecutive runs
        runs = []
        current_speaker = None
        current_run = 0
        
        for _, msg in messages.iterrows():
            speaker = 'customer' if msg['is_customer'] else 'agent'
            if speaker == current_speaker:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_speaker = speaker
                current_run = 1
        if current_run > 0:
            runs.append(current_run)
        
        return {
            'max_consecutive_customer_messages': max_consecutive_customer,
            'max_consecutive_agent_messages': max_consecutive_agent,
            'avg_consecutive_run_length': np.mean(runs) if runs else 0,
            'total_conversation_turns': len(runs)
        }
    
    def _calculate_response_times(self, messages):
        """Calculate response time patterns"""
        if len(messages) < 2:
            return self._get_default_response_time_features()
        
        response_times = []
        customer_response_times = []
        agent_response_times = []
        
        for i in range(1, len(messages)):
            current_msg = messages.iloc[i]
            previous_msg = messages.iloc[i-1]
            
            # Only calculate if speakers are different (actual response)
            if current_msg['is_customer'] != previous_msg['is_customer']:
                time_diff = (current_msg['created_at'] - previous_msg['created_at']).total_seconds() / 60
                response_times.append(time_diff)
                
                if current_msg['is_customer']:
                    customer_response_times.append(time_diff)
                else:
                    agent_response_times.append(time_diff)
        
        # Calculate conversation duration and velocity
        total_duration = (messages.iloc[-1]['created_at'] - messages.iloc[0]['created_at']).total_seconds() / 60
        message_velocity = len(messages) / max(total_duration, 1)  # messages per minute
        
        return {
            'conversation_duration_minutes': total_duration,
            'avg_response_time_minutes': np.mean(response_times) if response_times else 0,
            'max_response_time_minutes': np.max(response_times) if response_times else 0,
            'min_response_time_minutes': np.min(response_times) if response_times else 0,
            'std_response_time_minutes': np.std(response_times) if response_times else 0,
            
            'avg_customer_response_time': np.mean(customer_response_times) if customer_response_times else 0,
            'avg_agent_response_time': np.mean(agent_response_times) if agent_response_times else 0,
            
            'message_velocity': message_velocity,
            'total_response_exchanges': len(response_times)
        }
    
    def _get_temporal_features(self, conversation_id):
        """Extract temporal patterns and features"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT rt.created_at, cm.is_customer
            FROM conversation_messages cm
            JOIN raw_tweets rt ON cm.tweet_id = rt.tweet_id
            WHERE cm.conversation_id = ?
            ORDER BY cm.sequence_number
        '''
        
        messages = pd.read_sql_query(query, conn, params=(conversation_id,))
        conn.close()
        
        if len(messages) == 0:
            return self._get_default_temporal_features()
        
        messages['created_at'] = pd.to_datetime(messages['created_at'])
        messages['hour'] = messages['created_at'].dt.hour
        messages['day_of_week'] = messages['created_at'].dt.dayofweek
        messages['is_weekend'] = messages['day_of_week'].isin([5, 6]).astype(int)
        
        features = {
            # Time of day patterns
            'conversation_start_hour': messages.iloc[0]['hour'],
            'conversation_spans_business_hours': self._spans_business_hours(messages),
            'conversation_during_weekend': messages.iloc[0]['is_weekend'],
            
            # Temporal urgency indicators
            'started_outside_business_hours': self._is_outside_business_hours(messages.iloc[0]['hour']),
            'continued_outside_business_hours': any(
                self._is_outside_business_hours(hour) for hour in messages['hour']
            ),
            
            # Activity patterns
            'messages_per_hour_distribution': self._calculate_hourly_distribution(messages),
            'peak_activity_hour': messages['hour'].mode()[0] if len(messages) > 0 else 9,
        }
        
        return features
    
    def _get_behavioral_features(self, conversation_id):
        """Extract behavioral patterns and engagement features"""
        conn = sqlite3.connect(self.db_path)
        
        # Get the customer's author_id for this conversation
        customer_query = '''
            SELECT DISTINCT cm.author_id
            FROM conversation_messages cm
            WHERE cm.conversation_id = ? AND cm.is_customer = 1
            LIMIT 1
        '''
        
        customer_result = pd.read_sql_query(customer_query, conn, params=(conversation_id,))
        
        if len(customer_result) == 0:
            conn.close()
            return self._get_default_behavioral_features()
        
        author_id = customer_result.iloc[0]['author_id']
        
        # Get historical conversation count for this customer
        history_query = '''
            SELECT COUNT(DISTINCT c.conversation_id) as conversation_count,
                   AVG(c.message_count) as avg_messages_per_conversation,
                   AVG(c.sentiment_score) as avg_sentiment,
                   COUNT(CASE WHEN c.is_resolved = 1 THEN 1 END) as resolved_conversations
            FROM conversations c
            WHERE c.participant_ids LIKE ?
        '''
        
        history_result = pd.read_sql_query(
            history_query, 
            conn, 
            params=(f'%{author_id}%',)
        )
        
        conn.close()
        
        if len(history_result) == 0:
            return self._get_default_behavioral_features()
        
        hist = history_result.iloc[0]
        
        features = {
            # Customer history
            'customer_conversation_count': hist['conversation_count'],
            'is_repeat_customer': 1 if hist['conversation_count'] > 1 else 0,
            'customer_avg_messages_per_conv': hist['avg_messages_per_conversation'],
            'customer_avg_sentiment': hist['avg_sentiment'],
            'customer_resolution_rate': hist['resolved_conversations'] / max(hist['conversation_count'], 1),
            
            # Engagement indicators
            'customer_engagement_level': self._calculate_engagement_level(hist),
            'customer_satisfaction_indicator': self._calculate_satisfaction_indicator(hist),
        }
        
        return features
    
    def _spans_business_hours(self, messages):
        """Check if conversation spans business hours (9 AM - 5 PM)"""
        hours = messages['hour'].unique()
        business_hours = set(range(9, 17))
        return 1 if any(hour in business_hours for hour in hours) else 0
    
    def _is_outside_business_hours(self, hour):
        """Check if hour is outside business hours"""
        return hour < 9 or hour >= 17
    
    def _calculate_hourly_distribution(self, messages):
        """Calculate message distribution across hours"""
        hourly_counts = messages['hour'].value_counts()
        total_hours = len(messages['hour'].unique())
        return hourly_counts.std() / hourly_counts.mean() if len(hourly_counts) > 1 else 0
    
    def _calculate_engagement_level(self, customer_history):
        """Calculate customer engagement level (0-1 scale)"""
        conv_count = customer_history['conversation_count']
        avg_messages = customer_history['avg_messages_per_conversation']
        
        # Higher engagement = more conversations with reasonable message counts
        engagement_score = min(conv_count / 10, 1.0) * 0.5  # Conversation frequency
        engagement_score += min(avg_messages / 5, 1.0) * 0.5  # Message engagement
        
        return engagement_score
    
    def _calculate_satisfaction_indicator(self, customer_history):
        """Calculate customer satisfaction indicator (0-1 scale)"""
        resolution_rate = customer_history['resolved_conversations'] / max(customer_history['conversation_count'], 1)
        avg_sentiment = customer_history['avg_sentiment']
        
        # Normalize sentiment from [-1, 1] to [0, 1]
        normalized_sentiment = (avg_sentiment + 1) / 2
        
        # Combine resolution rate and sentiment
        satisfaction = (resolution_rate * 0.7) + (normalized_sentiment * 0.3)
        
        return satisfaction
    
    def _get_default_message_features(self):
        """Return default message features when no data available"""
        return {
            'total_messages': 0,
            'customer_message_count': 0,
            'agent_message_count': 0,
            'customer_agent_ratio': 0,
            'avg_message_length': 0,
            'avg_customer_message_length': 0,
            'avg_agent_message_length': 0,
            'max_message_length': 0,
            'min_message_length': 0,
            'last_message_by_customer': 0,
            'first_message_by_customer': 1,
            'max_consecutive_customer_messages': 0,
            'max_consecutive_agent_messages': 0,
            'avg_consecutive_run_length': 0,
            'total_conversation_turns': 0
        }
    
    def _get_default_response_time_features(self):
        """Return default response time features"""
        return {
            'conversation_duration_minutes': 0,
            'avg_response_time_minutes': 0,
            'max_response_time_minutes': 0,
            'min_response_time_minutes': 0,
            'std_response_time_minutes': 0,
            'avg_customer_response_time': 0,
            'avg_agent_response_time': 0,
            'message_velocity': 0,
            'total_response_exchanges': 0
        }
    
    def _get_default_temporal_features(self):
        """Return default temporal features"""
        current_hour = datetime.now().hour
        return {
            'conversation_start_hour': current_hour,
            'conversation_spans_business_hours': 1 if 9 <= current_hour <= 17 else 0,
            'conversation_during_weekend': 1 if datetime.now().weekday() >= 5 else 0,
            'started_outside_business_hours': 1 if current_hour < 9 or current_hour >= 17 else 0,
            'continued_outside_business_hours': 0,
            'messages_per_hour_distribution': 0,
            'peak_activity_hour': current_hour
        }
    
    def _get_default_behavioral_features(self):
        """Return default behavioral features"""
        return {
            'customer_conversation_count': 1,
            'is_repeat_customer': 0,
            'customer_avg_messages_per_conv': 1,
            'customer_avg_sentiment': 0,
            'customer_resolution_rate': 0,
            'customer_engagement_level': 0.5,
            'customer_satisfaction_indicator': 0.5
        }
    
    def extract_batch_features(self, conversation_ids):
        """Extract features for multiple conversations efficiently"""
        all_features = []
        
        for conv_id in conversation_ids:
            try:
                features = self.extract_customer_features(conv_id)
                all_features.append(features)
            except Exception as e:
                print(f"Error extracting features for conversation {conv_id}: {e}")
                # Add minimal features to keep batch processing going
                all_features.append({
                    'conversation_id': conv_id,
                    'error': str(e),
                    **self._get_minimal_features()
                })
        
        return pd.DataFrame(all_features)
    
    def _get_minimal_features(self):
        """Get minimal feature set for error cases"""
        return {
            'message_count': 1,
            'is_resolved': 0,
            'sentiment_score': 0,
            'urgency_score': 0,
            'complexity_score': 0,
            'cohort': 'standard_support'
        }
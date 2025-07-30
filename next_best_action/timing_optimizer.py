import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class TimingOptimizer:
    """
    Timing optimization for NBA actions using ML-enhanced rules
    
    Considers:
    - Customer urgency patterns
    - Channel-specific optimal timing
    - Business hours and timezone
    - Customer response patterns
    - Escalation prevention
    """
    
    def __init__(self):
        self.urgency_model = LinearRegression()
        self.response_time_model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Business rules for timing
        self.business_hours = {
            'start': 9,  # 9 AM
            'end': 17,   # 5 PM
            'timezone': 'UTC'
        }
        
        self.channel_timing_rules = {
            'twitter_dm_reply': {
                'immediate_threshold': 0.7,  # urgency threshold for immediate response
                'standard_delay_minutes': 30,
                'max_delay_hours': 2
            },
            'email_reply': {
                'immediate_threshold': 0.8,
                'standard_delay_hours': 2,
                'max_delay_hours': 24
            },
            'scheduling_phone_call': {
                'immediate_threshold': 0.9,
                'standard_delay_minutes': 60,
                'max_delay_hours': 4,
                'business_hours_only': True
            }
        }
    
    def train(self, training_data):
        """
        Train timing optimization models
        
        Args:
            training_data: DataFrame with conversation features and timing patterns
        """
        print("Training timing optimization models...")
        
        # Prepare features for urgency prediction
        timing_features = [
            'sentiment_score', 'complexity_score', 'message_count',
            'customer_agent_ratio', 'max_consecutive_customer_messages'
        ]
        
        if len(training_data) < 10:
            print("Insufficient data for timing model training. Using rule-based approach.")
            return
        
        try:
            X = training_data[timing_features].fillna(0)
            
            # Create synthetic urgency escalation score based on conversation patterns
            y_urgency = self._calculate_urgency_escalation(training_data)
            
            # Train urgency escalation model
            if len(np.unique(y_urgency)) > 1:  # Check for variance
                X_scaled = self.scaler.fit_transform(X)
                self.urgency_model.fit(X_scaled, y_urgency)
                
                # Calculate model performance
                score = self.urgency_model.score(X_scaled, y_urgency)
                print(f"Timing model R² score: {score:.3f}")
                
                self.is_trained = True
            else:
                print("No variance in urgency patterns. Using rule-based timing.")
                
        except Exception as e:
            print(f"Error training timing models: {e}")
    
    def _calculate_urgency_escalation(self, df):
        """Calculate urgency escalation score based on conversation patterns"""
        escalation_scores = []
        
        for _, row in df.iterrows():
            score = 0
            
            # Negative sentiment increases urgency over time
            if row.get('sentiment_score', 0) < -0.3:
                score += 0.3
            
            # Long conversations suggest increasing urgency
            msg_count = row.get('message_count', 1)
            if msg_count > 5:
                score += min(msg_count / 10, 0.4)
            
            # Multiple consecutive customer messages = escalation
            consecutive = row.get('max_consecutive_customer_messages', 0)
            if consecutive > 2:
                score += min(consecutive / 5, 0.3)
            
            # High complexity with duration suggests urgency growth
            if row.get('complexity_score', 0) > 0.6 and row.get('conversation_duration_minutes', 0) > 60:
                score += 0.2
            
            escalation_scores.append(min(score, 1.0))
        
        return np.array(escalation_scores)
    
    def calculate_optimal_time(self, customer_data, channel):
        """
        Calculate optimal send time for the action
        
        Args:
            customer_data: dict with customer features
            channel: selected channel for communication
            
        Returns:
            dict: optimal timing information
        """
        now = datetime.now()
        
        # Get base timing from rules
        base_timing = self._get_base_timing(customer_data, channel, now)
        
        # Apply ML adjustments if model is trained
        if self.is_trained:
            ml_adjustment = self._get_ml_timing_adjustment(customer_data)
            adjusted_time = base_timing['send_time'] - timedelta(minutes=ml_adjustment * 60)
        else:
            adjusted_time = base_timing['send_time']
        
        # Ensure business hours compliance for phone calls
        final_time = self._ensure_business_hours(adjusted_time, channel)
        
        return {
            'send_time': final_time,
            'reasoning': base_timing['reasoning'],
            'urgency_level': self._classify_urgency(customer_data),
            'estimated_delay_minutes': (final_time - now).total_seconds() / 60
        }
    
    def _get_base_timing(self, customer_data, channel, current_time):
        """Get base timing using rule-based logic"""
        urgency = customer_data.get('urgency_score', 0)
        sentiment = customer_data.get('sentiment_score', 0)
        message_count = customer_data.get('message_count', 1)
        
        rules = self.channel_timing_rules.get(channel, self.channel_timing_rules['twitter_dm_reply'])
        
        # Immediate response conditions
        if (urgency > rules['immediate_threshold'] or 
            sentiment < -0.6 or 
            message_count > 8):
            
            delay_minutes = 5  # Almost immediate
            reasoning = "Immediate response due to high urgency/negative sentiment"
            
        # Standard timing
        else:
            if 'standard_delay_minutes' in rules:
                delay_minutes = rules['standard_delay_minutes']
            else:
                delay_minutes = rules.get('standard_delay_hours', 1) * 60
            
            reasoning = f"Standard {channel.replace('_', ' ')} response timing"
        
        # Apply sentiment-based adjustments
        if sentiment < -0.4:
            delay_minutes = max(delay_minutes * 0.5, 5)  # Faster for negative sentiment
            reasoning += " (accelerated for negative sentiment)"
        
        # Apply complexity adjustments
        complexity = customer_data.get('complexity_score', 0)
        if complexity > 0.7 and channel == 'email_reply':
            delay_minutes *= 1.5  # More time for complex email preparation
            reasoning += " (extended for complex issue preparation)"
        
        send_time = current_time + timedelta(minutes=delay_minutes)
        
        return {
            'send_time': send_time,
            'reasoning': reasoning
        }
    
    def _get_ml_timing_adjustment(self, customer_data):
        """Get ML-based timing adjustment (in hours)"""
        try:
            timing_features = [
                'sentiment_score', 'complexity_score', 'message_count',
                'customer_agent_ratio', 'max_consecutive_customer_messages'
            ]
            
            features = []
            for feature in timing_features:
                features.append(customer_data.get(feature, 0))
            
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Predict urgency escalation (0-1 scale)
            urgency_escalation = self.urgency_model.predict(X_scaled)[0]
            
            # Convert to timing adjustment (hours to subtract from base timing)
            # Higher escalation = faster response
            return urgency_escalation * 2  # Max 2 hours acceleration
            
        except Exception:
            return 0  # No adjustment if ML fails
    
    def _ensure_business_hours(self, proposed_time, channel):
        """Ensure timing complies with business hours for phone calls"""
        if channel != 'scheduling_phone_call':
            return proposed_time
        
        # Check if proposed time is during business hours
        hour = proposed_time.hour
        
        if self.business_hours['start'] <= hour <= self.business_hours['end']:
            return proposed_time
        
        # If outside business hours, move to next business day
        if hour < self.business_hours['start']:
            # Too early - move to start of business day
            next_business_time = proposed_time.replace(
                hour=self.business_hours['start'], 
                minute=0, 
                second=0
            )
        else:
            # Too late - move to next business day start
            next_business_time = (proposed_time + timedelta(days=1)).replace(
                hour=self.business_hours['start'], 
                minute=0, 
                second=0
            )
        
        return next_business_time
    
    def _classify_urgency(self, customer_data):
        """Classify customer urgency level"""
        urgency = customer_data.get('urgency_score', 0)
        sentiment = customer_data.get('sentiment_score', 0)
        message_count = customer_data.get('message_count', 1)
        
        # High urgency conditions
        if urgency > 0.8 or sentiment < -0.6 or message_count > 10:
            return "critical"
        elif urgency > 0.6 or sentiment < -0.4 or message_count > 6:
            return "high"
        elif urgency > 0.3 or sentiment < -0.2 or message_count > 3:
            return "medium"
        else:
            return "low"
    
    def get_optimal_daily_schedule(self, customers_batch):
        """
        Generate optimal daily schedule for batch of customers
        
        Returns prioritized schedule with time slots
        """
        schedule = []
        
        for _, customer in customers_batch.iterrows():
            timing_info = self.calculate_optimal_time(
                customer.to_dict(), 
                customer.get('predicted_channel', 'twitter_dm_reply')
            )
            
            schedule.append({
                'customer_id': customer.get('conversation_id'),
                'send_time': timing_info['send_time'],
                'urgency_level': timing_info['urgency_level'],
                'channel': customer.get('predicted_channel', 'twitter_dm_reply'),
                'estimated_delay': timing_info['estimated_delay_minutes']
            })
        
        # Sort by urgency and send time
        urgency_priority = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        schedule.sort(key=lambda x: (urgency_priority[x['urgency_level']], x['send_time']))
        
        return pd.DataFrame(schedule)
    
    def get_metrics(self):
        """Return timing optimization metrics"""
        return {
            'is_trained': self.is_trained,
            'model_type': 'LinearRegression',
            'business_hours': self.business_hours,
            'channels_supported': list(self.channel_timing_rules.keys())
        }
    
    def analyze_timing_patterns(self, historical_data):
        """Analyze historical timing patterns for insights"""
        if len(historical_data) == 0:
            return {}
        
        analysis = {
            'avg_response_time_by_urgency': {},
            'resolution_rate_by_timing': {},
            'optimal_windows': {}
        }
        
        # Group by urgency levels
        for urgency_level in ['low', 'medium', 'high', 'critical']:
            urgency_data = historical_data[
                historical_data['urgency_score'].between(
                    self._get_urgency_range(urgency_level)[0],
                    self._get_urgency_range(urgency_level)[1]
                )
            ]
            
            if len(urgency_data) > 0:
                avg_response = urgency_data.get('avg_response_time_minutes', pd.Series()).mean()
                analysis['avg_response_time_by_urgency'][urgency_level] = avg_response
        
        return analysis
    
    def _get_urgency_range(self, level):
        """Get urgency score range for classification level"""
        ranges = {
            'low': (0, 0.3),
            'medium': (0.3, 0.6),
            'high': (0.6, 0.8),
            'critical': (0.8, 1.0)
        }
        return ranges.get(level, (0, 1.0))
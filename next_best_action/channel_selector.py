import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class ChannelSelector:
    """
    ML-based channel selection using Random Forest
    
    Features considered:
    - Urgency score
    - Sentiment score  
    - Complexity score
    - Message count
    - Conversation duration
    - Customer response patterns
    - Cohort information
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = [
            'urgency_score', 'sentiment_score', 'complexity_score',
            'message_count', 'conversation_duration_minutes',
            'avg_response_time_minutes', 'customer_agent_ratio',
            'max_consecutive_customer_messages', 'message_velocity',
            'avg_customer_message_length'
        ]
        self.is_trained = False
        self.channel_rules = self._define_fallback_rules()
        
    def _define_fallback_rules(self):
        """Define rule-based fallback for channel selection"""
        return {
            'scheduling_phone_call': {
                'conditions': [
                    ('urgency_score', '>', 0.7),
                    ('sentiment_score', '<', -0.4),
                    ('message_count', '>', 5),
                    ('complexity_score', '>', 0.6)
                ],
                'priority': 3
            },
            'email_reply': {
                'conditions': [
                    ('complexity_score', '>', 0.6),
                    ('avg_customer_message_length', '>', 100),
                    ('conversation_duration_minutes', '>', 60)
                ],
                'priority': 2
            },
            'twitter_dm_reply': {
                'conditions': [
                    ('urgency_score', '>', 0.3),
                    ('sentiment_score', '>', -0.3),
                    ('message_count', '<', 6)
                ],
                'priority': 1
            }
        }
    
    def train(self, training_data):
        """
        Train the channel selection model
        
        Args:
            training_data: DataFrame with conversation features
        """
        print("Training channel selection model...")
        
        # Create synthetic labels based on business rules (since we don't have historical channel data)
        X, y = self._create_training_data(training_data)
        
        if len(X) < 10:
            print("Insufficient training data for ML model. Using rule-based approach.")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Channel selection model accuracy: {accuracy:.3f}")
        print("\nFeature importance:")
        for feature, importance in zip(self.feature_names, self.model.feature_importances_):
            print(f"  {feature}: {importance:.3f}")
        
        self.is_trained = True
        
        # Save model
        self._save_model()
    
    def _create_training_data(self, df):
        """Create training data with synthetic labels based on business logic"""
        # Prepare features
        X = df[self.feature_names].fillna(0)
        
        # Create synthetic channel labels based on business rules
        y = []
        for _, row in df.iterrows():
            channel = self._apply_business_rules(row)
            y.append(channel)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X.values, y_encoded
    
    def _apply_business_rules(self, customer_data):
        """Apply business rules to determine optimal channel"""
        urgency = customer_data.get('urgency_score', 0)
        sentiment = customer_data.get('sentiment_score', 0)
        complexity = customer_data.get('complexity_score', 0)
        message_count = customer_data.get('message_count', 1)
        duration = customer_data.get('conversation_duration_minutes', 0)
        avg_msg_length = customer_data.get('avg_customer_message_length', 0)
        
        # High urgency + negative sentiment = phone call
        if urgency > 0.7 and sentiment < -0.4:
            return 'scheduling_phone_call'
        
        # Complex issue + frustrated customer = phone call
        if complexity > 0.7 and (sentiment < -0.3 or message_count > 6):
            return 'scheduling_phone_call'
        
        # Long conversation = phone call for efficiency
        if message_count > 8:
            return 'scheduling_phone_call'
        
        # Complex technical issue + patient customer = email
        if complexity > 0.6 and avg_msg_length > 100 and sentiment > -0.3:
            return 'email_reply'
        
        # Long duration + detailed responses = email
        if duration > 120 and avg_msg_length > 80:
            return 'email_reply'
        
        # Default to Twitter for quick, simple issues
        return 'twitter_dm_reply'
    
    def predict_channel(self, customer_data):
        """
        Predict optimal channel for customer
        
        Args:
            customer_data: dict or Series with customer features
            
        Returns:
            dict: channel prediction with confidence
        """
        if self.is_trained:
            return self._ml_predict(customer_data)
        else:
            return self._rule_based_predict(customer_data)
    
    def _ml_predict(self, customer_data):
        """Make ML-based prediction"""
        try:
            # Prepare features
            features = []
            for feature_name in self.feature_names:
                features.append(customer_data.get(feature_name, 0))
            
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Get prediction and probabilities
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get channel name
            channel = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            return {
                'channel': channel,
                'confidence': confidence,
                'method': 'ml'
            }
            
        except Exception as e:
            print(f"ML prediction failed: {e}. Falling back to rules.")
            return self._rule_based_predict(customer_data)
    
    def _rule_based_predict(self, customer_data):
        """Make rule-based prediction as fallback"""
        channel_scores = {}
        
        for channel, rules in self.channel_rules.items():
            score = 0
            max_score = len(rules['conditions'])
            
            for field, operator, threshold in rules['conditions']:
                if field in customer_data:
                    value = customer_data[field]
                    
                    if operator == '>' and value > threshold:
                        score += 1
                    elif operator == '<' and value < threshold:
                        score += 1
                    elif operator == '==' and value == threshold:
                        score += 1
            
            # Normalize score and apply priority weight
            normalized_score = score / max_score if max_score > 0 else 0
            priority_weight = 1 / rules['priority']  # Lower priority number = higher weight
            channel_scores[channel] = normalized_score * priority_weight
        
        # Select best channel
        best_channel = max(channel_scores.items(), key=lambda x: x[1])
        
        return {
            'channel': best_channel[0],
            'confidence': min(best_channel[1], 1.0),
            'method': 'rule_based'
        }
    
    def _save_model(self):
        """Save trained model to disk"""
        if not os.path.exists('models'):
            os.makedirs('models')
        
        joblib.dump(self.model, 'models/channel_selector_model.pkl')
        joblib.dump(self.scaler, 'models/channel_selector_scaler.pkl')
        joblib.dump(self.label_encoder, 'models/channel_selector_encoder.pkl')
        
        print("Channel selection model saved to models/")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            self.model = joblib.load('models/channel_selector_model.pkl')
            self.scaler = joblib.load('models/channel_selector_scaler.pkl')
            self.label_encoder = joblib.load('models/channel_selector_encoder.pkl')
            self.is_trained = True
            print("Channel selection model loaded successfully")
        except FileNotFoundError:
            print("No saved model found. Model will be trained on first use.")
    
    def get_metrics(self):
        """Return model performance metrics"""
        return {
            'is_trained': self.is_trained,
            'model_type': 'RandomForestClassifier',
            'feature_count': len(self.feature_names),
            'channels': ['twitter_dm_reply', 'email_reply', 'scheduling_phone_call']
        }
    
    def explain_prediction(self, customer_data, prediction_result):
        """Provide explanation for channel prediction"""
        channel = prediction_result['channel']
        confidence = prediction_result['confidence']
        method = prediction_result['method']
        
        explanation = f"Channel: {channel.replace('_', ' ').title()} (Confidence: {confidence:.2f}, Method: {method})\n"
        
        # Key factors
        urgency = customer_data.get('urgency_score', 0)
        sentiment = customer_data.get('sentiment_score', 0)
        complexity = customer_data.get('complexity_score', 0)
        message_count = customer_data.get('message_count', 1)
        
        explanation += f"Key factors:\n"
        explanation += f"  - Urgency: {urgency:.2f}\n"
        explanation += f"  - Sentiment: {sentiment:.2f}\n"
        explanation += f"  - Complexity: {complexity:.2f}\n"
        explanation += f"  - Message Count: {message_count}\n"
        
        # Channel-specific reasoning
        if channel == 'scheduling_phone_call':
            explanation += "Reasoning: High-touch interaction needed for complex/urgent/negative sentiment issues."
        elif channel == 'email_reply':
            explanation += "Reasoning: Complex issue requiring detailed documentation and comprehensive response."
        else:
            explanation += "Reasoning: Quick, efficient response appropriate for straightforward issues."
        
        return explanation
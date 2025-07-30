import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from next_best_action.channel_selector import ChannelSelector
from next_best_action.timing_optimizer import TimingOptimizer
from next_best_action.message_generator import MessageGenerator
from next_best_action.feature_engineer import FeatureEngineer

class NextBestActionEngine:
    """
    Hybrid NBA Engine combining ML models with rule-based logic
    
    Architecture:
    1. Feature Engineering: Extract advanced features from conversation data
    2. Channel Selection: ML model to predict optimal channel
    3. Timing Optimization: Rule-based with ML-informed urgency scoring
    4. Message Generation: Template-based with personalization
    """
    
    def __init__(self, db_path='riverline.db'):
        self.db_path = db_path
        self.feature_engineer = FeatureEngineer(db_path)
        self.channel_selector = ChannelSelector()
        self.timing_optimizer = TimingOptimizer()
        self.message_generator = MessageGenerator()
        
        # Train models on existing data
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize and train ML models using historical data"""
        print("Initializing NBA models...")
        
        # Load conversation features for training
        try:
            features_df = pd.read_csv('conversation_features.csv')
            if len(features_df) > 0:
                print(f"Training models on {len(features_df)} conversations")
                self.channel_selector.train(features_df)
                self.timing_optimizer.train(features_df)
                print("Models trained successfully")
            else:
                print("No training data available - using default models")
        except FileNotFoundError:
            print("No conversation features found - using default models")
    
    def predict_next_action(self, conversation_id=None, customer_data=None):
        """
        Main prediction method for NBA
        
        Args:
            conversation_id: ID of conversation to analyze
            customer_data: Pre-computed customer features (optional)
        
        Returns:
            dict: NBA recommendation in required JSON format
        """
        if customer_data is None and conversation_id is not None:
            customer_data = self.feature_engineer.extract_customer_features(conversation_id)
        elif customer_data is None:
            raise ValueError("Either conversation_id or customer_data must be provided")
        
        # Ensure customer_data is a dict
        if hasattr(customer_data, 'to_dict'):
            customer_data = customer_data.to_dict()
        
        # Step 1: Select optimal channel
        channel_prediction = self.channel_selector.predict_channel(customer_data)
        
        # Step 2: Optimize timing
        optimal_timing = self.timing_optimizer.calculate_optimal_time(
            customer_data, channel_prediction['channel']
        )
        
        # Step 3: Generate personalized message
        message = self.message_generator.generate_message(
            customer_data, channel_prediction['channel']
        )
        
        # Step 4: Create comprehensive reasoning
        reasoning = self._generate_comprehensive_reasoning(
            customer_data, channel_prediction, optimal_timing
        )
        
        return {
            "customer_id": str(customer_data.get('author_id', customer_data.get('customer_id', 'unknown'))),
            "channel": channel_prediction['channel'],
            "send_time": optimal_timing['send_time'].isoformat(),
            "message": message,
            "reasoning": reasoning
        }
    
    def _generate_comprehensive_reasoning(self, customer_data, channel_prediction, timing_info):
        """Generate detailed reasoning for the NBA decision"""
        reasoning_parts = []
        
        # Channel reasoning
        channel = channel_prediction['channel']
        confidence = channel_prediction.get('confidence', 0)
        
        if channel == 'scheduling_phone_call':
            reasoning_parts.append(
                f"Phone call selected (confidence: {confidence:.2f}) due to "
                f"high complexity (score: {customer_data.get('complexity_score', 0):.2f}) "
                f"and negative sentiment (score: {customer_data.get('sentiment_score', 0):.2f}). "
                f"Personal interaction will build trust and resolve complex issues efficiently."
            )
        elif channel == 'email_reply':
            reasoning_parts.append(
                f"Email selected (confidence: {confidence:.2f}) for detailed response capability. "
                f"Customer shows patience (avg message length: {customer_data.get('avg_customer_message_length', 0):.0f} chars) "
                f"and issue complexity ({customer_data.get('complexity_score', 0):.2f}) requires comprehensive documentation."
            )
        else:  # twitter_dm_reply
            reasoning_parts.append(
                f"Twitter reply selected (confidence: {confidence:.2f}) for quick resolution. "
                f"Moderate urgency (score: {customer_data.get('urgency_score', 0):.2f}) "
                f"and customer's platform preference suggest immediate social media response."
            )
        
        # Timing reasoning
        urgency = customer_data.get('urgency_score', 0)
        if urgency > 0.7:
            reasoning_parts.append(f"Immediate response scheduled due to high urgency (score: {urgency:.2f}).")
        elif customer_data.get('sentiment_score', 0) < -0.4:
            reasoning_parts.append("Quick response time to address negative sentiment and prevent escalation.")
        else:
            reasoning_parts.append(f"Standard response timing based on {channel.replace('_', ' ')} best practices.")
        
        # Cohort-specific reasoning
        cohort = customer_data.get('cohort', 'standard_support')
        cohort_insights = {
            'frustrated_customer': "Customer frustration detected - prioritizing empathetic, immediate response.",
            'escalation_needed': "Complex issue requiring senior support intervention.",
            'ping_pong': "Multiple exchanges detected - direct communication will be more efficient.",
            'patient_detailed': "Customer provides comprehensive information - detailed response warranted.",
            'abandoned_conversation': "Customer disengagement risk - proactive re-engagement strategy.",
            'urgent_technical': "Technical complexity requires specialized support channel."
        }
        
        if cohort in cohort_insights:
            reasoning_parts.append(cohort_insights[cohort])
        
        # Prediction confidence
        reasoning_parts.append(
            f"Decision based on conversation analysis of {customer_data.get('message_count', 1)} messages "
            f"over {customer_data.get('conversation_duration_minutes', 0):.0f} minutes."
        )
        
        return " ".join(reasoning_parts)
    
    def process_batch(self, customers_df=None, limit=1000, filters=None):
        """
        Process a batch of customers for NBA recommendations
        
        Args:
            customers_df: DataFrame of customers (optional, will load from DB if None)
            limit: Maximum number of customers to process
            filters: Additional filters for customer selection
        
        Returns:
            pandas.DataFrame: NBA recommendations with additional analysis columns
        """
        print(f"Processing batch NBA recommendations (limit: {limit})")
        
        # Load customer data if not provided
        if customers_df is None:
            customers_df = self._load_customer_data(limit, filters)
        
        # Filter to open conversations only
        open_conversations = customers_df[customers_df['is_resolved'] == 0].head(limit)
        
        if len(open_conversations) == 0:
            print("No open conversations found for NBA processing")
            return pd.DataFrame()
        
        print(f"Processing {len(open_conversations)} open conversations")
        
        results = []
        processed = 0
        
        for idx, customer in open_conversations.iterrows():
            try:
                # Get NBA recommendation
                action = self.predict_next_action(customer_data=customer)
                
                # Add additional analysis columns
                action['chat_log'] = self._generate_chat_log(customer)
                action['issue_status'] = self._predict_issue_status(customer, action)
                action['confidence_score'] = self._calculate_confidence_score(customer, action)
                action['expected_resolution_time'] = self._estimate_resolution_time(customer, action)
                
                results.append(action)
                processed += 1
                
                if processed % 100 == 0:
                    print(f"Processed {processed}/{len(open_conversations)} customers")
                    
            except Exception as e:
                print(f"Error processing customer {customer.get('conversation_id', 'unknown')}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        # Add batch analysis
        self._analyze_batch_results(results_df)
        
        return results_df
    
    def _load_customer_data(self, limit, filters):
        """Load customer data from conversation features"""
        try:
            features_df = pd.read_csv('conversation_features.csv')
            
            # Apply filters if provided
            if filters:
                for key, value in filters.items():
                    if key in features_df.columns:
                        if isinstance(value, list):
                            features_df = features_df[features_df[key].isin(value)]
                        else:
                            features_df = features_df[features_df[key] == value]
            
            return features_df.head(limit)
            
        except FileNotFoundError:
            print("Conversation features file not found. Loading from database...")
            return self._load_from_database(limit)
    
    def _load_from_database(self, limit):
        """Fallback to load data directly from database"""
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT c.*, 
                   COALESCE(c.cohort, 'standard_support') as cohort,
                   COALESCE(c.ml_cohort, 0) as ml_cohort
            FROM conversations c
            WHERE c.is_resolved = 0
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        return df
    
    def _generate_chat_log(self, customer_data):
        """Generate readable chat log from conversation messages"""
        conversation_id = customer_data.get('conversation_id')
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT cm.is_customer, cm.message_text, rt.created_at
                FROM conversation_messages cm
                JOIN raw_tweets rt ON cm.tweet_id = rt.tweet_id
                WHERE cm.conversation_id = ?
                ORDER BY cm.sequence_number
                LIMIT 20
            '''
            messages = pd.read_sql_query(query, conn, params=(conversation_id,))
            conn.close()
            
            if len(messages) == 0:
                return f"Customer: [Initial message for conversation {conversation_id}]"
            
            chat_log = []
            for _, msg in messages.iterrows():
                speaker = "Customer" if msg['is_customer'] else "Support_agent"
                # Truncate long messages
                message_text = msg['message_text'][:150] + "..." if len(msg['message_text']) > 150 else msg['message_text']
                chat_log.append(f"{speaker}: {message_text}")
            
            return "\n".join(chat_log)
            
        except Exception as e:
            return f"Customer: [Error loading conversation {conversation_id}: {e}]"
    
    def _predict_issue_status(self, customer_data, action):
        """Predict post-action issue status using decision logic"""
        channel = action['channel']
        urgency = customer_data.get('urgency_score', 0)
        complexity = customer_data.get('complexity_score', 0)
        sentiment = customer_data.get('sentiment_score', 0)
        message_count = customer_data.get('message_count', 1)
        
        # Phone call predictions
        if channel == 'scheduling_phone_call':
            if urgency > 0.8 and complexity > 0.7:
                return "escalated"
            elif sentiment < -0.6:
                return "pending_internal_review"
            else:
                return "resolved"
        
        # Email predictions
        elif channel == 'email_reply':
            if complexity > 0.8:
                return "pending_customer_reply"
            elif message_count > 8:
                return "escalated"
            else:
                return "resolved"
        
        # Twitter predictions
        else:  # twitter_dm_reply
            if urgency < 0.3 and sentiment > -0.2 and complexity < 0.5:
                return "resolved"
            elif sentiment < -0.5:
                return "pending_escalation"
            else:
                return "pending_customer_reply"
    
    def _calculate_confidence_score(self, customer_data, action):
        """Calculate confidence score for the NBA recommendation"""
        # Base confidence on data completeness and feature strength
        base_confidence = 0.5
        
        # Boost confidence based on clear indicators
        urgency = customer_data.get('urgency_score', 0)
        sentiment = customer_data.get('sentiment_score', 0)
        complexity = customer_data.get('complexity_score', 0)
        
        # Strong indicators boost confidence
        if urgency > 0.8 or urgency < 0.2:  # Very high or very low urgency
            base_confidence += 0.2
        if sentiment < -0.6 or sentiment > 0.6:  # Strong sentiment
            base_confidence += 0.2
        if complexity > 0.8 or complexity < 0.2:  # Very complex or very simple
            base_confidence += 0.1
        
        # Message count provides context
        message_count = customer_data.get('message_count', 1)
        if message_count > 5:  # Rich conversation history
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _estimate_resolution_time(self, customer_data, action):
        """Estimate expected resolution time based on action and customer profile"""
        channel = action['channel']
        complexity = customer_data.get('complexity_score', 0)
        urgency = customer_data.get('urgency_score', 0)
        
        # Base times by channel (in hours)
        base_times = {
            'twitter_dm_reply': 2,
            'email_reply': 8,
            'scheduling_phone_call': 4
        }
        
        base_time = base_times[channel]
        
        # Adjust based on complexity
        complexity_multiplier = 1 + (complexity * 2)  # 1x to 3x
        
        # Adjust based on urgency (high urgency = faster resolution)
        urgency_multiplier = 1 - (urgency * 0.5)  # 1x to 0.5x
        
        estimated_hours = base_time * complexity_multiplier * urgency_multiplier
        
        return f"{estimated_hours:.1f} hours"
    
    def _analyze_batch_results(self, results_df):
        """Analyze and print batch processing results"""
        if len(results_df) == 0:
            return
        
        print(f"\n{'='*50}")
        print("NBA BATCH ANALYSIS SUMMARY")
        print(f"{'='*50}")
        
        # Channel distribution
        print(f"\nChannel Distribution:")
        channel_dist = results_df['channel'].value_counts()
        for channel, count in channel_dist.items():
            percentage = count / len(results_df) * 100
            print(f"  {channel.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Issue status predictions
        print(f"\nPredicted Issue Status:")
        status_dist = results_df['issue_status'].value_counts()
        for status, count in status_dist.items():
            percentage = count / len(results_df) * 100
            print(f"  {status.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Resolution predictions
        resolved_predictions = len(results_df[results_df['issue_status'] == 'resolved'])
        resolution_rate = resolved_predictions / len(results_df) * 100
        print(f"\nExpected Resolution Rate: {resolution_rate:.1f}%")
        
        # Confidence analysis
        avg_confidence = results_df['confidence_score'].mean()
        print(f"Average Confidence Score: {avg_confidence:.2f}")
        
        # Save detailed results
        output_file = f"nba_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
    
    def get_model_performance_metrics(self):
        """Return performance metrics for the NBA models"""
        return {
            'channel_selector': self.channel_selector.get_metrics(),
            'timing_optimizer': self.timing_optimizer.get_metrics(),
            'last_updated': datetime.now().isoformat()
        }
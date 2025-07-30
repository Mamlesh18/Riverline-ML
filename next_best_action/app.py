import sqlite3
import pandas as pd
import csv
import logging
from .decision_engine import DecisionEngine
from .channel_optimizer import ChannelOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NextBestActionEngine:
    def __init__(self, db_name='riverline.db'):
        self.db_path = db_name
        self.conn = None
        self.decision_engine = DecisionEngine()
        self.channel_optimizer = ChannelOptimizer()
        self._initialize_connection()
        
    def _initialize_connection(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect(self.db_path)
        print("🔗 Connected to database for NBA analysis")
    
    def get_customer_data(self):
        """Get customer data with conversation analysis, cohort information, and complete chat history"""
        query = '''
            SELECT 
                ca.conversation_id,
                ca.is_resolved,
                ca.tags,
                ca.nature_of_request,
                ca.customer_sentiment,
                ca.urgency_level,
                ca.conversation_type,
                ca.customer_behavior,
                gc.author_id as customer_id,
                gc.message_text as latest_message,
                gc.created_at as latest_interaction,
                GROUP_CONCAT(cc.cohort_name, '; ') as customer_cohorts,
                COUNT(DISTINCT gc.tweet_id) as total_messages,
                GROUP_CONCAT(
                    gc.created_at || ' | ' || gc.author_id || ': ' || gc.message_text, 
                    ' ### '  -- Use a different separator that won't cause CSV issues
                ) as chat_history
            FROM conversation_analysis ca
            JOIN grouped_conversations gc ON ca.conversation_id = gc.conversation_id
            LEFT JOIN conversation_cohort_mapping ccm ON ca.conversation_id = ccm.conversation_id
            LEFT JOIN customer_cohorts cc ON ccm.cohort_id = cc.id
            WHERE ca.is_resolved = 0  -- Only unresolved conversations
            GROUP BY ca.conversation_id, ca.is_resolved, ca.tags, ca.nature_of_request, 
                    ca.customer_sentiment, ca.urgency_level, ca.conversation_type, 
                    ca.customer_behavior
            ORDER BY ca.conversation_id
        '''
        
        return pd.read_sql_query(query, self.conn)

    
    def generate_next_best_actions(self):
        """Generate next best actions for all unresolved conversations"""
        try:
            print("\n" + "="*80)
            print("NEXT BEST ACTION ENGINE - GENERATING RECOMMENDATIONS")
            print("="*80)
            
            # Get customer data
            customer_data = self.get_customer_data()
            
            if len(customer_data) == 0:
                print("❌ No unresolved conversations found for NBA analysis")
                return []
            
            print(f"📊 Found {len(customer_data)} unresolved conversations for analysis")
            
            nba_recommendations = []
            
            # Generate recommendations for each customer
            for idx, customer in customer_data.iterrows():
                print(f"\n🔍 Analyzing customer {idx + 1}/{len(customer_data)}")
                print(f"   Customer ID: {customer['customer_id']}")
                print(f"   Conversation ID: {customer['conversation_id']}")
                print(f"   Sentiment: {customer['customer_sentiment']}")
                print(f"   Urgency: {customer['urgency_level']}")
                print(f"   Cohorts: {customer['customer_cohorts']}")
                
                # Generate NBA recommendation
                recommendation = self.decision_engine.generate_recommendation(customer)
                
                if recommendation:
                    # Optimize channel and timing
                    optimized_recommendation = self.channel_optimizer.optimize_recommendation(
                        recommendation, customer
                    )
                    
                    nba_recommendations.append(optimized_recommendation)
                    
                    print(f"   ✅ Recommendation: {optimized_recommendation['channel']}")
                    print(f"   📅 Send Time: {optimized_recommendation['send_time']}")
                else:
                    print("   ❌ Failed to generate recommendation")
                
                # Small delay to avoid rate limiting
                import time
                time.sleep(1)
            
            # Save recommendations to CSV
            self.save_recommendations_to_csv(nba_recommendations)
            
            print(f"\n🎯 Generated {len(nba_recommendations)} NBA recommendations")
            print("💾 Results saved to 'nba_results.csv'")
            
            return nba_recommendations
            
        except Exception as e:
            logger.error(f"Error generating NBA recommendations: {str(e)}")
            return []
    
    def save_recommendations_to_csv(self, recommendations):
        """Save NBA recommendations to CSV file with proper formatting for multi-line fields"""
        if not recommendations:
            print("⚠️  No recommendations to save")
            return
        
        csv_filename = 'nba_results.csv'
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['customer_id', 'conversation_id', 'channel', 'send_time', 'message', 'reasoning', 'issue_status', 'chat_history']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)  # Quote all fields
            
            # Write header
            writer.writeheader()
            
            # Write recommendations
            for rec in recommendations:
                # Clean chat history - replace problematic separators with readable format
                chat_history = rec.get('chat_history', '')
                if chat_history:
                    # Replace the database separator with a more readable format
                    chat_history = chat_history.replace(' ### ', ' | ')
                    # Remove any remaining newlines that might cause CSV issues
                    chat_history = chat_history.replace('\n', ' | ').replace('\r', '')
                
                writer.writerow({
                    'customer_id': rec['customer_id'],
                    'conversation_id': rec.get('conversation_id', ''),
                    'channel': rec['channel'],
                    'send_time': rec['send_time'],
                    'message': rec['message'],
                    'reasoning': rec['reasoning'],
                    'issue_status': rec.get('issue_status', 'pending_customer_reply'),
                    'chat_history': chat_history
                })
        
        print(f"📄 Results saved to '{csv_filename}'")


    
    def print_recommendations_summary(self, recommendations):
        """Print summary of NBA recommendations"""
        if not recommendations:
            print("❌ No recommendations to display")
            return
        
        print("\n" + "="*80)
        print("NBA RECOMMENDATIONS SUMMARY")
        print("="*80)
        
        # Channel distribution
        channels = {}
        for rec in recommendations:
            channel = rec['channel']
            channels[channel] = channels.get(channel, 0) + 1
        
        print("\n📊 CHANNEL DISTRIBUTION:")
        for channel, count in channels.items():
            print(f"   📱 {channel}: {count} recommendations")
        
        # Show sample recommendations
        print("\n🔍 SAMPLE RECOMMENDATIONS:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"\n{i}. Customer: {rec['customer_id']}")
            print(f"   Channel: {rec['channel']}")
            print(f"   Send Time: {rec['send_time']}")
            print(f"   Message: {rec['message'][:100]}...")
            print(f"   Reasoning: {rec['reasoning'][:150]}...")
        
        if len(recommendations) > 3:
            print(f"\n... and {len(recommendations) - 3} more recommendations")
        
        print("\n💾 Full results available in 'nba_results.csv'")
    
    def run_nba_analysis(self):
        """Run complete NBA analysis and return results"""
        try:
            # Generate recommendations
            recommendations = self.generate_next_best_actions()
            
            # Print summary
            self.print_recommendations_summary(recommendations)
            
            return {
                'status': 'success',
                'recommendations_generated': len(recommendations),
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"NBA analysis failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("🔗 Database connection closed")
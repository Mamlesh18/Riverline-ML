import pandas as pd
import logging
from data_pipeline.data_clean import DataCleaner
from data_pipeline.data_store import DataStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, db_name='riverline.db'):
        self.cleaner = DataCleaner()
        self.store = DataStore(db_name)
        
    def run(self, csv_path):
        """Main pipeline execution"""
        try:
            # Load data
            print("Loading data from CSV...")
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} records")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample data:\n{df.head()}")
            
            # Clean data
            print("Cleaning data...")
            cleaned_df = self.cleaner.clean_data(df)
            print(f"Cleaned data: {len(cleaned_df)} records")
            
            # Store raw tweets
            print("Storing raw tweets...")
            self.store.insert_raw_tweets(cleaned_df)
            
            # Build and store conversations
            print("Building conversations...")
            conversations = self.cleaner.build_conversations(cleaned_df)
            print(f"Built {len(conversations)} conversations")
            
            self.store.insert_conversations(conversations)
            
            return {
                'status': 'success',
                'records_processed': len(cleaned_df),
                'conversations_created': len(conversations)
            }
            
        except Exception as e:
            import traceback
            logger.error(f"Pipeline failed: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def print_sample_conversations(self, limit=10):
        """Print conversations in simple row order"""
        
        # Get conversations in row order (first 10 conversations in database)
        conversations = self.store.get_conversations_with_messages(limit, order_by='row')
        
        print(f"\n{'='*80}")
        print(f"FIRST {limit} CONVERSATIONS (ROW ORDER)")
        print(f"{'='*80}")
        
        for i, (conv_id, messages) in enumerate(conversations.items(), 1):
            print(f"\n🔹 GROUP {i} - Conversation ID: {conv_id}")
            print(f"📊 Total Messages: {len(messages)}")
            print(f"👥 Participants: {len(set(msg['author_id'] for msg in messages))}")
            print(f"⏰ Duration: {messages[0]['created_at']} to {messages[-1]['created_at']}")
            print("─" * 80)
            
            for seq, msg in enumerate(messages, 1):
                msg_type = "🟦 CUSTOMER" if msg['is_customer'] else "🟩 AGENT"
                timestamp = msg['created_at'][:19] if isinstance(msg['created_at'], str) else str(msg['created_at'])[:19]
                
                print(f"{seq:2d}. [{timestamp}] {msg_type}")
                print(f"    👤 {msg['author_id']}")
                print(f"    💬 {msg['message_text']}")
                
                # Show response relationships if they exist
                if msg['in_response_to_tweet_id'] and msg['in_response_to_tweet_id'] != '':
                    print(f"    ↳ Replying to tweet: {msg['in_response_to_tweet_id']}")
                
                print()
            
            print("=" * 80) 
            print(f"⏰ Duration: {messages[0]['created_at']} to {messages[-1]['created_at']}")
            print("─" * 80)
            
            
    
    def get_basic_stats(self):
        """Get basic pipeline statistics"""
        return self.store.get_basic_stats()
    
    def close(self):
        """Close database connection"""
        self.store.close()
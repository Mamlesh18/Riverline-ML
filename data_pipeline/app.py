import pandas as pd
import hashlib
import logging
from datetime import datetime
from data_pipeline.data_clean import DataCleaner
from data_pipeline.data_store import DataStore
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self, db_name='riverline.db'):
        self.cleaner = DataCleaner()
        self.store = DataStore(db_name)
        self.batch_size = 1000
        
    def calculate_file_hash(self, filepath):
        hash_md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def run(self, csv_path):
        file_hash = self.calculate_file_hash(csv_path)
        
        if self.store.check_if_processed(file_hash):
            logger.info(f"File {csv_path} already processed. Skipping.")
            return {
                'status': 'skipped',
                'reason': 'already_processed'
            }
        
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.store.log_ingestion_start(batch_id, file_hash)
        
        try:
            total_records = self._process_file(csv_path)
            self.store.log_ingestion_complete(batch_id, total_records)
            
            return {
                'status': 'success',
                'batch_id': batch_id,
                'records_processed': total_records
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    def run_data_quality_checks(self, df):
        """
        Perform comprehensive data quality checks
        """
        print("\n" + "="*50)
        print("DATA QUALITY ANALYSIS")
        print("="*50)
        
        # Basic statistics
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
        
        # Check inbound/outbound distribution
        inbound_count = df['inbound'].sum()
        outbound_count = len(df) - inbound_count
        print(f"Inbound messages (customers): {inbound_count} ({inbound_count/len(df)*100:.1f}%)")
        print(f"Outbound messages (agents): {outbound_count} ({outbound_count/len(df)*100:.1f}%)")
        
        # Check response patterns
        has_response_to = df['in_response_to_tweet_id'].notna() & (df['in_response_to_tweet_id'] != '')
        print(f"Messages with response_to_tweet_id: {has_response_to.sum()}")
        
        # Check for conversation threads
        response_ids = set(df[has_response_to]['in_response_to_tweet_id'])
        tweet_ids = set(df['tweet_id'].astype(str))
        valid_responses = len(response_ids.intersection(tweet_ids))
        print(f"Valid response references (parent tweet exists): {valid_responses}")
        print(f"Orphaned responses (parent not found): {len(response_ids) - valid_responses}")
        
        # Sample conversation analysis
        print("\nSample conversation threading:")
        sample_conversations = df[has_response_to].head(5)
        for _, row in sample_conversations.iterrows():
            parent_exists = row['in_response_to_tweet_id'] in tweet_ids
            print(f"Tweet {row['tweet_id']} -> Parent {row['in_response_to_tweet_id']} (exists: {parent_exists})")
        
        # Check text quality
        empty_text = df['text'].isna() | (df['text'] == '')
        print(f"\nEmpty/null text messages: {empty_text.sum()}")
        
        avg_text_length = df[~empty_text]['text'].str.len().mean()
        print(f"Average text length: {avg_text_length:.1f} characters")
        
        # Check for potential resolution indicators
        all_text = ' '.join(df['text'].fillna('').astype(str))
        resolution_keywords = ['thank', 'thanks', 'resolved', 'fixed', 'solved', 'working']
        keyword_counts = {keyword: all_text.lower().count(keyword) for keyword in resolution_keywords}
        print("\nResolution keyword analysis:")
        for keyword, count in keyword_counts.items():
            print(f"  '{keyword}': {count} occurrences")
        
        return df
    def _process_file(self, csv_path):
        total_records = 0
        all_conversations = {}
        all_metadata = {}
        
        # Load all data first for better conversation threading
        print("Loading full dataset for conversation analysis...")
        full_df = pd.read_csv(csv_path)
        print(f"Loaded {len(full_df)} total records")
        
        # Run data quality checks
        full_df = self.run_data_quality_checks(full_df)
        
        # Clean the full dataset
        print("Cleaning full dataset...")
        cleaned_full_df = self.cleaner.clean_raw_data(full_df)
        
        # Build conversations on the full dataset
        print("Building conversations from full dataset...")
        conversations, metadata = self.cleaner.build_conversations(cleaned_full_df)
        
        # Validate conversations
        conversations = self.cleaner.validate_conversations(conversations, cleaned_full_df)
        
        # Process in chunks for database insertion
        for chunk in pd.read_csv(csv_path, chunksize=self.batch_size):
            logger.info(f"Processing chunk with {len(chunk)} records...")
            
            cleaned_chunk = self.cleaner.clean_raw_data(chunk)
            
            try:
                self.store.insert_tweets_batch(cleaned_chunk)
            except Exception as e:
                logger.warning(f"Some tweets already exist, continuing: {str(e)}")
            
            total_records += len(cleaned_chunk)
            logger.info(f"Processed {total_records} records so far...")
        
        # Store all conversations
        logger.info("Storing conversation data...")
        self._store_conversations(conversations, metadata)
        
        # Print resolution statistics
        resolved_count = sum(1 for meta in metadata.values() if meta['is_resolved'])
        print(f"\nCONVERSATION RESOLUTION ANALYSIS:")
        print(f"Total conversations: {len(conversations)}")
        print(f"Resolved conversations: {resolved_count}")
        print(f"Open conversations: {len(conversations) - resolved_count}")
        print(f"Resolution rate: {resolved_count/len(conversations)*100:.1f}%")
        
        return total_records
    
    def _store_conversations(self, conversations, metadata):
        for conv_id, conv_metadata in metadata.items():
            self.store.insert_conversation(conv_metadata)
            
            if conv_id in conversations:
                messages_df = self.cleaner.prepare_messages_for_storage(
                    conv_id, conversations[conv_id]
                )
                self.store.insert_conversation_messages(messages_df)
        
        logger.info(f"Stored {len(conversations)} conversations")
    
    def get_pipeline_stats(self):
        conn = self.store.conn
        
        stats = {
            'total_tweets': pd.read_sql_query(
                'SELECT COUNT(*) as count FROM raw_tweets', conn
            ).iloc[0]['count'],
            
            'total_conversations': pd.read_sql_query(
                'SELECT COUNT(*) as count FROM conversations', conn
            ).iloc[0]['count'],
            
            'resolved_conversations': pd.read_sql_query(
                'SELECT COUNT(*) as count FROM conversations WHERE is_resolved = TRUE', conn
            ).iloc[0]['count'],
            
            'open_conversations': pd.read_sql_query(
                'SELECT COUNT(*) as count FROM conversations WHERE is_resolved = FALSE', conn
            ).iloc[0]['count'],
            
            'avg_sentiment': pd.read_sql_query(
                'SELECT AVG(sentiment_score) as avg_sentiment FROM conversations', conn
            ).iloc[0]['avg_sentiment'],
            
            'avg_urgency': pd.read_sql_query(
                'SELECT AVG(urgency_score) as avg_urgency FROM conversations', conn
            ).iloc[0]['avg_urgency'],
            
            'avg_complexity': pd.read_sql_query(
                'SELECT AVG(complexity_score) as avg_complexity FROM conversations', conn
            ).iloc[0]['avg_complexity']
        }
        
        return stats
    
    def close(self):
        self.store.close()
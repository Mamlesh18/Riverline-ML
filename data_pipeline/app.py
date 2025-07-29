import pandas as pd
import hashlib
import logging
from datetime import datetime
from data_pipeline.data_clean import DataCleaner
from data_pipeline.data_store import DataStore

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
    
    def _process_file(self, csv_path):
        total_records = 0
        all_conversations = {}
        all_metadata = {}
        
        for chunk in pd.read_csv(csv_path, chunksize=self.batch_size):
            logger.info(f"Processing chunk with {len(chunk)} records...")
            
            cleaned_chunk = self.cleaner.clean_raw_data(chunk)
            
            try:
                self.store.insert_tweets_batch(cleaned_chunk)
            except Exception as e:
                logger.warning(f"Some tweets already exist, continuing: {str(e)}")
            
            conversations, metadata = self.cleaner.build_conversations(cleaned_chunk)
            all_conversations.update(conversations)
            all_metadata.update(metadata)
            
            total_records += len(cleaned_chunk)
            logger.info(f"Processed {total_records} records so far...")
        
        logger.info("Storing conversation data...")
        self._store_conversations(all_conversations, all_metadata)
        
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
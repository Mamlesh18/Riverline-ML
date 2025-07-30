import sqlite3
import pandas as pd
import json
import logging
from observe_user_behavior.ml_analyser import HybridAnalyzer
from observe_user_behavior.cohort_builder import CohortBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserBehaviorAnalyzer:
    def __init__(self, db_name='riverline.db'):
        self.db_path = db_name
        self.conn = None
        self.hybrid_analyzer = HybridAnalyzer()
        self.cohort_builder = CohortBuilder()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database connection and create analysis tables"""
        self.conn = sqlite3.connect(self.db_path)
        self._create_analysis_tables()
    
    def _create_analysis_tables(self):
        """Create tables for storing analysis results"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_analysis (
                conversation_id INTEGER PRIMARY KEY,
                is_resolved BOOLEAN,
                resolution_confidence REAL,
                tags TEXT,  -- JSON array of tags
                nature_of_request TEXT,
                customer_sentiment TEXT,
                urgency_level TEXT,
                conversation_type TEXT,
                customer_behavior TEXT,
                analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_method TEXT,  -- 'rule_based', 'ml_model', 'gemini_llm', 'hybrid'
                gemini_response TEXT  -- Full response for debugging
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customer_cohorts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cohort_name TEXT,
                cohort_description TEXT,
                cohort_criteria TEXT,  -- JSON criteria
                conversation_count INTEGER,
                created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_cohort_mapping (
                conversation_id INTEGER,
                cohort_id INTEGER,
                FOREIGN KEY (conversation_id) REFERENCES conversation_analysis(conversation_id),
                FOREIGN KEY (cohort_id) REFERENCES customer_cohorts(id)
            )
        ''')
        
        self.conn.commit()
        print("✅ Analysis tables created successfully")
    
    def get_conversations_for_analysis(self, limit=None):
        """Get conversations from database for analysis"""
        query = '''
            SELECT conversation_id, 
                   GROUP_CONCAT(
                       CASE WHEN is_customer = 1 THEN 'CUSTOMER: ' ELSE 'AGENT: ' END ||
                       author_id || ': ' || message_text, 
                       ' | '
                   ) as full_conversation,
                   COUNT(*) as message_count,
                   MIN(created_at) as start_time,
                   MAX(created_at) as end_time
            FROM grouped_conversations
            GROUP BY conversation_id
            ORDER BY conversation_id
        '''
        
        if limit:
            query += f' LIMIT {limit}'
            
        return pd.read_sql_query(query, self.conn)
    
    def analyze_conversation(self, conversation_data):
        """Analyze a single conversation using Hybrid approach"""
        try:
            conversation_id = conversation_data['conversation_id']
            full_conversation = conversation_data['full_conversation']
            
            print(f"🔍 Analyzing conversation {conversation_id}...")
            
            # Use Hybrid Analyzer (Rule-based + ML + Gemini)
            analysis_result = self.hybrid_analyzer.analyze_conversation(full_conversation)
            
            if analysis_result:
                # Store the analysis in database
                self._store_analysis_result(conversation_id, analysis_result)
                return analysis_result
            else:
                logger.error(f"Failed to analyze conversation {conversation_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing conversation {conversation_data['conversation_id']}: {str(e)}")
            return None
    
    def _store_analysis_result(self, conversation_id, analysis):
        """Store analysis result in database"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO conversation_analysis 
            (conversation_id, is_resolved, resolution_confidence, tags, nature_of_request,
             customer_sentiment, urgency_level, conversation_type, customer_behavior, 
             analysis_method, gemini_response)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            conversation_id,
            analysis.get('is_resolved', False),
            analysis.get('resolution_confidence', 0.0),
            json.dumps(analysis.get('tags', [])),
            analysis.get('nature_of_request', ''),
            analysis.get('customer_sentiment', ''),
            analysis.get('urgency_level', ''),
            analysis.get('conversation_type', ''),
            analysis.get('customer_behavior', ''),
            'hybrid',  # Analysis method
            json.dumps(analysis)  # Store full response
        ))
        
        self.conn.commit()
    
    def clear_previous_analysis(self):
        """Clear previous analysis data to start fresh"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM conversation_analysis')
        cursor.execute('DELETE FROM customer_cohorts')
        cursor.execute('DELETE FROM conversation_cohort_mapping')
        self.conn.commit()
        print("🧹 Cleared previous analysis data")
    
    
    def analyze_all_conversations(self, limit=10):
        """Analyze only the first 3 conversations and create cohorts"""
        try:
            self.clear_previous_analysis()
            
            conversations_df = self.get_conversations_for_analysis(limit=10)
            
            print("Starting analysis of FIRST 3 conversations only...")
            print("="*60)
            
            analyzed_count = 0
            resolved_count = 0
            open_count = 0
            
            for idx, (_, conv_data) in enumerate(conversations_df.iterrows(), 1):
                print(f"\n📋 ANALYZING CONVERSATION {idx}/3...")
                print(f"Conversation ID: {conv_data['conversation_id']}")
                print(f"Message Count: {conv_data['message_count']}")
                print(f"Preview: {conv_data['full_conversation'][:200]}...")
                print("-" * 60)
                
                analysis = self.analyze_conversation(conv_data)
                
                if analysis:
                    analyzed_count += 1
                    if analysis.get('is_resolved', False):
                        resolved_count += 1
                        print("✅ Status: RESOLVED")
                    else:
                        open_count += 1
                        print("🔓 Status: OPEN")
                    
                    print(f"🏷️  Tags: {analysis.get('tags', [])}")
                    print(f"📝 Nature: {analysis.get('nature_of_request', 'unknown')}")
                    print(f"😊 Sentiment: {analysis.get('customer_sentiment', 'unknown')}")
                    print(f"⚡ Urgency: {analysis.get('urgency_level', 'unknown')}")
                else:
                    print(f"❌ Analysis failed for conversation {conv_data['conversation_id']}")
                
                # Add delay between API calls
                import time
                time.sleep(2)
            
            # Create cohorts based on the 3 analyzed conversations
            print(f"\n🎯 Creating cohorts from {analyzed_count} analyzed conversations...")
            cohorts_created = self.cohort_builder.create_cohorts(self.conn)
            
            # Get unique tags
            unique_tags = self._get_unique_tags()
            
            return {
                'status': 'success',
                'total_analyzed': analyzed_count,
                'resolved_count': resolved_count,
                'open_count': open_count,
                'unique_tags': len(unique_tags),
                'cohorts_created': cohorts_created
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_all_conversations: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _get_unique_tags(self):
        """Get all unique tags from analysis"""
        cursor = self.conn.cursor()
        results = cursor.execute('SELECT tags FROM conversation_analysis WHERE tags IS NOT NULL').fetchall()
        
        all_tags = set()
        for (tags_json,) in results:
            try:
                tags = json.loads(tags_json)
                all_tags.update(tags)
            except Exception:
                continue
                
        return list(all_tags)
    
    def print_analysis_summary(self):
        """Print summary of analysis results for the 3 conversations"""
        print("\n" + "="*80)
        print("FIRST 3 CONVERSATIONS - ANALYSIS SUMMARY")
        print("="*80)
        
        # Get resolved vs open statistics
        cursor = self.conn.cursor()
        
        total_analyzed = cursor.execute('SELECT COUNT(*) FROM conversation_analysis').fetchone()[0]
        resolved_count = cursor.execute('SELECT COUNT(*) FROM conversation_analysis WHERE is_resolved = 1').fetchone()[0]
        open_count = total_analyzed - resolved_count
        
        print(f"📊 Total Conversations Analyzed: {total_analyzed}")
        print(f"✅ Resolved Conversations: {resolved_count}")
        print(f"🔓 Open Conversations: {open_count}")
        
        # Show all analyzed conversations (should be 3)
        detailed_query = '''
            SELECT conversation_id, is_resolved, tags, nature_of_request, 
                   customer_sentiment, urgency_level, conversation_type, customer_behavior
            FROM conversation_analysis 
            ORDER BY conversation_id
        '''
        
        results_df = pd.read_sql_query(detailed_query, self.conn)
        
        print("\n🔍 DETAILED ANALYSIS RESULTS:")
        print("=" * 80)
        
        for idx, row in results_df.iterrows():
            tags = json.loads(row['tags']) if row['tags'] else []
            resolved_status = "✅ RESOLVED" if row['is_resolved'] else "🔓 OPEN"
            
            print(f"\n📋 CONVERSATION {idx + 1} (ID: {row['conversation_id']})")
            print(f"   Status: {resolved_status}")
            print(f"   Nature: {row['nature_of_request']}")
            print(f"   Sentiment: {row['customer_sentiment']}")
            print(f"   Urgency: {row['urgency_level']}")
            print(f"   Type: {row['conversation_type']}")
            print(f"   Behavior: {row['customer_behavior']}")
            print(f"   Tags: {tags}")
        
        # Show cohorts created from these 3 conversations
        cohorts_df = pd.read_sql_query('SELECT * FROM customer_cohorts', self.conn)
        print(f"\n🎯 COHORTS CREATED: {len(cohorts_df)}")
        
        if len(cohorts_df) > 0:
            print("\nCohort Details:")
            for _, cohort in cohorts_df.iterrows():
                print(f"  🔹 {cohort['cohort_name']}: {cohort['conversation_count']} conversations")
                print(f"     {cohort['cohort_description']}")
        else:
            print("   No cohorts created (might need more diverse conversation patterns)")
        
        # Show unique tags found
        unique_tags = self._get_unique_tags()
        print(f"\n🏷️  UNIQUE TAGS FOUND: {len(unique_tags)}")
        if unique_tags:
            print(f"   Tags: {unique_tags}")
        
        print("\n" + "="*80)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")
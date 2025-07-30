"""
Simple Evaluation Script for Riverline NBA System

This script:
1. Runs the complete pipeline end-to-end
2. Processes 1000 customers through NBA engine
3. Generates required CSV outputs
4. Prints evaluation metrics
"""

import sys
import os

import pandas as pd
import json
from datetime import datetime
from data_pipeline import DataPipeline
from observe_user_behavior import UserBehaviorAnalyzer
from next_best_action import NextBestActionEngine

def run_complete_pipeline():
    """Run the complete pipeline from start to finish"""
    print("="*60)
    print("RIVERLINE NBA EVALUATION - COMPLETE PIPELINE")
    print("="*60)
    
    # Step 1: Data Pipeline
    print("\n1. Running Data Pipeline...")
    pipeline = DataPipeline(db_name='riverline.db')
    result = pipeline.run('./dataset/twcs.csv')
    pipeline.close()
    
    print(f"Pipeline Status: {result['status']}")
    
    # Step 2: User Behavior Analysis
    print("\n2. Running User Behavior Analysis...")
    analyzer = UserBehaviorAnalyzer(db_path='riverline.db')
    analysis_results = analyzer.run_analysis()
    analyzer.close()
    
    if analysis_results is None:
        print("❌ Analysis failed")
        return None
    
    print(f"✓ Analyzed {analysis_results['total_conversations']} conversations")
    print(f"✓ Resolved: {analysis_results['resolved_conversations']}")
    print(f"✓ Open: {analysis_results['open_conversations']}")
    
    return analysis_results

def run_nba_evaluation(limit=1000):
    """Run NBA engine on open conversations and generate evaluation metrics"""
    print(f"\n3. Running NBA Engine (processing {limit} customers)...")
    
    # Initialize NBA engine
    nba_engine = NextBestActionEngine(db_path='riverline.db')
    
    # Load conversation features
    try:
        features_df = pd.read_csv('./conversation_features.csv')
        print(f"✓ Loaded {len(features_df)} total conversations")
    except FileNotFoundError:
        print("❌ No conversation features found. Run user behavior analysis first.")
        return None
    
    # Filter conversations
    total_conversations = len(features_df)
    resolved_conversations = len(features_df[features_df['is_resolved'] == 1])
    open_conversations = len(features_df[features_df['is_resolved'] == 0])
    
    print(f"📊 Conversation Breakdown:")
    print(f"   Total conversations: {total_conversations}")
    print(f"   Already resolved: {resolved_conversations} (excluded from NBA)")
    print(f"   Open conversations: {open_conversations}")
    
    # Get open conversations for NBA processing
    open_df = features_df[features_df['is_resolved'] == 0].head(limit)
    
    if len(open_df) == 0:
        print("❌ No open conversations found for NBA processing")
        return None
    
    print(f"🎯 Processing {len(open_df)} open conversations through NBA engine...")
    
    # Process through NBA engine
    nba_results = nba_engine.process_batch(open_df, limit=limit)
    
    if len(nba_results) == 0:
        print("❌ No NBA recommendations generated")
        return None
    
    print(f"✓ Generated {len(nba_results)} NBA recommendations")
    
    return nba_results, resolved_conversations, open_conversations

def analyze_nba_results(nba_results):
    """Analyze and print NBA results"""
    print(f"\n4. NBA Results Analysis:")
    print("="*40)
    
    # Channel distribution
    channel_dist = nba_results['channel'].value_counts()
    print(f"📱 Channel Distribution:")
    for channel, count in channel_dist.items():
        percentage = count / len(nba_results) * 100
        print(f"   {channel.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Predicted issue status
    status_dist = nba_results['issue_status'].value_counts()
    print(f"\n📈 Predicted Issue Status:")
    for status, count in status_dist.items():
        percentage = count / len(nba_results) * 100
        print(f"   {status.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Resolution prediction
    predicted_resolved = len(nba_results[nba_results['issue_status'] == 'resolved'])
    resolution_rate = predicted_resolved / len(nba_results) * 100
    print(f"\n🎯 Expected Resolution Rate: {resolution_rate:.1f}%")
    print(f"   Predicted to resolve: {predicted_resolved} out of {len(nba_results)}")
    
    # Confidence analysis
    if 'confidence_score' in nba_results.columns:
        avg_confidence = nba_results['confidence_score'].mean()
        high_confidence = len(nba_results[nba_results['confidence_score'] > 0.7])
        print(f"\n🔍 Confidence Analysis:")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   High confidence (>0.7): {high_confidence} ({high_confidence/len(nba_results)*100:.1f}%)")

def create_evaluation_csv(nba_results):
    """Create the main evaluation CSV with all required columns"""
    print(f"\n5. Creating Evaluation CSV...")
    
    # Required columns for evaluation
    eval_columns = [
        'customer_id', 'channel', 'send_time', 'message', 'reasoning', 
        'chat_log', 'issue_status'
    ]
    
    # Ensure all columns exist
    eval_df = nba_results.copy()
    for col in eval_columns:
        if col not in eval_df.columns:
            eval_df[col] = ""
    
    # Select and reorder columns
    eval_df = eval_df[eval_columns]
    
    # Clean data
    eval_df['send_time'] = pd.to_datetime(eval_df['send_time']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    eval_df['message'] = eval_df['message'].astype(str)
    eval_df['reasoning'] = eval_df['reasoning'].astype(str)
    eval_df['chat_log'] = eval_df['chat_log'].astype(str)
    
    # Save evaluation CSV
    output_file = 'nba_evaluation_1000_customers.csv'
    eval_df.to_csv(output_file, index=False)
    
    print(f"✓ Saved evaluation CSV: {output_file}")
    print(f"   Records: {len(eval_df)}")
    print(f"   Columns: {list(eval_df.columns)}")
    
    return eval_df

def create_result_csv_format(nba_results, resolved_count, open_count):
    """Create result CSV in the exact format from the Google Sheet"""
    print(f"\n6. Creating Result CSV (Google Sheet Format)...")
    
    # Calculate metrics
    total_conversations = resolved_count + open_count
    processed_customers = len(nba_results)
    
    # Channel breakdown
    channel_counts = nba_results['channel'].value_counts()
    twitter_count = channel_counts.get('twitter_dm_reply', 0)
    email_count = channel_counts.get('email_reply', 0)
    phone_count = channel_counts.get('scheduling_phone_call', 0)
    
    # Status breakdown
    status_counts = nba_results['issue_status'].value_counts()
    resolved_predicted = status_counts.get('resolved', 0)
    pending_reply = status_counts.get('pending_customer_reply', 0)
    escalated = status_counts.get('escalated', 0)
    
    # Create result data matching Google Sheet format
    result_data = {
        'Metric': [
            'Total Conversations in Dataset',
            'Already Resolved (Excluded)',
            'Open Conversations Available',
            'Customers Processed by NBA',
            'Channel: Twitter DM Reply',
            'Channel: Email Reply', 
            'Channel: Phone Call',
            'Predicted Status: Resolved',
            'Predicted Status: Pending Customer Reply',
            'Predicted Status: Escalated',
            'Expected Resolution Rate (%)',
            'Average Confidence Score',
            'Processing Success Rate (%)'
        ],
        'Value': [
            total_conversations,
            resolved_count,
            open_count,
            processed_customers,
            twitter_count,
            email_count,
            phone_count,
            resolved_predicted,
            pending_reply,
            escalated,
            round(resolved_predicted / processed_customers * 100, 1) if processed_customers > 0 else 0,
            round(nba_results['confidence_score'].mean(), 2) if 'confidence_score' in nba_results.columns else 'N/A',
            round(processed_customers / min(open_count, 1000) * 100, 1)
        ],
        'Description': [
            'Total conversations found in the dataset',
            'Conversations already resolved, excluded from NBA processing',
            'Open conversations available for NBA recommendations',
            'Number of customers successfully processed by NBA engine',
            'Customers recommended for Twitter DM response',
            'Customers recommended for email response',
            'Customers recommended for phone call',
            'Customers predicted to have issues resolved after action',
            'Customers where we expect to wait for customer response',
            'Customers requiring escalation to senior support',
            'Percentage of customers expected to resolve after NBA action',
            'Average confidence score for NBA recommendations',
            'Percentage of available customers successfully processed'
        ]
    }
    
    result_df = pd.DataFrame(result_data)
    
    # Save result CSV
    result_file = 'result.csv'
    result_df.to_csv(result_file, index=False)
    
    print(f"✓ Saved result CSV: {result_file}")
    
    # Print key metrics
    print(f"\n📊 Key Results:")
    print(f"   Total Conversations: {total_conversations}")
    print(f"   Already Resolved (Excluded): {resolved_count}")
    print(f"   NBA Processed: {processed_customers}")
    print(f"   Expected Resolution Rate: {resolved_predicted/processed_customers*100:.1f}%")
    
    return result_df

def print_sample_recommendations(nba_results, num_samples=3):
    """Print sample NBA recommendations"""
    print(f"\n7. Sample NBA Recommendations:")
    print("="*50)
    
    sample_size = min(num_samples, len(nba_results))
    for i, (idx, rec) in enumerate(nba_results.head(sample_size).iterrows()):
        print(f"\nSample {i+1}:")
        print(f"Customer ID: {rec['customer_id']}")
        print(f"Channel: {rec['channel']}")
        print(f"Send Time: {rec['send_time']}")
        print(f"Message: {rec['message'][:100]}...")
        print(f"Reasoning: {rec['reasoning'][:100]}...")
        print(f"Issue Status: {rec['issue_status']}")
        if 'chat_log' in rec and rec['chat_log']:
            chat_preview = rec['chat_log'][:100].replace('\n', ' | ')
            print(f"Chat Log: {chat_preview}...")

def main():
    """Main evaluation function"""
    try:
        # Run complete pipeline
        analysis_results = run_complete_pipeline()
        if analysis_results is None:
            return
        
        # Run NBA evaluation
        nba_data = run_nba_evaluation(limit=1000)
        if nba_data is None:
            return
        
        nba_results, resolved_count, open_count = nba_data
        
        # Analyze results
        analyze_nba_results(nba_results)
        
        # Create evaluation CSV
        eval_df = create_evaluation_csv(nba_results)
        
        # Create result CSV in required format
        result_df = create_result_csv_format(nba_results, resolved_count, open_count)
        
        # Show sample recommendations
        print_sample_recommendations(nba_results)
        
        print(f"\n" + "="*60)
        print("✅ EVALUATION COMPLETE")
        print("="*60)
        print("Generated Files:")
        print("📄 nba_evaluation_1000_customers.csv - Main evaluation data")
        print("📄 result.csv - Summary metrics in required format")
        print(f"\n🎯 Final Result: {len(nba_results)} customers processed with {len(nba_results[nba_results['issue_status'] == 'resolved'])/len(nba_results)*100:.1f}% expected resolution rate")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
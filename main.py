from data_pipeline import DataPipeline
from observe_user_behavior import UserBehaviorAnalyzer
from next_best_action import NextBestActionEngine  # Import NBA engine
import json
import pandas as pd

def run_data_pipeline():
    pipeline = DataPipeline(db_name='riverline.db')
    result = pipeline.run('./dataset/twcs.csv')
    
    print(f"\nPipeline Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Batch ID: {result['batch_id']}")
        print(f"Records Processed: {result['records_processed']}")
        
        stats = pipeline.get_pipeline_stats()
        print("\nPipeline Statistics:")
        print(json.dumps(stats, indent=2, default=str))    
    
    pipeline.close()
    return result['status'] in ['skipped', 'success']  # Both are valid for continuing

def run_user_behavior_analysis():
    print("\n" + "="*50)
    print("Starting User Behavior Analysis")
    print("="*50)
    
    analyzer = UserBehaviorAnalyzer(db_path='riverline.db')
    
    results = analyzer.run_analysis()
    
    if results is None:
        print("Analysis failed - no results returned")
        return None
    
    print("\n" + "="*50)
    print("Analysis Results Summary")
    print("="*50)
    print(f"Total Conversations: {results['total_conversations']}")
    print(f"Resolved: {results['resolved_conversations']} ({results['resolved_conversations']/results['total_conversations']*100:.1f}%)")
    print(f"Open: {results['open_conversations']} ({results['open_conversations']/results['total_conversations']*100:.1f}%)")
    
    print("\nCohort Distribution:")
    for cohort, stats in results['cohort_distribution'].items():
        print(f"\n{cohort}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print("\nRecommendations by Cohort:")
    recommendations = analyzer.get_cohort_recommendations()
    for cohort, rec in recommendations.items():
        if cohort in results['cohort_distribution']:
            print(f"\n{cohort}: {rec['recommendation']}")
    
    cohort_summary = analyzer.export_cohort_summary()
    print("\nCohort summary exported to cohort_summary.csv")
    
    print("\nFiles Generated:")
    print("- user_behavior_analysis_results.json")
    print("- conversation_features.csv")
    print("- cohort_summary.csv")
    print("- visualizations/")
    print("  - cohort_distribution.png")
    print("  - resolution_analysis.png")
    print("  - behavioral_patterns.png")
    print("  - conversation_flows.png")
    
    return results

def run_nba_engine(analysis_results):
    print("\n" + "="*50)
    print("Starting Next-Best-Action Engine")
    print("="*50)
    
    # Initialize NBA engine
    nba_engine = NextBestActionEngine(db_path='riverline.db')
    
    # Load conversation features
    try:
        features_df = pd.read_csv('conversation_features.csv')
        print(f"Loaded {len(features_df)} conversations for NBA analysis")
    except FileNotFoundError:
        print("Conversation features file not found. Using results from analysis.")
        features_df = analysis_results['features_dataframe']
    
    # Filter to open conversations
    open_conversations = features_df[features_df['is_resolved'] == 0]
    print(f"Found {len(open_conversations)} open conversations")
    
    if len(open_conversations) == 0:
        print("No open conversations found for NBA recommendations")
        return None
    
    # Process batch of 1000 customers
    print("Processing NBA recommendations...")
    nba_results = nba_engine.process_batch(open_conversations, limit=1000)
    
    if len(nba_results) == 0:
        print("No NBA results generated")
        return None
    
    print(f"\nGenerated NBA recommendations for {len(nba_results)} customers")
    
    # Analyze channel distribution
    channel_dist = nba_results['channel'].value_counts()
    print(f"\nChannel Distribution:")
    for channel, count in channel_dist.items():
        print(f"  {channel.replace('_', ' ').title()}: {count} ({count/len(nba_results)*100:.1f}%)")
    
    # Analyze predicted issue status
    status_dist = nba_results['issue_status'].value_counts()
    print(f"\nPredicted Issue Status Distribution:")
    for status, count in status_dist.items():
        print(f"  {status.replace('_', ' ').title()}: {count} ({count/len(nba_results)*100:.1f}%)")
    
    # Calculate expected resolution rate
    resolved_predictions = len(nba_results[nba_results['issue_status'] == 'resolved'])
    expected_resolution_rate = resolved_predictions / len(nba_results) * 100
    print(f"\nExpected Resolution Rate: {expected_resolution_rate:.1f}%")
    
    # Show confidence analysis
    if 'confidence_score' in nba_results.columns:
        avg_confidence = nba_results['confidence_score'].mean()
        print(f"Average Confidence Score: {avg_confidence:.2f}")
    
    # Save main results file
    output_file = 'nba_recommendations.csv'
    nba_results.to_csv(output_file, index=False)
    print(f"\nNBA recommendations saved to {output_file}")
    
    # Create the required CSV format for evaluation
    create_evaluation_csv(nba_results)
    
    # Show sample recommendations
    print(f"\nSample NBA Recommendations:")
    print("="*50)
    sample_size = min(3, len(nba_results))
    for idx, rec in nba_results.head(sample_size).iterrows():
        print(f"\nCustomer {rec['customer_id']}:")
        print(f"  Channel: {rec['channel']}")
        print(f"  Send Time: {rec['send_time']}")
        print(f"  Message: {rec['message'][:100]}...")
        print(f"  Reasoning: {rec['reasoning'][:100]}...")
        print(f"  Predicted Status: {rec['issue_status']}")
        if 'confidence_score' in rec:
            print(f"  Confidence: {rec['confidence_score']:.2f}")
    
    return nba_results

def create_evaluation_csv(nba_results):
    """Create the evaluation CSV in the required format"""
    print("\nCreating evaluation CSV...")
    
    # Ensure all required columns are present
    required_columns = [
        'customer_id', 'channel', 'send_time', 'message', 
        'reasoning', 'chat_log', 'issue_status'
    ]
    
    eval_df = nba_results.copy()
    
    # Add any missing columns with defaults
    for col in required_columns:
        if col not in eval_df.columns:
            if col == 'chat_log':
                eval_df[col] = "Customer: [Initial message]\nSupport_agent: [Processing request]"
            elif col == 'issue_status':
                eval_df[col] = "pending_customer_reply"
            else:
                eval_df[col] = ""
    
    # Select and reorder columns to match requirements
    eval_df = eval_df[required_columns]
    
    # Clean and format data
    eval_df['send_time'] = pd.to_datetime(eval_df['send_time']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Truncate long text fields for readability
    eval_df['message'] = eval_df['message'].astype(str).str[:500]
    eval_df['reasoning'] = eval_df['reasoning'].astype(str).str[:300]
    eval_df['chat_log'] = eval_df['chat_log'].astype(str).str[:1000]
    
    # Save evaluation CSV
    eval_filename = 'nba_evaluation_results.csv'
    eval_df.to_csv(eval_filename, index=False)
    
    print(f"Evaluation CSV saved to {eval_filename}")
    print(f"Total records in evaluation file: {len(eval_df)}")
    
    # Print statistics for the evaluation
    print(f"\nEvaluation Statistics:")
    print(f"Already resolved conversations excluded: {len(nba_results)} processed")
    
    channel_stats = eval_df['channel'].value_counts()
    print(f"\nChannel breakdown:")
    for channel, count in channel_stats.items():
        print(f"  {channel}: {count}")
    
    status_stats = eval_df['issue_status'].value_counts()
    print(f"\nPredicted status breakdown:")
    for status, count in status_stats.items():
        print(f"  {status}: {count}")
    
    return eval_df

def main():
    """Main execution function"""
    print("="*60)
    print("RIVERLINE NBA SYSTEM - FULL PIPELINE EXECUTION")
    print("="*60)
    
    # Step 1: Run data pipeline
    pipeline_success = run_data_pipeline()
    
    if not pipeline_success:
        print("Data pipeline failed. Please check the logs.")
        return
    
    # Step 2: Run user behavior analysis
    analysis_results = run_user_behavior_analysis()
    
    if analysis_results is None:
        print("User behavior analysis failed. Cannot proceed with NBA engine.")
        return
    
    # Step 3: Run NBA engine
    nba_results = run_nba_engine(analysis_results)
    
    if nba_results is None:
        print("NBA engine failed to generate recommendations.")
        return
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*60)
    print(f"✓ Data Pipeline: {len(analysis_results.get('features_dataframe', []))} conversations processed")
    print(f"✓ User Behavior Analysis: {len(analysis_results['cohort_distribution'])} cohorts identified")
    print(f"✓ NBA Engine: {len(nba_results)} recommendations generated")
    print("\nGenerated Files:")
    print("- conversation_features.csv")
    print("- cohort_summary.csv")
    print("- user_behavior_analysis_results.json")
    print("- nba_recommendations.csv")
    print("- nba_evaluation_results.csv")
    print("- visualizations/ (folder)")
    
    print(f"\n🎯 Expected Resolution Rate: {len(nba_results[nba_results['issue_status'] == 'resolved']) / len(nba_results) * 100:.1f}%")

if __name__ == "__main__":
    main()
from data_pipeline import DataPipeline
from observe_user_behavior import UserBehaviorAnalyzer
import json
import pandas as pd

def run_data_pipeline():
    """Run the data pipeline"""
    print("="*60)
    print("RIVERLINE DATA PIPELINE - SIMPLIFIED VERSION")
    print("="*60)
    
    # Initialize and run pipeline
    pipeline = DataPipeline(db_name='riverline.db')
    result = pipeline.run('./dataset/twcs.csv')
    
    print(f"\nPipeline Status: {result['status']}")
    
    if result['status'] == 'success':
        print(f"Records Processed: {result['records_processed']}")
        print(f"Conversations Created: {result['conversations_created']}")
        
        # Get basic stats
        stats = pipeline.get_basic_stats()
        print("\nPipeline Statistics:")
        print(json.dumps(stats, indent=2, default=str))
    
    pipeline.close()
    return result['status'] == 'success'

def run_user_behavior_analysis():
    """Run user behavior analysis with Gemini LLM"""
    print("\n" + "="*60)
    print("USER BEHAVIOR ANALYSIS - GEMINI AI POWERED")
    print("="*60)
    
    # Initialize behavior analyzer
    analyzer = UserBehaviorAnalyzer(db_name='riverline.db')
    
    # Analyze conversations and create tags/cohorts
    result = analyzer.analyze_all_conversations()
    
    if result['status'] == 'success':
        print(f"\nAnalysis Complete!")
        print(f"Total Conversations Analyzed: {result['total_analyzed']}")
        print(f"Resolved Conversations (excluded): {result['resolved_count']}")
        print(f"Open Conversations (analyzed): {result['open_count']}")
        print(f"Unique Tags Found: {result['unique_tags']}")
        print(f"Cohorts Created: {result['cohorts_created']}")
        
        # Show sample results
        analyzer.print_analysis_summary()
        
    analyzer.close()
    return result['status'] == 'success'

def main():
    """Main execution function"""
    print("="*60)
    print("RIVERLINE NBA SYSTEM - FULL PIPELINE EXECUTION")
    print("="*60)
    
    # Step 1: Run data pipeline
    pipeline_success = run_data_pipeline()
    
    if pipeline_success:
        print("\n✅ Data Pipeline completed successfully!")
        
        # Step 2: Run user behavior analysis
        behavior_success = run_user_behavior_analysis()
        
        if behavior_success:
            print("\n✅ User Behavior Analysis completed successfully!")
            print("\n🎯 FULL SYSTEM EXECUTION COMPLETE!")
        else:
            print("\n❌ User Behavior Analysis failed!")
    else:
        print("\n❌ Data Pipeline failed! Skipping behavior analysis.")

if __name__ == "__main__":
    main()
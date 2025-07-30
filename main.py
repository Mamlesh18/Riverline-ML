
from data_pipeline import DataPipeline
from observe_user_behavior import UserBehaviorAnalyzer
from next_best_action import NextBestActionEngine
import json

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
    """Run user behavior analysis on FIRST 3 conversations only"""
    print("\n" + "="*60)
    print("USER BEHAVIOR ANALYSIS - FIRST 3 CONVERSATIONS ONLY")
    print("POWERED BY GEMINI AI")
    print("="*60)
    
    # Initialize behavior analyzer
    analyzer = UserBehaviorAnalyzer(db_name='riverline.db')
    
    # Analyze only first 3 conversations
    result = analyzer.analyze_all_conversations()  # limit=3 is set by default now
    
    if result['status'] == 'success':
        print(f"📊 Conversations Analyzed: {result['total_analyzed']}/3")
        print(f"✅ Resolved: {result['resolved_count']}")
        print(f"🔓 Open: {result['open_count']}")
        print(f"🏷️  Unique Tags: {result['unique_tags']}")
        print(f"🎯 Cohorts Created: {result['cohorts_created']}")
        
        # Show detailed results
        analyzer.print_analysis_summary()
        
    else:
        print(f"\n❌ Analysis Failed: {result.get('error', 'Unknown error')}")
        
    analyzer.close()
    return result['status'] == 'success'

def run_next_best_action():
    """Run Next Best Action engine for unresolved conversations"""
    print("\n" + "="*60)
    print("NEXT BEST ACTION ENGINE")
    print("POWERED BY GEMINI AI - NBA OPTIMIZATION")
    print("="*60)
    
    # Initialize NBA engine
    nba_engine = NextBestActionEngine(db_name='riverline.db')
    
    # Generate NBA recommendations
    result = nba_engine.run_nba_analysis()
    
    if result['status'] == 'success':

        
        if result['recommendations_generated'] > 0:
            sample = result['recommendations'][0]
            print(f"   Customer: {sample['customer_id']}")
            print(f"   Channel: {sample['channel']}")
            print(f"   Send Time: {sample['send_time']}")
            print(f"   Message: {sample['message'][:100]}...")
        
    else:
        print(f"\n❌ NBA Analysis Failed: {result.get('error', 'Unknown error')}")
    
    nba_engine.close()
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
            
            # Step 3: Run Next Best Action engine
            nba_success = run_next_best_action()
            
            if nba_success:
                print("\n✅ Next Best Action Analysis completed successfully!")
                print("\n🎉 FULL NBA SYSTEM EXECUTION COMPLETE!")
                print("\n📊 SUMMARY:")
                print("   1. ✅ Data Pipeline: Conversations grouped and stored")
                print("   2. ✅ Behavior Analysis: Customer tags and cohorts created")
                print("   3. ✅ NBA Engine: Channel recommendations generated")
                print("\n💾 Check 'nba_results.csv' for detailed recommendations!")
            else:
                print("\n❌ Next Best Action Analysis failed!")
        else:
            print("\n❌ User Behavior Analysis failed! Skipping NBA engine.")
    else:
        print("\n❌ Data Pipeline failed! Skipping subsequent steps.")

if __name__ == "__main__":
    main()
from data_pipeline import DataPipeline
from observe_user_behavior import UserBehaviorAnalyzer
import json

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
    return result['status'] == 'skipped'
    # return result['status'] == 'success'

def run_user_behavior_analysis():
    print("\n" + "="*50)
    print("Starting User Behavior Analysis")
    print("="*50)
    
    analyzer = UserBehaviorAnalyzer(db_path='riverline.db')
    
    results = analyzer.run_analysis()
    
    print("\n" + "="*50)
    print("Analysis Results Summary")
    print("="*50)
    print(f"Total Conversations: {results['total_conversations']}")
    print(f"Resolved: {results['resolved_conversations']} ({results['resolved_conversations']/results['total_conversations']*100:.1f}%)")
    print(f"Open: {results['open_conversations']} ({results['open_conversations']/results['total_conversations']*100:.1f}%)")
    
    print("\nCohort Distribution:")
    for cohort, stats in results['cohort_distribution'].items():
        print(f"\n{cohort}:")
    
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

if __name__ == "__main__":
    pipeline_success = run_data_pipeline()
    
    if pipeline_success:
        run_user_behavior_analysis()
    else:
        print("Data pipeline failed. Please check the logs.")
"""
Next-Best-Action (NBA) Engine Module

A comprehensive NBA system for customer support optimization using:
1. ML-based channel selection (Random Forest)
2. Intelligent timing optimization 
3. Personalized message generation
4. Advanced feature engineering

Author: Riverline AI Team
"""

from next_best_action.app import NextBestActionEngine
from next_best_action.channel_selector import ChannelSelector
from next_best_action.timing_optimizer import TimingOptimizer
from next_best_action.message_generator import MessageGenerator
from next_best_action.feature_engineer import FeatureEngineer

__version__ = "1.0.0"
__all__ = [
    'NextBestActionEngine',
    'ChannelSelector', 
    'TimingOptimizer',
    'MessageGenerator',
    'FeatureEngineer'
]
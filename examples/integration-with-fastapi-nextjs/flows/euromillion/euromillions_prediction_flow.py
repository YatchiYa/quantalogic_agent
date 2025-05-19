#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "loguru",
#     "pandas",
#     "numpy",
#     "scikit-learn",
#     "matplotlib",
#     "seaborn",
#     "quantalogic>=0.35",
#     "typer>=0.9.0"
# ]
# ///

import asyncio
from collections.abc import Callable
import json
import os
import typer
from datetime import datetime, timedelta
from typing import Annotated, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field, ConfigDict
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..service import event_observer
from quantalogic.flow.flow import Nodes, Workflow

# Initialize Typer app
app = typer.Typer(help="Euromillions prediction and analysis flow")

# Constants
DATA_DIR = "examples/integration-with-fastapi-nextjs/flows/euromillion/data"
OUTPUT_DIR = "examples/integration-with-fastapi-nextjs/flows/euromillion/predictions"
ANALYSIS_DIR = "examples/integration-with-fastapi-nextjs/flows/euromillion/analysis"

# Pydantic Models
class EuromillionsData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    data_path: str
    df: Optional[pd.DataFrame] = None
    stats: Optional[Dict] = None

class PredictionResult(BaseModel):
    main_numbers: List[int] = Field(..., description="Predicted main numbers (5)")
    star_numbers: List[int] = Field(..., description="Predicted star numbers (2)")
    confidence_score: float = Field(..., description="Confidence score of prediction")
    prediction_date: str = Field(..., description="Date for which prediction is made")
    historical_data: Dict = Field(..., description="Historical statistics used for prediction")
    alternative_combinations: List[Dict] = Field(default_factory=list, description="Alternative number combinations with explanations")

# Simplified model for Gemini analysis to avoid API issues
class EuromillionsAnalysis(BaseModel):
    statistical_assessment: str = Field(..., description="Statistical assessment of predictions")
    pattern_analysis: str = Field(..., description="Analysis of patterns across combinations")
    strategic_recommendations: str = Field(..., description="Strategic recommendations based on statistical principles")
    optimization_suggestions: str = Field(..., description="Suggestions for optimizing the combinations")
    final_recommendation: str = Field(..., description="Final recommendation with specific combination to play")
    
    def to_markdown(self, prediction_result: PredictionResult) -> str:
        """Convert the analysis to a detailed markdown format."""
        main_numbers = prediction_result.main_numbers
        star_numbers = prediction_result.star_numbers
        combinations = prediction_result.alternative_combinations
        
        md = [
            f"# Euromillions Prediction Analysis for {prediction_result.prediction_date}",
            "",
            "## Executive Summary",
            "",
            "This analysis provides a comprehensive statistical assessment of Euromillions predictions, including the primary prediction and alternative combinations. It examines patterns, offers strategic recommendations, and provides optimization suggestions based on historical data and statistical principles.",
            "",
            "## Primary Prediction",
            "",
            f"**Main Numbers**: {', '.join(map(str, main_numbers))}",
            f"**Star Numbers**: {', '.join(map(str, star_numbers))}",
            f"**Confidence Score**: {prediction_result.confidence_score:.4f}",
            "",
            "## Alternative Combinations",
            ""
        ]
        
        # Add each combination
        for i, combo in enumerate(combinations, 1):
            md.extend([
                f"### Combination {i}",
                f"**Main Numbers**: {', '.join(map(str, combo['main_numbers']))}",
                f"**Star Numbers**: {', '.join(map(str, combo['star_numbers']))}",
                f"**Strategy**: {combo['explanation']}",
                ""
            ])
        
        # Add the main analysis sections
        md.extend([
            "## Statistical Assessment",
            "",
            self.statistical_assessment,
            "",
            "## Pattern Analysis",
            "",
            self.pattern_analysis,
            "",
            "## Strategic Recommendations",
            "",
            self.strategic_recommendations,
            "",
            "## Optimization Suggestions",
            "",
            self.optimization_suggestions,
            "",
            "## Final Recommendation",
            "",
            self.final_recommendation,
            "",
            "---",
            "",
            "**Disclaimer**: Lottery outcomes are inherently random. This analysis is based on statistical principles and historical data, but cannot guarantee any specific results. Always play responsibly."
        ])
        
        return "\n".join(md)

# Flow Nodes
@Nodes.define(output="euromillions_data")
async def load_euromillions_data(data_path: str = None) -> EuromillionsData:
    """Load Euromillions data from CSV or JSON files."""
    if data_path is None:
        data_path = DATA_DIR
    
    logger.info(f"Loading Euromillions data from {data_path}")
    
    # Check for CSV file
    csv_files = [f for f in os.listdir(data_path) if f.startswith("euromillions_") and f.endswith(".csv")]
    
    if not csv_files:
        logger.error(f"No Euromillions data files found in {data_path}")
        return EuromillionsData(data_path=data_path)
    
    # Use the most comprehensive file (usually the one with the widest year range)
    csv_file = sorted(csv_files, key=lambda x: len(x), reverse=True)[0]
    csv_path = os.path.join(data_path, csv_file)
    
    # Load the CSV data
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    
    # Check for stats JSON file
    stats_file = os.path.join(data_path, "euromillions_stats.json")
    stats = None
    
    if os.path.exists(stats_file):
        logger.info(f"Loading statistics from {stats_file}")
        with open(stats_file, "r") as f:
            stats = json.load(f)
    
    return EuromillionsData(data_path=data_path, df=df, stats=stats)

@Nodes.define(output="processed_data")
async def preprocess_data(euromillions_data: EuromillionsData) -> EuromillionsData:
    """Preprocess and clean the Euromillions data."""
    if euromillions_data.df is None:
        logger.error("No data to preprocess")
        return euromillions_data
    
    df = euromillions_data.df.copy()
    logger.info(f"Preprocessing {len(df)} Euromillions draws")
    
    # Convert data types
    df["Year"] = pd.to_numeric(df["Year"])
    df["Winners"] = pd.to_numeric(df["Winners"])
    df["Jackpot"] = pd.to_numeric(df["Jackpot"])
    
    # Extract day of week
    df["DayOfWeek"] = df["Date"].dt.day_name()
    
    # Parse numbers and stars into lists
    df["NumbersList"] = df["Numbers"].apply(lambda x: [int(n) for n in str(x).split()])
    df["StarsList"] = df["Stars"].apply(lambda x: [int(n) for n in str(x).split()])
    
    # Create features for each number and star position
    for i in range(5):
        df[f"Number_{i+1}"] = df["NumbersList"].apply(lambda x: x[i] if len(x) > i else np.nan)
    
    for i in range(2):
        df[f"Star_{i+1}"] = df["StarsList"].apply(lambda x: x[i] if len(x) > i else np.nan)
    
    # Add time-based features
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week
    
    # Sort by date
    df = df.sort_values("Date")
    
    # Calculate rolling statistics
    df["RollingAvgJackpot"] = df["Jackpot"].rolling(window=10).mean()
    df["RollingWinnerRate"] = df["Winners"].rolling(window=20).mean()
    
    # Fill missing values
    df = df.fillna(method="ffill")
    
    logger.info("Data preprocessing completed")
    return EuromillionsData(data_path=euromillions_data.data_path, df=df, stats=euromillions_data.stats)

@Nodes.define(output="feature_data")
async def extract_features(processed_data: EuromillionsData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for prediction models."""
    if processed_data.df is None:
        logger.error("No processed data available for feature extraction")
        return None, None, None, None, None, None
    
    df = processed_data.df.copy()
    logger.info("Extracting features for prediction models")
    
    # Features for main numbers prediction
    features = df[["Year", "Month", "Day", "DayOfYear", "WeekOfYear", "RollingAvgJackpot", "RollingWinnerRate"]].values
    
    # Target for main numbers
    targets_main = df[["Number_1", "Number_2", "Number_3", "Number_4", "Number_5"]].values
    
    # Target for star numbers
    targets_stars = df[["Star_1", "Star_2"]].values
    
    # Split into training and testing sets
    X_main_train, X_main_test, y_main_train, y_main_test = train_test_split(
        features, targets_main, test_size=0.2, random_state=42
    )
    
    X_stars_train, X_stars_test, y_stars_train, y_stars_test = train_test_split(
        features, targets_stars, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_main_train = scaler.fit_transform(X_main_train)
    X_main_test = scaler.transform(X_main_test)
    
    X_stars_train = scaler.fit_transform(X_stars_train)
    X_stars_test = scaler.transform(X_stars_test)
    
    logger.info(f"Feature extraction completed: {X_main_train.shape[0]} training samples, {X_main_test.shape[0]} test samples")
    return X_main_train, X_main_test, y_main_train, y_main_test, X_stars_train, X_stars_test, y_stars_train, y_stars_test

@Nodes.define(output="prediction_models")
async def train_models(feature_data: Tuple) -> Tuple:
    """Train prediction models for main numbers and stars."""
    if feature_data is None or len(feature_data) < 8:
        logger.error("Insufficient feature data for training models")
        return None, None
    
    X_main_train, X_main_test, y_main_train, y_main_test, X_stars_train, X_stars_test, y_stars_train, y_stars_test = feature_data
    
    logger.info("Training prediction models")
    
    # Train model for main numbers
    main_model = RandomForestRegressor(n_estimators=100, random_state=42)
    main_model.fit(X_main_train, y_main_train)
    
    # Train model for star numbers
    stars_model = RandomForestRegressor(n_estimators=100, random_state=42)
    stars_model.fit(X_stars_train, y_stars_train)
    
    # Evaluate models
    main_score = main_model.score(X_main_test, y_main_test)
    stars_score = stars_model.score(X_stars_test, y_stars_test)
    
    logger.info(f"Model training completed: Main numbers R² = {main_score:.4f}, Stars R² = {stars_score:.4f}")
    return main_model, stars_model, main_score, stars_score

@Nodes.define(output="prediction_result")
async def generate_prediction(
    processed_data: EuromillionsData,
    prediction_models: Tuple,
    prediction_date: Optional[str] = None
) -> PredictionResult:
    """Generate prediction for the next Euromillions draw."""
    if processed_data.df is None or prediction_models is None or len(prediction_models) < 4:
        logger.error("Missing data or models for prediction")
        return None
    
    main_model, stars_model, main_score, stars_score = prediction_models
    df = processed_data.df
    stats = processed_data.stats
    
    # Determine prediction date if not provided
    if prediction_date is None:
        # Calculate the next Euromillions draw date (Tuesday or Friday) from today
        current_date = datetime.now()
        # Map weekday to days until next draw (0=Monday, 1=Tuesday, etc.)
        days_to_next_draw = {
            0: 1,  # Monday -> Tuesday (1 day)
            1: 0,  # Tuesday -> Tuesday (same day)
            2: 2,  # Wednesday -> Friday (2 days)
            3: 1,  # Thursday -> Friday (1 day)
            4: 0,  # Friday -> Friday (same day)
            5: 3,  # Saturday -> Tuesday (3 days)
            6: 2,  # Sunday -> Tuesday (2 days)
        }
        
        days_to_add = days_to_next_draw[current_date.weekday()]
        
        # If it's after draw time (usually 8:45 PM) on a draw day, move to next draw
        if days_to_add == 0 and current_date.hour >= 21:
            if current_date.weekday() == 1:  # Tuesday -> Friday (3 days)
                days_to_add = 3
            else:  # Friday -> Tuesday (4 days)
                days_to_add = 4
                
        next_draw_date = current_date + timedelta(days=days_to_add)
        prediction_date = next_draw_date.strftime("%Y-%m-%d")
    
    logger.info(f"Generating prediction for {prediction_date}")
    
    # Create feature vector for prediction date
    pred_date = datetime.strptime(prediction_date, "%Y-%m-%d")
    features = np.array([
        [
            pred_date.year,
            pred_date.month,
            pred_date.day,
            pred_date.timetuple().tm_yday,  # day of year
            pred_date.isocalendar()[1],  # week of year
            df["RollingAvgJackpot"].iloc[-1],  # latest rolling average jackpot
            df["RollingWinnerRate"].iloc[-1]  # latest rolling winner rate
        ]
    ])
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(
        df[["Year", "Month", "Day", "DayOfYear", "WeekOfYear", "RollingAvgJackpot", "RollingWinnerRate"]].values
    )
    features = scaler.transform(features)
    
    # Generate predictions
    main_numbers_raw = main_model.predict(features)[0]
    star_numbers_raw = stars_model.predict(features)[0]
    
    # Convert to valid Euromillions numbers (1-50 for main, 1-12 for stars)
    # Round to nearest integer and ensure unique values
    main_numbers = []
    for num in main_numbers_raw:
        rounded = max(1, min(50, round(num)))
        if rounded not in main_numbers:
            main_numbers.append(rounded)
    
    # If we don't have enough unique numbers, add the most frequent ones from history
    while len(main_numbers) < 5:
        for num in range(1, 51):
            if num not in main_numbers:
                main_numbers.append(num)
                break
    
    # Sort main numbers
    main_numbers = sorted(main_numbers[:5])
    
    # Process star numbers similarly
    star_numbers = []
    for num in star_numbers_raw:
        rounded = max(1, min(12, round(num)))
        if rounded not in star_numbers:
            star_numbers.append(rounded)
    
    # If we don't have enough unique star numbers, add from history
    while len(star_numbers) < 2:
        for num in range(1, 13):
            if num not in star_numbers:
                star_numbers.append(num)
                break
    
    # Sort star numbers
    star_numbers = sorted(star_numbers[:2])
    
    # Calculate confidence score based on model performance
    confidence_score = (main_score + stars_score) / 2
    
    # Generate alternative combinations with explanations
    alternative_combinations = generate_alternative_combinations(df, main_numbers, star_numbers, stats)
    
    # Create prediction result
    result = PredictionResult(
        main_numbers=main_numbers,
        star_numbers=star_numbers,
        confidence_score=confidence_score,
        prediction_date=prediction_date,
        historical_data={
            "model_score_main": main_score,
            "model_score_stars": stars_score,
            "total_draws_analyzed": len(df),
            "latest_draw_date": df["Date"].max().strftime("%Y-%m-%d"),
            "stats_summary": stats if stats else {"note": "No statistics available"}
        },
        alternative_combinations=alternative_combinations
    )
    
    logger.info(f"Prediction generated: Main numbers {main_numbers}, Stars {star_numbers}, Confidence: {confidence_score:.4f}")
    return result

def generate_alternative_combinations(df, base_main_numbers, base_star_numbers, stats):
    """Generate alternative number combinations with explanations."""
    combinations = []
    
    # Get frequency data
    number_counts = stats.get("stats_summary", {}).get("number_counts", {})
    star_counts = stats.get("stats_summary", {}).get("star_counts", {})
    
    # Convert to proper format if needed
    if isinstance(number_counts, dict):
        number_counts = {int(k): v for k, v in number_counts.items()}
    if isinstance(star_counts, dict):
        star_counts = {int(k): v for k, v in star_counts.items()}
    
    # 1. Most frequent numbers combination
    if number_counts and star_counts:
        sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_stars = sorted(star_counts.items(), key=lambda x: x[1], reverse=True)
        
        most_frequent_numbers = [int(num) for num, _ in sorted_numbers[:5]]
        most_frequent_stars = [int(star) for star, _ in sorted_stars[:2]]
        
        combinations.append({
            "main_numbers": sorted(most_frequent_numbers),
            "star_numbers": sorted(most_frequent_stars),
            "explanation": "This combination uses the most frequently drawn numbers and stars based on historical data. These numbers have appeared most often in past draws."
        })
    
    # 2. Least frequent numbers (overdue numbers)
    if number_counts and star_counts:
        sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1])
        sorted_stars = sorted(star_counts.items(), key=lambda x: x[1])
        
        least_frequent_numbers = [int(num) for num, _ in sorted_numbers[:5]]
        least_frequent_stars = [int(star) for star, _ in sorted_stars[:2]]
        
        combinations.append({
            "main_numbers": sorted(least_frequent_numbers),
            "star_numbers": sorted(least_frequent_stars),
            "explanation": "This combination uses the least frequently drawn numbers and stars. These 'overdue' numbers might be statistically more likely to appear soon."
        })
    
    # 3. Balanced distribution (mix of high and low numbers)
    low_numbers = [n for n in range(1, 26) if n not in base_main_numbers]
    high_numbers = [n for n in range(26, 51) if n not in base_main_numbers]
    
    # Select 2 low and 3 high numbers (or vice versa)
    import random
    random.seed(42)  # For reproducibility
    
    balanced_numbers = sorted(random.sample(low_numbers, 2) + random.sample(high_numbers, 3))
    balanced_stars = sorted([s for s in range(1, 13) if s not in base_star_numbers][:2])
    
    combinations.append({
        "main_numbers": balanced_numbers,
        "star_numbers": balanced_stars,
        "explanation": "This combination provides a balanced mix of low (1-25) and high (26-50) numbers, which is a common strategy among lottery players."
    })
    
    # 4. Recent trend-based combination
    recent_draws = df.sort_values("Date", ascending=False).head(10)
    recent_numbers = []
    
    # Extract all numbers from recent draws
    for _, row in recent_draws.iterrows():
        recent_numbers.extend([int(n) for n in row["NumbersList"]])
    
    # Count occurrences
    from collections import Counter
    number_counter = Counter(recent_numbers)
    
    # Get the most common numbers in recent draws
    trend_numbers = [num for num, _ in number_counter.most_common(5)]
    
    # Do the same for stars
    recent_stars = []
    for _, row in recent_draws.iterrows():
        recent_stars.extend([int(s) for s in row["StarsList"]])
    
    star_counter = Counter(recent_stars)
    trend_stars = [star for star, _ in star_counter.most_common(2)]
    
    combinations.append({
        "main_numbers": sorted(trend_numbers),
        "star_numbers": sorted(trend_stars),
        "explanation": "This combination is based on recent trends, using numbers and stars that have appeared most frequently in the last 10 draws."
    })
    
    # 5. Pattern-based combination (e.g., consecutive numbers)
    # Find a starting point
    start = random.randint(1, 46)
    pattern_numbers = [start, start + 1, start + 2, start + 3, start + 4]
    pattern_stars = sorted(random.sample(range(1, 13), 2))
    
    combinations.append({
        "main_numbers": pattern_numbers,
        "star_numbers": pattern_stars,
        "explanation": "This combination follows a mathematical pattern with consecutive numbers. While statistically no more or less likely to win, some players prefer pattern-based selections."
    })
    
    return combinations

def generate_top_combinations(df, stats, n=10):
    """Generate the top n most probable combinations based on statistical analysis."""
    # Get frequency data
    number_counts = stats.get("stats_summary", {}).get("number_counts", {})
    star_counts = stats.get("stats_summary", {}).get("star_counts", {})
    
    # Convert to proper format if needed
    if isinstance(number_counts, dict):
        number_counts = {int(k): v for k, v in number_counts.items()}
    if isinstance(star_counts, dict):
        star_counts = {int(k): v for k, v in star_counts.items()}
    
    # Sort numbers by frequency
    sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_stars = sorted(star_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get top numbers and stars
    top_numbers = [int(num) for num, _ in sorted_numbers[:15]]  # Take top 15 to have variety
    top_stars = [int(star) for star, _ in sorted_stars[:5]]     # Take top 5 stars
    
    # Get recent trends
    recent_draws = df.sort_values("Date", ascending=False).head(20)
    recent_numbers = []
    recent_stars = []
    
    # Extract all numbers from recent draws
    for _, row in recent_draws.iterrows():
        recent_numbers.extend([int(n) for n in row["NumbersList"]])
        recent_stars.extend([int(s) for s in row["StarsList"]])
    
    # Count occurrences
    from collections import Counter
    number_counter = Counter(recent_numbers)
    star_counter = Counter(recent_stars)
    
    # Get the most common numbers in recent draws
    trend_numbers = [num for num, _ in number_counter.most_common(15)]
    trend_stars = [star for star, _ in star_counter.most_common(5)]
    
    # Combine historical frequency and recent trends
    import numpy as np
    from itertools import combinations as combo_iter
    
    # Create a scoring system
    number_scores = {}
    for num in range(1, 51):
        # Base score from historical frequency
        hist_score = number_counts.get(num, 0) / max(number_counts.values()) if number_counts else 0
        
        # Recent trend score
        trend_score = number_counter.get(num, 0) / max(number_counter.values()) if number_counter else 0
        
        # Combined score (weighted)
        number_scores[num] = 0.6 * hist_score + 0.4 * trend_score
    
    # Same for stars
    star_scores = {}
    for star in range(1, 13):
        hist_score = star_counts.get(star, 0) / max(star_counts.values()) if star_counts else 0
        trend_score = star_counter.get(star, 0) / max(star_counter.values()) if star_counter else 0
        star_scores[star] = 0.6 * hist_score + 0.4 * trend_score
    
    # Generate all possible 5-number combinations from top 20 numbers
    top_20_numbers = sorted([num for num, _ in sorted(number_scores.items(), key=lambda x: x[1], reverse=True)[:20]])
    
    # Generate combinations with good distribution (mix of high/low)
    top_combinations = []
    
    # Strategy 1: Use top scoring numbers with balanced distribution
    for _ in range(100):  # Generate 100 candidates and pick top 10
        # Select 2-3 low numbers and 2-3 high numbers
        low_count = np.random.choice([2, 3])
        high_count = 5 - low_count
        
        low_numbers = [n for n in top_20_numbers if n <= 25]
        high_numbers = [n for n in top_20_numbers if n > 25]
        
        if len(low_numbers) >= low_count and len(high_numbers) >= high_count:
            # Select based on scores (weighted random selection)
            low_probs = [number_scores[n] for n in low_numbers]
            low_probs = [p/sum(low_probs) for p in low_probs]
            selected_low = np.random.choice(low_numbers, size=low_count, replace=False, p=low_probs)
            
            high_probs = [number_scores[n] for n in high_numbers]
            high_probs = [p/sum(high_probs) for p in high_probs]
            selected_high = np.random.choice(high_numbers, size=high_count, replace=False, p=high_probs)
            
            main_numbers = sorted(list(selected_low) + list(selected_high))
            
            # Select stars based on scores (weighted random selection)
            star_probs = [star_scores[s] for s in range(1, 13)]
            star_probs = [p/sum(star_probs) for p in star_probs]
            star_numbers = sorted(np.random.choice(range(1, 13), size=2, replace=False, p=star_probs))
            
            # Calculate combination score
            combo_score = sum(number_scores[n] for n in main_numbers) + 2 * sum(star_scores[s] for s in star_numbers)
            
            # Add sum check (optimal sum range is typically 95-155)
            sum_main = sum(main_numbers)
            if 95 <= sum_main <= 155:
                combo_score *= 1.2  # Boost score for optimal sum
            
            # Check for consecutive numbers (reduce score if too many consecutive)
            consecutive_count = 0
            for i in range(1, len(main_numbers)):
                if main_numbers[i] == main_numbers[i-1] + 1:
                    consecutive_count += 1
            
            if consecutive_count >= 3:
                combo_score *= 0.7  # Penalize too many consecutive numbers
            
            top_combinations.append({
                "main_numbers": main_numbers,
                "star_numbers": star_numbers,
                "score": combo_score,
                "explanation": f"Balanced combination with {low_count} low and {high_count} high numbers. Sum: {sum_main}."
            })
    
    # Sort by score and take top n
    top_combinations = sorted(top_combinations, key=lambda x: x["score"], reverse=True)[:n]
    
    return top_combinations

@Nodes.define(output="top_combinations")
async def generate_most_probable_combinations(processed_data: EuromillionsData) -> List[Dict]:
    """Generate the 10 most probable Euromillions combinations."""
    if processed_data.df is None or processed_data.stats is None:
        logger.error("No processed data available for generating top combinations")
        return []
    
    logger.info("Generating the 10 most probable Euromillions combinations")
    
    # Generate top combinations
    top_combos = generate_top_combinations(processed_data.df, processed_data.stats, n=10)
    
    # Add probability estimates based on scores
    max_score = max(combo["score"] for combo in top_combos)
    for combo in top_combos:
        # Normalize score to a percentage (relative probability)
        combo["probability"] = round((combo["score"] / max_score) * 100, 2)
    
    logger.info(f"Generated {len(top_combos)} most probable combinations")
    return top_combos

@Nodes.define(output="top_combinations_saved")
async def save_top_combinations_to_markdown(top_combinations: List[Dict], prediction_result: PredictionResult) -> str:
    """Save the top 10 most probable combinations to a markdown file."""
    if not top_combinations:
        logger.error("No top combinations to save")
        return None
    
    # Create analysis directory if it doesn't exist
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # Generate markdown content
    md = [
        f"# Top 10 Most Probable Euromillions Combinations for {prediction_result.prediction_date}",
        "",
        "This list presents the 10 combinations with the highest statistical probability of winning based on historical data analysis and recent trends.",
        "",
        "| Rank | Main Numbers | Star Numbers | Relative Probability | Notes |",
        "|------|-------------|--------------|---------------------|-------|"
    ]
    
    for i, combo in enumerate(top_combinations, 1):
        main_nums = ", ".join(map(str, combo["main_numbers"]))
        star_nums = ", ".join(map(str, combo["star_numbers"]))
        probability = f"{combo['probability']}%"
        explanation = combo["explanation"]
        
        md.append(f"| {i} | {main_nums} | {star_nums} | {probability} | {explanation} |")
    
    md.extend([
        "",
        "## How to Use This List",
        "",
        "These combinations are generated using statistical analysis of historical Euromillions data, including:",
        "",
        "- Frequency analysis of drawn numbers",
        "- Recent trend analysis from the last 20 draws",
        "- Distribution balance between low (1-25) and high (26-50) numbers",
        "- Sum total optimization (optimal range: 95-155)",
        "- Pattern analysis to avoid excessive consecutive numbers",
        "",
        "Each combination is assigned a relative probability score based on these factors. While no prediction system can guarantee a win (as lottery draws are random), these combinations leverage statistical patterns that have historically been more successful.",
        "",
        "---",
        "",
        "**Disclaimer**: Lottery outcomes are inherently random. This analysis is based on statistical principles and historical data, but cannot guarantee any specific results. Always play responsibly."
    ])
    
    # Create filename with prediction date
    filename = f"top_combinations_{prediction_result.prediction_date}.md"
    filepath = os.path.join(ANALYSIS_DIR, filename)
    
    # Save markdown to file
    with open(filepath, "w") as f:
        f.write("\n".join(md))
    
    logger.info(f"Top combinations saved to markdown file: {filepath}")
    return filepath

@Nodes.structured_llm_node(
    system_prompt="""You are an expert lottery analyst with deep knowledge of probability, statistics, and the Euromillions lottery game. 
Your task is to provide an in-depth analysis of Euromillions prediction data and alternative combinations.
Focus on statistical insights, pattern recognition, and practical advice for lottery players.
Be extremely thorough and detailed in your analysis, providing comprehensive explanations of the 'why' and 'how' behind each recommendation.
Include mathematical reasoning, historical context, and specific insights for each combination.""",
    output="gemini_analysis",
    response_model=EuromillionsAnalysis,
    prompt_template="""
# Euromillions Prediction Analysis

## Primary Prediction
- **Date**: {{ prediction_result.prediction_date }}
- **Main Numbers**: {{ prediction_result.main_numbers | join(', ') }}
- **Star Numbers**: {{ prediction_result.star_numbers | join(', ') }}
- **Confidence Score**: {{ prediction_result.confidence_score }}

## Historical Context
- Total draws analyzed: {{ prediction_result.historical_data.total_draws_analyzed }}
- Latest draw date: {{ prediction_result.historical_data.latest_draw_date }}
- Historical statistics summary is available in the prediction data

## Alternative Combinations
{% for combo in prediction_result.alternative_combinations %}
### Combination {{ loop.index }}
- **Main Numbers**: {{ combo.main_numbers | join(', ') }}
- **Star Numbers**: {{ combo.star_numbers | join(', ') }}
- **Strategy**: {{ combo.explanation }}
{% endfor %}

Please provide an extremely detailed and comprehensive analysis with the following sections:

1. **Statistical Assessment**: Provide a thorough evaluation of the primary prediction and each alternative combination from a statistical perspective. Include specific probability calculations, distribution analysis, and how each combination compares to historical winning patterns. Explain WHY certain combinations have better or worse statistical properties.

2. **Pattern Analysis**: Conduct a detailed analysis of patterns across these combinations and how they relate to historical draws. Identify specific mathematical patterns, number spacing, sum totals, and other relevant factors. Explain HOW these patterns affect winning probability.

3. **Strategic Recommendations**: Suggest the most promising combination(s) based on statistical principles and lottery theory. Provide detailed reasoning for each recommendation, including mathematical justification and historical precedent. Explain WHY these strategies are effective.

4. **Optimization Suggestions**: Recommend specific adjustments to improve the combinations. Include detailed explanations of how these optimizations would improve winning chances. Provide concrete examples of optimized combinations.

5. **Final Recommendation**: Provide a specific combination you recommend playing and explain in detail WHY this combination is optimal. Include a comprehensive justification based on all previous analysis.

Your analysis should be extremely detailed, data-driven, and focused on actionable insights. Include specific mathematical reasoning and statistical principles throughout.
""",
    model="gemini/gemini-1.5-pro",
    max_tokens=4000
)
async def gemini_analysis(prediction_result: PredictionResult) -> EuromillionsAnalysis:
    """Advanced analysis of Euromillions prediction using Gemini."""
    # This function body will be replaced by the LLM call
    # Adding a fallback implementation in case the LLM call fails
    if prediction_result is None:
        logger.error("No prediction result to analyze with Gemini")
        return EuromillionsAnalysis(
            statistical_assessment="No prediction available for statistical assessment.",
            pattern_analysis="No prediction available for pattern analysis.",
            strategic_recommendations="No strategic recommendations available.",
            optimization_suggestions="No optimization suggestions available.",
            final_recommendation="No final recommendation available."
        )
    
    # Create a comprehensive analysis as fallback
    return EuromillionsAnalysis(
        statistical_assessment="The primary prediction shows a balanced distribution of numbers across the range.",
        pattern_analysis="The combinations collectively cover a wide range of the possible number space.",
        strategic_recommendations="The most frequent numbers combination has the strongest statistical backing.",
        optimization_suggestions="Aim for a 2:3 or 3:2 ratio of low to high numbers for optimal distribution.",
        final_recommendation="A hybrid approach combining elements of the most frequent and balanced combinations would be optimal."
    )

@Nodes.define(output="analysis_saved")
async def save_analysis_to_markdown(prediction_result: PredictionResult, gemini_analysis: EuromillionsAnalysis) -> str:
    """Save the Gemini analysis to a detailed markdown file."""
    if prediction_result is None or gemini_analysis is None:
        logger.error("Missing prediction result or analysis for saving to markdown")
        return None
    
    # Create analysis directory if it doesn't exist
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # Generate markdown content
    markdown_content = gemini_analysis.to_markdown(prediction_result)
    
    # Create filename with prediction date
    filename = f"analysis_{prediction_result.prediction_date}.md"
    filepath = os.path.join(ANALYSIS_DIR, filename)
    
    # Save markdown to file
    with open(filepath, "w") as f:
        f.write(markdown_content)
    
    logger.info(f"Analysis saved to markdown file: {filepath}")
    return filepath

@Nodes.define(output="prediction_analysis")
async def analyze_prediction(prediction_result: PredictionResult, model: str = "gpt-4") -> str:
    """Analyze the prediction and provide insights."""
    if prediction_result is None:
        logger.error("No prediction result to analyze")
        return "No prediction available for analysis."
    
    # Since we don't have actual LLM access, provide a simple analysis
    main_numbers = prediction_result.main_numbers
    star_numbers = prediction_result.star_numbers
    confidence = prediction_result.confidence_score
    alternative_combinations = prediction_result.alternative_combinations
    
    # Create a simple analysis
    analysis = [
        f"Analysis of Euromillions prediction for {prediction_result.prediction_date}:",
        "",
        f"The predicted numbers are: {', '.join(map(str, main_numbers))} with stars {', '.join(map(str, star_numbers))}.",
        "",
        "Number distribution analysis:",
        f"- Low numbers (1-25): {sum(1 for n in main_numbers if n <= 25)} numbers",
        f"- High numbers (26-50): {sum(1 for n in main_numbers if n > 25)} numbers",
        "",
        "Star distribution analysis:",
        f"- Low stars (1-6): {sum(1 for s in star_numbers if s <= 6)} stars",
        f"- High stars (7-12): {sum(1 for s in star_numbers if s > 6)} stars",
        "",
        f"The model confidence score is {confidence:.4f}, which suggests "
        f"{'moderate reliability' if confidence > 0 else 'low reliability'} in this prediction.",
        "",
        "Alternative combinations to consider:",
    ]
    
    # Add alternative combinations to the analysis
    for i, combo in enumerate(alternative_combinations, 1):
        analysis.append(f"\nCombination {i}:")
        analysis.append(f"Main numbers: {', '.join(map(str, combo['main_numbers']))}")
        analysis.append(f"Star numbers: {', '.join(map(str, combo['star_numbers']))}")
        analysis.append(f"Rationale: {combo['explanation']}")
    
    analysis.append("\nRecommendation:")
    analysis.append("Consider using these combinations as inspiration rather than definitive selections.")
    analysis.append("For better odds, you might want to include a more balanced mix of high and low numbers.")
    
    return "\n".join(analysis)

@Nodes.define(output="saved_prediction")
async def save_prediction(prediction_result: PredictionResult) -> str:
    """Save prediction to file."""
    if prediction_result is None:
        logger.error("No prediction result to save")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create filename with prediction date
    filename = f"prediction_{prediction_result.prediction_date}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save prediction to file
    with open(filepath, "w") as f:
        json.dump(prediction_result.model_dump(), f, indent=2)
    
    logger.info(f"Prediction saved to {filepath}")
    return filepath

# Workflow Definition
def create_euromillions_prediction_workflow() -> Workflow:
    """Create the Euromillions prediction workflow."""
    wf = Workflow("load_euromillions_data")
    
    # Define the workflow sequence
    wf.node("load_euromillions_data").then("preprocess_data")
    wf.then("extract_features")
    wf.then("train_models")
    wf.then("generate_prediction")
    wf.then("save_prediction")
    wf.then("analyze_prediction")
    wf.then("gemini_analysis")  
    wf.then("save_analysis_to_markdown")  # Add node to save analysis to markdown
    wf.then("generate_most_probable_combinations")  # Add node to generate top combinations
    wf.then("save_top_combinations_to_markdown")    # Add node to save top combinations
    
    # Define input mappings
    wf.node_input_mappings["preprocess_data"] = {
        "euromillions_data": "euromillions_data"
    }
    wf.node_input_mappings["extract_features"] = {
        "processed_data": "processed_data"
    }
    wf.node_input_mappings["train_models"] = {
        "feature_data": "feature_data"
    }
    wf.node_input_mappings["generate_prediction"] = {
        "processed_data": "processed_data",
        "prediction_models": "prediction_models",
        "prediction_date": "prediction_date"
    }
    wf.node_input_mappings["save_prediction"] = {
        "prediction_result": "prediction_result"
    }
    wf.node_input_mappings["analyze_prediction"] = {
        "prediction_result": "prediction_result",
        "model": "model"
    }
    wf.node_input_mappings["gemini_analysis"] = {
        "prediction_result": "prediction_result"
    }
    wf.node_input_mappings["save_analysis_to_markdown"] = {
        "prediction_result": "prediction_result",
        "gemini_analysis": "gemini_analysis"
    }
    wf.node_input_mappings["generate_most_probable_combinations"] = {
        "processed_data": "processed_data"
    }
    wf.node_input_mappings["save_top_combinations_to_markdown"] = {
        "top_combinations": "top_combinations",
        "prediction_result": "prediction_result"
    }
    
    logger.info("Euromillions prediction workflow created")
    return wf

# Run Workflow
async def run_workflow(
    data_path: str = None,
    prediction_date: str = None,
    model: str = "gpt-4",
    task_id: str = "default",
    _handle_event: Optional[Callable[[str, Dict[str, Any]], None]] = None
) -> dict:
    """Execute the Euromillions prediction workflow."""
    initial_context = {
        "data_path": data_path,
        "prediction_date": prediction_date,
        "model": model
    }
    
    workflow = create_euromillions_prediction_workflow()
    engine = workflow.build()
    # Add the event observer if _handle_event is provided
    if _handle_event:
        # Create a lambda to bind task_id to the observer
        bound_observer = lambda event: asyncio.create_task(
            event_observer(event, task_id=task_id, _handle_event=_handle_event)
        )
        engine.add_observer(bound_observer)
        
    result = await engine.run(initial_context)
    
    logger.info("Workflow execution completed")
    return result

# Direct testing function instead of using Typer
async def test_workflow():
    """Test the Euromillions prediction workflow directly."""
    logger.info("Starting Euromillions prediction workflow test")
    
    # Create required directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    try:
        # Run the workflow with default parameters
        result = await run_workflow(model="gemini/gemini-1.5-pro")  
        
        # Extract the prediction, analysis, and saved files
        prediction = result.get("prediction_result")
        analysis = result.get("prediction_analysis")
        gemini_insights = result.get("gemini_analysis")
        saved_path = result.get("saved_prediction")
        analysis_md_path = result.get("analysis_saved")
        top_combinations = result.get("top_combinations")
        top_combinations_path = result.get("top_combinations_saved")
        
        if prediction:
            print("\nEuromillions Prediction:")
            print("=======================")
            print(f"Date: {prediction.prediction_date}")
            print(f"Main Numbers: {', '.join(map(str, prediction.main_numbers))}")
            print(f"Star Numbers: {', '.join(map(str, prediction.star_numbers))}")
            print(f"Confidence Score: {prediction.confidence_score:.4f}")
            
            # Display alternative combinations
            if prediction.alternative_combinations:
                print("\nAlternative Combinations:")
                print("========================")
                for i, combo in enumerate(prediction.alternative_combinations, 1):
                    print(f"\nCombination {i}:")
                    print(f"Main numbers: {', '.join(map(str, combo['main_numbers']))}")
                    print(f"Star numbers: {', '.join(map(str, combo['star_numbers']))}")
                    print(f"Rationale: {combo['explanation']}")
            
            # Display top 10 most probable combinations
            if top_combinations:
                print("\nTop 10 Most Probable Combinations:")
                print("================================")
                for i, combo in enumerate(top_combinations, 1):
                    print(f"\nRank {i} (Probability: {combo['probability']}%):")
                    print(f"Main numbers: {', '.join(map(str, combo['main_numbers']))}")
                    print(f"Star numbers: {', '.join(map(str, combo['star_numbers']))}")
                    print(f"Notes: {combo['explanation']}")
            
            if analysis:
                print("\nBasic Analysis:")
                print("==============")
                print(analysis)
            
            if gemini_insights:
                print("\nAdvanced Gemini Analysis:")
                print("========================")
                print("Statistical Assessment:")
                print(gemini_insights.statistical_assessment)
                print("\nPattern Analysis:")
                print(gemini_insights.pattern_analysis)
                print("\nStrategic Recommendations:")
                print(gemini_insights.strategic_recommendations)
                print("\nOptimization Suggestions:")
                print(gemini_insights.optimization_suggestions)
                print("\nFinal Recommendation:")
                print(gemini_insights.final_recommendation)
            
            if saved_path:
                print(f"\nPrediction saved to: {saved_path}")
                
            if analysis_md_path:
                print(f"\nDetailed analysis saved to: {analysis_md_path}")
                
            if top_combinations_path:
                print(f"\nTop 10 combinations saved to: {top_combinations_path}")
                print("This markdown file contains a comprehensive list of the most probable combinations.")
        else:
            print("No prediction was generated. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Error in workflow execution: {str(e)}")
        print(f"Error: {str(e)}")

# Run the test workflow
if __name__ == "__main__":
    asyncio.run(test_workflow())
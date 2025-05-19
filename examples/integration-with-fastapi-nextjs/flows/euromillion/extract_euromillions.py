from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime
import requests
import os
from loguru import logger
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

# Set up logging
logger.remove()
logger.add(
    "/home/yarab/Bureau/trash_agents_tests/f1/examples/flow/euromillion/euromillions_extraction.log",
    format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
    level="INFO",
    rotation="10 MB"
)

BASE_URL = "https://www.tirage-euromillions.net/euromillions/annees/annee-{year}/"
OUTPUT_DIR = "/home/yarab/Bureau/trash_agents_tests/f1/examples/flow/euromillion/data"

def fetch_html_content(url, max_retries=3, retry_delay=2):
    """
    Fetch HTML content from a URL with retry logic.
    
    Args:
        url (str): URL to fetch
        max_retries (int): Maximum number of retry attempts
        retry_delay (int): Delay between retries in seconds
        
    Returns:
        str: HTML content or None if failed
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed for {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                return None

def extract_euromillions_data(html_content, year):
    """
    Extract Euromillions lottery data from HTML content.
    
    Args:
        html_content (str): HTML content containing Euromillions data
        year (int): Year of the data
        
    Returns:
        pandas.DataFrame: DataFrame containing extracted data
    """
    if not html_content:
        logger.error(f"No HTML content to extract for year {year}")
        return pd.DataFrame()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the table containing the data
    table = soup.find('table', class_='blue_table')
    if not table:
        logger.error(f"No table found in HTML content for year {year}")
        return pd.DataFrame()
    
    # Initialize lists to store data
    dates = []
    numbers = []
    stars = []
    winners = []
    jackpots = []
    months = []
    years = []
    
    current_month = None
    
    # Iterate through table rows
    for row in table.find_all('tr'):
        # Check if this is a month header row
        month_header = row.find('td', colspan='4')
        if month_header and 'strong' in str(month_header):
            current_month = month_header.get_text().strip()
            continue
            
        # Check if this is a data row (has date)
        date_cell = row.find('td', attrs={'data-order': True})
        if date_cell:
            try:
                # Extract date
                date_text = date_cell.get_text().strip()
                date_order = date_cell.get('data-order')
                formatted_date = datetime.strptime(date_order, '%Y%m%d').strftime('%Y-%m-%d')
                dates.append(formatted_date)
                
                # Add current month and year
                months.append(current_month)
                years.append(year)
                
                # Extract numbers and stars
                numbers_cell = row.find('td', class_='nowrap')
                if numbers_cell:
                    # Extract the 5 main numbers
                    main_numbers = [span.get_text() for span in numbers_cell.find_all('span', class_='ball_small')]
                    numbers.append(' '.join(main_numbers))
                    
                    # Extract the 2 star numbers
                    star_numbers = [span.get_text() for span in numbers_cell.find_all('span', class_='star_small')]
                    stars.append(' '.join(star_numbers))
                else:
                    numbers.append('')
                    stars.append('')
                
                # Extract winners
                winner_cell = row.find('td', class_='nomobile')
                if winner_cell:
                    # Extract number of winners
                    if 'strong' in str(winner_cell):
                        winners.append(1)  # There was a winner
                    else:
                        winners.append(0)  # No winner
                else:
                    winners.append(0)
                
                # Extract jackpot
                jackpot_cell = row.find('td', class_='jackpot')
                if jackpot_cell:
                    # Extract jackpot amount (remove non-digit characters except for decimal point)
                    jackpot_text = jackpot_cell.get_text().strip()
                    jackpot_value = re.sub(r'[^\d]', '', jackpot_text)
                    jackpots.append(jackpot_value)
                else:
                    jackpots.append('0')
            except Exception as e:
                logger.error(f"Error processing row in year {year}: {str(e)}")
                continue
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Month': months,
        'Year': years,
        'Numbers': numbers,
        'Stars': stars,
        'Winners': winners,
        'Jackpot': jackpots
    })
    
    # Convert data types
    df['Date'] = pd.to_datetime(df['Date'])
    df['Winners'] = pd.to_numeric(df['Winners'])
    df['Jackpot'] = pd.to_numeric(df['Jackpot'])
    
    logger.info(f"Extracted {len(df)} draws for year {year}")
    return df

def fetch_all_years(start_year=2004, end_year=2025):
    """
    Fetch Euromillions data for a range of years.
    
    Args:
        start_year (int): Start year
        end_year (int): End year
        
    Returns:
        pandas.DataFrame: Combined DataFrame for all years
    """
    all_data = []
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for year in range(start_year, end_year + 1):
        logger.info(f"Fetching data for year {year}")
        url = BASE_URL.format(year=year)
        
        # Check if we already have data for this year
        year_file = os.path.join(OUTPUT_DIR, f"euromillions_{year}.csv")
        if os.path.exists(year_file):
            logger.info(f"Loading existing data for year {year} from {year_file}")
            year_df = pd.read_csv(year_file)
            all_data.append(year_df)
            continue
        
        html_content = fetch_html_content(url)
        if html_content:
            year_df = extract_euromillions_data(html_content, year)
            if not year_df.empty:
                # Save year data to CSV
                year_df.to_csv(year_file, index=False)
                logger.info(f"Saved data for year {year} to {year_file}")
                all_data.append(year_df)
            else:
                logger.warning(f"No data extracted for year {year}")
        else:
            logger.warning(f"Failed to fetch data for year {year}")
        
        # Be nice to the server
        time.sleep(1)
    
    # Combine all years
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined data for {len(all_data)} years, total {len(combined_df)} draws")
        return combined_df
    else:
        logger.error("No data extracted for any year")
        return pd.DataFrame()

def clean_and_validate_data(df):
    """
    Clean and validate the Euromillions data.
    
    Args:
        df (pandas.DataFrame): DataFrame to clean
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Convert date to datetime if not already
    if not pd.api.types.is_datetime64_dtype(df_clean['Date']):
        df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
    
    # Drop rows with invalid dates
    invalid_dates = df_clean['Date'].isna()
    if invalid_dates.any():
        logger.warning(f"Dropping {invalid_dates.sum()} rows with invalid dates")
        df_clean = df_clean.dropna(subset=['Date'])
    
    # Ensure Winners is numeric
    df_clean['Winners'] = pd.to_numeric(df_clean['Winners'], errors='coerce').fillna(0).astype(int)
    
    # Ensure Jackpot is numeric
    df_clean['Jackpot'] = pd.to_numeric(df_clean['Jackpot'], errors='coerce').fillna(0)
    
    # Sort by date
    df_clean = df_clean.sort_values('Date')
    
    # Add day of week
    df_clean['DayOfWeek'] = df_clean['Date'].dt.day_name()
    
    logger.info(f"Data cleaned and validated: {len(df_clean)} rows")
    return df_clean

def analyze_data(df):
    """
    Analyze Euromillions data and generate insights.
    
    Args:
        df (pandas.DataFrame): DataFrame containing Euromillions data
        
    Returns:
        dict: Dictionary of analysis results
    """
    if df.empty:
        logger.error("No data to analyze")
        return {}
    
    logger.info("Analyzing Euromillions data")
    
    # Basic statistics
    total_draws = len(df)
    total_winners = df['Winners'].sum()
    avg_jackpot = df['Jackpot'].mean()
    max_jackpot = df['Jackpot'].max()
    max_jackpot_date = df.loc[df['Jackpot'].idxmax(), 'Date'].strftime('%Y-%m-%d')
    
    # Winners by year
    winners_by_year = df.groupby('Year')['Winners'].sum()
    
    # Jackpot trends by year
    avg_jackpot_by_year = df.groupby('Year')['Jackpot'].mean()
    
    # Most frequent numbers
    all_numbers = []
    for num_set in df['Numbers']:
        if isinstance(num_set, str):
            all_numbers.extend(num_set.split())
    
    number_counts = Counter(all_numbers)
    most_common_numbers = number_counts.most_common(10)
    
    # Most frequent stars
    all_stars = []
    for star_set in df['Stars']:
        if isinstance(star_set, str):
            all_stars.extend(star_set.split())
    
    star_counts = Counter(all_stars)
    most_common_stars = star_counts.most_common(5)
    
    # Print summary
    logger.info(f"Total draws: {total_draws}")
    logger.info(f"Total jackpot winners: {total_winners}")
    logger.info(f"Average jackpot: €{avg_jackpot:,.2f}")
    logger.info(f"Maximum jackpot: €{max_jackpot:,.2f} on {max_jackpot_date}")
    
    logger.info("Most frequent numbers:")
    for num, count in most_common_numbers:
        logger.info(f"Number {num}: appeared {count} times")
    
    logger.info("Most frequent stars:")
    for star, count in most_common_stars:
        logger.info(f"Star {star}: appeared {count} times")
    
    return {
        'total_draws': total_draws,
        'total_winners': total_winners,
        'avg_jackpot': avg_jackpot,
        'max_jackpot': max_jackpot,
        'max_jackpot_date': max_jackpot_date,
        'winners_by_year': winners_by_year,
        'avg_jackpot_by_year': avg_jackpot_by_year,
        'number_counts': number_counts,
        'star_counts': star_counts
    }

def create_visualizations(df, stats, output_dir=None):
    """
    Create visualizations from Euromillions data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing Euromillions data
        stats (dict): Statistics from analyze_data function
        output_dir (str): Directory to save visualizations
    """
    if df.empty:
        logger.error("No data to visualize")
        return
    
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating visualizations in {output_dir}")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Jackpot trend over time
    plt.figure(figsize=(14, 7))
    plt.plot(df['Date'], df['Jackpot'] / 1000000, marker='.', linestyle='-', alpha=0.7, color='#1f77b4')
    plt.title('Euromillions Jackpot Trend Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Jackpot (Million €)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'jackpot_trend.png'))
    plt.close()
    
    # 2. Number frequency
    number_counts = stats['number_counts']
    numbers = list(range(1, 51))
    frequencies = [number_counts.get(str(num), 0) for num in numbers]
    
    plt.figure(figsize=(14, 7))
    bars = plt.bar(numbers, frequencies, color='#2ca02c', alpha=0.8)
    
    # Highlight most frequent numbers
    most_common = [int(num) for num, _ in number_counts.most_common(5)]
    for i, num in enumerate(numbers):
        if num in most_common:
            bars[i-1].set_color('#d62728')
    
    plt.title('Frequency of Euromillions Numbers', fontsize=16)
    plt.xlabel('Number', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(numbers[::5] + [50])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'number_frequency.png'))
    plt.close()
    
    # 3. Star frequency
    star_counts = stats['star_counts']
    stars = list(range(1, 13))  # Stars range from 1 to 12
    star_frequencies = [star_counts.get(str(star), 0) for star in stars]
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(stars, star_frequencies, color='#ff7f0e', alpha=0.8)
    
    # Highlight most frequent stars
    most_common_stars = [int(star) for star, _ in star_counts.most_common(3)]
    for i, star in enumerate(stars):
        if star in most_common_stars:
            bars[i-1].set_color('#d62728')
    
    plt.title('Frequency of Euromillions Stars', fontsize=16)
    plt.xlabel('Star', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(stars)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'star_frequency.png'))
    plt.close()
    
    # 4. Winners by year
    winners_by_year = stats['winners_by_year']
    
    plt.figure(figsize=(14, 7))
    winners_by_year.plot(kind='bar', color='#9467bd', alpha=0.8)
    plt.title('Euromillions Winners by Year', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Winners', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'winners_by_year.png'))
    plt.close()
    
    # 5. Average jackpot by year
    avg_jackpot_by_year = stats['avg_jackpot_by_year']
    
    plt.figure(figsize=(14, 7))
    (avg_jackpot_by_year / 1000000).plot(kind='bar', color='#8c564b', alpha=0.8)
    plt.title('Average Euromillions Jackpot by Year', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Jackpot (Million €)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_jackpot_by_year.png'))
    plt.close()
    
    # 6. Heatmap of number frequencies
    plt.figure(figsize=(12, 10))
    number_matrix = np.zeros((5, 10))
    for num, count in number_counts.items():
        if num.isdigit():
            num_int = int(num)
            if 1 <= num_int <= 50:
                row = (num_int - 1) // 10
                col = (num_int - 1) % 10
                number_matrix[row, col] = count
    
    sns.heatmap(number_matrix, annot=True, fmt='g', cmap='YlGnBu')
    plt.title('Heatmap of Euromillions Number Frequencies', fontsize=16)
    plt.xlabel('Last Digit (0-9)', fontsize=12)
    plt.ylabel('Tens Digit (0-4)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'number_heatmap.png'))
    plt.close()
    
    logger.info(f"Created 6 visualizations in {output_dir}")

def save_to_csv(df, output_file=None):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        output_file (str): Output file path
    """
    if df.empty:
        logger.error("No data to save")
        return
    
    if output_file is None:
        output_file = os.path.join(OUTPUT_DIR, "euromillions_all_years.csv")
    
    df.to_csv(output_file, index=False)
    logger.info(f"Data saved to {output_file}")

def main(start_year=2004, end_year=2025, html_file_path=None):
    """
    Main function to extract and analyze Euromillions data.
    
    Args:
        start_year (int): Start year for data extraction
        end_year (int): End year for data extraction
        html_file_path (str): Path to HTML file (optional, for local processing)
    """
    logger.info(f"Starting Euromillions data extraction from {start_year} to {end_year}")
    
    if html_file_path:
        # Process local file
        logger.info(f"Processing local file: {html_file_path}")
        try:
            with open(html_file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            year = int(re.search(r'annee-(\d{4})', html_file_path).group(1))
            df = extract_euromillions_data(html_content, year)
        except Exception as e:
            logger.error(f"Error processing local file: {str(e)}")
            return
    else:
        # Fetch data from website
        df = fetch_all_years(start_year, end_year)
    
    if df.empty:
        logger.error("No data extracted, exiting")
        return
    
    # Clean and validate data
    df_clean = clean_and_validate_data(df)
    
    # Save to CSV
    save_to_csv(df_clean)
    
    # Analyze data
    stats = analyze_data(df_clean)
    
    # Create visualizations
    create_visualizations(df_clean, stats)
    
    logger.info("Euromillions data extraction and analysis completed")
    print(f"Extracted and analyzed Euromillions data from {start_year} to {end_year}")
    print(f"Total draws: {stats.get('total_draws', 0)}")
    print(f"Total jackpot winners: {stats.get('total_winners', 0)}")
    print(f"Maximum jackpot: €{stats.get('max_jackpot', 0):,.2f} on {stats.get('max_jackpot_date', 'unknown')}")
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract and analyze Euromillions data')
    parser.add_argument('--start-year', type=int, default=2004, help='Start year for data extraction')
    parser.add_argument('--end-year', type=int, default=2025, help='End year for data extraction')
    parser.add_argument('--html-file', type=str, help='Path to local HTML file (optional)')
    
    args = parser.parse_args()
    
    main(args.start_year, args.end_year, args.html_file)

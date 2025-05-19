from bs4 import BeautifulSoup
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from collections import Counter
import os

def extract_euromillions_data(html_content):
    """
    Extract Euromillions lottery data from HTML content.
    
    Args:
        html_content (str): HTML content containing Euromillions data
        
    Returns:
        pandas.DataFrame: DataFrame containing extracted data
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the table containing the data
    table = soup.find('table', class_='blue_table')
    
    # Initialize lists to store data
    dates = []
    numbers = []
    stars = []
    winners = []
    jackpots = []
    months = []
    
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
            # Extract date
            date_text = date_cell.get_text().strip()
            date_order = date_cell.get('data-order')
            formatted_date = datetime.strptime(date_order, '%Y%m%d').strftime('%Y-%m-%d')
            dates.append(formatted_date)
            
            # Add current month
            months.append(current_month)
            
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
                # Extract number of winners (remove any non-digit characters)
                winner_text = winner_cell.get_text().strip()
                if 'strong' in str(winner_cell):
                    winners.append(1)  # There was a winner
                else:
                    winners.append(0)  # No winner
            else:
                winners.append('')
            
            # Extract jackpot
            jackpot_cell = row.find('td', class_='jackpot')
            if jackpot_cell:
                # Extract jackpot amount (remove non-digit characters except for decimal point)
                jackpot_text = jackpot_cell.get_text().strip()
                jackpot_value = re.sub(r'[^\d]', '', jackpot_text)
                jackpots.append(jackpot_value)
            else:
                jackpots.append('')
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Month': months,
        'Numbers': numbers,
        'Stars': stars,
        'Winners': winners,
        'Jackpot': jackpots
    })
    
    # Convert jackpot to numeric
    df['Jackpot'] = pd.to_numeric(df['Jackpot'])
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    return df

def analyze_data(df):
    """
    Analyze Euromillions data and generate insights.
    
    Args:
        df (pandas.DataFrame): DataFrame containing Euromillions data
    """
    print("\n===== EUROMILLIONS 2004 ANALYSIS =====")
    
    # Basic statistics
    total_draws = len(df)
    total_winners = df['Winners'].sum()
    avg_jackpot = df['Jackpot'].mean()
    max_jackpot = df['Jackpot'].max()
    max_jackpot_date = df.loc[df['Jackpot'].idxmax(), 'Date'].strftime('%Y-%m-%d')
    
    print(f"\nTotal draws: {total_draws}")
    print(f"Total jackpot winners: {total_winners}")
    print(f"Average jackpot: €{avg_jackpot:,.2f}")
    print(f"Maximum jackpot: €{max_jackpot:,.2f} on {max_jackpot_date}")
    
    # Most frequent numbers
    all_numbers = []
    for num_set in df['Numbers']:
        all_numbers.extend(num_set.split())
    
    number_counts = Counter(all_numbers)
    most_common_numbers = number_counts.most_common(5)
    
    print("\nMost frequent numbers:")
    for num, count in most_common_numbers:
        print(f"Number {num}: appeared {count} times")
    
    # Most frequent stars
    all_stars = []
    for star_set in df['Stars']:
        all_stars.extend(star_set.split())
    
    star_counts = Counter(all_stars)
    most_common_stars = star_counts.most_common(3)
    
    print("\nMost frequent stars:")
    for star, count in most_common_stars:
        print(f"Star {star}: appeared {count} times")
    
    return {
        'total_draws': total_draws,
        'total_winners': total_winners,
        'avg_jackpot': avg_jackpot,
        'max_jackpot': max_jackpot,
        'max_jackpot_date': max_jackpot_date,
        'number_counts': number_counts,
        'star_counts': star_counts
    }

def create_visualizations(df, stats, output_dir='euromillions_analysis'):
    """
    Create visualizations from Euromillions data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing Euromillions data
        stats (dict): Statistics from analyze_data function
        output_dir (str): Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Jackpot trend over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Jackpot'] / 1000000, marker='o', linestyle='-', color='#1f77b4')
    plt.title('Euromillions Jackpot Trend in 2004', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Jackpot (Million €)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/jackpot_trend.png')
    
    # 2. Number frequency
    number_counts = stats['number_counts']
    numbers = list(range(1, 51))
    frequencies = [number_counts.get(str(num), 0) for num in numbers]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(numbers, frequencies, color='#2ca02c')
    
    # Highlight most frequent numbers
    most_common = [int(num) for num, _ in number_counts.most_common(5)]
    for i, num in enumerate(numbers):
        if num in most_common:
            bars[i].set_color('#d62728')
    
    plt.title('Frequency of Euromillions Numbers in 2004', fontsize=16)
    plt.xlabel('Number', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(numbers[::5] + [50])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/number_frequency.png')
    
    # 3. Star frequency
    star_counts = stats['star_counts']
    stars = list(range(1, 10))
    star_frequencies = [star_counts.get(str(star), 0) for star in stars]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stars, star_frequencies, color='#ff7f0e')
    
    # Highlight most frequent stars
    most_common_stars = [int(star) for star, _ in star_counts.most_common(3)]
    for i, star in enumerate(stars):
        if star in most_common_stars:
            bars[i].set_color('#d62728')
    
    plt.title('Frequency of Euromillions Stars in 2004', fontsize=16)
    plt.xlabel('Star', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(stars)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/star_frequency.png')
    
    # 4. Winners by month
    monthly_winners = df.groupby(df['Date'].dt.strftime('%B'))['Winners'].sum()
    
    plt.figure(figsize=(12, 6))
    monthly_winners.plot(kind='bar', color='#9467bd')
    plt.title('Euromillions Winners by Month in 2004', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Winners', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/winners_by_month.png')
    
    print(f"\nVisualizations saved to '{output_dir}' directory")

def save_to_csv(df, output_file="/home/yarab/Bureau/trash_agents_tests/f1/legal_graph/euromillion/euromillions_2004.csv"):
    """
    Save DataFrame to CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame to save
        output_file (str): Output file path
    """
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

def main(html_file_path):
    """
    Main function to extract and analyze Euromillions data from HTML file.
    
    Args:
        html_file_path (str): Path to HTML file
    """
    # Read HTML file
    with open(html_file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    # Extract data
    df = extract_euromillions_data(html_content)
    
    # Print summary
    print(f"Extracted {len(df)} Euromillions draws from 2004")
    print(df.head())
    
    # Save to CSV
    save_to_csv(df)
    
    # Analyze data
    stats = analyze_data(df)
    
    # Create visualizations
    create_visualizations(df, stats)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        html_file_path = sys.argv[1]
    else:
        html_file_path = "/home/yarab/Bureau/trash_agents_tests/f1/legal_graph/euromillion/euromillions_2004.html"  # Default file path
    
    main(html_file_path)

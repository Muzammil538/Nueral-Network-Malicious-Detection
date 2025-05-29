import csv

def create_labeled_dataset(safe_urls_file, malicious_urls_file, output_csv):
    """
    Creates a labeled dataset CSV from safe and malicious URL files.
    
    Args:
        safe_urls_file (str): Path to file containing safe URLs
        malicious_urls_file (str): Path to file containing malicious URLs
        output_csv (str): Path for the output CSV file
    """
    try:
        with open(safe_urls_file, 'r', encoding='utf-8') as safe_file, \
             open(malicious_urls_file, 'r', encoding='utf-8') as malicious_file, \
             open(output_csv, 'w', newline='', encoding='utf-8') as csv_file:
            
            writer = csv.writer(csv_file)
            # Write header
            writer.writerow(['url', 'label'])
            
            # Process safe URLs (label 0)
            safe_count = 0
            for url in safe_file:
                url = url.strip()
                if url:  # Skip empty lines
                    writer.writerow([url, 0])
                    safe_count += 1
            
            # Process malicious URLs (label 1)
            malicious_count = 0
            for url in malicious_file:
                url = url.strip()
                if url:  # Skip empty lines
                    writer.writerow([url, 1])
                    malicious_count += 1
            
            print(f"Dataset created successfully with {safe_count} safe URLs and {malicious_count} malicious URLs.")
            print(f"Output saved to: {output_csv}")
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    safe_file = r'safe_urls.txt'
    malicious_file = r'malicious_urls.txt'
    output_file = r'dataset/full_urls.csv'
    
    create_labeled_dataset(safe_file, malicious_file, output_file)
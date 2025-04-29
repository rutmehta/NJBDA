# %%
import pandas as pd
import requests
import os
from urllib.parse import urlparse
from pathlib import Path
import mimetypes

# %%
# Create a directory for syllabi if it doesn't exist
syllabi_dir = Path('syllabi')
syllabi_dir.mkdir(exist_ok=True)

# %%
# Read the CSV file
df = pd.read_csv('AT_JN083SC6.csv')

# Display basic information about the dataframe
print("\nDataframe Info:")
print(df.info())

# Display the first few rows
print("\nFirst few rows of the data:")
print(df.head())

# Display basic statistics
print("\nBasic statistics:")
print(df.describe())


# %%
for index, row in df.head().iterrows():
    url = row['LINK TO SYLLABUS']
    course_id = row['COURSE ID'].replace(' ', '_')  # Clean course ID for filename
    
    print(f"Downloading syllabus for {course_id}...")
    try:
        # Make a HEAD request first to check content type
        head_response = requests.head(url, allow_redirects=True)
        content_type = head_response.headers.get('content-type', '')
        
        # Determine file extension based on content type
        if 'pdf' in content_type.lower():
            extension = '.pdf'
        elif 'document' in content_type.lower() or 'docx' in content_type.lower():
            extension = '.docx'
        else:
            # If content-type is not clear, try to get extension from URL
            path = urlparse(url).path
            extension = os.path.splitext(path)[1]
            if not extension:
                extension = '.pdf'  # Default to PDF if no extension found
        
        # Create filename with appropriate extension
        filename = f"{course_id}_syllabus{extension}"
        filepath = syllabi_dir / filename
        
        # Download the file
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the file
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded: {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error with {url}: {str(e)}")

print("\nDownload complete! Files are saved in the 'syllabi' directory.")



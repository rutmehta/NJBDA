#!/usr/bin/env python3
# Course Syllabus Analysis - Combining course equivalency data with syllabus similarity

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pdfplumber
import docx
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from urllib.parse import urlparse
import mimetypes
from tqdm.notebook import tqdm_notebook
import pickle

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Installing spaCy model...")
    import subprocess
    subprocess.call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load sentence transformer model
print("Loading sentence transformer model...")
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Define paths
BASE_PATH = Path("/Users/rutmehta/Developer/NJBDA")
EXCEL_DIR = BASE_PATH / "course_equiv analysis" / "excels"
RUTGERS_SYLLABI_DIR = BASE_PATH / "syllabi_matching" / "rutgers_datascience_syllabi"
SYLLABI_EXCELS_DIR = BASE_PATH / "syllabi_matching" / "syllabi_excels"
DOWNLOADED_SYLLABI_DIR = SYLLABI_EXCELS_DIR / "downloaded_syllabi"
CACHE_DIR = BASE_PATH / "cache"

# This is where Rutgers syllabi embeddings are cached to avoid recomputing them
RUTGERS_EMBEDDINGS_CACHE = CACHE_DIR / "rutgers_embeddings.pkl"

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(exist_ok=True)

# Section weights for syllabus comparison (from syllabus_similarity.py)
SECTION_WEIGHTS = {
    "Course Description": 2.0,
    "Learning Outcomes": 3.0,
    "Objectives": 2.0,
    "Prerequisites": 1.0,
    "Grading": 0.5,
    "Schedule": 1.0,
}

ALL_SECTIONS = list(SECTION_WEIGHTS.keys())

# Define canonical section titles for syllabus parsing
SECTION_TITLES = [
    "Course Description", "Learning Outcomes", "Objectives", "Prerequisites",
    "Textbook", "Required Materials", "Grading", "Schedule", "Policies",
    "Attendance", "Assignments", "Instructor", "Contact", "Office Hours"
]

# Define patterns for CS, Math, and Data Science courses
CS_PATTERN = r'^(CS|COMP|CMPS|CSCI|CPE|CIS|IT|CSE|CPS|INF|COMS|CSIT)\d'  # Computer Science
MATH_PATTERN = r'^(MATH|MTH|MAT|MA)\d'  # Mathematics
DATA_PATTERN = r'^(DATA|DS|DSC|STAT|STA|DSCI)\d'  # Data Science

# ------------ SYLLABUS PROCESSING FUNCTIONS ------------

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting text from DOCX {docx_path}: {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file."""
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as txt_file:
            return txt_file.read()
    except Exception as e:
        print(f"Error extracting text from TXT {txt_path}: {e}")
        return ""

def clean_and_tokenize(text):
    """Clean and tokenize text using spaCy."""
    doc = nlp(text)
    return " ".join([token.text for token in doc])

def find_closest_section(line, section_titles, cutoff=0.75):
    """Return the closest matching section title or None."""
    import difflib
    matches = difflib.get_close_matches(line.strip().lower(), [s.lower() for s in section_titles], n=1, cutoff=cutoff)
    if matches:
        # Return canonical capitalization
        idx = [s.lower() for s in section_titles].index(matches[0])
        return section_titles[idx]
    return None

def section_syllabus_text(text):
    """Parse syllabus text into sections using fuzzy header matching."""
    sections = {}
    current_section = None
    buffer = []
    lines = text.splitlines()
    for line in lines:
        # Try to match a section header
        match = find_closest_section(line, SECTION_TITLES)
        if match:
            if current_section and buffer:
                sections[current_section] = "\n".join(buffer).strip()
                buffer = []
            current_section = match
        elif current_section:
            buffer.append(line)
    if current_section and buffer:
        sections[current_section] = "\n".join(buffer).strip()
    return sections

def parse_syllabus(file_path):
    """Extract and section text from a syllabus file."""
    # Extract and clean text
    if str(file_path).lower().endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif str(file_path).lower().endswith(".docx"):
        text = extract_text_from_docx(file_path)
    elif str(file_path).lower().endswith(".txt"):
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    # Section the syllabus
    sections = section_syllabus_text(text)
    return sections

# ------------ SIMILARITY COMPUTATION FUNCTIONS ------------

def embed_section(section_text, embeddings_cache=None, cache_key=None):
    """Embed section text using the sentence transformer, with caching."""
    # Check if we have a cached embedding
    if embeddings_cache and cache_key and cache_key in embeddings_cache:
        return embeddings_cache[cache_key]
    
    # Compute the embedding
    embedding = model.encode(section_text, show_progress_bar=False)
    
    # Cache the result if possible
    if embeddings_cache is not None and cache_key is not None:
        embeddings_cache[cache_key] = embedding
    
    return embedding

def compute_syllabus_similarity(s1, s2, embeddings_cache=None, s1_code=None):
    """Compute similarity between two syllabi with consistent processing.
    s1 should be the Rutgers syllabus, s2 should be the sending institution syllabus.
    """
    # First try section-by-section comparison
    try:
        # Validate syllabi are not empty
        if not s1 or not s2:
            print(f"Warning: Empty syllabus detected - s1_code: {s1_code}")
            return 0.0
            
        # Try to compute section-based similarity
        section_similarities = {}
        weighted_sum = 0.0
        total_weight = 0.0
        valid_sections = 0
        
        for section in ALL_SECTIONS:
            t1 = s1.get(section, "")
            t2 = s2.get(section, "")
            
            # Both sections must be valid strings with content
            if t1 and t2 and isinstance(t1, str) and isinstance(t2, str) and t1.strip() and t2.strip():
                # Create a cache key for the Rutgers syllabus section if possible
                cache_key = f"{s1_code}_{section}" if s1_code else None
                
                emb1 = embed_section(t1, embeddings_cache, cache_key)
                emb2 = embed_section(t2)
                sim = cosine_similarity([emb1], [emb2])[0][0]
                section_similarities[section] = sim
                valid_sections += 1
                
                if section in SECTION_WEIGHTS:
                    weighted_sum += sim * SECTION_WEIGHTS[section]
                    total_weight += SECTION_WEIGHTS[section]
        
        # If we have valid sections with weights, return the weighted score
        if total_weight > 0 and valid_sections > 0:
            section_score = weighted_sum / total_weight
            return section_score
            
        # If we get here, we need to fall back to full document comparison
        print(f"Section comparison failed for {s1_code}, falling back to full document...")
    
    except Exception as e:
        print(f"Error in section similarity calculation: {e}")
        print(f"Falling back to full document comparison for {s1_code}...")
    
    # FALLBACK: Compare full documents
    try:
        # Concatenate all text from each syllabus
        s1_text = " ".join([str(text) for text in s1.values() if text and isinstance(text, str)])
        s2_text = " ".join([str(text) for text in s2.values() if text and isinstance(text, str)])
        
        # If either full text is empty, return 0 similarity
        if not s1_text.strip() or not s2_text.strip():
            return 0.0
            
        # Get or compute the embedding for the full Rutgers syllabus
        full_cache_key = f"{s1_code}_full_document" if s1_code else None
        emb1 = embed_section(s1_text, embeddings_cache, full_cache_key)
        
        # Compute the embedding for the sending institution syllabus
        emb2 = embed_section(s2_text)
        
        # Compute and return cosine similarity
        full_sim = cosine_similarity([emb1], [emb2])[0][0]
        
        s1_preview = s1_text[:20] + "..." if len(s1_text) > 20 else s1_text
        s2_preview = s2_text[:20] + "..." if len(s2_text) > 20 else s2_text
        print(f"Computing full document similarity between {s1_preview} and {s2_preview}...")
        
        return full_sim
        
    except Exception as e:
        print(f"Error in full document comparison: {e}")
        return 0.0

def compute_document_similarity(s1_text, s2_text):
    """Legacy function for document similarity calculation."""
    # Only print the course names, not the full text
    s1_preview = s1_text[:20] + "..." if isinstance(s1_text, str) and len(s1_text) > 20 else s1_text
    s2_preview = s2_text[:20] + "..." if isinstance(s2_text, str) and len(s2_text) > 20 else s2_text
    print(f"Computing overall document similarity between {s1_preview} and {s2_preview}...")
    try:
        # Ensure inputs are strings and not empty
        s1_text = str(s1_text) if s1_text is not None else ""
        s2_text = str(s2_text) if s2_text is not None else ""
        
        # If either text is empty, return 0 similarity
        if not s1_text.strip() or not s2_text.strip():
            return 0.0
        
        # Encode texts
        emb1 = model.encode(s1_text, show_progress_bar=False)
        emb2 = model.encode(s2_text, show_progress_bar=False)
        
        # Compute cosine similarity
        return cosine_similarity([emb1], [emb2])[0][0]
    except Exception as e:
        print(f"Error computing document similarity: {e}")
        # Reduced verbosity in error output
        print(f"Text 1 type: {type(s1_text)}, Text 2 type: {type(s2_text)}")
        return 0.0  # Return 0 similarity on error

# ------------ COURSE EQUIVALENCY FUNCTIONS ------------

def extract_data_from_excel(file_path):
    """Extract data from a course equivalency Excel file."""
    try:
        df = pd.read_excel(file_path)
        
        # Look for required columns (case-insensitive search)
        required_cols = ['SI', 'Course ID', 'Course Title', 'RI', "EQ 'FN'"]
        col_mapping = {}
        
        for req_col in required_cols:
            for col in df.columns:
                if req_col.lower() in col.lower():
                    col_mapping[req_col] = col
                    break
        
        # Check if all required columns were found
        missing = set(required_cols) - set(col_mapping.keys())
        if missing:
            print(f"Warning: Couldn't find these columns in {file_path.name}: {missing}")
            print(f"Available columns: {df.columns.tolist()}")
        
        # Select only the required columns that were found
        if col_mapping:
            df_selected = df[[col_mapping[col] for col in col_mapping]]
            # Rename columns to standardized names
            df_selected.columns = list(col_mapping.keys())
            return df_selected
        else:
            print(f"No required columns found in {file_path.name}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        return pd.DataFrame()

def filter_cs_math_ds_courses(df):
    """Filter dataframe to keep only CS, Math, and Data Science courses."""
    # Get the Course ID column name
    course_id_col = [col for col in df.columns if 'Course ID' in col][0]
    
    # Create combined filter
    cs_filter = df[course_id_col].str.match(CS_PATTERN, case=False)
    math_filter = df[course_id_col].str.match(MATH_PATTERN, case=False)
    data_filter = df[course_id_col].str.match(DATA_PATTERN, case=False)
    
    combined_filter = cs_filter | math_filter | data_filter
    filtered_df = df[combined_filter].copy()
    
    return filtered_df

def identify_non_transferring_courses(filtered_dataframes):
    """Identify courses that don't transfer between institutions."""
    non_transferring_courses = []
    
    for file_name, df in filtered_dataframes.items():
        # Get the EQ column name
        eq_col = [col for col in df.columns if "EQ 'FN'" in col][0]
        
        # Filter for courses with NaN in the EQ column
        non_transferring = df[df[eq_col].isna()].copy()
        non_transferring_courses.append(non_transferring)
    
    # Combine all the non-transferring courses into a single dataframe
    if non_transferring_courses:
        combined_non_transferring = pd.concat(non_transferring_courses, ignore_index=True)
        return combined_non_transferring
    else:
        return pd.DataFrame()

def identify_transferring_courses(filtered_dataframes):
    """Identify courses that transfer between institutions."""
    transferring_courses = []
    
    for file_name, df in filtered_dataframes.items():
        # Get the EQ column name
        eq_col = [col for col in df.columns if "EQ 'FN'" in col][0]
        
        # Filter for courses with non-NaN in the EQ column
        transferring = df[~df[eq_col].isna()].copy()
        transferring_courses.append(transferring)
    
    # Combine all the transferring courses into a single dataframe
    if transferring_courses:
        combined_transferring = pd.concat(transferring_courses, ignore_index=True)
        return combined_transferring
    else:
        return pd.DataFrame()

def analyze_course_equivalencies():
    """Load and filter course equivalency data for CS, Math, and Data Science courses."""
    print("Loading course equivalency data...")
    
    # Step 1: Load course equivalency data from Rutgers Excel files
    all_dataframes = load_all_excels()
    if not all_dataframes:
        print("No data to analyze. Exiting.")
        return None, None
    
    # Step 2: Filter dataframes to keep only CS, Math, and Data Science courses
    filtered_dataframes = {}
    for file_name, df in all_dataframes.items():
        filtered_df = filter_cs_math_ds_courses(df)
        filtered_dataframes[file_name] = filtered_df
        
        # Print summary of filtering
        print(f"{file_name}:")
        print(f"  Original: {len(df)} courses")
        print(f"  Filtered: {len(filtered_df)} courses")
        
        # Count courses by discipline
        course_id_col = [col for col in filtered_df.columns if 'Course ID' in col][0]
        cs_count = sum(filtered_df[course_id_col].str.match(CS_PATTERN, case=False))
        math_count = sum(filtered_df[course_id_col].str.match(MATH_PATTERN, case=False))
        data_count = sum(filtered_df[course_id_col].str.match(DATA_PATTERN, case=False))
        
        print(f"  CS courses: {cs_count}")
        print(f"  Math courses: {math_count}")
        print(f"  Data Science courses: {data_count}")
    
    # Step 3: Identify non-transferring and transferring courses (simple identification only)
    non_transferring_df = identify_non_transferring_courses(filtered_dataframes)
    transferring_df = identify_transferring_courses(filtered_dataframes)
    
    print(f"\nIdentified {len(non_transferring_df)} non-transferring courses and {len(transferring_df)} transferring courses")
    
    return non_transferring_df, transferring_df

def load_all_excels():
    """Load all Excel files from the excels directory."""
    all_dataframes = {}
    
    # Iterate through all Excel files in the directory
    excel_files = list(EXCEL_DIR.glob("*.xlsx")) + list(EXCEL_DIR.glob("*.xls"))
    
    # Filter to keep only Rutgers files
    rutgers_excel_files = [file for file in excel_files if "Rutgers" in file.name]
    
    if not rutgers_excel_files:
        print(f"No Rutgers Excel files found in {EXCEL_DIR}")
        return {}
    
    print(f"Found {len(rutgers_excel_files)} Rutgers Excel files to process.")
    
    # Process each Rutgers Excel file
    for file_path in rutgers_excel_files:
        file_name = file_path.name
        print(f"Processing {file_name}...")
        
        # Extract data from the file
        df = extract_data_from_excel(file_path)
        
        # Store DataFrame in dictionary with filename as key
        if not df.empty:
            all_dataframes[file_name] = df
            print(f"  Extracted {len(df)} rows with {len(df.columns)} columns")
        else:
            print(f"  No data extracted from {file_name}")
    
    return all_dataframes

def load_rutgers_syllabi():
    """Load all Rutgers syllabi from the directory and cache embeddings."""
    syllabi = {}
    syllabi_files = list(RUTGERS_SYLLABI_DIR.glob("*.pdf")) + list(RUTGERS_SYLLABI_DIR.glob("*.docx")) + list(RUTGERS_SYLLABI_DIR.glob("*.txt"))
    
    if not syllabi_files:
        print(f"No syllabus files found in {RUTGERS_SYLLABI_DIR}")
        return {}
    
    print(f"Found {len(syllabi_files)} Rutgers syllabus files to process.")
    
    # Print all syllabus filenames for debugging
    if len(syllabi_files) < 5:  # Only print if there are few files
        print("Rutgers syllabus files:")
        for file_path in syllabi_files:
            print(f"  - {file_path.name}")
    else:
        print(f"Found {len(syllabi_files)} Rutgers syllabus files")
    
    # Check if we have a cache of embeddings - loaded from rutgers_embeddings.pkl
    embeddings_cache = {}
    try:
        if RUTGERS_EMBEDDINGS_CACHE.exists():
            print(f"Loading cached Rutgers embeddings from {RUTGERS_EMBEDDINGS_CACHE}")
            with open(RUTGERS_EMBEDDINGS_CACHE, 'rb') as f:
                embeddings_cache = pickle.load(f)
            print(f"Loaded {len(embeddings_cache)} cached embeddings")
    except Exception as e:
        print(f"Error loading embeddings cache: {e}")
        embeddings_cache = {}
    
    for file_path in tqdm(syllabi_files, desc="Loading Rutgers syllabi"):
        try:
            # Use file stem as the course code - we'll include all of them
            course_code = file_path.stem.replace(" ", "_").upper()
            
            # Check if we have a cached version
            if course_code in embeddings_cache and 'parsed_syllabus' in embeddings_cache[course_code]:
                # Reduced verbosity
                print(f"Using cached parse for: {course_code}")
                syllabi[course_code] = embeddings_cache[course_code]['parsed_syllabus']
            else:
                # Reduced verbosity
                print(f"Loading syllabus: {course_code}")
                syllabi[course_code] = parse_syllabus(file_path)
                
                # Cache the parsed syllabus
                if course_code not in embeddings_cache:
                    embeddings_cache[course_code] = {}
                embeddings_cache[course_code]['parsed_syllabus'] = syllabi[course_code]
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    # Save the updated embeddings cache to rutgers_embeddings.pkl
    try:
        print(f"Saving Rutgers embeddings cache to {RUTGERS_EMBEDDINGS_CACHE}")
        with open(RUTGERS_EMBEDDINGS_CACHE, 'wb') as f:
            pickle.dump(embeddings_cache, f)
        print(f"Saved embeddings cache with {len(embeddings_cache)} items")
    except Exception as e:
        print(f"Error saving embeddings cache: {e}")
    
    print(f"Successfully loaded {len(syllabi)} Rutgers syllabi.")
    return syllabi, embeddings_cache

def download_syllabi_from_excel(excel_files, output_dir):
    """Download syllabi from URLs in Excel/CSV files."""
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    downloaded_syllabi = {}
    download_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Get list of already downloaded files
    existing_files = []
    for ext in ['.pdf', '.docx', '.txt']:
        existing_files.extend(list(output_dir.glob(f"*{ext}")))
    existing_file_stems = [f.stem for f in existing_files]
    
    for excel_file in excel_files:
        try:
            print(f"Reading URLs from {excel_file.name}...")
            # Read Excel or CSV file
            if excel_file.suffix.lower() == '.csv':
                df = pd.read_csv(excel_file)
            else:
                df = pd.read_excel(excel_file)
            
            # Find the columns for institution ID, course ID, course title, and syllabus link
            inst_col = None
            course_col = None
            title_col = None
            link_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'inst' in col_lower or 'institution' in col_lower:
                    inst_col = col
                elif 'course id' in col_lower or 'course_id' in col_lower:
                    course_col = col
                elif 'title' in col_lower:
                    title_col = col
                elif 'link' in col_lower and ('syllabus' in col_lower or 'syll' in col_lower):
                    link_col = col
            
            if not (inst_col and course_col and link_col):
                print(f"  Could not find required columns in {excel_file.name}")
                print(f"  Available columns: {df.columns.tolist()}")
                continue
            
            print(f"  Found columns: Institution={inst_col}, Course ID={course_col}, Link={link_col}")
            
            # Filter for data science, computer science, mathematics, and statistics courses
            filtered_rows = []
            ds_math_cs_count = 0
            
            for idx, row in df.iterrows():
                course_id = str(row[course_col]).strip()
                
                # Check if course ID matches data science, CS, or math patterns
                is_ds = bool(re.match(DATA_PATTERN, course_id, re.IGNORECASE))
                is_cs = bool(re.match(CS_PATTERN, course_id, re.IGNORECASE))
                is_math = bool(re.match(MATH_PATTERN, course_id, re.IGNORECASE))
                
                # If title column exists, also check for keywords in title
                is_relevant_title = False
                if title_col and not pd.isna(row[title_col]):
                    title = str(row[title_col]).lower()
                    keywords = ['data', 'statistic', 'analytic', 'math', 'computer science', 
                               'programming', 'algorithm', 'calculus', 'linear algebra', 
                               'probability', 'machine learning', 'deep learning']
                    is_relevant_title = any(keyword in title for keyword in keywords)
                
                if is_ds or is_cs or is_math or is_relevant_title:
                    filtered_rows.append(idx)
                    ds_math_cs_count += 1
            
            print(f"  Found {ds_math_cs_count} data science, CS, or math courses out of {len(df)} total courses")
            
            # Create filtered dataframe
            filtered_df = df.loc[filtered_rows].copy()
            
            # Process each row in the filtered dataframe
            for idx, row in tqdm(list(filtered_df.iterrows()), desc=f"Processing {excel_file.name}"):
                try:
                    inst_id = str(row[inst_col]).strip()
                    course_id = str(row[course_col]).strip().replace(" ", "")
                    url = str(row[link_col]).strip()
                    
                    if pd.isna(url) or not url or url.lower() == 'nan':
                        continue
                    
                    # Skip if not a valid URL
                    if not url.startswith('http'):
                        continue
                    
                    # Create filename base based on institution and course ID
                    filename_base = f"{inst_id}_{course_id}_syllabus"
                    
                    # Check if we already have this syllabus
                    if filename_base in existing_file_stems:
                        print(f"  Syllabus for {inst_id}_{course_id} already exists, skipping download")
                        
                        # Find the existing file with this stem
                        existing_file = None
                        for ext in ['.pdf', '.docx', '.txt']:
                            potential_file = output_dir / f"{filename_base}{ext}"
                            if potential_file.exists():
                                existing_file = potential_file
                                break
                        
                        if existing_file:
                            try:
                                syllabus_data = parse_syllabus(existing_file)
                                downloaded_syllabi[f"{inst_id}_{course_id}"] = syllabus_data
                                skipped_count += 1
                                print(f"  Successfully parsed existing syllabus: {existing_file.name}")
                            except Exception as e:
                                print(f"  Error parsing existing syllabus {existing_file}: {e}")
                        
                        continue
                    
                    # Try to download the file
                    try:
                        print(f"  Downloading {url}")
                        response = requests.get(url, timeout=30)
                        response.raise_for_status()
                        
                        # Determine file extension
                        content_type = response.headers.get('content-type', '')
                        if 'application/pdf' in content_type:
                            ext = '.pdf'
                        elif 'application/msword' in content_type or 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type:
                            ext = '.docx'
                        elif 'text/plain' in content_type:
                            ext = '.txt'
                        else:
                            # Try to get extension from URL
                            parsed_url = urlparse(url)
                            path = parsed_url.path
                            ext = os.path.splitext(path)[1]
                            if not ext or ext.lower() not in ['.pdf', '.docx', '.txt']:
                                # Check content for PDF signature
                                if response.content.startswith(b'%PDF'):
                                    ext = '.pdf'
                                else:
                                    # Default to txt for plain text
                                    ext = '.txt'
                        
                        # Full path for the file
                        file_path = output_dir / f"{filename_base}{ext}"
                        
                        # Save the file
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        
                        # Parse the downloaded syllabus
                        try:
                            syllabus_data = parse_syllabus(file_path)
                            downloaded_syllabi[f"{inst_id}_{course_id}"] = syllabus_data
                            download_count += 1
                            existing_file_stems.append(filename_base)  # Add to our list of existing files
                            print(f"  Saved and parsed {file_path.name} (Total: {download_count})")
                        except Exception as e:
                            print(f"  Error parsing downloaded syllabus {file_path}: {e}")
                            failed_count += 1
                    
                    except Exception as e:
                        print(f"  Error downloading {url}: {e}")
                        failed_count += 1
                
                except Exception as e:
                    print(f"  Error processing row {idx}: {e}")
                    failed_count += 1
        
        except Exception as e:
            print(f"Error processing file {excel_file.name}: {e}")
    
    print(f"Downloaded and parsed {download_count} syllabi")
    print(f"Skipped {skipped_count} existing syllabi")
    print(f"Failed to process {failed_count} syllabi")
    return downloaded_syllabi

def analyze_syllabus_similarities(non_transferring_df, transferring_df):
    """Analyze similarities between Rutgers syllabi and other institution syllabi."""
    print("\nStarting syllabus similarity analysis...")
    
    # Step 1: Load Rutgers syllabi - all of them are considered data science related
    # since they were manually selected, along with their embeddings cache
    rutgers_syllabi, embeddings_cache = load_rutgers_syllabi()
    if not rutgers_syllabi:
        print("No Rutgers syllabi found. Exiting similarity analysis.")
        return pd.DataFrame()
    
    # Step 2: Prepare for similarity analysis
    similarity_results = []
    
    # Ensure the Course ID and other required columns exist
    required_cols = ['SI', 'Course ID', 'Course Title', 'Transfers']
    missing_cols = set(required_cols) - set(non_transferring_df.columns)
    if missing_cols:
        print(f"Warning: Missing columns in non_transferring_df: {missing_cols}")
        print(f"Available columns: {non_transferring_df.columns.tolist()}")
    
    missing_cols = set(required_cols) - set(transferring_df.columns)
    if missing_cols:
        print(f"Warning: Missing columns in transferring_df: {missing_cols}")
        print(f"Available columns: {transferring_df.columns.tolist()}")
    
    # Show the first few rows to debug
    print("\nSample of non-transferring courses:")
    if not non_transferring_df.empty:
        print(non_transferring_df.head(3).to_string())
    
    print("\nSample of transferring courses:")
    if not transferring_df.empty:
        print(transferring_df.head(3).to_string())
    
    # Get column names that contain "Course ID" regardless of case
    course_id_col = None
    for col in non_transferring_df.columns:
        if 'course id' in col.lower():
            course_id_col = col
            break
    
    if not course_id_col:
        print("Error: Could not find Course ID column. Trying to use default 'Course ID'")
        course_id_col = 'Course ID'
    
    print(f"Using '{course_id_col}' as the course ID column")
    
    # Rename columns to standard names to avoid case sensitivity issues
    non_transferring_df = non_transferring_df.rename(columns={course_id_col: 'Course_ID'})
    transferring_df = transferring_df.rename(columns={course_id_col: 'Course_ID'})
    
    # Combine transferring and non-transferring courses
    all_courses = pd.concat([
        non_transferring_df.assign(Transfers=0),
        transferring_df.assign(Transfers=1)
    ])
    
    # All Rutgers syllabi are considered data science related since they were hand-selected
    rutgers_ds_courses = rutgers_syllabi.copy()
    print(f"Using all {len(rutgers_ds_courses)} Rutgers syllabi for analysis.")
    
    # Filter sending courses to include data science, CS, and math courses
    sending_ds_courses = all_courses[
        all_courses['Course_ID'].str.match(DATA_PATTERN, case=False) | 
        all_courses['Course_ID'].str.match(r'^(CS|COMP|CMPS|CSCI)\d', case=False) |  # CS courses
        all_courses['Course_ID'].str.match(r'^(MATH|MTH|MAT)\d', case=False) |  # Math courses
        all_courses['Course Title'].str.contains('data|statistics|analytics|algorithm|probability|calculus|linear algebra', 
                                               case=False, regex=True)
    ].copy()
    
    print(f"Found {len(sending_ds_courses)} sending institution data science courses.")
    
    # Step 3: Download and load sending institution syllabi
    print("\nChecking for syllabi from sending institutions...")
    
    # List Excel/CSV files in the syllabi_excels directory
    syllabi_excel_files = list(SYLLABI_EXCELS_DIR.glob("*.xlsx")) + list(SYLLABI_EXCELS_DIR.glob("*.csv"))
    print(f"Found {len(syllabi_excel_files)} Excel/CSV files with syllabus links.")
    
    # Create a temporary directory for downloaded syllabi
    DOWNLOADED_SYLLABI_DIR.mkdir(exist_ok=True)
    
    # Download syllabi without a limit
    sending_syllabi = download_syllabi_from_excel(syllabi_excel_files, DOWNLOADED_SYLLABI_DIR)
    
    if sending_syllabi:
        print(f"Successfully downloaded and parsed {len(sending_syllabi)} sending institution syllabi.")
        
        # Step 4: Compare syllabi and compute similarity scores
        print("Comparing syllabi...")
        
        # Process each Rutgers course
        for rutgers_code, rutgers_syllabus in tqdm(rutgers_ds_courses.items(), desc="Processing Rutgers courses"):
            # Compare with each sending institution course
            for _, sending_course in sending_ds_courses.iterrows():
                try:
                    si = str(sending_course['SI']).strip()
                    course_id = str(sending_course['Course_ID']).strip() if 'Course_ID' in sending_course else "UNKNOWN_ID"
                    course_title = str(sending_course['Course Title']).strip() if 'Course Title' in sending_course else "UNKNOWN_TITLE"
                    transfers = int(sending_course['Transfers']) if 'Transfers' in sending_course else 0
                    
                    # Look for the sending syllabus
                    sending_key = f"{si}_{course_id}"
                    sending_syllabus = sending_syllabi.get(sending_key)
                    
                    # Compute similarity
                    if sending_syllabus:
                        # We have actual syllabus data, use proper section comparison
                        try:
                            similarity_score = compute_syllabus_similarity(
                                rutgers_syllabus, sending_syllabus, 
                                embeddings_cache, rutgers_code
                            )
                        except Exception as e:
                            print(f"Error computing similarity for {rutgers_code} and {sending_key}: {e}")
                            # Fallback to course title if syllabus comparison fails
                            pseudo_syllabus = {"Course Description": course_title}
                            similarity_score = compute_syllabus_similarity(
                                rutgers_syllabus, pseudo_syllabus,
                                embeddings_cache, rutgers_code
                            )
                    else:
                        # No syllabus available, use title as a pseudo-syllabus
                        pseudo_syllabus = {"Course Description": course_title}
                        similarity_score = compute_syllabus_similarity(
                            rutgers_syllabus, pseudo_syllabus,
                            embeddings_cache, rutgers_code
                        )
                    
                    # Record the result
                    similarity_results.append({
                        'Rutgers_Course': rutgers_code,
                        'SI': si,
                        'Course_ID': course_id,
                        'Course_Title': course_title,
                        'Similarity_Score': similarity_score,
                        'Transfers': transfers,
                        'Has_Syllabus': 1 if sending_syllabus else 0
                    })
                except Exception as e:
                    print(f"Error processing course comparison for {rutgers_code}: {e}")
                    # Continue with next course
                    continue
    else:
        print("No sending institution syllabi downloaded. Using course titles for comparison.")
        
        # Generate results based on course titles
        for rutgers_code, rutgers_syllabus in tqdm(rutgers_ds_courses.items(), desc="Processing Rutgers courses"):
            # Compare with each sending course
            for _, sending_course in sending_ds_courses.iterrows():
                try:
                    si = str(sending_course['SI']).strip()
                    course_id = str(sending_course['Course_ID']).strip() if 'Course_ID' in sending_course else "UNKNOWN_ID"
                    course_title = str(sending_course['Course Title']).strip() if 'Course Title' in sending_course else "UNKNOWN_TITLE"
                    transfers = int(sending_course['Transfers']) if 'Transfers' in sending_course else 0
                    
                    # Create a pseudo-syllabus with just the course title in the course description section
                    pseudo_sending_syllabus = {"Course Description": course_title}
                    
                    # Compute similarity using our consistent method, but with the pseudo-syllabus
                    similarity_score = compute_syllabus_similarity(
                        rutgers_syllabus, pseudo_sending_syllabus, 
                        embeddings_cache, rutgers_code
                    )
                    
                    # Record the result
                    similarity_results.append({
                        'Rutgers_Course': rutgers_code,
                        'SI': si,
                        'Course_ID': course_id,
                        'Course_Title': course_title,
                        'Similarity_Score': similarity_score,
                        'Transfers': transfers,
                        'Has_Syllabus': 0
                    })
                except Exception as e:
                    print(f"Error processing course title comparison for {rutgers_code}: {e}")
                    # Continue with next course
                    continue
    
    # Convert results to DataFrame
    if similarity_results:
        results_df = pd.DataFrame(similarity_results)
        print(f"Generated {len(results_df)} similarity comparisons.")
        
        # Save the full results
        results_df.to_csv(BASE_PATH / "similarity_results.csv", index=False)
        
        # Analyze the relationship between similarity scores and transfer status
        transfer_scores = results_df[results_df['Transfers'] == 1]['Similarity_Score']
        non_transfer_scores = results_df[results_df['Transfers'] == 0]['Similarity_Score']
        
        print("\nSimilarity Score Analysis:")
        print(f"Average similarity for transferring courses: {transfer_scores.mean():.4f}")
        print(f"Average similarity for non-transferring courses: {non_transfer_scores.mean():.4f}")
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Transfers', y='Similarity_Score', data=results_df)
        plt.title('Similarity Scores by Transfer Status')
        plt.xlabel('Transfers (0=No, 1=Yes)')
        plt.ylabel('Similarity Score')
        plt.savefig(BASE_PATH / "similarity_boxplot.png")
        print("Saved similarity visualization to similarity_boxplot.png")
        
        return results_df
    else:
        print("No similarity results generated.")
        return pd.DataFrame()

def create_transfer_matrix(results_df):
    """Create a matrix showing the transfer status and similarity scores."""
    if results_df.empty:
        print("No results to create matrix from.")
        return
    
    # Create pivot tables
    transfer_pivot = results_df.pivot_table(
        index=['SI', 'Course_ID', 'Course_Title'],
        columns='Rutgers_Course',
        values='Transfers',
        aggfunc='max',
        fill_value=0
    )
    
    similarity_pivot = results_df.pivot_table(
        index=['SI', 'Course_ID', 'Course_Title'],
        columns='Rutgers_Course',
        values='Similarity_Score',
        aggfunc='mean',
        fill_value=0
    )
    
    # Save the matrices
    transfer_pivot.to_csv(BASE_PATH / "transfer_status_matrix.csv")
    similarity_pivot.to_csv(BASE_PATH / "similarity_score_matrix.csv")
    
    print("Created transfer status and similarity score matrices.")
    
    # Create a heatmap visualization
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarity_pivot.astype(float), cmap="YlGnBu", annot=False)
    plt.title('Syllabus Similarity Scores Between Sending Institutions and Rutgers')
    plt.xlabel('Rutgers Course')
    plt.ylabel('Sending Institution Course')
    plt.tight_layout()
    plt.savefig(BASE_PATH / "similarity_heatmap.png")
    print("Saved similarity heatmap to similarity_heatmap.png")
    
    return transfer_pivot, similarity_pivot

if __name__ == "__main__":
    print("Starting course and syllabus similarity analysis...")
    
    # Step 1: Load course equivalency data for CS, Math, and Data Science courses
    non_transferring_df, transferring_df = analyze_course_equivalencies()
    
    # Handle case where no courses were found
    if non_transferring_df is None or transferring_df is None:
        print("No course data found. Please check your Excel files and directory paths.")
        exit(1)
    
    # Step 2: Analyze syllabus similarities (main focus)
    results_df = analyze_syllabus_similarities(non_transferring_df, transferring_df)
    
    # Step 3: Create transfer matrix and visualizations
    if results_df is not None and not results_df.empty:
        create_transfer_matrix(results_df)
        print("\nAnalysis complete! Similarity results have been saved to CSV files in your project directory.")
    else:
        print("\nNo similarity results were generated. Please check your syllabus files and directory paths.")

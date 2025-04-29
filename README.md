# Course Equivalency Syllabus Analysis

This project analyzes course equivalencies by comparing syllabi from various sending institutions against a set of Rutgers data science syllabi. The goal is to build a binary transfer matrix (1 = transfers, 0 = does not) and explore how syllabus similarity relates to transferability.

*DISCLAIMER: course_syllabus_analysis.py isn't currently able to embed each section (can't find similar sections across schema). Will update if I can debug, or another, more robust, schema consistency implementation might be needed (such as using an LLM to parse sections, return content for each section in a JSON output and use Pydantic to parse output.)*

*Another next step is to use a crew.ai or LangChain agentic workflow to dynamically create similarity scores, course equivalency assessments for any PDF/DOC/TXT syllabus pair you give it, outside of training data, to be able to be used with NJ Transfer.*

## Directory Structure

```
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── AT_JN083SC6.csv                 # Example CSV of syllabus URLs for one sending institution
├── download_syllabi.py             # Script to download syllabi from CSV URLs into `syllabi/`
├── course_syllabus_analysis.py     # Main pipeline: loads data, reads/downloads syllabi, computes similarity, builds matrices & visuals
├── syllabus_similarity.py          # Helper functions: text cleaning, tokenization, vectorization, similarity scoring
├── test.py                         # Quick, small-scale tests of core functions
├── course_equiv analysis/
│   └── course_equiv.ipynb          # Notebook exploring Excel course-equivalency data
├── syllabi_matching/
│   ├── download_syllabi.ipynb      # Notebook for testing download logic
│   ├── syllabi_excels/             # Source Excel files per SI
│   │   ├── *.csv                   # Columns: SI, Course ID, Eff. Term, …
│   │   └── downloaded_syllabi/     # 934 syllabus files (PDF/TXT/DOCX)
│   └── rutgers_datascience_syllabi/ # 17 pre-downloaded Rutgers data science syllabi
└── syllabus similarity/
    └── syllabus_similarity_demo.ipynb # Notebook for testing similarity functions
```

## File Descriptions

- **download_syllabi.py** Reads a CSV of syllabus URLs (`ex: AT_JN083SC6.csv`), creates a `syllabi/` folder, and downloads each URL, saving with the correct extension.
- **course_syllabus_analysis.py**

  1. Loads all SI Excel files under `syllabi_matching/syllabi_excels/`.
  2. Filters for data science, computer science, and mathematics courses.
  3. Ensures 934 syllabi in `.../downloaded_syllabi/`.
  4. Loads 17 Rutgers syllabi from `rutgers_datascience_syllabi/`.
  5. Cleans and tokenizes text via `syllabus_similarity.py`.
  6. Vectorizes with TF-IDF (term frequency and in)and computes cosine similarity for every pair (934×17).
  7. Exports:
     - `similarity_results.csv`
     - `transfer_status_matrix.csv`
     - `similarity_score_matrix.csv`
     - `similarity_boxplot.png`
     - `similarity_heatmap.png`
- **syllabus_similarity.py** Provides PDF/TXT/DOCX readers, text cleaning (stopwords, stemming), tokenization, TF-IDF vectorization, and cosine similarity.
- **test.py** Contains unit tests for core functions without full pipeline execution.
- **Notebooks (`*.ipynb`)**
  All `.ipynb` files are for exploratory testing and validation; not optimized for large-scale runs.

## Usage

1. Create & activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download syllabi (production):
   ```bash
   python download_syllabi.py
   ```
4. Run main analysis pipeline:
   ```bash
   python course_syllabus_analysis.py
   ```
5. Review outputs; use notebooks for deeper exploration.

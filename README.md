# NLP Text Analysis Project

## Project Purpose

This project performs comparative natural language processing (NLP) analysis on a collection of text files. The primary goals are:

1.  **Text Preprocessing:** Clean and prepare raw text data for analysis.
2.  **Token Analysis:** Identify the most frequent meaningful terms (lemmas) in each text.
3.  **Named Entity Recognition (NER):** Extract named entities (like persons, locations, organizations).
4.  **Subject Determination:** Infer the main subject or theme of each text.
5.  **N-gram Analysis:** Capture common word sequences.
6.  **Authorship Attribution:** Compare an "unknown" text against others to find the most likely author based on writing style and word patterns.

## How it Works (Simplified)

1.  **Setup:** The script first checks if necessary NLTK data (like word lists, grammar models) is downloaded. If not, it downloads them.
2.  **Read Files:** It finds all `.txt` files in the specified directory.
3.  **Process Each File:** For every text file:
    *   It reads the content.
    *   It cleans the text (lowercase, removes punctuation/numbers).
    *   It breaks the text into individual words (tokens).
    *   It removes common "stopwords" (like 'the', 'a', 'is').
    *   It reduces words to their base or dictionary form (lemmatization).
4.  **Analyze Each File:**
    *   It counts the frequency of the base words (lemmas) to find the most common ones.
    *   It identifies named entities (people, places, etc.).
    *   It finds common sequences of words (n-grams, typically 3 words long).
5.  **Compare Authorship:**
    *   It takes a designated "unknown" file (e.g., `RJ_Lovecraft.txt`).
    *   It compares the n-grams and calculated stylistic features (like average word length, vocabulary richness) of the unknown file to all other files.
    *   It calculates similarity scores (Jaccard for n-grams, Cosine for style) and determines which known text is the closest match.
6.  **Determine Subject:**
    *   It looks for keywords related to known themes (like 'Romeo and Juliet', or author styles like 'Lovecraft') within the common words and entities.
    *   It prints a guess about the subject based on these findings.
7.  **Output:** All results (common words, entities, n-grams, authorship comparison, subject guesses) are printed to the console.

## How to Use

1.  **Prerequisites:**
    *   Make sure you have Python 3 installed.
    *   Install the NLTK library: Open your terminal or command prompt and run: `pip install nltk`
2.  **Prepare Files:**
    *   Save the script code as `nlp_analysis_main.py`.
    *   Place the text files (`.txt`) you want to analyze in the same directory as the script, or in a separate directory.
3.  **Run the Script:**
    *   Open your terminal or command prompt.
    *   Navigate to the directory where you saved `nlp_analysis_main.py`.
    *   Execute the script using Python:
        *   **Basic Usage (analyzes files in the current directory):**
            ```bash
            python nlp_analysis_main.py
            ```
        *   **Specify Data Directory:** If your text files are in a different folder (e.g., `my_texts`), use the `--data_dir` argument:
            ```bash
            python nlp_analysis_main.py --data_dir my_texts
            ```
        *   **Specify Unknown File:** To change the file used for authorship comparison (default is `RJ_Lovecraft.txt`), use `--unknown_file`:
            ```bash
            python nlp_analysis_main.py --unknown_file Text_4.txt
            ```
4.  **View Output:** The script will print its progress and the analysis results directly to your terminal. The first time you run it, it might take a moment to download the necessary NLTK data.

## Class Design and Implementation

*(The detailed explanation of classes and functions remains here as before)*
...

## Running the Script (Detailed)

*(The detailed running instructions remain here as before)*
...

## Limitations

*(The limitations section remains here as before)*
...

## Sample Output

*(The sample output section remains here as before)*
...

===== PART 1: TEXT PROCESSING AND BASIC ANALYSIS =====

Processing Martin.txt...

Results for Martin.txt:
Top 20 most common lemmatized tokens (excluding stopwords):
  could: 3
  feel: 3
  eye: 3
  keep: 2
  ser: 2
  aldric: 2
  forgotten: 1
  corner: 1
  realm: 1
  whisper: 1
  ancient: 1
  lore: 1
  met: 1
  chilling: 1
  wind: 1
  swept: 1
  desolate: 1
  landscape: 1
  existed: 1
  tale: 1

Number of unique named entities found: 1
Sample named entities: ['Aldric']

--------------------------------------------------
Processing RJ_Lovecraft.txt...

Results for RJ_Lovecraft.txt:
Top 20 most common lemmatized tokens (excluding stopwords):
  verona: 5
  amidst: 5
  juliet: 5
  romeo: 5
  ancient: 4
  fate: 3
  love: 3
  whisper: 3
  shadowed: 3
  eldritch: 3
  forbidden: 3
  tale: 2
  city: 2
  stone: 2
  pact: 2
  secret: 2
  realm: 2
  tragedy: 2
  labyrinthine: 2
  alley: 2

Number of unique named entities found: 1
Sample named entities: ['Verona']

--------------------------------------------------
Processing RJ_Martin.txt...

Results for RJ_Martin.txt:
Top 20 most common lemmatized tokens (excluding stopwords):
  house: 6
  verona: 5
  juliet: 5
  romeo: 5
  capulet: 3
  montague: 3
  love: 3
  amidst: 3
  city: 2
  wall: 2
  street: 2
  saga: 2
  intrigue: 2
  blood: 2
  feud: 2
  star: 2
  crossed: 2
  passion: 2
  whisper: 2
  shadow: 2

Number of unique named entities found: 3
Sample named entities: ['Capulet', 'Montague', 'Verona']

--------------------------------------------------
Processing RJ_Tolkein.txt...

Results for RJ_Tolkein.txt:
Top 20 most common lemmatized tokens (excluding stopwords):
  juliet: 7
  verona: 5
  amidst: 5
  romeo: 5
  land: 4
  love: 4
  woven: 3
  fate: 3
  age: 3
  upon: 3
  discord: 3
  tragic: 3
  eternal: 3
  despair: 3
  among: 2
  ancient: 2
  tale: 2
  thread: 2
  tapestry: 2
  fair: 2

Number of unique named entities found: 0
  No named entities found.

--------------------------------------------------

===== PART 2: N-GRAM ANALYSIS AND AUTHORSHIP ATTRIBUTION =====


Top 20 3-grams for Martin.txt:
  could feel eyes: 3
  keep ser aldric: 2
  in forgotten corners: 1
  forgotten corners realm: 1
  corners realm whispers: 1
  realm whispers ancient: 1
  whispers ancient lore: 1
  ancient lore met: 1
  lore met chilling: 1
  met chilling winds: 1
  chilling winds swept: 1
  winds swept desolate: 1
  swept desolate landscapes: 1
  desolate landscapes existed: 1
  landscapes existed tale: 1
  existed tale shrouded: 1
  tale shrouded shadows: 1
  shrouded shadows bound: 1
  shadows bound fate: 1
  bound fate capricious: 1

Top 20 3-grams for RJ_Lovecraft.txt:
  in hallowed city: 1
  hallowed city verona: 1
  city verona ancient: 1
  verona ancient stones: 1
  ancient stones echoed: 1
  stones echoed whispers: 1
  echoed whispers forbidden: 1
  whispers forbidden pacts: 1
  forbidden pacts eldritch: 1
  pacts eldritch secrets: 1
  eldritch secrets unfolded: 1
  secrets unfolded tale: 1
  unfolded tale steeped: 1
  tale steeped shadowed: 1
  steeped shadowed realms: 1
  shadowed realms fate: 1
  realms fate tragedy: 1
  fate tragedy amidst: 1
  tragedy amidst labyrinthine: 1
  amidst labyrinthine alleys: 1

Top 20 3-grams for RJ_Martin.txt:
  in sprawling city: 1
  sprawling city verona: 1
  city verona towering: 1
  verona towering walls: 1
  towering walls house: 1
  walls house capulet: 1
  house capulet house: 1
  capulet house montague: 1
  house montague loomed: 1
  montague loomed bustling: 1
  loomed bustling streets: 1
  bustling streets unfolded: 1
  streets unfolded saga: 1
  unfolded saga dripping: 1
  saga dripping intrigue: 1
  dripping intrigue blood: 1
  intrigue blood feuds: 1
  blood feuds star: 1
  feuds star crossed: 1
  star crossed passion: 1

Top 20 3-grams for RJ_Tolkein.txt:
  in verdant lands: 1
  verdant lands verona: 1
  lands verona nestled: 1
  verona nestled among: 1
  nestled among rolling: 1
  among rolling hills: 1
  rolling hills ancient: 1
  hills ancient woods: 1
  ancient woods dwelled: 1
  woods dwelled two: 1
  dwelled two souls: 1
  two souls bound: 1
  souls bound tale: 1
  bound tale woven: 1
  tale woven delicate: 1
  woven delicate threads: 1
  delicate threads fate: 1
  threads fate love: 1
  fate love amidst: 1
  love amidst grand: 1

Performing authorship analysis for RJ_Lovecraft.txt...

Authorship attribution results (compared to other files):
RJ_Tolkein.txt:
  N-gram Similarity (Jaccard): 0.0041
  Stylistic Similarity (Cosine): 0.9999
  Combined Score (Average): 0.5020
RJ_Martin.txt:
  N-gram Similarity (Jaccard): 0.0022
  Stylistic Similarity (Cosine): 0.9997
  Combined Score (Average): 0.5010
Martin.txt:
  N-gram Similarity (Jaccard): 0.0000
  Stylistic Similarity (Cosine): 0.9996
  Combined Score (Average): 0.4998

---> Based on combined score, the style of 'RJ_Lovecraft.txt' is most similar to 'RJ_Tolkein.txt'.

===== PART 3: SUBJECT DETERMINATION =====

Attempting to determine the subject based on common tokens and named entities:

Subject analysis for Martin.txt:
  -> Subject might be related to a 'Game of Thrones'-like fantasy setting.

Subject analysis for RJ_Lovecraft.txt:
  -> Subject appears to be related to 'Romeo and Juliet'.
     Style convention suggests: H.P. Lovecraft

Subject analysis for RJ_Martin.txt:
  -> Subject appears to be related to 'Romeo and Juliet'.
     Style convention suggests: George R.R. Martin

Subject analysis for RJ_Tolkein.txt:
  -> Subject appears to be related to 'Romeo and Juliet'.
     Style convention not recognized from filename.

Analysis complete!

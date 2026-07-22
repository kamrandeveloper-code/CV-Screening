
# AI Recruitment System - Intelligent Candidate-Job Matching Platform

An advanced AI-powered recruitment system that automatically extracts information from resumes, matches candidates with job postings using hybrid scoring, and provides intelligent shortlisting recommendations.

## 🚀 Features

- **Smart Resume Parsing**: Automatic extraction of candidate information (name, email, phone, skills, education, experience)
- **Advanced Document Analysis**: LayoutLMv3-powered PDF layout analysis for better text extraction
- **Intelligent Skill Extraction**: Semantic skill matching using embeddings and comprehensive skill taxonomy
- **Hybrid Matching Algorithm**: 70% embedding similarity + 30% skill overlap for accurate candidate-job matching
- **Batch Processing**: Match multiple candidates against job postings in background
- **Duplicate Detection**: Smart deduplication to prevent duplicate candidate entries
- **RESTful API**: Complete API endpoints for candidates, jobs, and matching

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                         │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Candidate   │  │     Job      │  │    Match     │      │
│  │   Endpoints  │  │  Endpoints   │  │  Endpoints   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│  ┌──────▼─────────────────▼─────────────────▼───────┐      │
│  │              Service Layer                        │      │
│  │  CandidateService | JobService | MatchService   │      │
│  └──────┬─────────────────┬─────────────────┬───────┘      │
│         │                 │                 │               │
│  ┌──────▼─────────────────▼─────────────────▼───────┐      │
│  │              Model Layer                          │      │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────┐ │      │
│  │  │  Embedding   │  │     NER      │  │ Layout  │ │      │
│  │  │    Model     │  │    Model     │  │  LMv3   │ │      │
│  │  └──────────────┘  └──────────────┘  └─────────┘ │      │
│  └───────────────────────────────────────────────────┘      │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐         │
│  │              SQLite Database                   │         │
│  │    Candidates | Jobs | Matches                 │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 🤖 AI Models Used

### 1. **Embedding Model: BAAI/bge-small-en-v1.5**
- **Purpose**: Converts text into numerical vectors (embeddings) for similarity comparison
- **Dimension**: 384-dimensional vectors
- **Use Case**: 
  - Convert resume text to embeddings
  - Convert job descriptions to embeddings
  - Calculate semantic similarity between candidates and jobs
- **Why BGE?**: Optimized for retrieval tasks, excellent semantic understanding, fast inference

### 2. **NER Model: xlm-roberta-base-finetuned-conll03-english**
- **Purpose**: Named Entity Recognition for extracting structured information from resumes
- **Entities Extracted**:
  - **PER** (Person): Candidate names
  - **ORG** (Organization): Previous companies, universities
  - **LOC/GPE** (Location): Geographic locations
- **Use Case**:
  - Extract candidate names from resume headers
  - Identify company names for work experience
  - Extract educational institutions
- **Why XLM-RoBERTa?**: Multilingual support, excellent NER performance, robust to formatting variations

### 3. **LayoutLMv3: microsoft/layoutlmv3-base**
- **Purpose**: Document layout analysis for better text extraction from complex PDFs
- **Features**:
  - Detects document structure (headers, sections, paragraphs)
  - Identifies section boundaries (Education, Experience, Skills)
  - Preserves layout information for better context
- **Use Case**:
  - Intelligent section segmentation
  - Better text extraction from multi-column resumes
  - Improved handling of tables and lists
- **Why LayoutLMv3?**: State-of-the-art for document understanding, combines text and layout

### 4. **Skill Taxonomy & Semantic Matching**
- **Comprehensive Skill Database**: Over 200+ predefined technical and soft skills
- **Semantic Matching**: Uses embeddings to match similar skills (e.g., "Python" matches "python3", "py")
- **Categories**:
  - Programming Languages (Python, Java, JavaScript, etc.)
  - Frameworks (Django, React, Spring Boot, etc.)
  - Databases (PostgreSQL, MongoDB, Redis, etc.)
  - Cloud & DevOps (AWS, Docker, Kubernetes, etc.)
  - Machine Learning (TensorFlow, PyTorch, scikit-learn, etc.)
  - Soft Skills (Leadership, Communication, Problem Solving, etc.)

## 🔄 System Workflow

### Workflow 1: Candidate Resume Upload

```
┌─────────────┐
│   Upload    │
│   Resume    │
│   (PDF/     │
│    DOCX)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 1: File Processing                     │
│  ├─ Validate file type & size               │
│  ├─ Extract raw text                        │
│  └─ For PDFs: LayoutLMv3 layout analysis    │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 2: Information Extraction             │
│  ├─ NER Model: Extract name, organizations  │
│  ├─ Regex patterns: Email, phone            │
│  ├─ Semantic extractor: Skills              │
│  └─ Section analysis: Education, Experience │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 3: Embedding Generation               │
│  └─ BGE Model: Create resume embedding      │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 4: Deduplication Check                │
│  ├─ Check by email                          │
│  ├─ Check by phone                          │
│  └─ Check by name similarity                │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 5: Database Storage                   │
│  └─ Store candidate with all extracted data │
└─────────────────────────────────────────────┘
```

### Workflow 2: Job Posting Creation

```
┌─────────────┐
│ Create Job  │
│  Posting    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 1: Job Data Processing                │
│  ├─ Job title, description, company         │
│  └─ Required skills (manual or auto)        │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 2: Skill Extraction                   │
│  ├─ Auto-extract from description           │
│  └─ Semantic matching against taxonomy      │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 3: Embedding Generation               │
│  └─ BGE Model: Create job embedding         │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Step 4: Database Storage                   │
│  └─ Store job with embedding and skills     │
└─────────────────────────────────────────────┘
```

### Workflow 3: Candidate-Job Matching

```
┌─────────────┐     ┌─────────────┐
│  Candidate  │     │    Job      │
│  Embedding  │     │  Embedding  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       └─────────┬─────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│  Component 1: Embedding Similarity (70%)    │
│  ├─ Cosine similarity between embeddings    │
│  └─ Score range: 0.0 - 1.0                 │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Component 2: Skill Matching (30%)          │
│  ├─ Semantic skill comparison               │
│  ├─ Matched skills found                    │
│  ├─ Missing skills identified               │
│  └─ Skill overlap score calculated          │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Hybrid Score Calculation                   │
│  Final = (0.7 × Embedding) +                │
│          (0.3 × Skill Match)                │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Decision Logic                             │
│  ├─ Score ≥ 75% → Shortlisted              │
│  ├─ Score ≥ 60% → Review                   │
│  └─ Score < 60% → Rejected                 │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│  Store Match Result                         │
│  ├─ Final score                             │
│  ├─ Matched skills                          │
│  ├─ Missing skills                          │
│  └─ Interview ready flag                    │
└─────────────────────────────────────────────┘
```

## 📊 Matching Algorithm Details

### Hybrid Scoring Formula

```
Final Score = (Embedding_Weight × Embedding_Score) + (Skill_Weight × Skill_Score)

Where:
- Embedding_Weight = 0.7 (70% importance)
- Skill_Weight = 0.3 (30% importance)
- Embedding_Score = Cosine similarity between resume and job embeddings
- Skill_Score = Matched skills / Total required skills
```

### Decision Criteria

| Score Range | Status | Action |
|------------|--------|--------|
| 75% - 100% | Shortlisted | Ready for interview |
| 60% - 74% | Review | Human review recommended |
| 0% - 59% | Rejected | Not suitable |

## 📁 Project Structure

```
fyp-ai-recruitment-system/
│
├── app/
│   ├── api/
│   │   └── endpoints/
│   │       ├── candidate.py      # Candidate CRUD operations
│   │       ├── job.py            # Job CRUD operations
│   │       └── match.py          # Matching endpoints
│   │
│   ├── core/
│   │   └── config.py             # Configuration settings
│   │
│   ├── db/
│   │   ├── database.py           # Database connection
│   │   └── models.py             # SQLAlchemy models
│   │
│   ├── models/
│   │   ├── embedding_model.py    # BGE embedding model
│   │   ├── resume_parser.py      # NER model for extraction
│   │   ├── document_layout.py    # LayoutLMv3 analyzer
│   │   ├── skill_extractor.py    # Semantic skill extraction
│   │   ├── matcher.py            # Hybrid matching logic
│   │   └── model_manager.py      # Model management
│   │
│   ├── services/
│   │   ├── candidate_service.py  # Candidate business logic
│   │   ├── job_service.py        # Job business logic
│   │   └── match_service.py      # Matching business logic
│   │
│   ├── schemas/
│   │   ├── candidate_schema.py   # Candidate data models
│   │   ├── job_schema.py         # Job data models
│   │   └── match_schema.py       # Match data models
│   │
│   ├── utils/
│   │   ├── file_handler.py       # File processing
│   │   ├── text_cleaner.py       # Text preprocessing
│   │   ├── id_generator.py       # Unique ID generation
│   │   └── deduplicator.py       # Duplicate detection
│   │
│   └── main.py                   # FastAPI application
│
├── models/                       # Downloaded model cache
├── uploads/                      # Uploaded resume storage
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── run.py                        # Application runner
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended (for model loading)
- 5GB free disk space (for models)

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd fyp-ai-recruitment-system
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Create Directories

```bash
mkdir models uploads logs
```

### Step 5: Run Application

```bash
python run.py
```

The server will start at `http://127.0.0.1:8000`

## 📚 API Documentation

Once running, access interactive API docs at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### API Endpoints

#### Candidates

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/candidate/upload-resume` | Upload resume (PDF/DOCX) |
| GET | `/candidate/` | Get all candidates |
| GET | `/candidate/{candidate_id}` | Get candidate by ID |
| DELETE | `/candidate/{candidate_id}` | Delete candidate |

#### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/job/create` | Create job posting |
| GET | `/job/` | Get all jobs |
| GET | `/job/{job_id}` | Get job by ID |
| DELETE | `/job/{job_id}` | Delete job |

#### Matching

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/match/` | Create single match |
| POST | `/match/job/{job_id}/batch` | Batch match all candidates |
| GET | `/match/job/{job_id}` | Get matches for job |
| GET | `/match/job/{job_id}/top` | Get top N candidates |
| GET | `/match/job/{job_id}/ready` | Get interview-ready candidates |
| GET | `/match/candidate/{candidate_id}` | Get matches for candidate |
| POST | `/match/job/{job_id}/recalculate` | Recalculate all matches |

## 💡 Usage Examples

### 1. Upload Resume

```bash
curl -X POST "http://127.0.0.1:8000/candidate/upload-resume" \
  -F "file=@/path/to/resume.pdf"
```

Response:
```json
{
  "success": true,
  "candidate_id": "cand_20240502_abc123",
  "is_new": true,
  "message": "New candidate created",
  "candidate": {
    "candidate_id": "cand_20240502_abc123",
    "name": "John Doe",
    "email": "john.doe@example.com",
    "phone": "+1234567890",
    "skills": ["Python", "Machine Learning", "SQL"],
    "created_at": "2024-05-02T10:30:00"
  }
}
```

### 2. Create Job

```bash
curl -X POST "http://127.0.0.1:8000/job/create" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior Python Developer",
    "description": "Looking for Python expert with ML experience...",
    "company": "Tech Corp",
    "skills_required": ["Python", "Machine Learning", "PostgreSQL"]
  }'
```

### 3. Match Candidate to Job

```bash
curl -X POST "http://127.0.0.1:8000/match/" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "cand_20240502_abc123",
    "job_id": "job_20240502_xyz789"
  }'
```

Response:
```json
{
  "match_id": "match_20240502_def456",
  "candidate_id": "cand_20240502_abc123",
  "job_id": "job_20240502_xyz789",
  "similarity_score": 0.85,
  "matched_skills": ["Python", "Machine Learning"],
  "missing_skills": ["PostgreSQL"],
  "ready_for_interview": true,
  "created_at": "2024-05-02T10:35:00"
}
```

### 4. Get Top Candidates for Job

```bash
curl "http://127.0.0.1:8000/match/job/job_20240502_xyz789/top?top_n=5"
```

## 🔧 Configuration

Edit `app/core/config.py` to customize:

```python
# API Settings
api_host = "127.0.0.1"
api_port = 8000

# Database
database_url = "sqlite:///recruitment.db"

# Model Settings
embedding_model = "BAAI/bge-small-en-v1.5"
ner_model = "xlm-roberta-base-finetuned-conll03-english"

# Matching Weights
embedding_weight = 0.7  # 70% embedding similarity
skill_weight = 0.3       # 30% skill overlap

# Thresholds
match_shortlist_threshold = 0.75  # 75% for shortlist
match_review_threshold = 0.60      # 60% for review
```

## 📈 Performance Optimization

### Model Caching
- Models are downloaded once and cached locally
- Singleton pattern prevents redundant loading
- Models persist across API requests

### Batch Processing
- Use background tasks for batch matching
- Prevents blocking on large operations
- Ideal for matching hundreds of candidates

### Database Indexing
- Indexes on candidate_id, job_id, match scores
- Optimized queries for fast retrieval
- SQLite with connection pooling

## 🐛 Troubleshooting

### Common Issues

**Issue**: `cannot import name 'cached_download'`
**Solution**: Upgrade huggingface_hub
```bash
pip install --upgrade huggingface_hub==0.20.3
```

**Issue**: Database missing columns
**Solution**: Delete database and restart
```bash
del recruitment.db  # Windows
rm recruitment.db   # Linux/Mac
python run.py
```

**Issue**: Out of memory loading models
**Solution**: Use smaller models in config
```python
embedding_model = "all-MiniLM-L6-v2"  # Smaller, 384-dim
```

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) for transformer models
- [Sentence Transformers](https://www.sbert.net/) for embedding models
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [LayoutLMv3](https://arxiv.org/abs/2204.08387) team for document AI
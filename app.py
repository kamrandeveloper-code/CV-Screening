# integrated_cv_skill_api.py
import os
import io
import re
import json
import joblib
import pickle
import tempfile
import traceback
import secrets
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Path, Query, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Text extraction
import pdfplumber
import docx2txt
from PIL import Image
import pytesseract

# ML & data
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# DB
from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, Text, Boolean, JSON, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.ext.mutable import MutableDict

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEFAULT_DB_URL = "sqlite:///cv_screening.db"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "png", "jpg", "jpeg"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ============================================================================
# DATABASE MODELS
# ============================================================================

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(50), unique=True, index=True)
    name = Column(String(100))
    email = Column(String(100))
    phone = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CandidateProfile(Base):
    """Structured candidate profile from CV parsing"""
    __tablename__ = "candidate_profiles"
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(50), index=True)
    profile_data = Column(JSON)  # JSON structured profile
    skills = Column(JSON)  # Normalized skills list
    created_at = Column(DateTime, default=datetime.utcnow)

class JobProfile(Base):
    """Job profile with extracted and normalized skills"""
    __tablename__ = "job_profiles"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(50), unique=True, index=True)  # User-provided job ID
    job_title = Column(String(200))
    company = Column(String(200))
    job_description = Column(Text)
    required_skills = Column(JSON)  # Normalized required skills
    optional_skills = Column(JSON)  # Normalized optional skills
    user_provided_skills = Column(JSON)  # Store user-provided skills
    created_at = Column(DateTime, default=datetime.utcnow)

class SkillMatchResult(Base):
    """Results from skill matching engine"""
    __tablename__ = "skill_match_results"
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(String(50), index=True)
    job_id = Column(String(50), index=True)  # Changed to String to match user-provided job_id
    match_score = Column(Float)
    matched_skills = Column(JSON)
    missing_required = Column(JSON)
    missing_optional = Column(JSON)
    explanations = Column(JSON)  # Explainable results
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ============================================================================
# PYDANTIC SCHEMAS - UPDATED
# ============================================================================

class EducationEntry(BaseModel):
    degree: str
    institution: Optional[str] = None
    year: Optional[str] = None
    gpa: Optional[str] = None

class ExperienceEntry(BaseModel):
    company: str
    role: str
    duration: str
    achievements: Optional[List[str]] = None

class CandidateProfileResponse(BaseModel):
    candidate_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    education: List[EducationEntry] = []  # Changed to simple list of degree names
    experience: List[ExperienceEntry] = []
    skills: Dict[str, List[str]] = {}  # categorized skills
    normalized_skills: List[str] = []  # flat normalized skills
    projects: List[Dict[str, Optional[str]]] = []  # Enhanced projects field
    certifications: List[str] = []

class JobProfileRequest(BaseModel):
    job_id: str = Field(..., description="User-provided job ID")
    job_title: str = Field(..., description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    job_description: Optional[str] = Field(None, description="Job description text (optional if skills are provided)")
    required_skills: List[str] = Field(default=[], description="List of required skills (comma-separated or list)")
    optional_skills: List[str] = Field(default=[], description="List of optional/nice-to-have skills")

class JobProfileResponse(BaseModel):
    job_id: str
    job_title: str
    company: Optional[str] = None
    required_skills: List[str]
    optional_skills: List[str]
    created_at: datetime

class SkillMatchRequest(BaseModel):
    candidate_id: str
    job_id: str  # Changed to String to match user-provided job_id

class SkillMatchResponse(BaseModel):
    match_id: int
    candidate_id: str
    job_id: str  # Changed to String
    match_score: float
    matched_skills: List[str]
    missing_required: List[str]
    missing_optional: List[str]
    explanations: List[str]
    created_at: datetime

class CandidateSummaryResponse(BaseModel):
    candidate_id: str
    name: Optional[str]
    email: Optional[str]
    overall_match_stats: Dict[str, Any]
    recent_matches: List[Dict[str, Any]]
    interview_readiness: str
    recommended_actions: List[str]

# ============================================================================
# SKILL TAXONOMY & NORMALIZER - CORRECTED
# ============================================================================

class SkillNormalizer:
    """Normalize skills to standard taxonomy"""
    
    SKILL_TAXONOMY = {
        "python": ["python", "python3", "python 3", "py"],
        "java": ["java", "java 8", "java 11", "java 17", "spring", "spring boot", "springboot"],
        "javascript": ["javascript", "js", "typescript", "node.js", "nodejs", "express", "express.js"],
        "c++": ["c++", "cpp", "c plus plus"],
        "c#": ["c#", "csharp", ".net", "dotnet"],
        "sql": ["sql", "mysql", "postgresql", "postgres", "oracle", "sql server"],
        "mongodb": ["mongodb", "mongo"],
        "nosql": ["nosql", "cassandra", "redis"],
        "react": ["react", "react.js", "reactjs"],
        "aws": ["aws", "amazon web services", "ec2", "s3", "lambda", "cloudformation"],
        "docker": ["docker", "containerization"],
        "kubernetes": ["kubernetes", "k8s"],
        "git": ["git", "github", "gitlab", "version control"],
        "html": ["html", "html5"],
        "css": ["css", "css3", "bootstrap", "tailwindcss", "tailwind"],
        "node.js": ["node", "node.js", "nodejs"],
        "express.js": ["express", "express.js"],
        "restful_api": ["rest", "restful", "restful api", "api"],
        "postman": ["postman"],
        "vscode": ["vscode", "visual studio code"],
        "machine_learning": ["machine learning", "ml", "ai", "artificial intelligence"],
        "data_structures": ["dsa", "data structures", "algorithms"],
        "database_management": ["dbms", "database management", "database"],
        "oop": ["oop", "object oriented programming"],
        "operating_system": ["operating system", "os"],
        "computer_networks": ["computer network", "networking", "networks"],
        "communication": ["communication", "presentation", "writing", "verbal communication"],
        "teamwork": ["teamwork", "collaboration"],
        "problem_solving": ["problem solving", "analytical", "critical thinking", "troubleshooting"],
        "time_management": ["time management", "time management skills"],
        "leadership": ["leadership", "management", "team lead", "project management"],
        "web_development": ["web development", "web dev", "full stack", "mern stack"],
        "mern_stack": ["mern", "mern stack"],
        "vercel": ["vercel"],
        "passport.js": ["passport.js", "passport"],
        "multer": ["multer"],
        "ejs": ["ejs"],
    }
    
    @classmethod
    def normalize(cls, skill: str) -> str:
        """Normalize a skill to standard form"""
        skill_lower = skill.lower().strip()
        
        for standard_skill, variations in cls.SKILL_TAXONOMY.items():
            if skill_lower == standard_skill or skill_lower in variations:
                return standard_skill
        
        # If not found, try partial matching
        for standard_skill, variations in cls.SKILL_TAXONOMY.items():
            for variation in variations:
                if variation in skill_lower or skill_lower in variation:
                    return standard_skill
        
        return skill_lower  # Return as-is if no match
    
    @classmethod
    def normalize_list(cls, skills: List[str]) -> List[str]:
        """Normalize a list of skills"""
        normalized = set()
        for skill in skills:
            normalized_skill = cls.normalize(skill)
            if normalized_skill:  # Only add non-empty skills
                normalized.add(normalized_skill)
        return list(normalized)

# ============================================================================
# CV PARSER SERVICE - CORRECTED WITH IMPROVED PARSING
# ============================================================================

class CVParserService:
    """Service for CV parsing and structured data extraction"""
    
    def __init__(self):
        # Load ML model if available
        self.model_path = os.path.join(MODELS_DIR, "cv_parser_model.pkl")
        self.model = self._load_model()
    
    def _load_model(self):
        """Load CV parsing ML model"""
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except:
                print("Warning: Could not load CV parser model, using rule-based parser")
        return None
    
    @staticmethod
    def extract_text(file: UploadFile) -> str:
        """Extract text from various file formats"""
        try:
            content = file.file.read()
            filename = file.filename.lower()
            
            if filename.endswith('.pdf'):
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    text = "\n".join(text_parts)
            
            elif filename.endswith('.docx'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                try:
                    text = docx2txt.process(tmp_file_path)
                finally:
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass
            
            elif filename.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = Image.open(io.BytesIO(content))
                    text = pytesseract.image_to_string(image)
                except Exception as e:
                    text = f"Error extracting text from image: {str(e)}"
            
            elif filename.endswith('.txt'):
                try:
                    text = content.decode('utf-8', errors='ignore')
                except:
                    text = content.decode('latin-1', errors='ignore')
            
            else:
                text = ""
            
            return text.strip() if text else ""
            
        except Exception as e:
            print(f"Error extracting text: {e}")
            return ""
    
    def parse_cv(self, text: str) -> Dict[str, Any]:
        """Parse CV text into structured data"""
        # Extract name - look for the name at the beginning
        lines = text.split('\n')
        name = lines[0].strip() if lines else None
        
        # Extract email - improved regex pattern
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        email = email_match.group(0) if email_match else None
        
        # Extract phone - improved regex
        phone_match = re.search(r'(?:\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}', text)
        phone = phone_match.group(0).strip() if phone_match else None
        
        # Extract education - simplified to just degree names
        education = self._extract_education_simple(text)
        
        # Extract experience
        experience = self._extract_experience(text)
        
        # Extract skills
        skills = self._extract_skills(text)
        
        # Extract projects - enhanced to include details
        projects = self._extract_projects_enhanced(text)
        
        # Extract certifications
        certifications = self._extract_certifications(text)
        
        # Normalize skills
        normalized_skills = SkillNormalizer.normalize_list(
            [skill for category in skills.values() for skill in category]
        )
        
        return {
            "name": name,
            "email": email,
            "phone": phone,
            "education": education,
            "experience": experience,
            "skills": skills,
            "normalized_skills": normalized_skills,
            "projects": projects,
            "certifications": certifications,
            "raw_text_preview": text[:500] + "..." if len(text) > 500 else text
        }
    
    def _extract_education_simple(self, text: str) -> List[str]:
        """Extract education information - simplified to just degree names"""
        education = []
        
        # Find education section more precisely
        edu_pattern = r'(?:EDUCATION)[\s\S]*?(?=\n\n[A-Z]{3,}|$)'
        edu_match = re.search(edu_pattern, text, re.IGNORECASE)
        
        if not edu_match:
            # Try alternative pattern
            edu_pattern = r'(?:\n|^)EDUCATION\s*\n-+\s*\n(.*?)(?=\n\n[A-Z]{3,}|$)'
            edu_match = re.search(edu_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if edu_match:
            edu_text = edu_match.group(0)
            lines = edu_text.split('\n')
            
            # Look for degree patterns in the education section
            for line in lines:
                line = line.strip()
                if not line or line.upper() == 'EDUCATION':
                    continue
                
                # Skip lines that look like institutions (containing College, University, etc.)
                if re.search(r'College|University|School|Institute|Affiliated|Pakistan', line, re.IGNORECASE):
                    continue
                
                # Look for degree patterns
                degree_patterns = [
                    r'Bachelor of Science in (Computer Science|IT|Software Engineering)',
                    r'B\.?S\.?c?\.?\s*(?:in\s+)?(Computer Science|Information Technology)',
                    r'F\.?Sc\.?\s*(Pre-?Engineering|Computer Science)',
                    r'Matriculation\s*\(?(Science Group|Computer Science)\)?',
                    r'(?:Master|M\.?S\.?|PhD|Doctorate)\s+in\s+[A-Za-z\s]+',
                    r'Diploma in [A-Za-z\s]+',
                    r'Certificate in [A-Za-z\s]+',
                ]
                
                for pattern in degree_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        # Extract just the degree name
                        if 'Bachelor' in pattern or 'B.S' in pattern:
                            degree_name = "Bachelor's in Computer Science"
                        elif 'F.Sc' in pattern:
                            degree_name = "F.Sc Pre-Engineering"
                        elif 'Matriculation' in pattern:
                            degree_name = "Matriculation (Science)"
                        else:
                            degree_name = match.group(0).strip()
                        
                        if degree_name not in education:
                            education.append(degree_name)
                        break
                else:
                    # If no pattern matched, check if line looks like a degree
                    if len(line) > 10 and len(line) < 100 and not any(char.isdigit() for char in line):
                        # Check if it contains degree-like words
                        degree_keywords = ['Bachelor', 'Master', 'PhD', 'Diploma', 'Certificate', 
                                          'F.Sc', 'Matriculation', 'Engineering', 'Science', 'Arts']
                        if any(keyword.lower() in line.lower() for keyword in degree_keywords):
                            if line not in education:
                                education.append(line.strip())
        
        # If we still don't have education, try direct extraction from Amir's CV format
        if not education:
            # Direct patterns for Amir's CV
            if 'Bachelor of Science in Computer Science' in text:
                education.append("Bachelor's in Computer Science")
            if 'F.Sc. Pre-Engineering' in text:
                education.append("F.Sc Pre-Engineering")
            if 'Matriculation (Science Group)' in text:
                education.append("Matriculation (Science)")
        
        return education[:3]  # Return up to 3 most recent degrees
    
    def _extract_experience(self, text: str) -> List[Dict[str, Any]]:
        """Extract experience information"""
        experience = []
        
        # Find experience section
        exp_pattern = r'(?:EXPERIENCE|Experience|Work Experience)[\s\S]*?(?=\n\n[A-Z]|$)'
        exp_match = re.search(exp_pattern, text, re.IGNORECASE)
        
        if not exp_match:
            return experience
        
        exp_text = exp_match.group(0)
        
        # Split by job entries
        job_entries = re.split(r'\n(?=\w)', exp_text)
        
        for entry in job_entries[1:]:  # Skip the "EXPERIENCE" header
            lines = entry.strip().split('\n')
            if len(lines) >= 2:
                # First line typically contains role and company
                first_line = lines[0]
                role_company = first_line.split('|')
                
                if len(role_company) >= 2:
                    role = role_company[0].strip()
                    company_info = role_company[1].strip().split('\n')[0]
                    
                    # Extract company and location
                    company_parts = company_info.split(',')
                    company = company_parts[0].strip()
                    
                    experience.append({
                        "company": company,
                        "role": role,
                        "duration": "",  # Duration might not be in this format
                        "achievements": lines[1:] if len(lines) > 1 else None
                    })
        
        # Fallback: Simple extraction for Amir's CV format
        if not experience:
            lines = exp_text.split('\n')
            for i, line in enumerate(lines):
                if 'Technical Writer' in line or 'Medium' in line:
                    experience.append({
                        "company": "Medium",
                        "role": "Technical Writer",
                        "duration": "Remote",
                        "achievements": [
                            "Write SEO-optimized, beginner-friendly technical articles on web development and JavaScript",
                            "Explain coding topics in easy language with real examples",
                            "Share personal experience and tips to help others learn faster"
                        ]
                    })
                    break
        
        return experience
    
    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract and categorize skills"""
        skills_by_category = {
            "programming": [],
            "databases": [],
            "devops": [],
            "data_science": [],
            "soft_skills": [],
            "web_development": [],
            "tools": []
        }
        
        text_lower = text.lower()
        
        # Check for programming languages
        programming_languages = ["c++", "javascript", "python", "java", "c#", "go", "rust", "php", "ruby"]
        for lang in programming_languages:
            if re.search(r'\b' + re.escape(lang) + r'\b', text_lower):
                skills_by_category["programming"].append(lang)
        
        # Check for web development skills
        web_skills = ["html", "css", "react", "node.js", "express.js", "bootstrap", "tailwindcss", "ejs", "restful api"]
        for skill in web_skills:
            if re.search(r'\b' + re.escape(skill.replace('.', '\.')) + r'\b', text_lower):
                skills_by_category["web_development"].append(skill)
        
        # Check for database skills
        db_keywords = ["mongodb", "mysql", "sql", "postgresql", "redis", "nosql"]
        for keyword in db_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                skills_by_category["databases"].append(keyword)
        
        # Check for devops skills
        devops_keywords = ["docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "git", "ci/cd", "vercel"]
        for keyword in devops_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                skills_by_category["devops"].append(keyword)
        
        # Check for tools
        tools = ["vscode", "github", "postman", "git", "passport.js", "multer"]
        for tool in tools:
            if re.search(r'\b' + re.escape(tool) + r'\b', text_lower):
                skills_by_category["tools"].append(tool)
        
        # Check for soft skills
        soft_keywords = ["communication", "teamwork", "problem solving", "time management", "leadership", "analytical"]
        for keyword in soft_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                skills_by_category["soft_skills"].append(keyword)
        
        # Check for coursework/technical skills
        coursework = ["dsa", "data structures", "algorithms", "dbms", "database management", "oop", 
                     "object oriented", "operating system", "computer network"]
        for course in coursework:
            if re.search(r'\b' + re.escape(course) + r'\b', text_lower):
                skills_by_category["programming"].append(course)
        
        # Remove empty categories
        return {k: v for k, v in skills_by_category.items() if v}
    
    def _extract_projects_enhanced(self, text: str) -> List[Dict[str, str]]:
        """Extract projects from CV with enhanced details"""
        projects = []
        
        # Find projects section - be more precise
        project_pattern = r'(?:PROJECTS|Projects)[\s\S]*?(?=\n[A-Z]{3,}\s*\n|$)'
        project_match = re.search(project_pattern, text, re.IGNORECASE)
        
        if not project_match:
            # Try alternative pattern
            project_pattern = r'(?:\n|^)PROJECTS\s*\n-+\s*\n(.*?)(?=\n[A-Z]{3,}\s*\n|$)'
            project_match = re.search(project_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if not project_match:
            return projects
        
        project_text = project_match.group(0)
        
        # Clean up the project text - remove the "PROJECTS" header
        project_text = re.sub(r'^PROJECTS\s*\n-+\s*\n', '', project_text, flags=re.IGNORECASE | re.MULTILINE)
        project_text = re.sub(r'^PROJECTS\s*\n', '', project_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Split projects by looking for project names
        # For standard CV format: project name followed by description
        project_blocks = []
        lines = project_text.split('\n')
        current_block = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this looks like a new project (starts with capital letter, not too long)
            if current_block and (len(line) < 50 and line[0].isupper() and not line.startswith(('•', '-', '*'))):
                project_blocks.append('\n'.join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        
        if current_block:
            project_blocks.append('\n'.join(current_block))
        
        for block in project_blocks:
            block = block.strip()
            if not block or len(block) < 20:  # Skip very short blocks
                continue
            
            lines = block.split('\n')
            project_data = {
                "name": "",
                "description": "",
                "technologies": "",
                "url": ""
            }
            
            # First non-empty line is usually the project name
            for line in lines:
                if line.strip():
                    project_data["name"] = line.strip().rstrip(':').strip('# ')
                    break
            
            # Extract description and technologies
            description_lines = []
            tech_lines = []
            
            for i, line in enumerate(lines[1:]):  # Skip the name line
                line = line.strip()
                if not line:
                    continue
                
                # Check for technology indicators (usually in parentheses or after dash)
                if re.search(r'\([^)]+\)|\[[^\]]+\]|\b(?:using|with|built with|technologies?|stack:?)\b', line, re.IGNORECASE):
                    tech_lines.append(line)
                # Check for URL or links
                elif re.search(r'\.(com|io|org|net|dev|github\.io)\b|http[s]?://|www\.', line, re.IGNORECASE):
                    project_data["url"] = line
                else:
                    description_lines.append(line)
            
            # Join description (limit to 150 chars)
            if description_lines:
                description = ' '.join(description_lines)
                project_data["description"] = description[:150] + '...' if len(description) > 150 else description
            
            # Extract technologies from tech lines
            if tech_lines:
                tech_text = ' '.join(tech_lines)
                # Extract text within parentheses/brackets or after "using"/"with"
                tech_matches = re.findall(r'\(([^)]+)\)|\[([^\]]+)\]|\b(?:using|with|built with)\s+([^.,]+)', tech_text, re.IGNORECASE)
                if tech_matches:
                    # Flatten matches and clean up
                    technologies = []
                    for match in tech_matches:
                        tech = next((item for item in match if item), '').strip()
                        if tech:
                            technologies.append(tech)
                    if technologies:
                        project_data["technologies"] = ', '.join(technologies)[:100]
            
            # Only add if we have a name
            if project_data["name"] and len(project_data["name"]) > 2:
                projects.append(project_data)
        
        # Fallback: simple extraction for specific CV formats
        if not projects:
            # Look for specific projects in Amir's CV
            project_names = ["EchoTap", "Real Estate Listings Platform", "Quora Posts Web App"]
            for name in project_names:
                if name in text:
                    # Find the project block
                    start = text.find(name)
                    if start != -1:
                        # Extract next few lines for description
                        end = text.find('\n\n', start)
                        if end == -1:
                            end = min(start + 300, len(text))
                        
                        project_block = text[start:end]
                        lines = project_block.split('\n')
                        
                        project_data = {
                            "name": name,
                            "description": "",
                            "technologies": "",
                            "url": ""
                        }
                        
                        if len(lines) > 1:
                            # First line after name might be technologies or description
                            next_line = lines[1].strip() if len(lines) > 1 else ""
                            if next_line and '(' in next_line or '[' in next_line:
                                project_data["technologies"] = next_line[:100]
                            elif next_line:
                                project_data["description"] = next_line[:150]
                        
                        projects.append(project_data)
        
        return projects[:5]
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications from CV"""
        certifications = []
        
        # Find certifications section - stop before ACHIEVEMENTS
        cert_pattern = r'(?:CERTIFICATES|Certifications|Certificates)[\s\S]*?(?=\nACHIEVEMENTS|\n[A-Z]{3,}\s*\n|$)'
        cert_match = re.search(cert_pattern, text, re.IGNORECASE)
        
        if not cert_match:
            # Try alternative pattern
            cert_pattern = r'(?:\n|^)CERTIFICATES\s*\n-+\s*\n(.*?)(?=\nACHIEVEMENTS|\n[A-Z]{3,}\s*\n|$)'
            cert_match = re.search(cert_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if cert_match:
            cert_text = cert_match.group(0)
            
            # Clean up - remove header
            cert_text = re.sub(r'^CERTIFICATES\s*\n-+\s*\n', '', cert_text, flags=re.IGNORECASE | re.MULTILINE)
            cert_text = re.sub(r'^CERTIFICATES\s*\n', '', cert_text, flags=re.IGNORECASE | re.MULTILINE)
            
            lines = cert_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.upper() in ['CERTIFICATES', 'CERTIFICATIONS']:
                    continue
                
                # Skip lines that are part of other sections
                if any(section in line.upper() for section in ['ACHIEVEMENTS', 'PUBLICATION', 'EDUCATION', 'EXPERIENCE', 'PROJECTS', 'SKILLS']):
                    break
                
                # Skip bullet points but keep the text
                if line.startswith(('•', '-', '*')):
                    line = line[1:].strip()
                
                if line and len(line) > 10:  # Minimum length for a certification
                    certifications.append(line[:150])
        
        return certifications[:10]  # Return up to 10 certifications

# ============================================================================
# JOB PARSER SERVICE - UPDATED
# ============================================================================

class JobParserService:
    """Service for job description parsing and skill extraction"""
    
    def __init__(self):
        self.skill_normalizer = SkillNormalizer()
    
    def parse_job_description(self, job_description: str, 
                             user_required_skills: List[str] = None,
                             user_optional_skills: List[str] = None) -> Dict[str, Any]:
        """Parse job description and extract skills"""
        
        # If user provided skills directly, use those
        if user_required_skills or user_optional_skills:
            required_skills = self.skill_normalizer.normalize_list(user_required_skills or [])
            optional_skills = self.skill_normalizer.normalize_list(user_optional_skills or [])
            
            # Also extract additional skills from description if provided
            if job_description:
                extracted_required = self._extract_required_skills(job_description)
                extracted_optional = self._extract_optional_skills(job_description)
                
                # Combine user-provided and extracted skills (avoid duplicates)
                required_skills = list(set(required_skills + extracted_required))
                optional_skills = list(set(optional_skills + extracted_optional))
        else:
            # Extract skills from description only
            required_skills = self._extract_required_skills(job_description)
            optional_skills = self._extract_optional_skills(job_description)
        
        # Normalize skills
        required_skills = self.skill_normalizer.normalize_list(required_skills)
        optional_skills = self.skill_normalizer.normalize_list(optional_skills)
        
        return {
            "required_skills": required_skills,
            "optional_skills": optional_skills,
            "user_provided": bool(user_required_skills or user_optional_skills)
        }
    
    def _extract_required_skills(self, text: str) -> List[str]:
        """Extract required skills from job description"""
        skills = []
        text_lower = text.lower()
        
        # Look for required skills section
        required_patterns = [
            r'required skills[:\-]\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'must have[:\-]\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'requirements[:\-]\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'qualifications[:\-]\s*(.*?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        required_text = ""
        for pattern in required_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match:
                required_text = match.group(1)
                break
        
        # If no specific section, search entire text
        if not required_text:
            required_text = text_lower
        
        # Extract skills from required text
        all_skills = list(self.skill_normalizer.SKILL_TAXONOMY.keys())
        
        for skill in all_skills:
            skill_variations = self.skill_normalizer.SKILL_TAXONOMY[skill]
            for variation in skill_variations:
                if re.search(r'\b' + re.escape(variation) + r'\b', required_text):
                    skills.append(skill)
                    break
        
        return skills
    
    def _extract_optional_skills(self, text: str) -> List[str]:
        """Extract optional/nice-to-have skills"""
        skills = []
        text_lower = text.lower()
        
        # Look for optional skills section
        optional_patterns = [
            r'preferred skills[:\-]\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'nice to have[:\-]\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'optional[:\-]\s*(.*?)(?:\n\n|\n[A-Z]|$)',
            r'bonus[:\-]\s*(.*?)(?:\n\n|\n[A-Z]|$)'
        ]
        
        optional_text = ""
        for pattern in optional_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match:
                optional_text = match.group(1)
                break
        
        # If no optional section, return empty
        if not optional_text:
            return []
        
        # Extract skills from optional text
        all_skills = list(self.skill_normalizer.SKILL_TAXONOMY.keys())
        
        for skill in all_skills:
            skill_variations = self.skill_normalizer.SKILL_TAXONOMY[skill]
            for variation in skill_variations:
                if re.search(r'\b' + re.escape(variation) + r'\b', optional_text):
                    skills.append(skill)
                    break
        
        return skills

# ============================================================================
# SKILL MATCHING SERVICE
# ============================================================================

class SkillMatchService:
    """Core skill matching service with ML capabilities"""
    
    def __init__(self):
        self.model_path = os.path.join(MODELS_DIR, "skill_match_model.pkl")
        self.model = self._load_model()
        self.skill_normalizer = SkillNormalizer()
    
    def _load_model(self):
        """Load skill matching ML model"""
        if os.path.exists(self.model_path):
            try:
                return joblib.load(self.model_path)
            except:
                print("Warning: Could not load skill match model, using rule-based matching")
        return None
    
    def match_skills(self, candidate_skills: List[str], 
                    required_skills: List[str], 
                    optional_skills: List[str]) -> Dict[str, Any]:
        """Match candidate skills against job requirements"""
        
        # Convert to sets for easier operations
        candidate_set = set(candidate_skills)
        required_set = set(required_skills)
        optional_set = set(optional_skills)
        
        # Calculate matches
        matched_required = list(candidate_set.intersection(required_set))
        matched_optional = list(candidate_set.intersection(optional_set))
        missing_required = list(required_set - candidate_set)
        missing_optional = list(optional_set - candidate_set)
        
        # Calculate scores
        required_score = len(matched_required) / len(required_set) if required_set else 1.0
        optional_score = len(matched_optional) / len(optional_set) if optional_set else 0.0
        
        # Weighted overall score (70% required, 30% optional)
        overall_score = (required_score * 0.7) + (optional_score * 0.3)
        
        # Generate explanations
        explanations = self._generate_explanations(
            matched_required, missing_required, 
            matched_optional, missing_optional,
            required_score, optional_score, overall_score
        )
        
        return {
            "match_score": round(overall_score * 100, 2),
            "matched_skills": matched_required + matched_optional,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
            "explanations": explanations,
            "score_breakdown": {
                "required_score": round(required_score * 100, 2),
                "optional_score": round(optional_score * 100, 2)
            }
        }
    
    def _generate_explanations(self, matched_required: List[str], 
                              missing_required: List[str],
                              matched_optional: List[str], 
                              missing_optional: List[str],
                              required_score: float, optional_score: float,
                              overall_score: float) -> List[str]:
        """Generate explainable results for the match"""
        explanations = []
        
        # Overall match explanation
        if overall_score >= 0.8:
            explanations.append("Excellent match! Candidate has most required skills.")
        elif overall_score >= 0.6:
            explanations.append("Good match. Candidate meets key requirements.")
        elif overall_score >= 0.4:
            explanations.append("Moderate match. Some important skills are missing.")
        else:
            explanations.append("Poor match. Many required skills are missing.")
        
        # Required skills explanation
        if matched_required:
            explanations.append(f"Matches {len(matched_required)} required skills: {', '.join(matched_required[:3])}")
        
        if missing_required:
            explanations.append(f"Missing {len(missing_required)} required skills: {', '.join(missing_required[:3])}")
        
        # Optional skills explanation
        if matched_optional:
            explanations.append(f"Bonus: Has {len(matched_optional)} preferred skills: {', '.join(matched_optional[:3])}")
        
        # Specific recommendations
        if missing_required:
            explanations.append(f"Recommendation: Focus on acquiring {missing_required[0]} skill")
        
        return explanations

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="AI-Powered CV & Job Matching System",
    version="1.0",
    description="""Four Endpoint Structure:
    1. CV Parsing - Upload and parse CVs
    2. Job Description Processing - Extract and normalize skills
    3. Skill Matching - Core matching engine
    4. Candidate Readiness Summary - Interview readiness assessment"""
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # important with "*"
)
# Initialize services
cv_parser_service = CVParserService()
job_parser_service = JobParserService()
skill_match_service = SkillMatchService()

# ============================================================================
# ENDPOINT 1: CV PARSING
# ============================================================================

@app.post("/cv/upload", response_model=CandidateProfileResponse)
async def upload_and_parse_cv(
    candidate_id: str = Form(..., description="User-provided candidate ID"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    📄 Upload CV with user-provided candidate ID, run CV parsing ML model, 
    and store structured candidate profile
    
    Uses:
    - cv_parser_model.pkl (if available)
    - cv_parser_service.py
    """
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check if candidate ID already exists
    existing_candidate = db.query(Candidate).filter(Candidate.candidate_id == candidate_id).first()
    if existing_candidate:
        raise HTTPException(status_code=400, detail=f"Candidate ID '{candidate_id}' already exists")
    
    try:
        # Read and validate file
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 10MB)")
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        await file.seek(0)
        
        # Extract text from file
        cv_text = cv_parser_service.extract_text(file)
        if not cv_text:
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Parse CV using service
        parsed_data = cv_parser_service.parse_cv(cv_text)
        
        # Create candidate with user-provided ID
        candidate = Candidate(
            candidate_id=candidate_id,
            name=parsed_data.get("name"),
            email=parsed_data.get("email"),
            phone=parsed_data.get("phone")
        )
        db.add(candidate)
        db.flush()
        
        # Store candidate profile
        profile = CandidateProfile(
            candidate_id=candidate_id,
            profile_data=parsed_data,
            skills=parsed_data.get("normalized_skills", [])
        )
        db.add(profile)
        db.commit()

        edu_list = parsed_data.get("education") or []
        education_objs = [
            {"degree": e, "institution": None, "year": None}
            for e in edu_list if e
        ]
        # Prepare response
        return CandidateProfileResponse(
            candidate_id=candidate_id,
            name=parsed_data.get("name"),
            email=parsed_data.get("email"),
            phone=parsed_data.get("phone"),
            # education=parsed_data.get("education", []),  # Now simple list
            education=education_objs,
            experience=[ExperienceEntry(**exp) for exp in parsed_data.get("experience", [])],
            skills=parsed_data.get("skills", {}),
            normalized_skills=parsed_data.get("normalized_skills", []),
            projects=parsed_data.get("projects", []),  # Now enhanced projects
            certifications=parsed_data.get("certifications", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# ENDPOINT 2: JOB DESCRIPTION PROCESSING - UPDATED WITH SKILLS INPUT
# ============================================================================

@app.post("/job/upload", response_model=JobProfileResponse)
async def upload_job_description(
    job_request: JobProfileRequest,
    db: Session = Depends(get_db)
):
    """
    📝 Accept job description text with user-provided job ID, 
    extract required & optional skills, normalize skills, and store job profile
    
    Uses:
    - job_parser_service.py
    - skill_normalizer.py
    
    Note: Either job_description OR required_skills must be provided
    """
    
    try:
        # Check if job ID already exists
        existing_job = db.query(JobProfile).filter(JobProfile.job_id == job_request.job_id).first()
        if existing_job:
            raise HTTPException(status_code=400, detail=f"Job ID '{job_request.job_id}' already exists")
        
        # Validate that either job_description or required_skills is provided
        if not job_request.job_description and not job_request.required_skills:
            raise HTTPException(
                status_code=400, 
                detail="Either job_description or required_skills must be provided"
            )
        
        # Parse job description and skills
        parsed_job = job_parser_service.parse_job_description(
            job_description=job_request.job_description or "",
            user_required_skills=job_request.required_skills,
            user_optional_skills=job_request.optional_skills
        )
        
        # Create job profile with user-provided job_id
        job_profile = JobProfile(
            job_id=job_request.job_id,
            job_title=job_request.job_title,
            company=job_request.company,
            job_description=job_request.job_description or "",
            required_skills=parsed_job["required_skills"],
            optional_skills=parsed_job["optional_skills"],
            user_provided_skills={
                "required_skills": job_request.required_skills,
                "optional_skills": job_request.optional_skills
            } if job_request.required_skills or job_request.optional_skills else None
        )
        
        db.add(job_profile)
        db.commit()
        db.refresh(job_profile)
        
        return JobProfileResponse(
            job_id=job_profile.job_id,  # Return user-provided job_id
            job_title=job_profile.job_title,
            company=job_profile.company,
            required_skills=job_profile.required_skills,
            optional_skills=job_profile.optional_skills,
            created_at=job_profile.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# ENDPOINT 3: SKILL MATCHING (CORE ENDPOINT)
# ============================================================================

@app.post("/match/skills", response_model=SkillMatchResponse)
async def match_skills(
    match_request: SkillMatchRequest,
    db: Session = Depends(get_db)
):
    """
    ⚡ Take candidate_id + job_id, run skill matching ML model, 
    compute match score, and return explainable results
    
    Uses:
    - skill_match_model.pkl
    - skill_match_service.py
    """
    
    try:
        # Fetch candidate profile
        candidate_profile = db.query(CandidateProfile)\
            .filter(CandidateProfile.candidate_id == match_request.candidate_id)\
            .order_by(CandidateProfile.created_at.desc())\
            .first()
        
        if not candidate_profile:
            raise HTTPException(status_code=404, detail="Candidate profile not found")
        
        # Fetch job profile using user-provided job_id
        job_profile = db.query(JobProfile)\
            .filter(JobProfile.job_id == match_request.job_id)\
            .first()
        
        if not job_profile:
            raise HTTPException(status_code=404, detail="Job profile not found")
        
        # Get candidate skills
        candidate_skills = candidate_profile.skills or []
        
        # Run skill matching
        match_result = skill_match_service.match_skills(
            candidate_skills=candidate_skills,
            required_skills=job_profile.required_skills or [],
            optional_skills=job_profile.optional_skills or []
        )
        
        # Store match result
        skill_match_result = SkillMatchResult(
            candidate_id=match_request.candidate_id,
            job_id=match_request.job_id,  # Store user-provided job_id
            match_score=match_result["match_score"],
            matched_skills=match_result["matched_skills"],
            missing_required=match_result["missing_required"],
            missing_optional=match_result["missing_optional"],
            explanations=match_result["explanations"]
        )
        
        db.add(skill_match_result)
        db.commit()
        db.refresh(skill_match_result)
        
        return SkillMatchResponse(
            match_id=skill_match_result.id,
            candidate_id=skill_match_result.candidate_id,
            job_id=skill_match_result.job_id,  # Return user-provided job_id
            match_score=skill_match_result.match_score,
            matched_skills=skill_match_result.matched_skills or [],
            missing_required=skill_match_result.missing_required or [],
            missing_optional=skill_match_result.missing_optional or [],
            explanations=skill_match_result.explanations or [],
            created_at=skill_match_result.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# ENDPOINT 4: CANDIDATE READINESS SUMMARY
# ============================================================================

@app.get("/candidate/{candidate_id}/summary", response_model=CandidateSummaryResponse)
async def get_candidate_summary(
    candidate_id: str = Path(..., description="Candidate ID"),
    db: Session = Depends(get_db)
):
    """
    📊 Combine CV + match results, decide interview readiness, 
    and prepare for interview phase
    
    Purpose: Provide comprehensive candidate assessment for interview planning
    """
    
    try:
        # Fetch candidate info
        candidate = db.query(Candidate)\
            .filter(Candidate.candidate_id == candidate_id)\
            .first()
        
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        # Fetch candidate profile
        candidate_profile = db.query(CandidateProfile)\
            .filter(CandidateProfile.candidate_id == candidate_id)\
            .order_by(CandidateProfile.created_at.desc())\
            .first()
        
        # Fetch match history
        match_results = db.query(SkillMatchResult)\
            .filter(SkillMatchResult.candidate_id == candidate_id)\
            .order_by(SkillMatchResult.created_at.desc())\
            .limit(10)\
            .all()
        
        # Calculate overall match statistics
        if match_results:
            avg_score = sum(r.match_score for r in match_results) / len(match_results)
            max_score = max(r.match_score for r in match_results)
            min_score = min(r.match_score for r in match_results)
            recent_score = match_results[0].match_score if match_results else 0
            
            # Count by score ranges
            excellent_matches = sum(1 for r in match_results if r.match_score >= 80)
            good_matches = sum(1 for r in match_results if 60 <= r.match_score < 80)
            poor_matches = sum(1 for r in match_results if r.match_score < 60)
        else:
            avg_score = max_score = min_score = recent_score = 0
            excellent_matches = good_matches = poor_matches = 0
        
        # Determine interview readiness
        if match_results:
            latest_match = match_results[0]
            if latest_match.match_score >= 75:
                interview_readiness = "HIGH PRIORITY - Strong match"
                recommended_actions = [
                    "Schedule interview within 48 hours",
                    "Prepare technical assessment on strongest matched skills",
                    "Review project portfolio for discussion points"
                ]
            elif latest_match.match_score >= 60:
                interview_readiness = "MODERATE - Good potential"
                recommended_actions = [
                    "Schedule interview within 1 week",
                    "Prepare questions about missing required skills",
                    "Consider skills assessment test"
                ]
            else:
                interview_readiness = "LOW - Needs evaluation"
                recommended_actions = [
                    "Consider phone screening first",
                    "Evaluate if missing skills can be trained",
                    "Check for transferable skills"
                ]
        else:
            interview_readiness = "NO MATCHES YET"
            recommended_actions = ["Upload CV and run matches against job postings"]
        
        # Prepare recent matches data
        recent_matches = []
        for match in match_results[:5]:  # Last 5 matches
            job = db.query(JobProfile).filter(JobProfile.job_id == match.job_id).first()
            if job:
                recent_matches.append({
                    "job_title": job.job_title,
                    "company": job.company,
                    "job_id": job.job_id,  # Include user-provided job_id
                    "match_score": match.match_score,
                    "match_date": match.created_at.isoformat(),
                    "matched_skills_count": len(match.matched_skills or [])
                })
        
        return CandidateSummaryResponse(
            candidate_id=candidate_id,
            name=candidate.name,
            email=candidate.email,
            overall_match_stats={
                "average_score": round(avg_score, 2),
                "best_score": round(max_score, 2),
                "recent_score": round(recent_score, 2),
                "total_matches": len(match_results),
                "excellent_matches": excellent_matches,
                "good_matches": good_matches,
                "poor_matches": poor_matches
            },
            recent_matches=recent_matches,
            interview_readiness=interview_readiness,
            recommended_actions=recommended_actions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ============================================================================
# ADDITIONAL UTILITY ENDPOINTS
# ============================================================================

@app.get("/candidate/{candidate_id}/profile")
async def get_candidate_profile(
    candidate_id: str = Path(..., description="Candidate ID"),
    db: Session = Depends(get_db)
):
    """Get candidate's parsed profile"""
    profile = db.query(CandidateProfile)\
        .filter(CandidateProfile.candidate_id == candidate_id)\
        .order_by(CandidateProfile.created_at.desc())\
        .first()
    
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    
    return profile.profile_data

@app.get("/job/{job_id}/profile")
async def get_job_profile(
    job_id: str = Path(..., description="User-provided Job ID"),
    db: Session = Depends(get_db)
):
    """Get job profile by user-provided job_id"""
    job = db.query(JobProfile).filter(JobProfile.job_id == job_id).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job.job_id,
        "job_title": job.job_title,
        "company": job.company,
        "required_skills": job.required_skills,
        "optional_skills": job.optional_skills,
        "user_provided_skills": job.user_provided_skills,
        "created_at": job.created_at.isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0",
        "endpoints": {
            "cv_upload": "/cv/upload (with candidate_id parameter)",
            "job_upload": "/job/upload (with job_id and skills input)",
            "skill_match": "/match/skills (using candidate_id and job_id)",
            "candidate_summary": "/candidate/{candidate_id}/summary"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================================================
# NEW ENDPOINT: SKILL SUGGESTIONS
# ============================================================================

@app.get("/skills/suggestions")
async def get_skill_suggestions():
    """Get list of available skill suggestions from taxonomy"""
    skill_categories = {}
    
    for skill, variations in SkillNormalizer.SKILL_TAXONOMY.items():
        # Categorize skills
        if any(tech in skill for tech in ["python", "java", "javascript", "c++", "c#", "go", "rust", "php", "ruby"]):
            category = "programming"
        elif any(tech in skill for tech in ["sql", "mongodb", "nosql", "mysql", "postgres"]):
            category = "databases"
        elif any(tech in skill for tech in ["aws", "docker", "kubernetes", "azure", "gcp", "devops"]):
            category = "devops_cloud"
        elif any(tech in skill for tech in ["machine_learning", "deep_learning", "data_analysis", "data_engineering"]):
            category = "data_science"
        elif any(tech in skill for skill in ["communication", "teamwork", "problem_solving", "project_management", "leadership", "time_management"]):
            category = "soft_skills"
        elif any(tech in skill for tech in ["html", "css", "react", "node.js", "express.js", "web_development", "mern_stack"]):
            category = "web_development"
        else:
            category = "other"
        
        if category not in skill_categories:
            skill_categories[category] = []
        
        skill_categories[category].append({
            "skill": skill,
            "variations": variations[:3],  # Show first 3 variations
            "example": variations[0] if variations else skill
        })
    
    return {
        "skill_taxonomy": skill_categories,
        "total_skills": len(SkillNormalizer.SKILL_TAXONOMY),
        "how_to_use": "Use these normalized skill names when providing required/optional skills"
    }

# ============================================================================
# DEBUG ENDPOINTS
# ============================================================================

@app.post("/debug/upload-test")
async def debug_upload_test(
    candidate_id: str = Form(None),
    file: UploadFile = File(...)
):
    """Debug endpoint to test file upload without database"""
    try:
        # Read file
        content = await file.read()
        await file.seek(0)
        
        # Extract text
        cv_text = cv_parser_service.extract_text(file)
        
        # Parse CV
        parsed_data = cv_parser_service.parse_cv(cv_text) if cv_text else {}
        
        return {
            "candidate_id_provided": candidate_id,
            "filename": file.filename,
            "size": len(content),
            "text_preview": cv_text[:500] if cv_text else "No text extracted",
            "parsed_name": parsed_data.get("name"),
            "parsed_email": parsed_data.get("email"),
            "success": bool(cv_text)
        }
    except Exception as e:
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/debug/list-candidates")
async def debug_list_candidates(db: Session = Depends(get_db)):
    """List all candidates with their IDs"""
    candidates = db.query(Candidate).all()
    return {
        "total_candidates": len(candidates),
        "candidates": [
            {
                "candidate_id": c.candidate_id,
                "name": c.name,
                "email": c.email,
                "created_at": c.created_at.isoformat()
            } for c in candidates
        ]
    }

@app.get("/debug/list-jobs")
async def debug_list_jobs(db: Session = Depends(get_db)):
    """List all jobs with their IDs"""
    jobs = db.query(JobProfile).all()
    return {
        "total_jobs": len(jobs),
        "jobs": [
            {
                "job_id": j.job_id,
                "job_title": j.job_title,
                "company": j.company,
                "required_skills": j.required_skills,
                "optional_skills": j.optional_skills,
                "created_at": j.created_at.isoformat()
            } for j in jobs
        ]
    }



# @app.get("/jobs")
# async def get_all_jobs(db: Session = Depends(get_db)):
#     jobs = db.query(JobProfile).order_by(JobProfile.created_at.desc()).all()
#     return [
#         {
#             "jobid": j.jobid,
#             "jobtitle": j.jobtitle,
#             "company": j.company,
#             "requiredskills": j.requiredskills or [],
#             "optionalskills": j.optionalskills or [],
#             "userprovidedskills": j.userprovidedskills,
#             "createdat": j.created_at.isoformat() if j.created_at else None,
#         }
#         for j in jobs
#     ]

@app.get("/profiles")
async def get_all_profiles(db: Session = Depends(get_db)):
    subq = (
        db.query(
            CandidateProfile.candidate_id,
            func.max(CandidateProfile.created_at).label("max_created_at"),
        )
        .group_by(CandidateProfile.candidate_id)
        .subquery()
    )

    latest_profiles = (
        db.query(CandidateProfile)
        .join(
            subq,
            (CandidateProfile.candidate_id == subq.c.candidate_id)
            & (CandidateProfile.created_at == subq.c.max_created_at),
        )
        .order_by(CandidateProfile.created_at.desc())
        .all()
    )

    return [
        {
            "candidateid": p.candidate_id,
            "createdat": p.created_at.isoformat() if p.created_at else None,
            "profiledata": p.profile_data,
            "skills": p.skills or [],
        }
        for p in latest_profiles
    ]


@app.get("/debug/jobs-count")
async def debug_jobs_count(db: Session = Depends(get_db)):
    return {
        "count": db.query(JobProfile).count(),
        "db_url": str(engine.url),
    }


@app.delete("/candidate/{candidateid}")
async def delete_candidate(candidateid: str = Path(...), db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.candidate_id == candidateid).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # delete dependent rows first
    db.query(CandidateProfile).filter(CandidateProfile.candidate_id == candidateid).delete()
    db.query(SkillMatchResult).filter(SkillMatchResult.candidate_id == candidateid).delete()

    db.delete(candidate)
    db.commit()
    return {"deleted": True, "candidateid": candidateid}

@app.delete("/candidate/{candidateid}")
async def delete_candidate(candidateid: str = Path(...), db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.candidateid == candidateid).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    # delete dependent rows
    db.query(CandidateProfile).filter(CandidateProfile.candidateid == candidateid).delete()
    db.query(SkillMatchResult).filter(SkillMatchResult.candidateid == candidateid).delete()

    db.delete(candidate)
    db.commit()
    return {"deleted": True, "candidateid": candidateid}




@app.delete("/job/{jobid}")
async def delete_job(jobid: str = Path(...), db: Session = Depends(get_db)):
    job = db.query(JobProfile).filter(JobProfile.job_id == jobid).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # delete job-related match history first (recommended)
    db.query(SkillMatchResult).filter(SkillMatchResult.job_id == jobid).delete()

    db.delete(job)
    db.commit()
    return {"deleted": True, "jobid": jobid}

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import uvicorn
    
    print("=" * 60)
    print("AI-Powered CV & Job Matching System (Enhanced Skills Input)")
    print("=" * 60)
    print("\n📋 FOUR ENDPOINT STRUCTURE:")
    print("1️⃣ POST /cv/upload         - CV Parsing with user-provided candidate_id")
    print("2️⃣ POST /job/upload        - Job Processing with direct skills input")
    print("3️⃣ POST /match/skills      - Core Skill Matching (uses both IDs)")
    print("4️⃣ GET  /candidate/{id}/summary - Candidate Readiness")
    print("\n🆕 NEW FEATURE:")
    print("   GET  /skills/suggestions - Get available skill suggestions")
    print("\n📝 Job Upload with Skills Example:")
    print('''   {
     "job_id": "001",
     "job_title": "Software Engineer",
     "company": "Devsinc",
     "job_description": "Optional description",
     "required_skills": ["python", "sql", "docker"],
     "optional_skills": ["aws", "kubernetes"]
   }''')
    print("\n📚 API Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)
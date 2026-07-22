"""Resume parser using XLM-RoBERTa for NER"""
import re
import logging
from typing import Dict, Any, Optional, List
from transformers import pipeline
from app.core.config import settings

logger = logging.getLogger(__name__)

class ResumeParser:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResumeParser, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ResumeParser._initialized:
            return
        
        logger.info(f"Loading NER model: {settings.ner_model}...")
        
        # Try to load the configured model
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model=settings.ner_model,
                tokenizer=settings.ner_model,
                aggregation_strategy="simple",
                device=-1  # Use CPU, change to 0 for GPU
            )
            logger.info(f"[OK] NER model loaded: {settings.ner_model}")
        except Exception as e:
            logger.error(f"Failed to load {settings.ner_model}: {e}")
            logger.info("Falling back to dslim/bert-base-NER")
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device=-1
                )
                logger.info("[OK] Fallback NER model loaded: dslim/bert-base-NER")
            except Exception as e2:
                logger.error(f"Failed to load fallback NER model: {e2}")
                self.ner_pipeline = None
        
        ResumeParser._initialized = True
    
    def extract_all(self, text: str, sections: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Extract all information from resume text"""
        result = {
            "name": None,
            "email": None,
            "phone": None,
            "skills": [],
            "education": [],
            "experience": [],
            "organizations": [],
            "locations": []
        }
        
        # Extract from header section if available, otherwise from full text
        header_text = sections.get("header", text) if sections else text
        result["name"] = self.extract_name(header_text)
        result["email"] = self.extract_email(text)
        result["phone"] = self.extract_phone(text)
        
        # Extract from specific sections if available
        edu_text = sections.get("education", text) if sections else text
        exp_text = sections.get("experience", text) if sections else text
        
        result["education"] = self.extract_education(edu_text)
        result["experience"] = self.extract_experience(exp_text)
        
        # Extract entities from the entire text
        entities = self.extract_entities(text)
        result["organizations"] = entities.get("organizations", [])
        result["locations"] = entities.get("locations", [])
        
        return result
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract candidate name using NER and pattern matching"""
        if not text or not self.ner_pipeline:
            return self._extract_name_fallback(text)
        
        # Look at the first few lines for the name (usually at the top)
        header = "\n".join(text.split("\n")[:30])
        try:
            ner_results = self.ner_pipeline(header)
            persons = [e["word"].strip() for e in ner_results if e["entity_group"] == "PER"]
            
            for name in persons:
                if self._is_valid_name(name):
                    logger.info(f"NER extracted name: {name}")
                    return name
        except Exception as e:
            logger.warning(f"NER name extraction failed: {e}")
        
        # Fallback: try to find name with pattern matching
        return self._extract_name_fallback(header)
    
    def _extract_name_fallback(self, text: str) -> Optional[str]:
        """Fallback name extraction using regex patterns"""
        # Pattern 1: Look for common name patterns at the beginning of text
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if not line or len(line) > 50 or len(line) < 3:
                continue
            
            # Check if line looks like a name (two or three capitalized words)
            words = line.split()
            if 2 <= len(words) <= 4:
                # Check if each word starts with uppercase or is a common name prefix
                valid = True
                for w in words:
                    if w and w[0].isalpha():
                        if not (w[0].isupper() or w.upper() in ['JR', 'SR', 'II', 'III', 'IV']):
                            valid = False
                            break
                
                if valid:
                    # Avoid false positives
                    false_indicators = ['email', 'phone', 'address', 'resume', 'curriculum', 
                                       'vitae', 'linkedin', 'github', 'objective', 'summary']
                    if not any(ind in line.lower() for ind in false_indicators):
                        return line
        
        # Pattern 2: Look for "Name: John Doe" format
        name_pattern = r'(?:Name|Candidate)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
        match = re.search(name_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _is_valid_name(self, name: str) -> bool:
        """Validate if extracted name is likely a real person name"""
        if not name or len(name) < 3 or len(name) > 50:
            return False
        
        name_lower = name.lower()
        false_positives = [
            "linkedin", "github", "profile", "page", "resume", "curriculum", "vitae",
            "experience", "education", "skills", "projects", "summary", "contact",
            "street", "avenue", "road", "city", "state", "email", "phone", "ongoing",
            "objective", "qualification", "certification", "employment", "reference",
            "address", "telephone", "mobile", "professional", "work", "job"
        ]
        
        if any(fp in name_lower for fp in false_positives):
            return False
        
        words = name.split()
        if len(words) < 2:
            return False
        
        # Check if each word is either capitalized or a known initial
        for w in words:
            if w and w[0].isalpha() and not w[0].isupper() and w.upper() not in ['JR', 'SR', 'II', 'III', 'IV']:
                return False
        
        return True
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities (organizations, locations) from text"""
        if not self.ner_pipeline:
            return {"persons": [], "organizations": [], "locations": []}
        
        results = {"persons": [], "organizations": [], "locations": []}
        seen_orgs, seen_locs = set(), set()
        
        try:
            # Process text in chunks to avoid token limits
            chunk_size = 1000
            chunks = [text[i:i+chunk_size] for i in range(0, min(len(text), 50000), chunk_size)]
            
            for chunk in chunks:
                if not chunk.strip():
                    continue
                    
                ner_results = self.ner_pipeline(chunk)
                for ent in ner_results:
                    word = re.sub(r"[^\w\s\-\.]", "", ent["word"]).strip()
                    if not word or len(word) <= 2:
                        continue
                    
                    if ent["entity_group"] == "ORG" and word not in seen_orgs:
                        seen_orgs.add(word)
                        results["organizations"].append(word)
                    elif ent["entity_group"] in ("LOC", "GPE") and word not in seen_locs:
                        seen_locs.add(word)
                        results["locations"].append(word)
        except Exception as e:
            logger.error(f"NER extraction error: {e}")
        
        return results
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text"""
        # Comprehensive email pattern
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, text)
        
        # Filter out invalid email domains
        invalid_domains = ["example.com", "test.com", "email.com", "domain.com", 
                          "sample.com", "placeholder.com", "demo.com"]
        
        for match in matches:
            domain = match.split("@")[1].lower()
            if any(inv in domain for inv in invalid_domains):
                continue
            
            # Check if the email looks legitimate
            local_part = match.split("@")[0].lower()
            if len(local_part) < 3:
                continue
            
            return match
        
        return None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text"""
        # Multiple phone number patterns
        patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # International
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format (123) 456-7890
            r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',  # 123-456-7890
            r'\b\d{10}\b',  # 1234567890
            r'\b\d{11,12}\b',  # 12345678901 (with country code)
            r'\+?[0-9]{1,3}[-. ]?[0-9]{3}[-. ]?[0-9]{3}[-. ]?[0-9]{4}',  # General
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Clean the match to get only digits
                digits = re.sub(r'\D', '', match)
                if 10 <= len(digits) <= 15:
                    return match.strip()
        
        return None
    
    def extract_education(self, text: str, max_items: int = 5) -> List[str]:
        """Extract education information from text"""
        education_keywords = [
            r'\b(?:b\.?tech|m\.?tech|b\.?e\.?|m\.?e\.?|b\.?sc|m\.?sc|ph\.?d|phd|mba|bca|mca|bs|ms|ba|ma)\b',
            r'\b(?:bachelor|master|diploma|degree|doctorate|ph\.?d|phd)\b',
            r'\b(?:university|college|institute|academy|school)\s+of\s+\w+',
            r'\b(?:high school|secondary school|higher secondary)\b',
            r'\b(?:computer science|engineering|business administration|arts|science|commerce)\b',
            r'\b(?:B\.S\.|M\.S\.|B\.A\.|M\.A\.|B\.Com|M\.Com)\b'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        found, seen = [], set()
        
        for sent in sentences:
            sent = sent.strip()
            if 10 < len(sent) < 300:  # Avoid very short or very long sentences
                for pattern in education_keywords:
                    if re.search(pattern, sent, re.IGNORECASE):
                        if sent not in seen:
                            seen.add(sent)
                            found.append(sent)
                        break
        
        return found[:max_items]
    
    def extract_experience(self, text: str, max_items: int = 5) -> List[str]:
        """Extract work experience information from text"""
        experience_patterns = [
            r'\b(?:experience|worked as|employed as|intern|trainee)\b.*?\b(?:developer|engineer|analyst|manager|consultant|architect|designer|specialist)\b',
            r'\b(?:software|web|full[-\s]stack|data|system|cloud|devops|security|network|machine learning|ai)\s+(?:developer|engineer|architect|analyst|scientist)\b',
            r'\b(?:years?|months?)\s+of\s+experience\b',
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\b.*?\b(?:to|–|-|until)\b.*?\b(?:present|current|\d{4})\b',
            r'\b(?:lead|senior|junior|associate|principal|staff)\s+(?:software|web|data|system|devops)\s+(?:developer|engineer|architect)\b',
            r'\b(?:worked|responsible|managed|developed|designed|implemented|created|built|led)\b'
        ]
        
        sentences = re.split(r'[.!?]+', text)
        found, seen = [], set()
        
        for sent in sentences:
            sent = sent.strip()
            if 15 < len(sent) < 400:  # Experience sentences are usually longer
                for pattern in experience_patterns:
                    if re.search(pattern, sent, re.IGNORECASE):
                        if sent not in seen:
                            seen.add(sent)
                            found.append(sent)
                        break
        
        return found[:max_items]

_resume_parser = None

def get_resume_parser():
    """Get singleton instance of ResumeParser"""
    global _resume_parser
    if _resume_parser is None:
        _resume_parser = ResumeParser()
    return _resume_parser
"""Semantic skill extractor using taxonomy + embeddings"""
import re
import logging
from typing import List, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.models.embedding_model import get_embedding_model

logger = logging.getLogger(__name__)

# Comprehensive skill taxonomy database
COMPREHENSIVE_SKILL_TAXONOMY = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", 
    "php", "ruby", "swift", "kotlin", "scala", "r programming", "matlab", 
    "sql", "html", "css", "sass", "less", "bash", "powershell", "perl", "lua", "dart", "julia",
    
    # Frameworks & Libraries
    "django", "flask", "fastapi", "tornado", "pyramid", "react", "vue.js", "angular", 
    "svelte", "express.js", "next.js", "nuxt.js", "laravel", "symfony", "ruby on rails",
    "spring boot", "asp.net", "bootstrap", "tailwind css", "material ui", "jquery",
    "redux", "webpack", "babel", "vite", "react native", "flutter", "ionic",
    
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra", 
    "dynamodb", "sqlite", "oracle database", "microsoft sql server", "mariadb", 
    "firebase", "supabase", "neo4j", "influxdb", "prisma", "sequelize", "typeorm", "hibernate",
    
    # Machine Learning & AI
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm", 
    "catboost", "huggingface transformers", "pandas", "numpy", "scipy", 
    "matplotlib", "seaborn", "plotly", "opencv", "spacy", "nltk", "gensim",
    "mlflow", "kubeflow", "apache airflow", "dagster", "weights and biases",
    "data preprocessing", "feature engineering", "model training", "model evaluation",
    "hyperparameter tuning", "cross validation", "deep learning", "machine learning",
    "natural language processing", "computer vision", "reinforcement learning",
    "time series analysis", "statistical modeling", "a/b testing", "experimental design",
    "data visualization", "big data", "data mining", "predictive modeling",
    
    # Cloud & DevOps
    "aws", "amazon web services", "google cloud platform", "microsoft azure", 
    "docker", "kubernetes", "jenkins", "gitlab ci", "github actions",
    "circleci", "travis ci", "terraform", "ansible", "puppet", "chef", 
    "nginx", "apache http server", "istio", "envoy", "prometheus", "grafana", 
    "aws lambda", "ec2", "s3", "rds", "ecs", "eks", "gke", "azure devops",
    "ci/cd", "continuous integration", "continuous deployment",
    
    # Big Data
    "apache spark", "hadoop", "kafka", "flink", "storm", "hive", "presto", 
    "databricks", "snowflake", "google bigquery", "amazon redshift", "etl", "data pipeline",
    
    # Tools & Version Control
    "git", "github", "gitlab", "bitbucket", "rest api", "graphql", "grpc", 
    "soap", "websocket", "oauth", "jwt", "json", "xml", "yaml",
    "linux", "ubuntu", "centos", "windows server", "macos", "unix",
    "postman", "insomnia", "swagger", "openapi", "figma",
    "jira", "confluence", "trello", "asana", "notion", "slack",
    "vscode", "vim", "intellij idea", "pycharm", "eclipse", "visual studio",
    
    # Testing
    "selenium", "cypress", "jest", "pytest", "mocha", "junit", "cucumber",
    "unit testing", "integration testing", "end to end testing", "test driven development",
    
    # Soft Skills
    "problem solving", "teamwork", "communication", "time management", 
    "leadership", "critical thinking", "adaptability", "creativity", 
    "attention to detail", "project management", "agile", "scrum", "kanban", 
    "devops culture", "pair programming", "code review",
    
    # Emerging Technologies
    "blockchain", "ethereum", "solidity", "web3", "smart contracts",
    "microservices", "serverless architecture", "event driven architecture", 
    "domain driven design", "system design", "object oriented programming", 
    "functional programming", "design patterns", "algorithms", "data structures",
    
    # Mobile Development
    "android", "ios", "swiftui", "jetpack compose", "kotlin multiplatform", "xamarin",
    
    # Cybersecurity
    "cybersecurity", "network security", "application security", "penetration testing",
    "vulnerability assessment", "security auditing", "encryption", "authentication"
]

class SemanticSkillExtractor:
    def __init__(self, similarity_threshold: float = 0.65):
        self.embedding_model = get_embedding_model()
        self.similarity_threshold = similarity_threshold
        self.skill_taxonomy = [s.lower() for s in COMPREHENSIVE_SKILL_TAXONOMY]
        self.taxonomy_words = self._build_taxonomy_words()
        self._skill_embeddings = None
        
    def _build_taxonomy_words(self) -> Set[str]:
        """Build a set of individual words from skill taxonomy for fast filtering"""
        words = set()
        for skill in self.skill_taxonomy:
            words.update(skill.split())
        return words
    
    @property
    def skill_embeddings(self):
        """Lazy load skill embeddings (cached after first use)"""
        if self._skill_embeddings is None:
            logger.info(f"Precomputing embeddings for {len(self.skill_taxonomy)} skills...")
            self._skill_embeddings = self.embedding_model.encode_batch(
                self.skill_taxonomy, 
                batch_size=64
            )
            logger.info("Skill taxonomy embeddings ready")
        return self._skill_embeddings
    
    def _extract_candidate_phrases(self, text: str) -> List[str]:
        """Extract potential skill phrases from text"""
        # Clean text
        text = re.sub(r"http\S+|www\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        
        phrases = []
        
        # Split into sentences
        sentences = re.split(r"[.!?;\n]+", text)
        
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 3:
                continue
            
            # Add whole sentence
            phrases.append(sent)
            
            # Extract n-grams (1-4 words)
            words = re.findall(r"\b[a-zA-Z0-9+#.()\-]+\b", sent)
            for n in range(1, min(5, len(words) + 1)):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i:i+n]).lower().strip()
                    # Filter by length and relevance
                    if 2 <= len(ngram) <= 50:
                        # Quick filter: check if any word matches taxonomy words
                        if any(w in self.taxonomy_words for w in ngram.split()):
                            phrases.append(ngram)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_phrases = []
        for p in phrases:
            if p not in seen:
                seen.add(p)
                unique_phrases.append(p)
        
        return unique_phrases
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using semantic similarity"""
        if not text or len(text.strip()) < 10:
            return []
        
        # Extract candidate phrases
        phrases = self._extract_candidate_phrases(text)
        if not phrases:
            return []
        
        # Get embeddings for all phrases
        phrase_embeddings = self.embedding_model.encode_batch(phrases, batch_size=32)
        
        # Compute similarities with skill taxonomy
        similarities = cosine_similarity(phrase_embeddings, self.skill_embeddings)
        
        # Find matching skills
        found_skills = set()
        for i, phrase_sims in enumerate(similarities):
            best_idx = int(np.argmax(phrase_sims))
            best_score = float(phrase_sims[best_idx])
            
            if best_score >= self.similarity_threshold:
                matched_skill = self.skill_taxonomy[best_idx]
                found_skills.add(matched_skill)
                logger.debug(f"Skill match: {phrases[i]} -> {matched_skill} (score: {best_score:.3f})")
        
        return sorted(list(found_skills))
    
    def extract_skills_batch(self, texts: List[str]) -> List[List[str]]:
        """Extract skills from multiple texts efficiently"""
        results = []
        for text in texts:
            results.append(self.extract_skills(text))
        return results

# Create singleton instance
skill_extractor = SemanticSkillExtractor()
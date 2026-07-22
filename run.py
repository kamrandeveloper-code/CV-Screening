"""Application runner with model diagnostics"""
import os
import sys
import logging
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings
os.makedirs(settings.logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.logs_dir / "app.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

def diagnose_models():
    print("\n" + "="*70)
    print("MODEL DIAGNOSIS")
    print("="*70)
    
    print("\n[1] EMBEDDING MODEL STATUS:")
    print("-" * 40)
    try:
        from app.models.embedding_model import get_embedding_model
        embedding = get_embedding_model()
        info = embedding.get_model_info()
        print(f"   Model: {info['model_name']}")
        print(f"   Dimension: {info['dimension']}")
        print(f"   Device: {info['device']}")
        print("   Status: LOADED")
    except Exception as e:
        print(f"   Status: ERROR - {e}")
    
    print("\n[2] NER MODEL STATUS:")
    print("-" * 40)
    try:
        from app.models.resume_parser import get_resume_parser
        parser = get_resume_parser()
        if parser.ner_pipeline:
            print("   Model: xlm-roberta-base-finetuned-conll03-english")
            print("   Status: LOADED")
        else:
            print("   Status: NOT LOADED")
    except Exception as e:
        print(f"   Status: ERROR - {e}")
    
    print("\n[3] SKILL MATCHING TEST:")
    print("-" * 40)
    try:
        from app.models.matcher import get_matcher
        matcher = get_matcher()
        candidate_skills = ["python", "react", "mongodb"]
        job_skills = ["python", "django", "postgresql"]
        matched, missing, skill_score = matcher.match_skills(candidate_skills, job_skills)
        print(f"   Candidate: {candidate_skills}")
        print(f"   Job: {job_skills}")
        print(f"   Matched: {matched}")
        print(f"   Missing: {missing}")
        print(f"   Skill overlap: {skill_score:.2f}")
    except Exception as e:
        print(f"   Status: ERROR - {e}")
    
    print("\n" + "="*70)

def main():
    print("\n" + "="*70)
    print("AI RECRUITMENT SYSTEM - STARTING")
    print("="*70 + "\n")
    
    diagnose_models()
    
    print("\n" + "="*70)
    print("STARTING API SERVER...")
    print("="*70 + "\n")
    
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
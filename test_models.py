"""Test model loading"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.models.embedding_model import get_embedding_model
from app.models.resume_parser import get_resume_parser
from app.models.document_layout import get_layout_analyzer
from app.models.skill_extractor import skill_extractor

def test_models():
    print("\n" + "="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    # Test Embedding Model
    print("\n[1] Testing Embedding Model...")
    try:
        emb = get_embedding_model()
        info = emb.get_model_info()
        print(f"   ✓ Loaded: {info['model_name']}")
        print(f"   Dimension: {info['dimension']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test NER Model
    print("\n[2] Testing NER Model...")
    try:
        parser = get_resume_parser()
        if parser.ner_pipeline:
            print(f"   ✓ Loaded: {settings.ner_model}")
        else:
            print("   ✗ Failed to load")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test Layout Model
    print("\n[3] Testing LayoutLMv3 Model...")
    try:
        analyzer = get_layout_analyzer()
        if analyzer.model is not None:
            print("   ✓ LayoutLMv3 loaded successfully")
        else:
            print("   ⚠ LayoutLMv3 not available (will use fallback)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test Skill Extractor
    print("\n[4] Testing Skill Extractor...")
    try:
        test_text = "Python programming and machine learning"
        skills = skill_extractor.extract_skills(test_text)
        print(f"   ✓ Skills extracted: {skills[:3] if skills else 'none'}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_models()
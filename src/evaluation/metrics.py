import jiwer
import re
from rapidfuzz import fuzz
from typing import List, Set
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

class MetricEvaluator:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Robust normalization: lowercase and remove punctuation."""
        if not text:
            return ""
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower().strip()

    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        ref = MetricEvaluator.normalize_text(reference)
        hyp = MetricEvaluator.normalize_text(hypothesis)
        if not ref:
            return 1.0 if hyp else 0.0
        return jiwer.wer(ref, hyp)

    @staticmethod
    def calculate_cer(reference: str, hypothesis: str) -> float:
        ref = MetricEvaluator.normalize_text(reference)
        hyp = MetricEvaluator.normalize_text(hypothesis)
        if not ref:
            return 1.0 if hyp else 0.0
        return jiwer.cer(ref, hyp)

    @staticmethod
    def is_devanagari(text: str) -> bool:
        """Check if the text contains Devanagari (Hindi) characters."""
        return bool(re.search(r'[\u0900-\u097F]', text))

    @staticmethod
    def calculate_entity_accuracy(target_locality: str, hypothesis: str) -> bool:
        """
        Uses Transliteration + Fuzzy Logic to match entities across scripts.
        Matches "Whitefield" with "वाइट फील्ड" by Romanizing the Hindi first.
        """
        if not target_locality or not hypothesis:
            return False
            
        target = target_locality.lower()
        hyp = hypothesis.lower()
        
        # 1. Direct match (Fast path)
        if target in hyp:
            return True
            
        # 2. Transliteration path (for Hindi/Urdu/etc. results)
        if MetricEvaluator.is_devanagari(hypothesis):
            # Transliterate Devanagari to ITRANS (Romanized)
            romanized_hyp = transliterate(hypothesis, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
            # Also try a more common Romanization (like HK or IAST) if needed
            # For now, ITRANS is standard for Indian languages
            
            # Check if target is in the romanized version
            if target in romanized_hyp:
                return True
                
            # Fuzzy match on romanized version
            score = fuzz.partial_ratio(target, romanized_hyp)
            if score >= 80:
                return True

        # 3. Final Fuzzy fallback on original text
        score = fuzz.partial_ratio(target, hyp)
        return score >= 80

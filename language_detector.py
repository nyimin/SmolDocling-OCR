"""
Language Detector Module for DocFlow

Auto-detects document language for proper metadata and processing.
"""

from typing import Dict, Optional, List, Tuple
import re
from collections import Counter


# Unicode ranges for script detection
SCRIPT_RANGES = {
    'latin': (0x0000, 0x024F),
    'cyrillic': (0x0400, 0x04FF),
    'arabic': (0x0600, 0x06FF),
    'devanagari': (0x0900, 0x097F),
    'bengali': (0x0980, 0x09FF),
    'myanmar': (0x1000, 0x109F),
    'thai': (0x0E00, 0x0E7F),
    'cjk': (0x4E00, 0x9FFF),
    'hangul': (0xAC00, 0xD7AF),
    'hiragana': (0x3040, 0x309F),
    'katakana': (0x30A0, 0x30FF),
    'greek': (0x0370, 0x03FF),
    'hebrew': (0x0590, 0x05FF),
}

# Common words by language for Latin script
COMMON_WORDS = {
    'en': {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'that', 'it', 'with', 'as', 'was', 'be'},
    'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für'},
    'fr': {'le', 'la', 'les', 'de', 'un', 'une', 'et', 'est', 'en', 'que', 'des', 'du', 'qui', 'dans'},
    'es': {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'una', 'es', 'se', 'los', 'las', 'con', 'por'},
    'it': {'il', 'la', 'di', 'che', 'e', 'un', 'una', 'in', 'è', 'per', 'non', 'con', 'del', 'si'},
    'pt': {'o', 'a', 'de', 'que', 'e', 'do', 'da', 'em', 'um', 'uma', 'para', 'é', 'com', 'não'},
    'nl': {'de', 'het', 'een', 'van', 'en', 'in', 'is', 'op', 'te', 'dat', 'die', 'voor', 'met'},
    'id': {'yang', 'dan', 'di', 'untuk', 'dengan', 'dari', 'ini', 'itu', 'ke', 'dalam', 'adalah'},
    'tr': {'bir', 've', 'bu', 'için', 'ile', 'de', 'da', 'olarak', 'olan', 'var', 'kadar', 'gibi'},
    'vi': {'và', 'của', 'là', 'có', 'trong', 'được', 'cho', 'với', 'này', 'các', 'những', 'đã'},
}

# Script to primary language mapping (for non-Latin scripts)
SCRIPT_LANGUAGES = {
    'cyrillic': 'ru',
    'arabic': 'ar',
    'devanagari': 'hi',
    'bengali': 'bn',
    'myanmar': 'my',
    'thai': 'th',
    'hangul': 'ko',
    'hebrew': 'he',
    'greek': 'el',
}


class LanguageDetector:
    """
    Detect document language from text content.
    
    Uses character script analysis and common word matching.
    """
    
    def __init__(self, default_language: str = 'en'):
        """
        Initialize detector.
        
        Args:
            default_language: Fallback language code
        """
        self.default_language = default_language
    
    def detect(self, text: str, min_confidence: float = 0.3) -> Dict[str, any]:
        """
        Detect language from text.
        
        Args:
            text: Text to analyze
            min_confidence: Minimum confidence threshold
        
        Returns:
            {
                'language': ISO 639-1 code,
                'confidence': 0.0-1.0,
                'script': Detected script,
                'method': Detection method used
            }
        """
        if not text or len(text.strip()) < 10:
            return self._result(self.default_language, 0.0, 'unknown', 'insufficient_text')
        
        # 1. Detect script
        script, script_conf = self._detect_script(text)
        
        # 2. If non-Latin script, map directly
        if script != 'latin' and script in SCRIPT_LANGUAGES:
            return self._result(SCRIPT_LANGUAGES[script], script_conf, script, 'script')
        
        # 3. CJK needs special handling
        if script == 'cjk':
            lang = self._detect_cjk(text)
            return self._result(lang, script_conf, 'cjk', 'cjk_analysis')
        
        # 4. For Latin, use word frequency
        if script == 'latin':
            lang, word_conf = self._detect_latin_language(text)
            return self._result(lang, word_conf, 'latin', 'word_frequency')
        
        return self._result(self.default_language, 0.0, script, 'fallback')
    
    def _detect_script(self, text: str) -> Tuple[str, float]:
        """Detect primary script in text."""
        script_counts = Counter()
        letter_count = 0
        
        for char in text:
            code = ord(char)
            if not char.isalpha():
                continue
            letter_count += 1
            
            for script, (start, end) in SCRIPT_RANGES.items():
                if start <= code <= end:
                    script_counts[script] += 1
                    break
        
        if not script_counts or letter_count == 0:
            return ('unknown', 0.0)
        
        primary_script, count = script_counts.most_common(1)[0]
        confidence = count / letter_count
        
        return (primary_script, confidence)
    
    def _detect_latin_language(self, text: str) -> Tuple[str, float]:
        """Detect language for Latin script text using word frequency."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return (self.default_language, 0.0)
        
        word_set = set(words)
        scores = {}
        
        for lang, common in COMMON_WORDS.items():
            matches = len(word_set & common)
            scores[lang] = matches
        
        if not scores or max(scores.values()) == 0:
            return (self.default_language, 0.0)
        
        best_lang = max(scores, key=scores.get)
        confidence = scores[best_lang] / len(word_set) if word_set else 0
        
        # Normalize confidence
        confidence = min(1.0, confidence * 5)
        
        return (best_lang, confidence)
    
    def _detect_cjk(self, text: str) -> str:
        """Distinguish between Chinese, Japanese, and Korean in CJK text."""
        hiragana = sum(1 for c in text if 0x3040 <= ord(c) <= 0x309F)
        katakana = sum(1 for c in text if 0x30A0 <= ord(c) <= 0x30FF)
        hangul = sum(1 for c in text if 0xAC00 <= ord(c) <= 0xD7AF)
        
        if hiragana + katakana > 5:
            return 'ja'
        if hangul > 5:
            return 'ko'
        return 'zh'
    
    def _result(self, lang: str, conf: float, script: str, method: str) -> Dict:
        return {
            'language': lang,
            'confidence': round(conf, 4),
            'script': script,
            'method': method
        }
    
    def detect_batch(self, pages: List[str]) -> Dict[str, any]:
        """
        Detect language from multiple pages.
        
        Args:
            pages: List of page texts
        
        Returns:
            Aggregated detection result
        """
        if not pages:
            return self._result(self.default_language, 0.0, 'unknown', 'no_pages')
        
        combined = ' '.join(pages[:5])  # Use first 5 pages
        return self.detect(combined)


def detect_language(text: str) -> str:
    """
    Quick language detection.
    
    Args:
        text: Text to analyze
    
    Returns:
        ISO 639-1 language code
    """
    detector = LanguageDetector()
    result = detector.detect(text)
    return result['language']


# Example usage
if __name__ == "__main__":
    samples = [
        "The quick brown fox jumps over the lazy dog.",
        "Le renard brun rapide saute par-dessus le chien paresseux.",
        "Der schnelle braune Fuchs springt über den faulen Hund.",
        "ミャンマー語のテキストサンプル",
        "မြန်မာဘာသာစကား နမူနာ စာသား",
        "Đây là một văn bản mẫu tiếng Việt.",
    ]
    
    detector = LanguageDetector()
    for sample in samples:
        result = detector.detect(sample)
        print(f"'{sample[:40]}...' -> {result['language']} ({result['confidence']:.2f})")

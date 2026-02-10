"""
Multi-Language Support for NPC AI

Supports multiple languages with automatic translation and
language-specific semantic evaluation.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class LanguageConfig:
    """Language configuration"""
    code: str  # ISO 639-1 code
    name: str
    bert_model: str  # Language-specific BERT model
    supported: bool = True


class MultiLanguageSupport:
    """
    Multi-language support for NPC conversations
    
    Supports:
    - English (en)
    - Vietnamese (vi)
    - Chinese (zh)
    - Japanese (ja)
    - Korean (ko)
    - Spanish (es)
    - French (fr)
    - German (de)
    """
    
    # Language configurations
    LANGUAGES = {
        'en': LanguageConfig(
            code='en',
            name='English',
            bert_model='sentence-transformers/all-MiniLM-L6-v2'
        ),
        'vi': LanguageConfig(
            code='vi',
            name='Vietnamese',
            bert_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ),
        'zh': LanguageConfig(
            code='zh',
            name='Chinese',
            bert_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ),
        'ja': LanguageConfig(
            code='ja',
            name='Japanese',
            bert_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ),
        'ko': LanguageConfig(
            code='ko',
            name='Korean',
            bert_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ),
        'es': LanguageConfig(
            code='es',
            name='Spanish',
            bert_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ),
        'fr': LanguageConfig(
            code='fr',
            name='French',
            bert_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        ),
        'de': LanguageConfig(
            code='de',
            name='German',
            bert_model='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
    }
    
    def __init__(self, default_language: str = 'en'):
        """
        Initialize multi-language support
        
        Args:
            default_language: Default language code
        """
        self.default_language = default_language
        self._translators = {}
        self._models = {}
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (ISO 639-1)
        """
        try:
            from langdetect import detect
            lang = detect(text)
            
            # Map to supported languages
            if lang in self.LANGUAGES:
                return lang
            
            # Default to English for unsupported
            return self.default_language
            
        except ImportError:
            print("⚠ langdetect not installed. Install with: pip install langdetect")
            return self.default_language
        except:
            return self.default_language
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Translate text between languages
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if source_lang == target_lang:
            return text
        
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # Get or create translator
            key = f"{source_lang}-{target_lang}"
            if key not in self._translators:
                model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                try:
                    tokenizer = MarianTokenizer.from_pretrained(model_name)
                    model = MarianMTModel.from_pretrained(model_name)
                    self._translators[key] = (tokenizer, model)
                except:
                    print(f"⚠ Translation model {model_name} not available")
                    return text
            
            tokenizer, model = self._translators[key]
            
            # Translate
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            outputs = model.generate(**inputs)
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return translated
            
        except ImportError:
            print("⚠ transformers not installed for translation")
            return text
        except Exception as e:
            print(f"⚠ Translation error: {e}")
            return text
    
    def get_language_model(self, language: str) -> str:
        """
        Get appropriate BERT model for language
        
        Args:
            language: Language code
            
        Returns:
            Model name
        """
        if language in self.LANGUAGES:
            return self.LANGUAGES[language].bert_model
        return self.LANGUAGES[self.default_language].bert_model
    
    def is_supported(self, language: str) -> bool:
        """Check if language is supported"""
        return language in self.LANGUAGES
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes"""
        return list(self.LANGUAGES.keys())
    
    def get_language_info(self, language: str) -> Optional[LanguageConfig]:
        """Get language configuration"""
        return self.LANGUAGES.get(language)


# Convenience functions
def detect_language(text: str) -> str:
    """Detect language of text"""
    ml = MultiLanguageSupport()
    return ml.detect_language(text)


def translate_text(text: str, source: str, target: str) -> str:
    """Translate text"""
    ml = MultiLanguageSupport()
    return ml.translate(text, source, target)


if __name__ == "__main__":
    # Example usage
    print("Multi-Language Support - Example")
    print("="*60)
    
    ml = MultiLanguageSupport()
    
    # Show supported languages
    print("\nSupported Languages:")
    for code, config in ml.LANGUAGES.items():
        print(f"  {code}: {config.name}")
    
    # Example detection
    texts = {
        "Hello, how are you?": "en",
        "Xin chào, bạn khỏe không?": "vi",
        "こんにちは、元気ですか？": "ja"
    }
    
    print("\nLanguage Detection:")
    for text, expected in texts.items():
        detected = ml.detect_language(text)
        print(f"  '{text[:30]}...' -> {detected}")

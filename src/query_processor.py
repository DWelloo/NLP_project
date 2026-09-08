"""
Query Processor for improved graph search using NLP techniques.

Implements lemmatization, stop word removal, and keyword extraction
for better matching between user queries and graph nodes.
"""

import spacy
from typing import Optional
from src.logger import get_logger


class QueryProcessor:
    """Processes user queries using NLP techniques for better graph search."""
    
    # Polish stop words and common question words
    STOP_WORDS = {
        # Question words
        'co', 'to', 'jest', 'jaki', 'jaka', 'jakie', 'jakim', 'jakiej', 'jaką',
        'kiedy', 'gdzie', 'który', 'która', 'które', 'którym', 'której',
        'czym', 'kto', 'jak', 'dlaczego', 'czy', 'ile',
        # Conjunctions and prepositions
        'i', 'a', 'ale', 'lub', 'oraz', 'albo', 'ani', 'bo', 'więc', 'jednak',
        'przez', 'przed', 'przy', 'po', 'pod', 'nad', 'za', 'do', 'od', 'ze', 'z',
        'na', 'w', 'we', 'dla', 'bez', 'o', 'u', 'ku',
        # Pronouns and articles
        'ten', 'ta', 'to', 'te', 'tym', 'tej', 'tego', 'tych', 'temu',
        'on', 'ona', 'ono', 'oni', 'one', 'ich', 'im', 'ją', 'jej', 'jego', 'nimi',
        'się', 'sobie', 'siebie', 'mnie', 'mi', 'my', 'nas', 'nam', 'wy', 'was', 'wam',
        # Common verbs
        'być', 'mieć', 'móc', 'musieć', 'chcieć', 'powinien',
        'może', 'można', 'należy', 'trzeba', 'warto',
        # Other common words
        'tylko', 'także', 'również', 'nawet', 'już', 'jeszcze', 'bardzo', 'bardziej',
        'tak', 'nie', 'czy', 'tutaj', 'tam', 'teraz', 'potem', 'zawsze', 'nigdy',
        'wszystko', 'nic', 'coś', 'niektóre', 'inne', 'każdy', 'żaden',
    }
    
    def __init__(self):
        self._logger = get_logger("QueryProcessor")
        self._nlp: Optional[spacy.Language] = None
        self._load_model()
    
    def _load_model(self):
        """Load Polish spaCy model."""
        try:
            self._nlp = spacy.load("pl_core_news_sm")
            self._logger.debug("Loaded Polish spaCy model")
        except OSError:
            self._logger.warning("Polish spaCy model not found, using fallback")
            self._nlp = None
    
    def extract_keywords(self, query: str) -> list[str]:
        """
        Extract keywords from query using lemmatization and filtering.
        
        Args:
            query: User query string
            
        Returns:
            List of keyword lemmas
        """
        if not self._nlp:
            return self._fallback_extract(query)
        
        doc = self._nlp(query)
        
        keywords = []
        for token in doc:
            # Skip punctuation, stop words, and very short words
            if (token.is_punct or 
                token.is_space or 
                token.lemma_.lower() in self.STOP_WORDS or
                len(token.lemma_) < 3):
                continue
            
            # Prefer nouns, proper nouns, and adjectives
            if token.pos_ in ('NOUN', 'PROPN', 'ADJ', 'NUM'):
                keywords.append(token.lemma_.lower())
            # Also include verbs but with lower priority
            elif token.pos_ == 'VERB' and len(token.lemma_) > 4:
                keywords.append(token.lemma_.lower())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)
        
        self._logger.debug(f"Extracted keywords: {unique_keywords}")
        return unique_keywords
    
    def _fallback_extract(self, query: str) -> list[str]:
        """Fallback keyword extraction without spaCy."""
        words = query.lower().replace('?', '').replace('.', '').replace(',', '').split()
        return [w for w in words if len(w) > 3 and w not in self.STOP_WORDS]
    
    def get_search_variants(self, keyword: str) -> list[str]:
        """
        Generate search variants for a keyword (original + lemma + partial).
        
        Args:
            keyword: Keyword to generate variants for
            
        Returns:
            List of search variants
        """
        variants = [keyword]
        
        if self._nlp:
            doc = self._nlp(keyword)
            for token in doc:
                if token.lemma_.lower() != keyword:
                    variants.append(token.lemma_.lower())
        
        # Add partial match for long keywords (first 5 characters)
        if len(keyword) > 6:
            variants.append(keyword[:5])
        
        return list(set(variants))
    
    def extract_entities(self, query: str) -> dict[str, list[str]]:
        """
        Extract named entities from query.
        
        Returns:
            Dictionary with entity types as keys and entity texts as values
        """
        if not self._nlp:
            return {}
        
        doc = self._nlp(query)
        entities = {}
        
        for ent in doc.ents:
            ent_type = ent.label_
            if ent_type not in entities:
                entities[ent_type] = []
            entities[ent_type].append(ent.text)
        
        self._logger.debug(f"Extracted entities: {entities}")
        return entities


# Singleton instance for reuse
_processor_instance: Optional[QueryProcessor] = None


def get_query_processor() -> QueryProcessor:
    """Get or create singleton QueryProcessor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = QueryProcessor()
    return _processor_instance

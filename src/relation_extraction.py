"""
Gemini-based relation extraction for knowledge graph construction.

This module provides relation extraction using Google's Gemini API,
supporting both text input and file upload for document processing.
"""

import json
import re
from typing import List, Dict, Optional
import google.generativeai as genai
from llama_index.core.schema import TextNode

from src.logger import get_logger


class GeminiExtractor:
    """
    Extracts subject-predicate-object relations from text using Gemini API.

    Supports:
    - Direct text extraction from TextNode objects
    - File upload for larger documents
    - Batch processing of multiple nodes
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self._logger = get_logger("GeminiExtractor")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)
        self._model_name = model_name

        self._extract_relation_prompt = """Jesteś automatem wyodrębniającym dane. Twoim jedynym zadaniem jest wypisanie relacji zgodnie z formatem.

ZASADY:
- Zwróć dane WYŁĄCZNIE jako listę obiektów JSON (NIE dodawaj wstępu ani podsumowania (np. "Oto wyniki")).
- Każdy obiekt musi mieć klucze: "subject", "predicate", "object".
- Używaj języka polskiego.
- NIE powtarzaj przykładów z instrukcji.
- Wyodrębniaj jak najwięcej sensownych relacji z tekstu.
- Relacje powinny być konkretne i informacyjne.

PRZYKŁAD:
Tekst: Anna mieszka w Warszawie. Marek pracuje w Google.
Relacje:
[
    {{"subject": "Anna", "predicate": "mieszka w", "object": "Warszawa"}},
    {{"subject": "Marek", "predicate": "pracuje w", "object": "Google"}}
]

TWOJE ZADANIE:
Tekst: {text}

Wyodrębnij relacje i zwróć je jako listę JSON:"""

        self._file_prompt = """Jesteś automatem wyodrębniającym dane. Twoim jedynym zadaniem jest wypisanie relacji zgodnie z formatem.

ZASADY:
- Zwróć dane WYŁĄCZNIE jako listę obiektów JSON (NIE dodawaj wstępu ani podsumowania (np. "Oto wyniki")).
- Każdy obiekt musi mieć klucze: "subject", "predicate", "object".
- Używaj języka polskiego.
- NIE powtarzaj przykładów z instrukcji.
- Wyodrębniaj jak najwięcej sensownych relacji z tekstu.
- Relacje powinny być konkretne i informacyjne.

PRZYKŁAD:
Tekst: Anna mieszka w Warszawie. Marek pracuje w Google.
Relacje:
[
    {{"subject": "Anna", "predicate": "mieszka w", "object": "Warszawa"}},
    {{"subject": "Marek", "predicate": "pracuje w", "object": "Google"}}
]

TWOJE ZADANIE:
Tekst znajduje się w załączonym pliku. Wyodrębnij relacje i zwróć je w formacie JSON, opisanym w zasadach powyżej."""

    def extract_relations(self, nodes: List[TextNode]) -> List[Dict]:
        """
        Extract relations from a list of TextNode objects.

        Args:
            nodes: List of TextNode objects containing text to process

        Returns:
            List of relation dictionaries with subject, predicate, object keys
        """
        if not nodes:
            return []

        # Combine all node content
        combined_text = " ".join([node.get_content() for node in nodes])
        self._logger.debug(
            f"Extracting relations from {len(nodes)} nodes ({len(combined_text)} chars)"
        )

        prompt = self._extract_relation_prompt.format(text=combined_text)

        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", temperature=0.1
                ),
            )

            self._logger.debug(f"LLM response received: {len(response.text)} chars")
            relations = self._parse_relations(response.text)
            self._logger.info(
                f"Extracted {len(relations)} relations from {len(nodes)} nodes"
            )
            return relations

        except Exception as e:
            self._logger.error(f"Error extracting relations: {e}")
            return []

    def extract_from_file(self, filepath: str) -> List[Dict]:
        """
        Extract relations from a file using Gemini's file upload API.

        Args:
            filepath: Path to the file to process

        Returns:
            List of relation dictionaries
        """
        self._logger.info(f"Extracting relations from file: {filepath}")

        try:
            # Upload file to Gemini
            uploaded_file = genai.upload_file(filepath)
            self._logger.debug(f"Uploaded file to Gemini API: {uploaded_file.name}")

            response = self._model.generate_content(
                [uploaded_file, self._file_prompt],
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", temperature=0.1
                ),
            )

            relations = self._parse_relations(response.text)
            self._logger.info(
                f"Extracted {len(relations)} relations from file {filepath}"
            )
            return relations

        except Exception as e:
            self._logger.error(f"Error extracting relations from file: {e}")
            return []

    def _parse_relations(self, response_text: str) -> List[Dict]:
        """
        Parse LLM response into structured relations.

        Handles various response formats including:
        - Direct JSON arrays
        - JSON wrapped in markdown code blocks
        - Malformed JSON with text around it
        """
        if not response_text:
            return []

        clean_text = response_text.strip()

        # Remove markdown code blocks
        if clean_text.startswith("```json"):
            clean_text = clean_text.replace("```json", "", 1)
        if clean_text.startswith("```"):
            clean_text = clean_text.replace("```", "", 1)
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        clean_text = clean_text.strip()

        # Try direct JSON parse
        try:
            relations = json.loads(clean_text)
            if isinstance(relations, list):
                valid_relations = self._validate_relations(relations)
                self._logger.debug(
                    f"Parsed {len(valid_relations)} valid relations from JSON"
                )
                return valid_relations
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in response
        match = re.search(r"\[.*\]", clean_text, re.DOTALL)
        if match:
            try:
                relations = json.loads(match.group(0))
                if isinstance(relations, list):
                    valid_relations = self._validate_relations(relations)
                    self._logger.debug(
                        f"Parsed {len(valid_relations)} relations from extracted JSON"
                    )
                    return valid_relations
            except json.JSONDecodeError as e:
                self._logger.warning(f"Failed to parse extracted JSON: {e}")

        self._logger.warning(
            f"Could not parse relations from response: {clean_text[:200]}..."
        )
        return []

    def _validate_relations(self, relations: List[Dict]) -> List[Dict]:
        """Validate and filter relations to ensure they have required keys."""
        valid = []
        required_keys = {"subject", "predicate", "object"}

        for rel in relations:
            if isinstance(rel, dict) and required_keys.issubset(rel.keys()):
                # Ensure all values are non-empty strings
                if all(
                    rel.get(k) and isinstance(rel.get(k), str) and rel.get(k).strip()
                    for k in required_keys
                ):
                    valid.append(
                        {
                            "subject": rel["subject"].strip(),
                            "predicate": rel["predicate"].strip(),
                            "object": rel["object"].strip(),
                        }
                    )
            else:
                self._logger.debug(f"Skipping invalid relation: {rel}")

        return valid

    def extract_from_nodes_batch(
        self, nodes: List[TextNode], batch_size: int = 5
    ) -> List[Dict]:
        """
        Extract relations from nodes in batches to handle large documents.

        Args:
            nodes: List of TextNode objects
            batch_size: Number of nodes to process per batch

        Returns:
            Combined list of all extracted relations
        """
        all_relations = []

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]
            self._logger.debug(
                f"Processing batch {i//batch_size + 1} ({len(batch)} nodes)"
            )

            batch_relations = self.extract_relations(batch)

            # Add source node info
            for rel in batch_relations:
                rel["batch_index"] = i // batch_size

            all_relations.extend(batch_relations)

        self._logger.info(
            f"Extracted total {len(all_relations)} relations from {len(nodes)} nodes in {(len(nodes) + batch_size - 1) // batch_size} batches"
        )
        return all_relations

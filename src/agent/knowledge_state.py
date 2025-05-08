"""
Knowledge state module for agent cognitive capabilities
"""
import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import json
from langchain.schema import Document

logger = logging.getLogger(__name__)

class KnowledgeState:
    """Tracks the agent's knowledge state and cognitive capabilities"""
    
    def __init__(self):
        # Core knowledge state
        self.known_facts = []
        self.unknown_aspects = []
        self.confidence_levels = {}
        self.reasoning_chains = []
        self.detected_conflicts = []
        
        # Document tracking
        self.document_sources = {}
        self.cited_documents = set()
        
        # Interaction state
        self.interactions = []
        self.current_session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.last_query_timestamp = None
        self.response_metadata = {}
        
        # Initialize state
        self._initialize_state()
    
    def _initialize_state(self) -> None:
        """Initialize the knowledge state"""
        # Set default unknown aspects
        self.unknown_aspects = [
            "User's specific context",
            "Information not present in provided documents"
        ]
        
        # Set base confidence levels for different knowledge domains
        self.confidence_levels = {
            "document_content": 0.9,  # High confidence in document content
            "inferences": 0.7,        # Medium confidence in inferences
            "user_intents": 0.5       # Lower confidence in user intents
        }
        
        logger.info("Initialized knowledge state")
    
    def update_with_documents(self, documents: List[Document]) -> None:
        """
        Update knowledge state with information from documents
        
        Args:
            documents: List of Document objects
        """
        logger.info(f"Updating knowledge state with {len(documents)} documents")
        
        # Extract and store key facts from documents
        extracted_facts = []
        
        for doc in documents:
            # Get document ID and source information
            doc_id = doc.metadata.get("document_id", "unknown")
            source = doc.metadata.get("file_name", "unknown source")
            
            # Store document source info
            self.document_sources[doc_id] = {
                "source": source,
                "confidence": doc.metadata.get("confidence_score", 0.8),
                "creation_date": doc.metadata.get("creation_date", None),
                "accessed_timestamp": datetime.now().isoformat()
            }
            
            # Extract facts from document
            facts = self._extract_facts_from_document(doc)
            
            # Add source attribution to facts
            for fact in facts:
                fact["source"] = doc_id
                fact["confidence"] = doc.metadata.get("confidence_score", 0.8)
                extracted_facts.append(fact)
        
        # Deduplicate facts and add to known facts
        self._update_facts(extracted_facts)
        
        # Update confidence levels based on document quality
        avg_confidence = sum(doc.metadata.get("confidence_score", 0.8) for doc in documents) / max(1, len(documents))
        self.confidence_levels["document_content"] = min(1.0, avg_confidence + 0.1)
        
        logger.info(f"Knowledge state updated with {len(extracted_facts)} facts")
    
    def _extract_facts_from_document(self, document: Document) -> List[Dict[str, Any]]:
        """Extract key facts from document content"""
        facts = []
        content = document.page_content
        
        # Simple approach: split by sentences and filter for fact-like statements
        sentences = re.split(r'[.!?]', content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
                
            # Check if sentence likely contains a fact
            if self._is_likely_fact(sentence):
                fact = {
                    "text": sentence,
                    "extracted_at": datetime.now().isoformat(),
                    "type": "document_fact"
                }
                facts.append(fact)
                
        return facts
    
    def _is_likely_fact(self, text: str) -> bool:
        """Determine if text likely contains a factual statement"""
        # Simple heuristic: check for factual indicators
        factual_indicators = [
            r"\b(?:is|are|was|were)\b",
            r"\b(?:has|have|had)\b",
            r"\b(?:will|must|should)\b",
            r"\b(?:increased|decreased)\b",
            r"\b(?:according to|based on)\b",
            r"\b(?:shown|demonstrated|confirmed|verified)\b"
        ]
        
        for indicator in factual_indicators:
            if re.search(indicator, text, re.IGNORECASE):
                return True
                
        return False
    
    def _update_facts(self, new_facts: List[Dict[str, Any]]) -> None:
        """Update known facts, handling duplicates and conflicts"""
        for new_fact in new_facts:
            # Check if this fact is already known (or similar)
            is_duplicate = False
            for i, existing_fact in enumerate(self.known_facts):
                if self._are_facts_similar(new_fact["text"], existing_fact["text"]):
                    is_duplicate = True
                    
                    # Check for contradiction
                    if self._are_facts_contradicting(new_fact["text"], existing_fact["text"]):
                        # Record the conflict
                        conflict = {
                            "fact1": existing_fact["text"],
                            "fact2": new_fact["text"],
                            "source1": existing_fact.get("source", "unknown"),
                            "source2": new_fact.get("source", "unknown"),
                            "detected_at": datetime.now().isoformat()
                        }
                        self.detected_conflicts.append(conflict)
                        
                        # Prefer the fact with higher confidence
                        if new_fact.get("confidence", 0) > existing_fact.get("confidence", 0):
                            self.known_facts[i] = new_fact
                    
                    break
            
            # Add new fact if not a duplicate
            if not is_duplicate:
                self.known_facts.append(new_fact)
    
    def _are_facts_similar(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are semantically similar"""
        # Normalize and tokenize
        words1 = set(re.findall(r'\b\w+\b', fact1.lower()))
        words2 = set(re.findall(r'\b\w+\b', fact2.lower()))
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        # Calculate overlap
        if not words1 or not words2:
            return False
            
        overlap = len(words1.intersection(words2))
        similarity = overlap / len(words1.union(words2))
        
        return similarity > 0.6  # Consider similar if 60% overlap
    
    def _are_facts_contradicting(self, fact1: str, fact2: str) -> bool:
        """Check if two facts contradict each other"""
        # Simple contradiction detection based on negation
        
        # First check if they're about the same topic
        if not self._are_facts_similar(fact1, fact2):
            return False
        
        # Check for negation words
        negations = ["not", "never", "no", "isn't", "aren't", "wasn't", "weren't"]
        
        has_negation1 = any(neg in fact1.lower().split() for neg in negations)
        has_negation2 = any(neg in fact2.lower().split() for neg in negations)
        
        # If one has negation and the other doesn't, likely contradiction
        return has_negation1 != has_negation2
    
    def add_reasoning_chain(
        self, 
        query: str,
        reasoning_steps: List[str],
        conclusion: str
    ) -> None:
        """
        Add a reasoning chain to track agent's thought process
        
        Args:
            query: The original query
            reasoning_steps: List of reasoning steps
            conclusion: The conclusion reached
        """
        reasoning_chain = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "reasoning_steps": reasoning_steps,
            "conclusion": conclusion
        }
        
        self.reasoning_chains.append(reasoning_chain)
        logger.info("Added reasoning chain to knowledge state")
    
    def record_interaction(
        self,
        query: str,
        response: str,
        cited_sources: List[str],
        confidence: float
    ) -> None:
        """
        Record an interaction with the user
        
        Args:
            query: User query
            response: Agent response
            cited_sources: List of cited source document IDs
            confidence: Confidence in the response
        """
        interaction = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "response_length": len(response),
            "cited_sources": cited_sources,
            "confidence": confidence,
            "session_id": self.current_session_id
        }
        
        self.interactions.append(interaction)
        self.last_query_timestamp = datetime.now().isoformat()
        
        # Track cited documents
        for source in cited_sources:
            self.cited_documents.add(source)
            
        logger.info("Recorded interaction in knowledge state")
    
    def update_unknown_aspects(self, unknown_items: List[str]) -> None:
        """
        Update the list of things the agent doesn't know
        
        Args:
            unknown_items: List of unknown aspects
        """
        # Add new unknown aspects, avoid duplicates
        for item in unknown_items:
            item = item.strip()
            if item and item not in self.unknown_aspects:
                self.unknown_aspects.append(item)
                
        logger.info(f"Updated unknown aspects: {len(self.unknown_aspects)} items")
    
    def identify_knowledge_gaps(self, query: str) -> List[str]:
        """
        Identify potential knowledge gaps related to a query
        
        Args:
            query: User query
            
        Returns:
            List of identified knowledge gaps
        """
        # Extract key entities and concepts from query
        query_concepts = self._extract_query_concepts(query)
        
        # Check if we have facts related to these concepts
        gaps = []
        for concept in query_concepts:
            concept_covered = False
            
            # Check for concept coverage in known facts
            for fact in self.known_facts:
                if concept.lower() in fact["text"].lower():
                    concept_covered = True
                    break
            
            if not concept_covered:
                gaps.append(f"Missing information about: {concept}")
                
        return gaps
    
    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract key entities and concepts from a query"""
        # Simple approach: extract noun phrases and entities
        # In a real implementation, this would use NER and more sophisticated NLP
        
        # Extract words that are likely nouns or named entities
        words = query.split()
        concepts = []
        
        # 1. Look for capitalized words (possible named entities)
        for i, word in enumerate(words):
            # Skip first word (might be capitalized as sentence start)
            if i > 0 and word and word[0].isupper():
                concepts.append(word.strip(".,;:?!"))
                
        # 2. Look for domain-specific terms
        domain_terms = [
            "compliance", "regulation", "policy", "document", "report",
            "standard", "requirement", "rule", "law", "guideline"
        ]
        
        for term in domain_terms:
            if term.lower() in query.lower():
                concepts.append(term)
                
        # 3. Extract noun phrases using simple patterns
        noun_phrase_patterns = [
            r"the ([a-z]+ [a-z]+)",
            r"a ([a-z]+ [a-z]+)",
            r"an ([a-z]+ [a-z]+)"
        ]
        
        for pattern in noun_phrase_patterns:
            matches = re.findall(pattern, query.lower())
            concepts.extend(matches)
                
        return list(set(concepts))  # Deduplicate
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current knowledge state
        
        Returns:
            Dictionary with knowledge state summary
        """
        return {
            "known_facts_count": len(self.known_facts),
            "known_facts_sample": self.known_facts[:5] if self.known_facts else [],
            "unknown_aspects": self.unknown_aspects,
            "confidence_levels": self.confidence_levels,
            "conflicts_detected": len(self.detected_conflicts),
            "document_sources_count": len(self.document_sources),
            "interactions_count": len(self.interactions),
            "reasoning_chains_count": len(self.reasoning_chains)
        }
    
    def identify_decision_triggers(self, query: str, response: str) -> Dict[str, Any]:
        """
        Identify if the agent should trigger a decision suggestion
        
        Args:
            query: User query
            response: Agent response
            
        Returns:
            Dictionary with decision trigger information
        """
        # This method determines if the agent should suggest the user
        # take some action based on the information provided
        
        # Look for action-oriented language in the query
        action_triggers = [
            "should i", "can i", "how do i", "what steps",
            "recommend", "advise", "suggest", "option"
        ]
        
        has_action_request = any(trigger in query.lower() for trigger in action_triggers)
        
        # Look for uncertain language in the response
        uncertainty_markers = [
            "uncertain", "unclear", "might", "maybe",
            "possibly", "could be", "not sure", "don't know"
        ]
        
        has_uncertainty = any(marker in response.lower() for marker in uncertainty_markers)
        
        # Look for strong recommendations in the response
        recommendation_markers = [
            "recommend", "should", "must", "need to",
            "important to", "advised", "suggested"
        ]
        
        has_recommendation = any(marker in response.lower() for marker in recommendation_markers)
        
        # Decision trigger assessment
        is_triggered = has_action_request and (has_recommendation or has_uncertainty)
        trigger_confidence = 0.8 if has_recommendation else 0.6 if has_uncertainty else 0.4
        
        return {
            "decision_triggered": is_triggered,
            "trigger_type": "recommendation" if has_recommendation else "uncertainty" if has_uncertainty else "none",
            "confidence": trigger_confidence,
            "has_action_request": has_action_request
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert knowledge state to dictionary for serialization
        
        Returns:
            Dictionary representation of knowledge state
        """
        return {
            "known_facts": self.known_facts,
            "unknown_aspects": self.unknown_aspects,
            "confidence_levels": self.confidence_levels,
            "detected_conflicts": self.detected_conflicts,
            "document_sources": self.document_sources,
            "cited_documents": list(self.cited_documents),
            "interactions": self.interactions,
            "reasoning_chains": self.reasoning_chains,
            "current_session_id": self.current_session_id,
            "last_query_timestamp": self.last_query_timestamp,
            "response_metadata": self.response_metadata
        }
    
    def from_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load knowledge state from dictionary
        
        Args:
            state_dict: Dictionary representation of knowledge state
        """
        self.known_facts = state_dict.get("known_facts", [])
        self.unknown_aspects = state_dict.get("unknown_aspects", [])
        self.confidence_levels = state_dict.get("confidence_levels", {})
        self.detected_conflicts = state_dict.get("detected_conflicts", [])
        self.document_sources = state_dict.get("document_sources", {})
        self.cited_documents = set(state_dict.get("cited_documents", []))
        self.interactions = state_dict.get("interactions", [])
        self.reasoning_chains = state_dict.get("reasoning_chains", [])
        self.current_session_id = state_dict.get("current_session_id", self.current_session_id)
        self.last_query_timestamp = state_dict.get("last_query_timestamp")
        self.response_metadata = state_dict.get("response_metadata", {})
        
        logger.info("Loaded knowledge state from dictionary")
    
    def update_from_context(self, document_metadata_list: List[Dict[str, Any]], query: str) -> None:
        """
        Update knowledge state based on retrieved context documents
        
        Args:
            document_metadata_list: List of document metadata dictionaries
            query: The user query
        """
        logger.info(f"Updating knowledge state from context with {len(document_metadata_list)} documents")
        
        # Extract key information from document metadata
        for metadata in document_metadata_list:
            doc_id = metadata.get("document_id", "unknown")
            source = metadata.get("file_name", metadata.get("source", "unknown"))
            
            # Store document source info
            self.document_sources[doc_id] = {
                "source": source,
                "confidence": metadata.get("confidence_score", 0.8),
                "creation_date": metadata.get("creation_date", None),
                "accessed_timestamp": datetime.now().isoformat(),
                "retrieved_for_query": query
            }
            
        # Identify potential knowledge gaps related to the query
        gaps = self.identify_knowledge_gaps(query)
        if gaps:
            self.update_unknown_aspects(gaps)
            
        logger.info("Knowledge state updated from context")
    
    def update_from_response(self, response: str, sources: List[Dict[str, Any]]) -> None:
        """
        Update knowledge state based on generated response
        
        Args:
            response: The generated response text
            sources: List of source information dictionaries
        """
        logger.info("Updating knowledge state from response")
        
        # Extract potential facts from response
        extracted_facts = self._extract_facts_from_text(response)
        
        # Add source attribution to facts
        cited_sources = []
        for fact in extracted_facts:
            # Assign the most relevant source
            for source in sources:
                source_id = source.get("id", "unknown")
                fact["source"] = source_id
                fact["confidence"] = source.get("confidence", 0.8)
                cited_sources.append(source_id)
                break
                
            fact["type"] = "response_fact"
            
        # Add unique, non-duplicate facts
        self._update_facts(extracted_facts)
        
        # Track cited documents
        for source in sources:
            source_id = source.get("id", "unknown")
            self.cited_documents.add(source_id)
        
        # Update response metadata
        self.response_metadata = {
            "generated_at": datetime.now().isoformat(),
            "response_length": len(response),
            "sources_count": len(sources),
            "extracted_facts": len(extracted_facts)
        }
        
        logger.info(f"Knowledge state updated with {len(extracted_facts)} facts from response")
    
    def _extract_facts_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract key facts from any text content"""
        facts = []
        
        # Simple approach: split by sentences and filter for fact-like statements
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
                
            # Check if sentence likely contains a fact
            if self._is_likely_fact(sentence):
                fact = {
                    "text": sentence,
                    "extracted_at": datetime.now().isoformat(),
                    "type": "extracted_fact"
                }
                facts.append(fact)
                
        return facts
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current knowledge state for reporting
        
        Returns:
            Knowledge state dictionary with relevant insights
        """
        # Return a condensed version of the state
        return {
            "facts_count": len(self.known_facts),
            "recent_facts": self.known_facts[-3:] if self.known_facts else [],
            "unknown_aspects": self.unknown_aspects[:5],
            "document_sources": len(self.document_sources),
            "conflicts": len(self.detected_conflicts),
            "interactions": len(self.interactions),
            "confidence": self.confidence_levels
        }
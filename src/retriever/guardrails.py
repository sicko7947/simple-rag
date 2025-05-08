"""
Guardrails module for fact-checking and validation
"""
import os
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from langchain.schema import Document
from src.language_model.model_interface import ModelInterface

logger = logging.getLogger(__name__)

class GuardrailsValidator:
    """Validates and fact-checks information using guardrails"""
    
    def __init__(self):
        self.enabled = os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true"
        self.conflict_threshold = float(os.getenv("CONFLICT_THRESHOLD", "0.8"))  # Similarity threshold for conflict detection
        self.fact_check_enabled = os.getenv("ENABLE_FACT_CHECK", "true").lower() == "true"
        self.risk_rules_enabled = os.getenv("ENABLE_RISK_RULES", "true").lower() == "true"
        
        try:
            # Initialize model for semantic evaluation if needed
            if self.fact_check_enabled:
                self.model_interface = ModelInterface()
        except Exception as e:
            logger.warning(f"Failed to initialize ModelInterface for fact checking: {str(e)}")
            self.fact_check_enabled = False
        
        # Initialize risk rules
        self.risk_rules = self._initialize_risk_rules()
        
    def validate_context(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Validate and filter retrieved documents
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Tuple of filtered documents and validation metadata
        """
        if not self.enabled or not documents:
            return documents, {"guardrails_applied": False}
            
        logger.info(f"Applying guardrails to {len(documents)} documents")
        
        # Step 1: Apply conflict detection
        filtered_docs, conflicts = self._detect_conflicts(documents)
        
        # Step 2: Apply risk rules filtering
        if self.risk_rules_enabled:
            filtered_docs, risk_flags = self._apply_risk_rules(filtered_docs)
        else:
            risk_flags = []
            
        # Step 3: Update confidence scores based on validation results
        scored_docs = self._apply_confidence_scoring(filtered_docs)
        
        # Step 4: Cross-verify facts across documents
        if self.fact_check_enabled and len(scored_docs) > 1:
            verified_docs, verification_results = self._cross_verify_facts(scored_docs)
        else:
            verified_docs = scored_docs
            verification_results = {"performed": False}
        
        validation_metadata = {
            "guardrails_applied": True,
            "original_count": len(documents),
            "filtered_count": len(verified_docs),
            "conflicts_detected": conflicts,
            "risk_flags": risk_flags,
            "fact_verification": verification_results,
            "validation_level": "full" if self.fact_check_enabled else "basic"
        }
        
        logger.info(f"Guardrails applied: filtered {len(documents) - len(verified_docs)} documents")
        return verified_docs, validation_metadata
        
    def _detect_conflicts(self, documents: List[Document]) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        Detect conflicting information across documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of filtered documents and list of detected conflicts
        """
        conflicts = []
        excluded_docs = set()
        
        # Extract claims from documents
        doc_claims = {}
        
        for i, doc in enumerate(documents):
            # Extract key claims/facts from the document
            claims = self._extract_claims(doc)
            if claims:
                doc_claims[i] = claims
        
        # Compare claims across documents to detect conflicts
        for i, claims_i in doc_claims.items():
            for j, claims_j in doc_claims.items():
                if i >= j:  # Skip self-comparison and avoid duplicate comparisons
                    continue
                    
                # Compare overlapping claims
                for claim_i in claims_i:
                    for claim_j in claims_j:
                        if self._are_claims_related(claim_i, claim_j):
                            if self._are_claims_conflicting(claim_i, claim_j):
                                # Record the conflict
                                conflict = {
                                    "doc1_id": documents[i].metadata.get("document_id", f"doc_{i}"),
                                    "doc2_id": documents[j].metadata.get("document_id", f"doc_{j}"),
                                    "claim1": claim_i,
                                    "claim2": claim_j,
                                    "severity": "high" if "high_confidence" in claim_i and "high_confidence" in claim_j else "medium"
                                }
                                conflicts.append(conflict)
                                
                                # If both documents have high confidence in their claims,
                                # exclude the one with lower overall confidence
                                if "high_confidence" in claim_i and "high_confidence" in claim_j:
                                    conf_i = documents[i].metadata.get("confidence_score", 0.5)
                                    conf_j = documents[j].metadata.get("confidence_score", 0.5)
                                    
                                    if conf_i < conf_j:
                                        excluded_docs.add(i)
                                    else:
                                        excluded_docs.add(j)
        
        # Filter out excluded documents
        filtered_documents = [doc for i, doc in enumerate(documents) if i not in excluded_docs]
        
        # If no filtering occurred, return the original documents
        if len(filtered_documents) == len(documents):
            return documents, conflicts
            
        return filtered_documents, conflicts
    
    def _extract_claims(self, document: Document) -> List[Dict[str, Any]]:
        """
        Extract key claims/facts from a document
        
        Args:
            document: Document object
            
        Returns:
            List of extracted claims with metadata
        """
        # In a real implementation, this would use NLP techniques to extract claims
        # For this implementation, we'll use a simplified approach
        
        claims = []
        content = document.page_content
        
        # Simple pattern matching for statements of fact
        # Look for sentences with factual indicators
        fact_patterns = [
            r"(.*?is\s+(?:definitely|certainly|absolutely)\s+.*?\.)",
            r"(.*?will\s+(?:always|never)\s+.*?\.)",
            r"(.*?(?:all|every|no)\s+.*?\.)",
            r"(.*?(?:must|should|have to)\s+.*?\.)",
            r"(.*?(?:increased|decreased|improved|reduced)\s+by\s+\d+%.*?\.)",
            r"(.*?as\s+of\s+\d{4}.*?\.)"
        ]
        
        for pattern in fact_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                claim = {
                    "text": match.strip(),
                    "source": document.metadata.get("file_name", "unknown"),
                    "high_confidence": "certainly" in match.lower() or "definitely" in match.lower()
                }
                claims.append(claim)
                
        return claims
    
    def _are_claims_related(self, claim1: Dict[str, Any], claim2: Dict[str, Any]) -> bool:
        """Check if two claims are about the same topic"""
        # In a real implementation, this would use semantic similarity
        # For simplicity, we'll use basic text overlap
        
        text1 = claim1["text"].lower()
        text2 = claim2["text"].lower()
        
        # Extract keywords from each text
        keywords1 = set(re.findall(r'\b\w{4,}\b', text1))  # Words of at least 4 chars
        keywords2 = set(re.findall(r'\b\w{4,}\b', text2))
        
        # Calculate overlap ratio
        if not keywords1 or not keywords2:
            return False
            
        overlap = len(keywords1.intersection(keywords2))
        overlap_ratio = overlap / min(len(keywords1), len(keywords2))
        
        return overlap_ratio > 0.4  # At least 40% keyword overlap
    
    def _are_claims_conflicting(self, claim1: Dict[str, Any], claim2: Dict[str, Any]) -> bool:
        """Check if two related claims are conflicting"""
        # In a real implementation, this would use a more sophisticated 
        # approach to identify logical contradictions
        
        text1 = claim1["text"].lower()
        text2 = claim2["text"].lower()
        
        # Simple contradiction detection based on opposing terms
        contradiction_pairs = [
            ("increase", "decrease"),
            ("higher", "lower"),
            ("more", "less"),
            ("always", "never"),
            ("must", "cannot"),
            ("all", "none"),
            ("positive", "negative")
        ]
        
        for term1, term2 in contradiction_pairs:
            if (term1 in text1 and term2 in text2) or (term2 in text1 and term1 in text2):
                return True
                
        return False
    
    def _initialize_risk_rules(self) -> List[Dict[str, Any]]:
        """Initialize risk rules for content filtering"""
        # These are example rules that might be used in a compliance context
        return [
            {
                "name": "financial_advice",
                "pattern": r"(?:invest|buy|sell).*(?:stock|bond|security|securities)",
                "action": "flag",
                "risk_level": "high"
            },
            {
                "name": "medical_advice",
                "pattern": r"(?:diagnose|treat|cure|prevent).*(?:disease|condition|illness)",
                "action": "flag",
                "risk_level": "high"
            },
            {
                "name": "legal_advice",
                "pattern": r"(?:should|must|legally required).*(?:lawsuit|court|legal)",
                "action": "flag",
                "risk_level": "medium"
            },
            {
                "name": "pii_data",
                "pattern": r"\b(?:\d{3}[-.]?\d{2}[-.]?\d{4}|(?:\d{4}[- ]?){3}\d{4}|\d{9})\b",  # SSN or credit card pattern
                "action": "exclude",
                "risk_level": "critical"
            }
        ]
    
    def _apply_risk_rules(self, documents: List[Document]) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        Apply risk rules filtering to documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of filtered documents and list of risk flags
        """
        if not self.risk_rules_enabled:
            return documents, []
            
        filtered_docs = []
        risk_flags = []
        
        for doc in documents:
            content = doc.page_content
            exclude_doc = False
            
            # Check content against each risk rule
            for rule in self.risk_rules:
                if re.search(rule["pattern"], content, re.IGNORECASE):
                    # Create risk flag record
                    flag = {
                        "document_id": doc.metadata.get("document_id", "unknown"),
                        "rule_name": rule["name"],
                        "risk_level": rule["risk_level"],
                        "action_taken": rule["action"]
                    }
                    risk_flags.append(flag)
                    
                    # If rule action is 'exclude', mark document for exclusion
                    if rule["action"] == "exclude":
                        exclude_doc = True
                        break
            
            # Add document to filtered list if not excluded
            if not exclude_doc:
                filtered_docs.append(doc)
        
        return filtered_docs, risk_flags
    
    def _apply_confidence_scoring(self, documents: List[Document]) -> List[Document]:
        """
        Apply confidence scoring to documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Documents with updated confidence scores
        """
        for doc in documents:
            # Start with the base confidence score
            base_confidence = doc.metadata.get("confidence_score", 1.0)
            
            # Adjust confidence based on various factors
            
            # Age penalty: newer documents are more trustworthy
            age_factor = 1.0  # Default no penalty
            if "creation_date" in doc.metadata and doc.metadata["creation_date"]:
                try:
                    # Simple age penalty based on creation year
                    creation_year = int(doc.metadata["creation_date"].split("-")[0])
                    current_year = 2023  # Use actual current year in production
                    years_old = max(0, current_year - creation_year)
                    
                    # Apply mild penalty for older documents
                    # 0-2 years: no penalty
                    # 3-5 years: 5% penalty
                    # 6+ years: 10% + 1% per year penalty
                    if years_old <= 2:
                        age_factor = 1.0
                    elif years_old <= 5:
                        age_factor = 0.95
                    else:
                        age_factor = max(0.5, 0.9 - (0.01 * (years_old - 5)))
                except (ValueError, IndexError):
                    pass
                
            # Source reliability factor
            source_factor = 1.0  # Default full reliability
            if "file_name" in doc.metadata:
                file_name = doc.metadata["file_name"].lower()
                
                # Example source reliability factors
                if any(x in file_name for x in ["official", "report", "whitepaper"]):
                    source_factor = 1.05  # Bonus for official sources
                elif any(x in file_name for x in ["blog", "forum", "comment"]):
                    source_factor = 0.9  # Penalty for less formal sources
            
            # Update confidence score
            adjusted_confidence = min(1.0, base_confidence * age_factor * source_factor)
            doc.metadata["confidence_score"] = adjusted_confidence
            
            # Add confidence factors for transparency
            doc.metadata["confidence_factors"] = {
                "base": base_confidence,
                "age_factor": age_factor,
                "source_factor": source_factor
            }
            
        return documents
    
    def _cross_verify_facts(self, documents: List[Document]) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Cross-verify facts across multiple documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of verified documents and verification results
        """
        if not self.fact_check_enabled or len(documents) < 2:
            return documents, {"performed": False}
            
        logger.info("Performing cross-document fact verification")
        
        # Extract key facts from all documents
        all_facts = {}
        for i, doc in enumerate(documents):
            facts = self._extract_facts(doc.page_content)
            if facts:
                all_facts[i] = facts
        
        # Track fact verification results
        verification_results = {
            "performed": True,
            "total_facts": sum(len(facts) for facts in all_facts.values()),
            "verified_facts": 0,
            "contradicted_facts": 0,
            "unverified_facts": 0
        }
        
        # Cross-verify facts
        for i, facts_i in all_facts.items():
            for fact in facts_i:
                # Check if this fact is verified by other documents
                verified = False
                contradicted = False
                
                for j, facts_j in all_facts.items():
                    if i == j:  # Skip self-verification
                        continue
                        
                    for other_fact in facts_j:
                        if self._are_facts_similar(fact, other_fact):
                            verified = True
                            break
                        elif self._are_facts_contradicting(fact, other_fact):
                            contradicted = True
                            break
                
                # Update verification counts
                if contradicted:
                    verification_results["contradicted_facts"] += 1
                elif verified:
                    verification_results["verified_facts"] += 1
                else:
                    verification_results["unverified_facts"] += 1
        
        # No need to filter documents here, just return verification stats
        return documents, verification_results
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text"""
        # Simple fact extraction - in a real implementation this would use NLP
        # Here we'll just use sentences that have fact indicators
        
        sentences = re.split(r'[.!?]', text)
        facts = []
        
        fact_indicators = [
            r"\b(?:is|was|are|were)\b",
            r"\b(?:has|have|had)\b",
            r"\b(?:percent|%)\b",
            r"\b(?:in \d{4})\b",
            r"\b(?:increased|decreased)\b"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            for indicator in fact_indicators:
                if re.search(indicator, sentence, re.IGNORECASE):
                    facts.append(sentence)
                    break
                    
        return facts
    
    def _are_facts_similar(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are semantically similar"""
        # Simple implementation - in a real system would use semantic similarity
        
        # Normalize and tokenize
        words1 = set(re.findall(r'\b\w+\b', fact1.lower()))
        words2 = set(re.findall(r'\b\w+\b', fact2.lower()))
        
        # Calculate word overlap
        common_words = words1.intersection(words2)
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return False
            
        similarity = len(common_words) / len(words1.union(words2))
        return similarity > 0.6  # Consider similar if 60% word overlap
    
    def _are_facts_contradicting(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are contradicting each other"""
        # Simple negation detection - would use more sophisticated NLP in production
        
        # First check if they're about the same topic
        if not self._are_facts_similar(fact1, fact2):
            return False
            
        # Then look for opposing claims
        negations = ["not", "no", "never", "isn't", "aren't", "wasn't", "weren't", "doesn't", "don't"]
        
        has_negation1 = any(neg in fact1.lower().split() for neg in negations)
        has_negation2 = any(neg in fact2.lower().split() for neg in negations)
        
        # If one fact has negation and the other doesn't, they might be contradicting
        return has_negation1 != has_negation2
        
    def fact_check_response(self, response: str, context_docs: List[Document]) -> Dict[str, Any]:
        """
        Fact check a generated response against source documents
        
        Args:
            response: Generated response text
            context_docs: Source documents used for generation
            
        Returns:
            Fact checking results
        """
        if not self.fact_check_enabled:
            return {
                "fact_checked": False,
                "reason": "Fact checking disabled"
            }
            
        logger.info("Fact checking generated response")
        
        # Extract claims from the response
        response_claims = self._extract_response_claims(response)
        
        # Track fact checking results
        supported_claims = []
        unsupported_claims = []
        
        # For each claim, check if it's supported by the source documents
        for claim in response_claims:
            support_found = False
            supporting_docs = []
            
            for doc in context_docs:
                support_score = self._calculate_claim_support(claim, doc.page_content)
                if support_score > 0.7:  # Threshold for support
                    support_found = True
                    supporting_docs.append({
                        "id": doc.metadata.get("document_id", "unknown"),
                        "file_name": doc.metadata.get("file_name", "unknown"),
                        "support_score": support_score
                    })
                    
            if support_found:
                supported_claims.append({
                    "claim": claim,
                    "supporting_docs": supporting_docs
                })
            else:
                unsupported_claims.append({
                    "claim": claim,
                    "severity": "high" if "certainly" in claim.lower() or "definitely" in claim.lower() else "medium"
                })
                
        # Calculate factuality score
        if response_claims:
            factuality_score = len(supported_claims) / len(response_claims)
        else:
            factuality_score = 1.0  # No claims to check
            
        return {
            "fact_checked": True,
            "factuality_score": factuality_score,
            "claim_count": len(response_claims),
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims
        }
        
    def _extract_response_claims(self, response: str) -> List[str]:
        """Extract factual claims from the response"""
        # Similar to _extract_facts but specifically for responses
        
        sentences = re.split(r'[.!?]', response)
        claims = []
        
        # Skip sentences that are clearly not claims
        skip_patterns = [
            r"^\s*I don't know",
            r"^\s*I'm not sure",
            r"^\s*It's unclear",
            r"^\s*The context doesn't"
        ]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Skip non-claims
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in skip_patterns):
                continue
                
            # Include sentences with specific claim indicators
            if (re.search(r"\b(?:is|was|are|were|has|have|had)\b", sentence, re.IGNORECASE) and
                len(sentence.split()) > 5):  # Avoid very short statements
                claims.append(sentence)
                
        return claims
        
    def _calculate_claim_support(self, claim: str, doc_content: str) -> float:
        """
        Calculate how well a document supports a claim
        
        Args:
            claim: Claim text
            doc_content: Document content
            
        Returns:
            Support score between 0 and 1
        """
        # Simplified support calculation - in a real system would use NLI/entailment
        
        # Normalize and tokenize
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        doc_words = set(re.findall(r'\b\w+\b', doc_content.lower()))
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by"}
        claim_words = claim_words - stop_words
        
        # Calculate word coverage
        covered_words = sum(1 for word in claim_words if word in doc_words)
        
        # Calculate support score
        if not claim_words:
            return 0.0
            
        return covered_words / len(claim_words)
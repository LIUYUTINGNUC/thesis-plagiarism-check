"""Knowledge-graph construction and matching for idea-plagiarism detection.

Builds a directed graph of entities and their co-occurrence relations from
academic text, then compares two graphs to detect structural similarity in
arguments (idea plagiarism).
"""

from __future__ import annotations

import re
from collections import Counter

import networkx as nx
import numpy as np

# ======================================================================
# Domain terminology dictionaries
# ======================================================================

DOMAIN_TERMS: dict[str, list[str]] = {
    "medicine": [
        "clinical trial",
        "diagnosis",
        "treatment",
        "pathogenesis",
        "etiology",
        "prognosis",
        "symptom",
        "syndrome",
        "biomarker",
        "therapy",
        "patient",
        "surgery",
        "medication",
        "infection",
        "inflammation",
        "chronic disease",
        "acute",
        "epidemiology",
        "mortality",
        "morbidity",
        "gene expression",
        "protein",
        "cell",
        "tissue",
        "organ",
        "immune response",
        "antibody",
        "pathogen",
        "virus",
        "bacteria",
        "metabolism",
        "pharmacology",
        "toxicology",
        "genetics",
        "genomics",
        "clinical outcome",
        "risk factor",
        "prevalence",
        "incidence",
        "diagnostic",
    ],
    "computer_science": [
        "algorithm",
        "data structure",
        "machine learning",
        "deep learning",
        "neural network",
        "artificial intelligence",
        "database",
        "system architecture",
        "distributed system",
        "cloud computing",
        "encryption",
        "authentication",
        "protocol",
        "optimization",
        "classification",
        "regression",
        "clustering",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "computational complexity",
        "time complexity",
        "space complexity",
        "graph",
        "tree",
        "sorting",
        "search",
        "recursion",
        "dynamic programming",
        "greedy algorithm",
        "api",
        "microservice",
        "container",
        "virtualization",
        "parallel computing",
        "data mining",
        "big data",
        "stream processing",
        "fault tolerance",
        "load balancing",
    ],
    "humanities": [
        "discourse",
        "narrative",
        "ideology",
        "subjectivity",
        "identity",
        "culture",
        "modernity",
        "postmodernism",
        "phenomenology",
        "hermeneutics",
        "epistemology",
        "ontology",
        "aesthetics",
        "ethics",
        "dialectic",
        "semiotics",
        "structuralism",
        "deconstruction",
        "paradigm",
        "hegemony",
        "discourse analysis",
        "textual analysis",
        "critical theory",
        "cultural studies",
        "gender",
        "class",
        "race",
        "colonialism",
        "nationalism",
        "globalization",
        "representation",
        "power",
        "knowledge",
        "subject",
        "agency",
        "canon",
        "intertextuality",
        "historicism",
        "materialism",
        "idealism",
    ],
}

DEFAULT_TERMS: list[str] = [
    "analysis",
    "theory",
    "methodology",
    "framework",
    "approach",
    "system",
    "model",
    "process",
    "structure",
    "function",
    "data",
    "result",
    "evidence",
    "concept",
    "hypothesis",
    "research",
    "study",
    "experiment",
    "observation",
    "conclusion",
    "factor",
    "variable",
    "parameter",
    "criterion",
    "mechanism",
    "correlation",
    "causation",
    "distribution",
    "pattern",
    "trend",
]

_ENGLISH_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "although",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "them", "we", "you", "who", "whom", "which", "what",
}


# ======================================================================
# Public helpers
# ======================================================================


def load_domain_terms(discipline: str) -> list[str]:
    """Load domain-specific terminology for a given academic discipline.

    Built-in dictionaries are provided for:
        ``medicine``, ``computer_science``, ``humanities``.

    Falls back to a set of general academic terms for unknown disciplines.

    Args:
        discipline: Discipline name (case-insensitive, spaces become underscores).

    Returns:
        List of key terms.
    """
    key = discipline.lower().replace(" ", "_")
    return DOMAIN_TERMS.get(key, DEFAULT_TERMS)


# ======================================================================
# DomainKnowledgeGraph
# ======================================================================


class DomainKnowledgeGraph:
    """Domain-specific knowledge graph for detecting idea plagiarism.

    Builds a directed graph of entities (nodes) and their co-occurrence
    relations (edges) from academic text, then compares graphs to detect
    structural similarity in arguments.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._text: str = ""

    # ------------------------------------------------------------------
    # Public construction API
    # ------------------------------------------------------------------

    def build_from_text(self, text: str, domain_terms: list[str]) -> None:
        """Build the knowledge graph from raw text and domain terminology.

        Pipeline:
            1. Extract entities matching domain terms plus TF-IDF keywords.
            2. Build co-occurrence-based relation triples.
            3. Populate the internal ``nx.DiGraph``.

        Args:
            text: Source text to analyse.
            domain_terms: Domain-specific terms for entity extraction.
        """
        self._text = text

        entities = self.extract_entities(text, domain_terms)
        relations = self.build_relations(entities, text)

        # Reset graph
        self._graph.clear()

        # Add entity nodes with frequency attribute
        entity_counts = Counter(entities)
        for entity, count in entity_counts.items():
            self._graph.add_node(entity, frequency=count, type="entity")

        # Add co-occurrence edges (aggregate weight)
        edge_weights: dict[tuple[str, str], float] = {}
        for subj, _pred, obj in relations:
            if subj == obj or not (self._graph.has_node(subj) and self._graph.has_node(obj)):
                continue
            key = (subj, obj)
            edge_weights[key] = edge_weights.get(key, 0.0) + 1.0

        for (subj, obj), weight in edge_weights.items():
            self._graph.add_edge(subj, obj, predicate="co_occurs_with", weight=weight)

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self, text: str, domain_terms: list[str]) -> list[str]:
        """Extract domain entities from text.

        Two-pass strategy:
            - **Pass 1:** Match domain terms that appear (case-insensitive).
            - **Pass 2:** Frequency-based keyword extraction for additional terms.

        Returns a deduplicated list preserving the first-seen order.

        Args:
            text: Source text.
            domain_terms: Domain terminology list.

        Returns:
            Deduplicated list of entity strings.
        """
        text_lower = text.lower()
        entities: list[str] = []

        # --- Pass 1: domain term matching ----------------------------
        for term in domain_terms:
            if term.lower() in text_lower:
                entities.append(term)

        # --- Pass 2: frequency-based keywords ------------------------
        entities.extend(self._extract_keywords(text))

        # Deduplicate (case-insensitive), preserve order
        seen: set[str] = set()
        deduped: list[str] = []
        for entity in entities:
            key = entity.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(entity)

        return deduped

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract additional keywords via frequency analysis.

        Returns words (3+ characters) that appear at least twice, excluding
        common English stopwords.  Limited to the top 20 candidates.
        """
        words = re.findall(r"[a-zA-Z]+(?:[-_][a-zA-Z]+)*", text)
        counts: Counter[str] = Counter()
        for w in words:
            w_lower = w.lower()
            if w_lower not in _ENGLISH_STOPWORDS and len(w_lower) > 2:
                counts[w_lower] += 1

        # Preserve original casing of the first occurrence
        seen_case = {}
        for w in words:
            wl = w.lower()
            if wl not in seen_case and wl in counts:
                seen_case[wl] = w

        return [
            seen_case[word]
            for word, cnt in counts.most_common(20)
            if cnt >= 2 and word in seen_case
        ]

    # ------------------------------------------------------------------
    # Relation building
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on Chinese/English punctuation."""
        parts = re.split(r"[。！？.!?\n]+", text)
        return [s.strip() for s in parts if s.strip()]

    def build_relations(self, entities: list[str], text: str) -> list[tuple[str, str, str]]:
        """Build entity-relation triples based on co-occurrence.

        For every pair of distinct entities that co-occur within a sliding
        window of 5 sentences, a triple ``(entity1, "co_occurs_with", entity2)``
        is produced.

        Args:
            entities: List of entity strings.
            text: Source text to analyse.

        Returns:
            List of ``(subject, predicate, object)`` triples.
        """
        if len(entities) < 2:
            return []

        sentences = self._split_sentences(text)
        entity_set = set(e.lower() for e in entities)  # noqa: F841

        # Map sentence index -> entities present in that sentence
        sentence_entities: list[list[str]] = []
        for sent in sentences:
            s_lower = sent.lower()
            present = [e for e in entities if e.lower() in s_lower]
            sentence_entities.append(present)

        # Sliding window of 5 sentences
        relations_set: set[tuple[str, str, str]] = set()
        window_size = 5

        for i in range(len(sentence_entities)):
            # Collect all entities in window [i, i + window_size)
            window_ents: set[str] = set()
            for j in range(i, min(i + window_size, len(sentence_entities))):
                window_ents.update(sentence_entities[j])

            window_list = list(window_ents)
            for a in range(len(window_list)):
                for b in range(a + 1, len(window_list)):
                    e1, e2 = window_list[a], window_list[b]
                    if e1 != e2:
                        relations_set.add((e1, "co_occurs_with", e2))

        return list(relations_set)

    # ------------------------------------------------------------------
    # Graph matching
    # ------------------------------------------------------------------

    def match_graph(self, query_graph: DomainKnowledgeGraph) -> dict:
        """Match this (reference) graph against a query graph.

        Returns a dict with four metrics, each in ``[0, 1]``:

        - ``entity_overlap``: Jaccard similarity of entity node sets.
        - ``relation_similarity``: Jaccard similarity of relation triple sets.
        - ``graph_edit_distance``: Normalised symmetric-difference distance.
        - ``overall_score``: Weighted combination of the above (weights:
          0.35 / 0.35 / 0.30).

        Args:
            query_graph: Graph to compare against.

        Returns:
            Dictionary of similarity / distance metrics.
        """
        nodes_self = set(self._graph.nodes())
        nodes_query = set(query_graph._graph.nodes())

        # --- Entity overlap (Jaccard) ---------------------------------
        if not nodes_self and not nodes_query:
            entity_overlap = 1.0
        elif not nodes_self or not nodes_query:
            entity_overlap = 0.0
        else:
            intersection = nodes_self & nodes_query
            union = nodes_self | nodes_query
            entity_overlap = len(intersection) / len(union)

        # --- Relation similarity (Jaccard) ----------------------------
        rels_self = self._relation_set()
        rels_query = query_graph._relation_set()

        if not rels_self and not rels_query:
            relation_similarity = 1.0
        elif not rels_self or not rels_query:
            relation_similarity = 0.0
        else:
            intersection = rels_self & rels_query
            union = rels_self | rels_query
            relation_similarity = len(intersection) / len(union)

        # --- Normalised graph-edit distance ---------------------------
        edit_distance = self._normalized_edit_distance(query_graph)

        # --- Weighted overall score -----------------------------------
        overall_score = 0.35 * entity_overlap + 0.35 * relation_similarity + 0.30 * (1.0 - edit_distance)

        return {
            "entity_overlap": float(entity_overlap),
            "relation_similarity": float(relation_similarity),
            "graph_edit_distance": float(edit_distance),
            "overall_score": float(np.clip(overall_score, 0.0, 1.0)),
        }

    def _relation_set(self) -> set[tuple[str, str, str]]:
        """Return all relation triples stored in the graph."""
        triples: set[tuple[str, str, str]] = set()
        for u, v, data in self._graph.edges(data=True):
            pred = data.get("predicate", "co_occurs_with")
            triples.add((u, pred, v))
        return triples

    def _normalized_edit_distance(self, other: DomainKnowledgeGraph) -> float:
        """Approximate normalised graph edit distance.

        Uses symmetric difference of node and edge sets as a fast proxy for
        the true (NP-hard) graph-edit distance.
        """
        max_nodes = max(self.node_count, other.node_count)
        if max_nodes == 0:
            return 0.0

        nodes_self = set(self._graph.nodes())
        nodes_other = set(other._graph.nodes())
        node_diff = len(nodes_self.symmetric_difference(nodes_other))

        edges_self = set(self._graph.edges())
        edges_other = set(other._graph.edges())
        edge_diff = len(edges_self.symmetric_difference(edges_other))

        total_possible = max_nodes + (max_nodes * (max_nodes - 1) / 2.0)
        if total_possible == 0:
            return 0.0

        total_diff = node_diff + edge_diff
        return min(total_diff / (2.0 * total_possible), 1.0)

    def argument_flow_similarity(self, other: DomainKnowledgeGraph) -> float:
        """Compare the argument structure of two graphs.

        Uses the normalised graph-edit distance as a proxy:
        ``similarity = 1 - edit_distance``.

        Args:
            other: Another knowledge graph to compare against.

        Returns:
            Similarity score in ``[0, 1]``.
        """
        edit_dist = self._normalized_edit_distance(other)
        return float(np.clip(1.0 - edit_dist, 0.0, 1.0))

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def idea_plagiarism_score(
        self,
        original_text: str,
        suspect_text: str,
        domain: str = "default",
    ) -> dict:
        """Detect idea plagiarism between two texts.

        Pipeline:
            1. Load domain terms for the given discipline.
            2. Build knowledge graphs for both texts.
            3. Match graphs and compute argument-flow similarity.
            4. Return a comprehensive score dict with a verdict.

        Args:
            original_text: The original / source academic text.
            suspect_text: Text to check for idea plagiarism.
            domain: Academic discipline (used to load domain terms).

        Returns:
            Dictionary with keys:
            - ``entity_overlap``
            - ``relation_similarity``
            - ``graph_edit_distance``
            - ``argument_flow_score``
            - ``overall_idea_plagiarism_score``
            - ``verdict``: ``'highly_similar'``, ``'moderately_similar'``, or ``'distinct'``.
        """
        terms = load_domain_terms(domain)

        original_g = DomainKnowledgeGraph()
        original_g.build_from_text(original_text, terms)

        suspect_g = DomainKnowledgeGraph()
        suspect_g.build_from_text(suspect_text, terms)

        match = original_g.match_graph(suspect_g)
        arg_flow = original_g.argument_flow_similarity(suspect_g)

        overall = float(np.clip(0.4 * match["overall_score"] + 0.3 * arg_flow, 0.0, 1.0))

        if overall >= 0.7:
            verdict = "highly_similar"
        elif overall >= 0.35:
            verdict = "moderately_similar"
        else:
            verdict = "distinct"

        return {
            "entity_overlap": match["entity_overlap"],
            "relation_similarity": match["relation_similarity"],
            "graph_edit_distance": match["graph_edit_distance"],
            "argument_flow_score": float(arg_flow),
            "overall_idea_plagiarism_score": overall,
            "verdict": verdict,
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Number of nodes (entities) in the graph."""
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Number of edges (relations) in the graph."""
        return self._graph.number_of_edges()

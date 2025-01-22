import torch
from torch_geometric.data import Data
from transformers import AutoProcessor, AutoModel
import networkx as nx
import numpy as np
from datasets import load_dataset
from PIL import Image
import faiss
from typing import List, Dict, Tuple
import logging

class MathVisionGraphRAG:
    def __init__(self, vision_model_name: str = "microsoft/resnet-50", 
                 embedding_dim: int = 2048,
                 k_neighbors: int = 5):
        """
        Initialize the Graph RAG system for math vision problems.
        
        Args:
            vision_model_name: Name of the vision model to use
            embedding_dim: Dimension of the embeddings
            k_neighbors: Number of neighbors for graph construction
        """
        self.processor = AutoProcessor.from_pretrained(vision_model_name)
        self.vision_model = AutoModel.from_pretrained(vision_model_name)
        self.embedding_dim = embedding_dim
        self.k_neighbors = k_neighbors
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.graph = nx.Graph()
        self.image_embeddings = {}
        self.problem_metadata = {}
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_dataset(self, dataset_path: str = "MathLLMs/MathVision"):
        """
        Load and process the MathVision dataset.
        
        Args:
            dataset_path: Path to the dataset on HuggingFace
        """
        self.logger.info(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
        
        for split in dataset.keys():
            for idx, example in enumerate(dataset[split]):
                # Extract image and metadata
                image = Image.open(example['image']).convert('RGB')
                embedding = self._get_image_embedding(image)
                
                # Store embeddings and metadata
                self.image_embeddings[idx] = embedding
                self.problem_metadata[idx] = {
                    'question': example['question'],
                    'solution': example['solution'],
                    'answer': example['answer'],
                    'topic': example['topic']
                }
                
                # Add to FAISS index
                self.index.add(np.array([embedding]))
                
        self._build_graph()
        self.logger.info("Dataset processing complete")

    def _get_image_embedding(self, image: Image) -> np.ndarray:
        """
        Get embedding for an image using the vision model.
        
        Args:
            image: PIL Image
        Returns:
            numpy array of embedding
        """
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vision_model(**inputs)
        return outputs.pooler_output.numpy().flatten()

    def _build_graph(self):
        """
        Build a knowledge graph based on image embeddings and problem metadata.
        """
        self.logger.info("Building knowledge graph")
        
        # Find k-nearest neighbors for each problem
        for idx in self.image_embeddings.keys():
            embedding = self.image_embeddings[idx]
            _, neighbors = self.index.search(np.array([embedding]), self.k_neighbors + 1)
            
            # Add edges between similar problems
            for neighbor_idx in neighbors[0][1:]:  # Skip self
                if neighbor_idx < len(self.problem_metadata):
                    similarity = self._calculate_similarity(embedding, 
                                                         self.image_embeddings[neighbor_idx])
                    
                    self.graph.add_edge(idx, 
                                      neighbor_idx, 
                                      weight=similarity)
                    
                    # Add topic-based edges
                    if (self.problem_metadata[idx]['topic'] == 
                        self.problem_metadata[neighbor_idx]['topic']):
                        self.graph.add_edge(idx, 
                                          neighbor_idx, 
                                          weight=similarity * 1.2)  # Boost similar topics

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        """
        return float(np.dot(emb1, emb2) / 
                    (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def retrieve_similar_problems(self, 
                                query_image: Image, 
                                n_results: int = 3) -> List[Dict]:
        """
        Retrieve similar problems for a query image using graph-based retrieval.
        
        Args:
            query_image: PIL Image of the math problem
            n_results: Number of similar problems to retrieve
            
        Returns:
            List of similar problems with metadata
        """
        # Get embedding for query image
        query_embedding = self._get_image_embedding(query_image)
        
        # Find nearest neighbors
        _, indices = self.index.search(np.array([query_embedding]), n_results * 2)
        
        # Use graph to refine results
        candidates = set()
        for idx in indices[0]:
            if idx < len(self.problem_metadata):
                candidates.add(idx)
                # Add neighbors from graph
                candidates.update(self.graph.neighbors(idx))
        
        # Score candidates using graph structure
        scored_candidates = []
        for candidate_idx in candidates:
            score = self._calculate_similarity(query_embedding, 
                                            self.image_embeddings[candidate_idx])
            
            # Boost score based on graph centrality
            centrality = nx.degree_centrality(self.graph)[candidate_idx]
            score *= (1 + centrality)
            
            scored_candidates.append((score, candidate_idx))
        
        # Sort and return top results
        scored_candidates.sort(reverse=True)
        results = []
        for _, idx in scored_candidates[:n_results]:
            results.append({
                'metadata': self.problem_metadata[idx],
                'similarity_score': score
            })
            
        return results

    def enhance_solution(self, 
                        query_image: Image, 
                        base_solution: str) -> Tuple[str, List[Dict]]:
        """
        Enhance a solution using retrieved similar problems.
        
        Args:
            query_image: PIL Image of the math problem
            base_solution: Initial solution to enhance
            
        Returns:
            Tuple of (enhanced solution, list of similar problems used)
        """
        similar_problems = self.retrieve_similar_problems(query_image)
        
        # Analyze patterns in similar problems
        solution_patterns = []
        for problem in similar_problems:
            solution_patterns.append(problem['metadata']['solution'])
        
        # Enhance solution using patterns
        enhanced_solution = base_solution
        
        # Add relevant solution steps from similar problems
        for pattern in solution_patterns:
            if len(pattern.split()) > len(enhanced_solution.split()) * 1.5:
                # If similar solution is more detailed, incorporate its structure
                enhanced_solution += f"\n\nAlternative approach based on similar problem:\n{pattern}"
        
        return enhanced_solution, similar_problems

    def get_graph_statistics(self) -> Dict:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary of graph statistics
        """
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'average_degree': sum(dict(self.graph.degree()).values()) / 
                            self.graph.number_of_nodes(),
            'clustering_coefficient': nx.average_clustering(self.graph),
            'connected_components': nx.number_connected_components(self.graph)
        }

import torch
import networkx as nx
from PIL import Image
from llama_cpp import Llama
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json
from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class MathVisionGraphRAG:
    def __init__(self, model_path: str = "llava-v1.5-13b"):
        """
        Initialize the MathVision Graph RAG system
        
        Args:
            model_path: Path to the LLaVA model
        """
        # Initialize LLaVA model and processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path)
        
        # Initialize sentence transformer for text embeddings
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize knowledge graph
        self.knowledge_graph = nx.Graph()
        
        # Initialize cache for computed embeddings
        self.embedding_cache = {}

    def build_knowledge_graph(self, dataset_path: str):
        """
        Build knowledge graph from the MathVision dataset
        
        Args:
            dataset_path: Path to the MathVision dataset
        """
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
            
        # Process each problem in the dataset
        for problem in dataset:
            # Create node for the problem
            problem_id = problem['id']
            self.knowledge_graph.add_node(problem_id, 
                                       type='problem',
                                       image_path=problem['image_path'],
                                       question=problem['question'],
                                       solution=problem['solution'])
            
            # Extract mathematical concepts and create concept nodes
            concepts = self._extract_concepts(problem['solution'])
            for concept in concepts:
                if not self.knowledge_graph.has_node(concept):
                    self.knowledge_graph.add_node(concept, type='concept')
                self.knowledge_graph.add_edge(problem_id, concept)

    def _extract_concepts(self, solution: str) -> List[str]:
        """
        Extract mathematical concepts from solution text
        
        Args:
            solution: Solution text
            
        Returns:
            List of mathematical concepts
        """
        # This is a simplified version - in practice, you'd want more sophisticated
        # concept extraction, possibly using a specialized model
        common_math_concepts = [
            'algebra', 'geometry', 'trigonometry', 'calculus',
            'equations', 'functions', 'vectors', 'matrices'
        ]
        
        return [concept for concept in common_math_concepts 
                if concept.lower() in solution.lower()]

    def _compute_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding for text using cached values when possible
        
        Args:
            text: Input text
            
        Returns:
            Text embedding
        """
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.text_encoder.encode(text)
        return self.embedding_cache[text]

    def retrieve_similar_problems(self, 
                                query_image: Image.Image, 
                                query_text: str, 
                                k: int = 3) -> List[Dict]:
        """
        Retrieve similar problems using graph-based retrieval
        
        Args:
            query_image: Input image
            query_text: Input question text
            k: Number of problems to retrieve
            
        Returns:
            List of similar problems with their solutions
        """
        # Get image and text embeddings for the query
        query_embedding = self._compute_embedding(query_text)
        
        similar_problems = []
        
        # Find similar problems based on text similarity and graph structure
        for node in self.knowledge_graph.nodes():
            if self.knowledge_graph.nodes[node]['type'] == 'problem':
                problem_text = self.knowledge_graph.nodes[node]['question']
                problem_embedding = self._compute_embedding(problem_text)
                
                # Compute similarity score
                similarity = np.dot(query_embedding, problem_embedding)
                
                # Get connected concepts
                concepts = [n for n in self.knowledge_graph.neighbors(node)
                          if self.knowledge_graph.nodes[n]['type'] == 'concept']
                
                # Boost similarity score based on shared concepts
                concept_boost = len(concepts) * 0.1
                final_score = similarity + concept_boost
                
                similar_problems.append({
                    'id': node,
                    'score': final_score,
                    'problem': self.knowledge_graph.nodes[node]
                })
        
        # Sort by similarity score and return top k
        similar_problems.sort(key=lambda x: x['score'], reverse=True)
        return similar_problems[:k]

    def solve_problem(self, 
                     image: Image.Image, 
                     question: str) -> Tuple[str, List[Dict]]:
        """
        Solve a math problem using Graph RAG enhanced LLaVA
        
        Args:
            image: Problem image
            question: Problem question
            
        Returns:
            Tuple of (solution, similar problems used)
        """
        # Retrieve similar problems
        similar_problems = self.retrieve_similar_problems(image, question)
        
        # Prepare context from similar problems
        context = "Here are some similar problems and their solutions:\n"
        for prob in similar_problems:
            context += f"Problem: {prob['problem']['question']}\n"
            context += f"Solution: {prob['problem']['solution']}\n\n"
        
        # Prepare prompt for LLaVA
        prompt = f"{context}\nNow solve this problem:\n{question}\n"
        
        # Process image and text with LLaVA
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        # Generate solution
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            temperature=0.7
        )
        
        solution = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return solution, similar_problems

def main():
    # Initialize the system
    math_solver = MathVisionGraphRAG()
    
    # Build knowledge graph from dataset
    math_solver.build_knowledge_graph("path_to_mathvision_dataset.json")
    
    # Example usage
    image = Image.open("path_to_problem_image.jpg")
    question = "What is the area of the triangle shown in the image?"
    
    # Solve the problem
    solution, similar_problems = math_solver.solve_problem(image, question)
    
    print("Solution:", solution)
    print("\nSimilar problems used:")
    for prob in similar_problems:
        print(f"- Problem: {prob['problem']['question']}")

if __name__ == "__main__":
    main()

import time
import json
import numpy as np
import faiss
import networkx as nx
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os

class NeuroSymbolicPipeline:
    def __init__(self, model_path, raw_lore=None, embedding_model='all-MiniLM-L6-v2'):
        print(f"Initializing Neuro-Symbolic Pipeline with model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False
        )
        self.embedder = SentenceTransformer(embedding_model)
        
        # Default Lore (if none provided)
        self.raw_lore = raw_lore if raw_lore else """
        The Elder Stone is an ancient artifact located deep within the Whisper Woods. 
        It emits a faint blue glow used by the High Elves for healing rituals. 
        However, the Whisper Woods are guarded by the Shadow Stalkers, who obey the Dragon of the North. 
        King Alaric founded the Kingdom of Aethelgard in the year 300, but he was betrayed by his brother, Duke Varen, in 402. 
        Duke Varen sought the power of the Void Heart to usurp the throne. 
        Healing potions require Blue Moon Flowers, which only bloom under the light of a full moon in the Silver Glade.
        """
        
        self.setup_advanced_knowledge()

    def extract_triples_with_llm(self, text):
        """Information Extraction: Uses LLM to turn Text -> Graph Triples"""
        prompt = f"<|system|>Extract knowledge graph triples (Subject, Relation, Object) from the text. Return ONLY a JSON list of lists. Example: [[\"Paris\", \"is capital of\", \"France\"]]<|end|>\n<|user|>{text}<|end|><|assistant|>"
        try:
            output = self.llm(prompt, max_tokens=256, temperature=0.1)
            response = output['choices'][0]['text']
            
            # Robust JSON extraction
            match = re.search(r'\[\s*\[.*\]\s*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                triples = json.loads(json_str)
                return triples
            return []
        except Exception as e:
            print(f"LLM Extraction failed: {e}. using heuristic fallback.")
            # Heuristic fallback: Split by key verbs
            triples = []
            sentences = text.split('.')
            for s in sentences:
                if 'located in' in s:
                    parts = s.split('located in')
                    triples.append([parts[0].strip(), 'located in', parts[1].strip()])
                elif 'founded' in s:
                    parts = s.split('founded')
                    triples.append([parts[0].strip(), 'founded', parts[1].strip()])
            return triples

    def setup_advanced_knowledge(self):
        # 1. Recursive Chunking (Advanced RAG)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        )
        self.chunks = splitter.split_text(self.raw_lore)
        print(f"Split Lore into {len(self.chunks)} semantic chunks.")
        
        # Vector DB
        embeddings = self.embedder.encode(self.chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
        # 2. Dynamic Graph Construction (Neuro-Symbolic)
        print("Building Knowledge Graph via LLM Extraction...")
        self.graph = nx.Graph()
        triples = self.extract_triples_with_llm(self.raw_lore)
        for triple in triples:
            if len(triple) == 3:
                subj, rel, obj = triple
                self.graph.add_edge(subj, obj, relation=rel)
        print(f"Graph constructed with {self.graph.number_of_edges()} Neuro-Symbolic edges.")

    def retrieve_rag(self, query):
        vec = self.embedder.encode([query])
        D, I = self.index.search(vec, k=2)
        return [self.chunks[i] for i in I[0] if i < len(self.chunks)]

    def retrieve_graph(self, query):
        # Heuristic: Find entities in query matching graph nodes
        matches = []
        for node in self.graph.nodes:
            if node.lower() in query.lower():
                # Get 1-hop neighbors
                for neighbor in self.graph.neighbors(node):
                    rel = self.graph.edges[node, neighbor].get('relation', 'related to')
                    matches.append(f"{node} -> {rel} -> {neighbor}")
        return matches

    def generate(self, prompt, config):
        context_stack = []
        
        # Hybrid Retrieval
        if config.get('enable_rag', False):
            rag_hits = self.retrieve_rag(prompt)
            context_stack.append(f"[LORE]: {'; '.join(rag_hits)}")
            
        if config.get('enable_graph', False):
            graph_hits = self.retrieve_graph(prompt)
            if graph_hits:
                context_stack.append(f"[GRAPH]: {'; '.join(graph_hits)}")
        
        context_str = "\n".join(context_stack)
        
        # Prompt using Tuned Instruction Format
        full_prompt = f"### Instruction:\nUse this knowledge: {context_str}\nAnswer: {prompt}\n\n### Response:\n"
        
        start_t = time.perf_counter()
        output = self.llm(full_prompt, max_tokens=150)
        lat = (time.perf_counter() - start_t) * 1000
        
        text = output['choices'][0]['text']
        tk = output['usage']['completion_tokens']
        
        return {'text': text, 'latency_ms': lat, 'tps': tk/(lat/1000) if lat>0 else 0}

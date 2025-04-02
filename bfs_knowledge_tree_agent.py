from litellm import completion
from pydantic import BaseModel, ValidationError
from collections import deque
import json
import os
import re 

# Define the Pydantic model for the BFS tree
class BFSTree(BaseModel):
    bfs_tree: dict[str, list[str]]

# System prompt guiding the LLM's behavior
SYSTEM_PROMPT = """
You are a reasoning agent that traverses a knowledge graph for {topic} using Breadth-First Search (BFS). Your goal is to expand knowledge in a level-by-level manner.

At each step:
1. Start with the current node(s) in the queue.
2. For each node, fetch all directly connected neighbors (related concepts, definitions, or facts).
3. Add unvisited neighbors to the end of the queue.
4. Mark the current node as visited.
5. Output new knowledge gained at each layer before proceeding to the next.

You must avoid repeating already visited concepts and aim to explore as broadly as possible before going deeper. Prioritize clarity, completeness, and explain each relation.

Example:
- Start node: "Photosynthesis"
- Step 1: Expand to ["Chlorophyll", "Sunlight", "Glucose"]
- Step 2: From "Chlorophyll", "Sunlight", and "Glucose", expand the next layer of related concepts.

Continue until a stopping condition is reached or no new concepts remain.

##Output json format:
{{
  "bfs_tree": {{
    "Quantum Physics": ["Wave-Particle Duality", "Uncertainty Principle"],
    "Wave-Particle Duality": ["Double Slit Experiment", "de Broglie Hypothesis"]
  }}
}}
"""

def openai_model_generate_neighbors(node: str, topic: str) -> list:

    system_prompt = SYSTEM_PROMPT.format(topic=topic)

    response = completion(
        model="xai/grok-beta",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"List direct concepts or facts related to '{node}' in JSON format exactly as specified. Respond with valid JSON only."}
        ]
    )
    content = response['choices'][0]['message']['content']

    # Remove markdown code block formatting if present
    content = re.sub(r'^```json|```$', '', content.strip())

    try:
        response_json = json.loads(content.strip())
        if 'bfs_tree' in response_json:
            return response_json['bfs_tree'].get(node, [])
        else:
            print(f"Key 'bfs_tree' not found in the response for node '{node}'.")
            return []
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error for node '{node}':", e)
        print("Content received:", content)
        return []
    except Exception as e:
        print(f"Unexpected error for node '{node}':", e)
        print("Content received:", content)
        return []

def bfs_dynamic(start_node: str, topic: str, max_depth: int = 5) -> None:
    """
    Performs a dynamic BFS traversal starting from the given node up to the specified depth.

    Args:
        start_node (str): The starting concept node.
        topic (str): The topic of the tree.
        max_depth (int): Maximum depth to traverse.
    """
    bfs_tree = {}
    visited = set()
    queue = deque([(start_node, 0)])
    visited.add(start_node)

    while queue:
        current, depth = queue.popleft()
        if depth >= max_depth:
            continue

        neighbors = openai_model_generate_neighbors(current, topic)
        bfs_tree[current] = neighbors

        # Save the BFS tree to JSON after each step
        with open("chemistry_dynamic_bfs_tree.json", "w") as f:
            json.dump({"bfs_tree": bfs_tree}, f, indent=2)

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

        print(f"Step completed for node '{current}', depth {depth}. Current BFS tree saved.")

    print("Dynamic BFS traversal complete. Final tree saved to dynamic_bfs_tree.json.")

if __name__ == "__main__":

    bfs_dynamic("Chemistry History", "Chemistry History")

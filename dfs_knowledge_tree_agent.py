from litellm import completion
import json
import re

SYSTEM_PROMPT = """
You are a reasoning agent that traverses a knowledge graph using Depth-First Search (DFS). Your goal is to expand knowledge by exploring each branch as deeply as possible before backtracking.

At each step:
1. Start with the current node at the top of the stack.
2. Fetch all directly connected neighbors (related concepts, definitions, or facts).
3. Add unvisited neighbors to the stack.
4. Mark the current node as visited.
5. Output new knowledge gained before proceeding to the next node.

Avoid repeating already visited concepts and aim to explore each branch thoroughly before moving to another. Prioritize clarity and completeness, explaining each relation.

Example:
- Start node: "Photosynthesis"
- Step 1: Expand to ["Chlorophyll"]
- Step 2: From "Chlorophyll", expand to ["Thylakoid Membrane"]
- Continue this deep exploration before backtracking to explore other branches.

Continue until a stopping condition is reached or no new concepts remain.

## Output JSON format:
{
  "dfs_tree": {
    "Quantum Physics": ["Wave-Particle Duality"],
    "Wave-Particle Duality": ["Double Slit Experiment"],
    ...
  }
}
"""

def openai_model_generate_neighbors(node: str) -> list:
    response = completion(
        model="xai/grok-beta",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"List direct concepts or facts related to '{node}' in JSON format as specified."}
        ]
    )
    content = response['choices'][0]['message']['content']

    # Remove markdown code block formatting if present
    content = re.sub(r'^```json|```$', '', content.strip())

    try:
        response_json = json.loads(content)
        return response_json["dfs_tree"].get(node, [])
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error for node '{node}':", e)
        print("Content received:", content)
        return []

def dfs_traversal(start_node: str, max_depth: int = 10) -> None:
    dfs_tree = {}
    visited = set()
    stack = [(start_node, 0)]
    visited.add(start_node)

    while stack:
        current, depth = stack.pop()
        if depth >= max_depth:
            continue

        neighbors = openai_model_generate_neighbors(current)
        dfs_tree[current] = neighbors

        # Save the DFS tree to JSON after each step
        with open("dynamic_dfs_tree.json", "w") as f:
            json.dump({"dfs_tree": dfs_tree}, f, indent=2)

        for neighbor in reversed(neighbors):  # Reverse to maintain order similar to recursive DFS
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append((neighbor, depth + 1))

        print(f"Step completed for node '{current}', depth {depth}. Current DFS tree saved.")

    print("Dynamic DFS traversal complete. Final tree saved to dynamic_dfs_tree.json.")

if __name__ == "__main__":
    dfs_traversal("Quantum Physics")

from collections import deque
import json
import os
import re
from typing import Dict, List, Set

from litellm import completion

SYSTEM_PROMPT = """
You are an AI assistant tasked with creating comprehensive lecture materials using a Depth-First Search (DFS) approach. Your goal is to explore each topic branch thoroughly before moving to the next, ensuring clarity and completeness in the content.

At each step:
1. **Topic Exploration:** Begin with the main topic and delve deeply into each subtopic.
2. **Class Development:** For each subtopic, generate detailed class modules covering all relevant aspects.
3. **Slide Creation:** Within each class module, create a series of slides that sequentially present the information.
4. **Speaker Notes:** For each slide, provide comprehensive speaker notes that elaborate on the slide content, offering examples and explanations.

Continue this process until all subtopics are thoroughly explored or no new information can be generated. Avoid repeating previously covered concepts and ensure each branch is fully developed before backtracking. Prioritize clarity and completeness, explaining each relation and concept in detail.

**Output JSON Format:**
{
  "lecture_materials": {
    "Topic": {
      "Subtopic": {
        "Class Module": {
          "Slide Title": "Speaker Notes",
          ...
        },
        ...
      },
      ...
    }
  }
}
""" 

def openai_model_generate_content(node: str, level: str, structure: Dict) -> Dict:
    """
    Generates content for the given node at the specified level using the OpenAI model.
    """
    prompt = f"Based on the current structure: {json.dumps(structure, indent=2)}, generate {level} for '{node}' in JSON format as specified."
    response = completion(
        model="xai/grok-beta",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    content = response['choices'][0]['message']['content']

    # Remove markdown code block formatting if present
    content = re.sub(r'^```json|```$', '', content.strip())

    try:
        response_json = json.loads(content)
        return response_json.get("lecture_materials", {}).get(node, {})
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error for node '{node}':", e)
        print("Content received:", content)
        return {}
        
def dfs_generate_lecture(topic: str, max_depth: int = 3) -> None:
    lecture_structure: Dict[str, Dict] = {topic: {}}
    visited: Set[str] = set()
    stack = [(topic, 0, lecture_structure[topic])]
    visited.add(topic)

    while stack:
        current, depth, parent_structure = stack.pop()
        if depth >= max_depth:
            continue

        # Assign content generation level based on depth
        if depth == 0:
            level = "subtopics"
        elif depth == 1:
            level = "class modules"
        elif depth == 2:
            level = "slides"
        else:
            level = "speaker notes"

        content = openai_model_generate_content(current, level, lecture_structure)
        print(content)
        parent_structure.update(content)

        # Save the lecture structure to JSON after each step
        with open("lecture_structure.json", "w") as f:
            json.dump({"lecture_materials": lecture_structure}, f, indent=2)

        if depth < max_depth - 1:
            for child in reversed(list(content.keys())):
                if child not in visited:
                    visited.add(child)
                    stack.append((child, depth + 1, parent_structure[child]))

        print(f"Step completed for '{current}' at depth {depth}. Current lecture structure saved.")

    print("Lecture generation complete. Final structure saved to lecture_structure.json.")

if __name__ == "__main__":
    dfs_generate_lecture("Machine Learning")

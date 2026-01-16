import re
import os
import sys
import json
import time
import pickle
import argparse
import openai

# 1. Essential for Pickle and Project Imports
import mmagent.videograph
sys.modules["videograph"] = mmagent.videograph
from mmagent.retrieve import search
from mmagent.utils.general import load_video_graph

# 2. Configuration Loading
API_CONFIG = json.load(open("configs/api_config.json"))
PROC_CONFIG = json.load(open("configs/processing_config.json"))

# 3. Initialize Standard OpenAI Client (using GPT-4o for Reasoning)
GPT_MODEL = "gpt-5-mini"
client = openai.OpenAI(api_key=API_CONFIG[GPT_MODEL]["api_key"])

# 4. Agent Prompts (Official M3-Agent logic)
SYSTEM_PROMPT = (
    "You are given a question and some relevant knowledge. Your task is to reason about "
    "whether the provided knowledge is sufficient to answer the question. If it is sufficient, "
    "output [Answer] followed by the answer. If it is not sufficient, output [Search] and "
    "generate a query that will be encoded into embeddings for a vector similarity search. "
    "The query will help retrieve additional information from a memory bank.\n\n"
    "Question: {question}"
)

INSTRUCTION = """
Output the answer in the format:
Action: [Answer] or [Search]
Content: {content}

Rules:
1. If searching, {content} should be a single query.
2. If searching for people, use: "What is the name of <character_{i}>" or "What is the character id of {name}".
3. After getting a mapping, use Character IDs (e.g., <face_1>) instead of names for better searching.
4. If answering, {content} is the final answer. Use NAMES in the final answer, not IDs.
5. You must be concise.
"""

def get_ai_response(messages):
    """Calls GPT-4o for the Agent's reasoning loop."""
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            max_completion_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n[!] API Error: {e}")
        return None

def run_complex_agent(mem_path, question, total_rounds=5):
    # Load Memory Graph
    print(f"\n[*] Loading Memory Graph: {mem_path}...")
    vg = load_video_graph(mem_path)
    vg.refresh_equivalences()

    # Initial state
    current_clips = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(question=question)},
        {"role": "user", "content": "Searched knowledge: {}"}
    ]
    
    print(f"[*] Question: {question}")
    print("-" * 50)

    for r in range(total_rounds):
        print(f"\n[ROUND {r+1}/{total_rounds}] Reasoning...")
        
        # Format current prompt
        prompt_with_instr = messages[-1]["content"] + "\n" + INSTRUCTION
        if r == total_rounds - 1:
            prompt_with_instr += "\n(CRITICAL: This is the last round. You MUST provide an [Answer] now.)"
        
        current_messages = messages[:-1] + [{"role": "user", "content": prompt_with_instr}]

        # Get Agent Decision
        ai_output = get_ai_response(current_messages)
        if not ai_output: break
        
        print(f"| Agent Decision:\n{ai_output}")

        # Parse Logic
        pattern = r"Action: \[(.*)\].*Content: (.*)"
        match = re.search(pattern, ai_output, re.DOTALL)
        
        if not match:
            # Fallback parsing
            action = "Answer" if "[Answer]" in ai_output else "Search"
            content = ai_output.split("Content:")[-1].strip()
        else:
            action = match.group(1).strip()
            content = match.group(2).strip()

        if action == "Answer":
            print("\n" + "="*20 + " FINAL ANSWER " + "="*20)
            print(content)
            print("="*54)
            return content

        # Else Action is [Search]
        print(f"| Performing Graph Retrieval for: '{content}'")
        
        # Search the Multimodal Graph
        # We use a threshold of 0.5 to filter irrelevant clips
        if "character id" in content.lower() or "name" in content.lower():
            # Identity search
            new_info, _, _ = search(vg, content, [], mem_wise=True, topk=10)
        else:
            # Event search
            new_info, updated_clips, _ = search(vg, content, current_clips, threshold=0.45, topk=2)
            current_clips = updated_clips

        # Update knowledge for the next round
        search_summary = json.dumps(new_info, ensure_ascii=False)
        if not new_info:
            search_summary = "(No new information found for this query. Try a different search.)"
        
        messages.append({"role": "assistant", "content": ai_output})
        messages.append({"role": "user", "content": f"Searched knowledge: {search_summary}"})

    return "Agent failed to reach a conclusion."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mem_path", type=str, default="data/memory_graphs/test.pkl")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    # Start the loop
    run_complex_agent(args.mem_path, args.question, total_rounds=args.rounds)
import json
import argparse
import openai
from mmagent.unified_engine import UnifiedMemoryEngine

# --- CONFIG ---
with open("configs/api_config.json") as f:
    conf = json.load(f)

GPT_MODEL = "gpt-5-mini"

client = openai.OpenAI(
    api_key=conf[GPT_MODEL]["api_key"]
)

engine = UnifiedMemoryEngine(
    video_id="asfa_pro_session_01"
)

SYSTEM_PROMPT = """You are a Multimodal Intelligence Agent.
You have access to a video's Long-Term Memory.

The provided knowledge includes:
- moment: what was happening in the video
- related_entities: people, objects, or concepts
- timestamp: when it occurred

Answer the question using ONLY this knowledge.
If the information is missing, say exactly what is missing.
"""

def ask_pro(question: str):
    print(f"\n[*] Querying Cloud Brain for: {question}")
    
    # 1. Get Enriched Knowledge
    knowledge = engine.cloud_search(question)
    
    if not knowledge:
        print("\n[!] WARNING: No relevant memories found in Qdrant for this video_id.")
        print("Check if your VIDEO_ID in ask_agent_pro.py matches the one used in memorization_pro.py")
        return
    
    knowledge_str = json.dumps(knowledge, indent=2)

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Knowledge:\n{knowledge_str}\n\nQuestion:\n{question}"
            }
        ],
    )

    answer = response.choices[0].message.content

    print("\n" + "=" * 30 + " AGENT ANSWER " + "=" * 30)
    print(answer)
    print("=" * 74)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, type=str)
    args = parser.parse_args()

    ask_pro(args.q)

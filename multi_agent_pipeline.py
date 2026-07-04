import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


def call_model(system_prompt, user_prompt, max_retries=3):
    """Reusable helper — every agent uses this same reliable calling pattern."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️ Error, retrying... ({e})")
            time.sleep(1)
    return "Agent failed after retries."


# --- Simulated raw data, standing in for real web research ---
RAW_COMPETITOR_DATA = """
Competitor: RivalCloud Inc.
Pricing tiers found on their website:
- Starter: $9/month, 5 users, 10GB storage
- Growth: $29/month, 25 users, 100GB storage  
- Enterprise: Custom pricing, unlimited users, 1TB storage

Our current pricing:
- Starter: $12/month, 5 users, 10GB storage
- Growth: $39/month, 25 users, 100GB storage
- Enterprise: Custom pricing, unlimited users, 1TB storage

Recent news: RivalCloud raised $50M Series B funding 3 months ago.
Customer reviews mention RivalCloud's onboarding is slower than ours.
"""


# --- Agent 1: Researcher ---
def researcher_agent(topic):
    system_prompt = """You are a Researcher agent. Your only job is to organize raw 
    information clearly and factually. Do not analyze, interpret, or give opinions — 
    just organize the facts cleanly."""

    user_prompt = f"Organize this raw research data clearly:\n\n{RAW_COMPETITOR_DATA}"

    print("🔍 Researcher agent working...")
    result = call_model(system_prompt, user_prompt)
    print(f"   → Research complete\n")
    return result


# --- Agent 2: Analyzer ---
def analyzer_agent(research_output):
    system_prompt = """You are an Analyzer agent. Your job is to find patterns, risks, 
    and strategic implications in research data. Do not write a polished report — just 
    give clear, direct analytical insights and flag anything strategically important."""

    user_prompt = f"Analyze this research and identify key strategic implications:\n\n{research_output}"

    print("🧠 Analyzer agent working...")
    result = call_model(system_prompt, user_prompt)
    print(f"   → Analysis complete\n")
    return result


# --- Agent 3: Writer ---
def writer_agent(research_output, analysis_output):
    system_prompt = """You are a Writer agent. Your job is to turn research and analysis 
    into a clear, polished, executive-ready report. Use short paragraphs and a 
    professional but readable tone."""

    user_prompt = f"""Write a polished report using this research and analysis:

RESEARCH:
{research_output}

ANALYSIS:
{analysis_output}"""

    print("✍️  Writer agent working...")
    result = call_model(system_prompt, user_prompt)
    print(f"   → Report complete\n")
    return result


# --- The pipeline: research → analyze → write, in sequence ---
def run_pipeline(topic):
    print(f"{'='*60}")
    print(f"PIPELINE START: {topic}")
    print(f"{'='*60}\n")

    research = researcher_agent(topic)
    print("--- RESEARCHER OUTPUT ---")
    print(research)
    print()

    analysis = analyzer_agent(research)
    print("--- ANALYZER OUTPUT ---")
    print(analysis)
    print()
    
    report = writer_agent(research, analysis)

    print(f"{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(report)


if __name__ == "__main__":
    run_pipeline("RivalCloud competitor pricing analysis")
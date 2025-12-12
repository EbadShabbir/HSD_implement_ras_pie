import requests
import time
from colorama import Fore, Style, init
import re

init(autoreset=True)

OLLAMA_API = "http://localhost:11434/api/generate"

# ===== MENTAL HEALTH / SAFETY CONFIG =====

MENTAL_HEALTH_KEYWORDS = [
    "depression", "depressed", "anxiety", "anxious", "panic attack",
    "bipolar", "ptsd", "ocd", "adhd", "psychosis", "hallucination",
    "suicidal", "suicide", "kill myself", "end my life", "self harm",
    "self-harm", "cut myself", "hurt myself", "don't want to live",
    "don't wanna live", "life is pointless", "i want to die",
    "i wanna die", "worthless", "mental health", "therapist",
    "psychiatrist", "counsellor", "counselor"
]

RISK_KEYWORDS = {
    "high_risk_emergency": [
        "kill myself", "killing myself", "i want to die", "i wanna die",
        "end my life", "end it all", "i'm going to kill myself",
        "i am going to kill myself", "i will kill myself",
        "suicide note", "committing suicide", "commit suicide",
        "how do i kill myself", "how to kill myself", "how can i die",
        "best way to die", "painless way to die", "overdose to die",
        "i cut myself deeply", "i cut too deep", "bleeding a lot",
        "i'm going to hurt someone", "i'm going to kill them",
        "i will kill them", "i want to kill someone"
    ],
    "moderate_risk": [
        "i hurt myself", "i self harm", "i self-harm", "i cut myself",
        "i burned myself", "i starve myself", "i haven't eaten in days",
        "i wish i didn't exist", "i wish i was dead", "i don't care if i die",
        "i wouldn't mind if i died", "life is not worth living",
        "it would be better if i disappeared", "i think about killing myself",
        "suicidal thoughts", "thinking of suicide"
    ],
    "mild_concern": [
        "i feel empty", "i feel numb", "i feel hopeless", "i feel useless",
        "i hate myself", "i'm so tired of everything", "i'm done with everything",
        "what's the point of living", "my life is pointless",
        "i'm not okay", "i'm really struggling", "mentally exhausted"
    ]
}

MEDICAL_DISCLAIMER = (
    "\n\n⚠️ **Important:** I'm an AI system and not a doctor or therapist. "
    "I can't diagnose conditions or replace professional care. "
    "For any concerns about your mental health, please talk to a licensed mental health professional."
)

CRISIS_MESSAGE_HIGH = (
    "I'm really sorry you're feeling this way. Your safety is extremely important.\n\n"
    "I **can't** help with any instructions for self-harm or harming others.\n\n"
    "If you are in immediate danger or feel you might hurt yourself or someone else:\n"
    "• Please contact your local emergency services **right now**.\n"
    "• If possible, reach out to a trusted friend, family member, or someone nearby.\n"
    "• You can also look for crisis hotlines or mental health helplines in your country.\n\n"
    "You deserve support from real humans who can help you in this moment."
)

CRISIS_MESSAGE_MODERATE = (
    "Thank you for sharing this. It sounds like you're going through something really heavy.\n\n"
    "I'm not a substitute for a mental health professional, and I can't fully assess your situation.\n\n"
    "It would be really important to:\n"
    "• Reach out to a mental health professional (psychiatrist, psychologist, or counsellor).\n"
    "• Talk to someone you trust (friend, family member) about what you're feeling.\n"
    "• If your thoughts about self-harm or suicide get stronger or feel uncontrollable, "
    "please contact emergency services or a crisis helpline immediately.\n\n"
    "You don't have to deal with this alone."
)

CRISIS_MESSAGE_MILD = (
    "It sounds like you're having a really tough time emotionally.\n\n"
    "I can share general information and supportive guidance, but I'm not a doctor or therapist.\n\n"
    "It might help to:\n"
    "• Talk to someone you trust about how you're feeling.\n"
    "• Consider booking an appointment with a mental health professional.\n"
    "• Take small steps for self-care (sleep, food, movement, journaling), even if they feel hard.\n\n"
    "Your feelings are valid, and reaching out for help is a strong and important step."
)

def classify_risk(prompt: str) -> str:
    """Very simple keyword-based risk classification."""
    text = prompt.lower()
    
    for phrase in RISK_KEYWORDS["high_risk_emergency"]:
        if phrase in text:
            return "high_risk_emergency"
    
    for phrase in RISK_KEYWORDS["moderate_risk"]:
        if phrase in text:
            return "moderate_risk"
    
    for phrase in RISK_KEYWORDS["mild_concern"]:
        if phrase in text:
            return "mild_concern"
    
    return "no_risk"

def generate_safety_response(prompt: str, risk_level: str) -> str:
    """Return a safe, templated response for mental health / crisis content."""
    if risk_level == "high_risk_emergency":
        return CRISIS_MESSAGE_HIGH + "\n\n" + MEDICAL_DISCLAIMER
    elif risk_level == "moderate_risk":
        return CRISIS_MESSAGE_MODERATE + "\n\n" + MEDICAL_DISCLAIMER
    elif risk_level == "mild_concern":
        return CRISIS_MESSAGE_MILD + "\n\n" + MEDICAL_DISCLAIMER
    else:
        return MEDICAL_DISCLAIMER

def looks_like_mental_health_query(prompt: str) -> bool:
    """Check if the question is about mental health / psychiatry."""
    text = prompt.lower()
    return any(word in text for word in MENTAL_HEALTH_KEYWORDS)

def add_disclaimer_if_needed(prompt: str, response: str) -> str:
    """Append a medical disclaimer if the question is mental-health-related."""
    if looks_like_mental_health_query(prompt):
        if "i'm an ai system and not a doctor" not in response.lower():
            return response.strip() + MEDICAL_DISCLAIMER
    return response

def call_model_with_confidence(model, prompt, num_tokens=150):
    """Call model with safety instructions"""
    full_prompt = (
        "You are an assistant. You MUST NOT diagnose medical or psychiatric conditions, "
        "and you MUST NOT prescribe or suggest medication or dosages. "
        "You can provide general information and supportive guidance only.\n\n"
        f"User Question: {prompt}\n\n"
        "Please answer the question."
    )
    
    response = requests.post(OLLAMA_API, json={
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "num_predict": num_tokens,
            "temperature": 0.3
        }
    })
    return response.json()['response']

def is_draft_good_enough(response, prompt):
    """VERY STRICT validation"""
    response = response.strip()
    
    if len(response) < 10:
        return False, "too short"
    
    uncertain_phrases = [
        "i don't know", "i'm not sure", "i cannot", "i can't",
        "i apologize", "unable to", "i don't have", "not available",
        "as an ai", "i'm sorry", "i do not"
    ]
    if any(phrase in response.lower() for phrase in uncertain_phrases):
        return False, "uncertain/refused"
    
    words = response.split()
    
    # Code checks
    if "```
        if response.count("```") < 2:
            return False, "incomplete code block"
        if len(words) < 200:
            return False, "code too short"
    
    # Math checks
    if any(kw in prompt.lower() for kw in ["calculate", "solve", "step-by-step"]) and any(char.isdigit() for char in prompt):
        has_calculation = re.search(r'\d+\s*[+\-*/=]\s*\d+', response)
        if not has_calculation:
            return False, "no calculations shown"
    
    # Simple questions
    simple_keywords = ["what is", "who is", "when", "where", "capital", "define"]
    is_simple = any(kw in prompt.lower() for kw in simple_keywords)
    
    if is_simple and len(words) >= 3:
        return True, "good factual answer"
    
    if len(words) >= 50:
        return True, "passed quality checks"
    
    return False, "needs more detail"

def smart_routing(prompt, target="phi3:3.8b", draft="phi:2.7b"):
    print(f"\n{'='*70}")
    print(f"🧠 SMART ROUTING WITH SAFETY LAYER (Phi Family)")
    print(f"{'='*70}\n")
    print(f"📝 Your Question: {Fore.YELLOW}{prompt}{Style.RESET_ALL}\n")
    
    # ===== Step 0: Safety / mental health risk check =====
    risk_level = classify_risk(prompt)
    if risk_level != "no_risk":
        print(f"{Fore.RED}⚠ Detected mental health / crisis-related content (risk: {risk_level}){Style.RESET_ALL}\n")
        safe_response = generate_safety_response(prompt, risk_level)
        print(f"{Fore.GREEN}Safety Response:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{safe_response}{Style.RESET_ALL}\n")
        return safe_response, "safety", 0.0
    
    # Step 1: Try draft model (Phi-2 2.7B)
    print(f"{Fore.CYAN}⚡ Trying DRAFT model ({draft} - Phi-2 2.7B)...{Style.RESET_ALL}")
    start_draft = time.time()
    draft_response = call_model_with_confidence(draft, prompt, num_tokens=250)
    draft_time = time.time() - start_draft
    
    draft_response = add_disclaimer_if_needed(prompt, draft_response)
    
    print(f"\n{Fore.CYAN}Draft Response ({len(draft_response.split())} words, {draft_time:.2f}s):{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{draft_response}{Style.RESET_ALL}\n")
    
    is_good, reason = is_draft_good_enough(draft_response, prompt)
    
    if is_good:
        print(f"{Fore.GREEN}✅ Draft response PASSED! ({reason}){Style.RESET_ALL}")
        print(f"⏱️  Total time: {draft_time:.2f}s")
        return draft_response, "draft", draft_time
    
    else:
        print(f"{Fore.RED}❌ Draft FAILED: {reason}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎯 Routing to TARGET model ({target} - Phi-3-mini 3.8B)...{Style.RESET_ALL}\n")
        
        start_target = time.time()
        target_response = call_model_with_confidence(target, prompt, num_tokens=400)
        target_time = time.time() - start_target
        
        target_response = add_disclaimer_if_needed(prompt, target_response)
        
        print(f"{Fore.GREEN}Target Response ({len(target_response.split())} words, {target_time:.2f}s):{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{target_response}{Style.RESET_ALL}\n")
        print(f"⏱️  Total time: {draft_time + target_time:.2f}s")
        
        return target_response, "target", draft_time + target_time

if __name__ == "__main__":
    print(f"\n{Fore.MAGENTA}{'='*70}")
    print("AI ASSISTANT WITH MENTAL HEALTH SAFETY LAYER (Phi Family)")
    print(f"{'='*70}{Style.RESET_ALL}\n")
    print(f"{Fore.CYAN}💡 Draft: Phi-2 (2.7B) | Target: Phi-3-mini (3.8B){Style.RESET_ALL}")
    print(f"{Fore.CYAN}💡 Ask any question! (Type 'exit' to quit){Style.RESET_ALL}\n")
    
    while True:
        prompt = input(f"{Fore.YELLOW}Your question: {Style.RESET_ALL}")
        
        if prompt.lower() in ['exit', 'quit', 'q']:
            print(f"\n{Fore.GREEN}👋 Goodbye!{Style.RESET_ALL}")
            break
        
        if not prompt.strip():
            print(f"{Fore.RED}Please enter a question!{Style.RESET_ALL}\n")
            continue
        
        response, model_used, total_time = smart_routing(prompt)
        
        print(f"\n{Fore.MAGENTA}{'='*70}")
        print(f"RESULT:")
        print(f"{'='*70}{Style.RESET_ALL}")
        print(f"🤖 Model: {Fore.GREEN if model_used == 'draft' else (Fore.RED if model_used == 'target' else Fore.CYAN)}{model_used.upper()}{Style.RESET_ALL} | ⏱️  {total_time:.2f}s\n")

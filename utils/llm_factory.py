import os
from typing import Literal
from dotenv import load_dotenv

# Import both LLM providers
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

def get_llm(model_class: Literal["main", "fast"]):
    """
    Returns a resilient LLM Runnable wrapped with exponential backoff 
    (via max_retries) and cross-provider fallbacks.
    
    Primary Provider: Gemini
    Fallback Provider: Anthropic
    """
    
    # 1. Configuration Constants
    retries = 3
    
    gemini_main = "gemini-3.1-flash-lite"
    gemini_fast = "gemini-3.1-flash-lite"
    
    anthropic_main = "claude-3-5-sonnet-20240620"
    anthropic_fast = "claude-3-haiku-20240307"
    
    # 2. Extract API Keys
    google_key = os.environ.get("GOOGLE_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if google_key == "your_google_api_key_here":
        google_key = None
    if anthropic_key == "your_anthropic_api_key_here":
        anthropic_key = None
        
    # 3. Instantiate Providers
    primary_llm = None
    fallback_llm = None
    
    if google_key:
        model_name = gemini_main if model_class == "main" else gemini_fast
        primary_llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=0, 
            max_retries=retries,
            api_key=google_key
        )
        
    if anthropic_key:
        model_name = anthropic_main if model_class == "main" else anthropic_fast
        fallback_llm = ChatAnthropic(
            model=model_name, 
            temperature=0, 
            max_retries=retries,
            api_key=anthropic_key
        )
        
    # 4. Construct Resilience Chain
    if primary_llm and fallback_llm:
        # If both keys are present, bind the fallback to the primary
        # LangChain will automatically try the primary, backoff and retry `max_retries` times,
        # and if it still fails (e.g. strict rate limit), it will seamlessly fallback to Anthropic.
        return primary_llm.with_fallbacks([fallback_llm])
        
    elif primary_llm:
        return primary_llm
        
    elif fallback_llm:
        return fallback_llm
        
    else:
        # Neither key is present, returning None. The agents will handle this gracefully.
        return None

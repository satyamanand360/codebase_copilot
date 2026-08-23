import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        print("ERROR: Invalid API key found in .env")
        return
        
    print(f"Testing Gemini API with key ending in: ...{api_key[-5:]}")
    
    try:
        print("Sending test request to Gemini 3.7 Flash...")
        llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)
        response = llm.invoke("Respond with exactly: 'Hello World!'")
        print("✅ SUCCESS! API is working. Response:", response.content)
    except Exception as e:
        print("❌ FAILED! Error details:", type(e).__name__)
        print(e)

if __name__ == "__main__":
    main()

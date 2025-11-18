from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from config import Settings

settings = Settings()

llm = ChatOpenAI(
    model_name=settings.MODEL_NAME,
    openai_api_key=settings.OPENAI_API_KEY
)

# ============================================================
# A interactive prompt example, the user will be asked what country is on their mind and 
# what intrests them in life. Based on that input, the model give the country a sync score with the user.

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert travel advisor that rates how well a country matches a user's interests on a scale from 1 to 10."),
    ("human", "Given my interest in {interest}, how well does {country} match my interests? Provide a score from 1 to 10 and a brief explanation.")
])

# --- get user input ----
country = input("Enter a country you are interested in: ")
interest = input("Enter your main interest (e.g., history, nature, food): ")

pipeone = prompt | llm

# --- piping the prompt and llm together ---
response = pipeone.invoke({"country": country, "interest": interest})

print("Model Response:")
print(response.content)
# ============================================================
# This will use the pipe operator with lambda functions to extract the score 
# from the model's response. and give it an emoji based on the score.

def extract_score(response_text: str) -> int:
    # Simple extraction assuming the score is the first number in the response
    import re
    match = re.search(r'(\d+)', response_text)
    if match:
        return int(match.group(1))
    return 0

def score_to_emoji(score: int) -> str:
    if score >= 8:
        return "🌟"
    elif score >= 5:
        return "👍"
    else:
        return "👎"
    
pipeline = prompt | llm | RunnableLambda(lambda resp: extract_score(resp.content)) | RunnableLambda(lambda score: score_to_emoji(score))

emoji = pipeline.invoke({"country": country, "interest": interest})
print(f"Your Trip Emoji: {emoji}")

#==================================================================
# CLI chain to convince non-pet people to leave the dark side and adopt a pet from a shelter
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a highly skilled salesperson who left your job as a top performing salesperson to pursue your dream providing loving homes to shelter pets."),
    ('human',"I currently don't want a pet, but deep down I'm lonely and need purpose"),
    ("system","you also spent a few years in undergrad studying to become a consoler, you don't have a license and didn't finish your degree but you have knowledge and will use it to help this person see the good side of pet ownership."),
    ("human","this is my reasoning not to adopt a pet: {reason} convince me otherwise")
])

reason = input("Welcome to the animal shelter, what is a reason you can't adopt a pet today?\n")

pipeline3 = prompt | llm

response = pipeline3.invoke({"reason":reason})

print("\n"+response.content)
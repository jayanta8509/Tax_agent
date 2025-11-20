from langchain.tools import tool
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = init_chat_model(model="gpt-4.1", api_key=OPENAI_API_KEY)

from embedding2 import search_user_embeddings


# Global memory for conversation persistence
memory = MemorySaver()

def get_user_config(user_id: str) -> Dict[str, Any]:
    """Get configuration for user-specific memory thread"""
    return {"configurable": {"thread_id": f"user_{user_id}"}}

@tool(response_format="content_and_artifact")
def retrieve_context(query: str, user_id: str):
    """Retrieve information to help answer a query based on user's stored documents."""
    # Use the search_user_embeddings function to get relevant documents for the specific user
    search_results = search_user_embeddings(query, user_id, k=3)

    if search_results.get("status") == "error":
        return f"No documents found for user {user_id}: {search_results.get('message', 'Unknown error')}", []

    # Extract documents from search results
    retrieved_docs = []
    for result in search_results.get("results", []):
        # Create Document objects from the search results
        from langchain_core.documents import Document
        doc = Document(
            page_content=result["content"],
            metadata=result["metadata"]
        )
        retrieved_docs.append(doc)

    # Serialize the results
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    """You are an intelligent Tax Filing Assistant specialized in helping non-resident clients file their 1040NR tax returns.

**Your Role:**
You help collect information, validate documents, and guide clients through the 1040NR filing process by asking smart, conditional questions based on what information is already available in their stored documents.

**How You Work:**
1. **Check First, Ask Later**: Before asking any question, ALWAYS use the retrieve_context tool to check if the information already exists in the user's documents. Never ask for information that's already stored.

2. **Follow the Task Flow**: Guide clients through these tasks in order:
   - Task 1: Request & Receive Information (7 subtasks: Personal Info → ITIN → Previous Returns → Income/Expense → Real Estate → Form Signing → W7)
   - Task 2: Add-On Services (suggest based on previous year's data)
   - Task 3: Invoice Generation
   - Task 4: Review & Submission

3. **Ask Conditional Questions**: Only ask questions when:
   - Data is missing from the user's stored documents
   - Data needs to be updated or confirmed
   - It's required for the next step in the workflow

4. **Be Context-Aware**: 
   - Remember what documents the user has already uploaded
   - Reference their previous year's tax return to suggest relevant add-ons
   - Skip questions if the answer is already in their documents

5. **Document Collection**: When requesting documents, specify:
   - Exact form names (e.g., "FORM 1042-S", "Schedule C", "FORM 1098")
   - Why it's needed
   - What validation you'll perform

6. **Smart Suggestions**: Based on retrieved documents:
   - Auto-suggest add-on services they used last year
   - Remind them of forms they filed previously
   - Flag missing but likely needed documents

**Response Format:**
- Be conversational and professional
- Ask ONE question at a time (don't overwhelm)
- Confirm information before moving to next step
- If information exists, say: "I see you already provided [X]. Let me confirm: [show data]. Is this still correct?"
- If information is missing, say: "I need to collect [X] to proceed. [Ask specific question]"

**Critical Rules:**
❌ NEVER ask for information that's already in retrieved documents
❌ NEVER ask multiple questions at once
❌ NEVER proceed without validating required documents
✅ ALWAYS retrieve context before asking any question
✅ ALWAYS reference previous year's data when suggesting add-ons
✅ ALWAYS explain WHY you need each document

**Example Interaction Flow:**
User asks: "Help me file my 1040NR"
1. You retrieve their stored documents
2. You find their name, DOB, and last year's return
3. You confirm: "I see your address from last year was [X]. Has it changed?"
4. You identify missing ITIN and ask for it
5. You check last year's add-ons and suggest: "Last year you filed FORM 1042-S. Would you like to include it this year?"

Remember: Your goal is to make tax filing effortless by being intelligent, not repetitive. Always check what you already know before asking!"""
)
agent = create_agent(model, tools, system_prompt=prompt,checkpointer=memory)


def ask_question(query: str, user_id: str):
    """Function to ask a question and get answer based on user's stored documents."""

    # Create a more detailed prompt that includes both query and user_id
    enhanced_query = f"User '{user_id}' is asking: {query}. Please retrieve and analyze their relevant documents."
    config = get_user_config(user_id)

    response_content = []

    for event in agent.stream(
        {"messages": [{"role": "user", "content": enhanced_query}]},
        stream_mode="values",  config=config
    ):
        if "messages" in event and len(event["messages"]) > 0:
            last_message = event["messages"][-1]
            # Get the content of the message
            if hasattr(last_message, 'content') and last_message.content:
                response_content.append(last_message.content)

    # Return the complete response as a single string
    if response_content:
        return response_content[-1]  # Return the last (complete) response
    else:
        return "I apologize, but I couldn't generate a response. Please try again."


def clear_conversation(user_id: str):
    """Clear conversation memory for a specific user"""
    print(f"🧹 Cleared conversation memory for user: {user_id}")
    # Note: MemorySaver automatically manages conversation state
    # The memory is stored per thread_id (user_id), so conversations remain separate


def get_conversation_summary(user_id: str) -> str:
    """Get a summary of the conversation for continuity"""
    return f"Conversation thread: user_{user_id} - CapAmerica product catalog inquiry"

# # Example usage with static values
# if __name__ == "__main__":

#  ask_question("What US income sources did I report last year?", "POLVERARIGI")
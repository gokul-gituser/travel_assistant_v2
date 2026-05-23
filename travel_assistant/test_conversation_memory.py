# test_conversation_memory.py

from conversation_memory import store_chat_interaction
from retriever import retrieve_context

store_chat_interaction(
    user_id="rahul",
    user_message="I recently travelled to Kyoto in Japan",
    assistant_reply="Kyoto is famous for temples and cherry blossoms",
)

context = retrieve_context(
    user_id="rahul",
    query="Where did I travel?"
)

print(context)
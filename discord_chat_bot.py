"""
This will be a simple gpt based chatbot with discord integrations
The idea here is to allow a chatbot to participate in a discord and remember what each participant has said.
I kinda stole the idea from TheGuy on Twitter (@Guygies). he has a bot called Geepers on discord.
The bot has a medium term memory and can respond to things that were said in the past, but the memory is cleared
when the bot is restarted. I want to make a bot that can remember things for a long time and can respond to things
that were said in the past. I also want to make a bot that can learn from the conversation and improve itself over time.
Then I guess I want to run this on a cloud service so I don't have to keep it running on my laptop.
I don't fully understand how async works. do some research on that after some sleep

TODO:
    [ ] pass username to the bot so it can remember who said what
    [ ] make the bot remember things for a long time
    [ ] make a way to save memories with embeddings so that semantic search can be done to retrieve past memories
    [ ] save the memories separated by discord user name
    [ ] make it able to communicate with other bots (MidJourney)
    [ ] deploy in a way that it can run all the time. (maybe use heroku)
"""

import discord
import json
import openai
from langchain.llms import OpenAI
import os
from langchain import PromptTemplate, ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationalBufferWindowMemory, \
    ConversationSummaryBufferMemory, ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE


# Load Keys
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
    key = f.read().strip()
    openai.api_key = key
    os.environ["OPENAI_API_KEY"] = key
with open("/Users/alexthe5th/Documents/API Keys/GoogleSearchAPI_key.txt", "r") as f:
    key = f.read().strip()
    os.environ["GOOGLE_API_KEY"] = key
with open("/Users/alexthe5th/Documents/API Keys/GoogleSearch_ID.txt", "r") as f:
    key = f.read().strip()
    os.environ["GOOGLE_CSE_ID"] = key
with open("prompt_template.txt", "r") as f:
    template = f.read()
auth = load_json('/Users/alexthe5th/Documents/API Keys/Discord_auth.json')
TOKEN = auth['token'].strip()
CHAN_ID = int(auth['chan_id'].strip())

# Set up discord bot
intents = discord.Intents.all()
client = discord.Client(intents=intents)
llm = OpenAI(
    temperature=0,
    model_name='text-davinci-003',
    max_tokens=1024,
)
memory = ConversationEntityMemory(llm=llm)
prompt = PromptTemplate(
    input_variables=['entities', 'history', 'input'],
    template=template
)
bort = ConversationChain(
    llm=llm,
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=llm, memory=memory)
)


# Discord interface
@client.event
async def on_message(message):
    if not message.author.bot:
        discord_text = message.content
        user = message.author.name
        text = f"{user}: {discord_text}"
        reply = bort.run(input=text)
        await message.channel.send(reply)


@client.event
async def on_ready():
    channel = client.get_channel(CHAN_ID)
    if channel is not None:
        await channel.send(f"{client.user} has connected to Discord!")
    else:
        print("Channel not found")


if __name__ == "__main__":
    client.run(TOKEN)

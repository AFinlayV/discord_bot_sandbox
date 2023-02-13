"""
Starting from scratch with the disnake bot template uses /ask_gpt to send a question to the bot.
then the bot will either:
- do math with llm-math
- search google with google-search
- solve a problem with wolfram-alpha
"""
import asyncio
import disnake
from disnake.ext import commands
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.agents import load_tools, initialize_agent
from langchain.chains.conversation.memory import ConversationBufferMemory

TEMPLATE = """
The following is a question from a user on discord:

+++++++++++++++
Question:
{message}

+++++++++++++++

The following is a long, detailed and verbose response from a bot:
"""

intents = disnake.Intents.all()
CHAN = int(os.environ.get('SANDBOX_DISCORD_CHAN_ID'))
TOKEN = os.environ.get('SANDBOX_DISCORD_TOKEN')
bot = commands.Bot(
    intents=intents,
    command_prefix='/'
)
LLM = OpenAI(temperature=0.0)
MEMORY = ConversationBufferMemory(memory_key="chat_history")
TOOL_NAMES = ['llm-math', 'google-search', 'wolfram-alpha']
TOOLS = load_tools(TOOL_NAMES, llm=LLM)
AGENT = initialize_agent(llm=LLM, memory=MEMORY, tools=TOOLS, agent='conversational-react-description', verbose=True)


def process_message(message):
    try:
        prompt = PromptTemplate(input_variables=['message'], template=TEMPLATE)
        text = prompt.format(message=message)
        response = AGENT.run(text)
        return response
    except Exception as e:
        return f'<Error>: {e}'


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    channel = bot.get_channel(CHAN)
    await channel.send(f'{bot.user} has connected to Discord!')


@bot.command()
async def ask_gpt(ctx, *args):
    question = ' '.join(args)
    print(f'Question: {question}')
    if ctx.author == bot.user:
        return
    else:
        # use asyncio to run the process_message function in the background
        response = await asyncio.get_event_loop().run_in_executor(None, process_message, question)
        channel = bot.get_channel(int(os.environ.get('SANDBOX_DISCORD_CHAN_ID')))
        await ctx.send(response)


bot.run(TOKEN)

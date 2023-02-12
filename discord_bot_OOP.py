"""
Starting from scratch with the discord bot template using the Bot class instead of the Client class.
"""
import asyncio
import discord
from discord.ext import commands
import os


intents = discord.Intents.all()
bot = commands.Bot(
    intents=intents,
    command_prefix='!',
)


def process_message(message):
    try:
        """
        This is where the logic for the bot goes.
        """
        pass
    except Exception as e:
        return f'<Error>: {e}'


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    channel = bot.get_channel(int(os.environ.get('SANDBOX_DISCORD_CHAN_ID')))
    await channel.send(f'{bot.user} has connected to Discord!')


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    elif message.content.startswith('!') or message.content == "":
        return
    else:
        # use asyncio to run the process_message function in the background
        response = await asyncio.get_event_loop().run_in_executor(None, process_message, message)
        channel = bot.get_channel(int(os.environ.get('SANDBOX_DISCORD_CHAN_ID')))
        await channel.send(response)


bot.run(os.environ.get('SANDBOX_DISCORD_TOKEN'))

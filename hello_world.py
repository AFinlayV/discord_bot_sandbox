import discord
import json

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


auth = load_json('/Users/alexthe5th/Documents/API Keys/Discord_auth.json')
TOKEN = auth['token'].strip()
CHAN_ID = 1071975175802851461

intents = discord.Intents.default()
print(intents)

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    channel = client.get_channel(1071975175802851461)
    if channel is not None:
        await channel.send("Hello World!")
    else:
        print("Channel not found")


if __name__ == "__main__":
    client.run(TOKEN)

"""
This will be a simple gpt based chatbot with discord integrations
The idea here is to allow a chatbot to participate in a discord and remember what each participant has said.
I kinda stole the idea from TheGuy on Twitter (@Guygies). he has a bot called Geepers on discord.
The bot has a medium term memory and can respond to things that were said in the past, but the memory is cleared
when the bot is restarted. I want to make a bot that can remember things for a long time and can respond to things
that were said in the past. I also want to make a bot that can learn from the conversation and improve itself over time.
Then I guess I want to run this on a cloud service so I don't have to keep it running on my laptop.
I don't fully understand how async works. do some research on that after some sleep

For this version I am going to copy a bunch of code from LongtermChatExternalSources.py (RAVEN) by david shapiro to get the chat
memory working. This means abandoning langchain, but i think the embedded memory is going to be better for long term
memories and make it easier to silo the memories by user.

TODO:
    [ ] make it able to communicate with other bots (MidJourney)
    [ ] deploy in a way that it can run all the time. (maybe use heroku)
    [ ] discord timeouts!
"""
import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time, sleep
from uuid import uuid4
import datetime
import discord
import asyncio


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2) / (norm(v1) * norm(v2))  # return cosine similarity


def fetch_memories(vector, logs, count):
    scores = list()
    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = similarity(i['vector'], vector)
        i['score'] = score
        scores.append(i)
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


def load_convo():
    files = os.listdir('nexus')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json('nexus/%s' % file)
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return ordered


def summarize_memories(memories):  # summarize a block of memories into one payload
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    block = ''
    identifiers = list()
    timestamps = list()
    for mem in memories:
        block += mem['message'] + '\n\n'
        identifiers.append(mem['uuid'])
        timestamps.append(mem['time'])
    block = block.strip()
    prompt = open_file('prompt_notes.txt').replace('<<INPUT>>', block)
    # TODO - do this in the background over time to handle huge amounts of memories
    notes = gpt3_completion(prompt)
    #   SAVE NOTES
    vector = gpt3_embedding(block)
    info = {'notes': notes, 'uuids': identifiers, 'times': timestamps, 'uuid': str(uuid4()), 'vector': vector,
            'time': time()}
    filename = 'notes_%s.json' % time()
    save_json('internal_notes/%s' % filename, info)
    return notes


def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:-1]
    except:
        short = conversation
    output = ''
    for i in short:
        output += '%s\n\n' % i['message']
    output = output.strip()
    return output


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.5, top_p=1.0, tokens=1024, freq_pen=0.0, pres_pen=0.0):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


# Discord interface
intents = discord.Intents.all()
client = discord.Client(intents=intents,
                        shard_id=0,
                        shard_count=1,
                        reconnect=True)


def process_message(discord_message):
    discord_text = discord_message.content
    user = discord_message.author.name
    try:
        a = f'{user}: {discord_text}'
        user = discord_message.author.name
        timestamp = time()
        vector = gpt3_embedding(a)
        timestring = timestamp_to_datetime(timestamp)
        message = '%s: %s' % ([{user}], a)
        info = {'speaker': user, 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()),
                'timestring': timestring}
        filename = f'log_{timestamp}_{user}.json'
        save_json('nexus/%s' % filename, info)
        conversation = load_convo()
        memories = fetch_memories(vector, conversation, 10)
        notes = summarize_memories(memories)
        recent = get_last_messages(conversation, 4)
        prompt = open_file('prompt_response.txt') \
            .replace('<<NOTES>>', notes) \
            .replace('<<CONVERSATION>>', recent) \
            .replace('<<MESSAGE>>', a)
        output = gpt3_completion(prompt)
        timestamp = time()
        vector = gpt3_embedding(output)
        timestring = timestamp_to_datetime(timestamp)
        message = '%s: %s' % ('Bort', output)
        info = {
            'speaker': 'Bort',
            'time': timestamp,
            'vector': vector,
            'message': message,
            'uuid': str(uuid4()),
            'timestring': timestring
        }
        filename = 'log_%s_Bort.json' % time()
        save_json('nexus/%s' % filename, info)
        return {'output': output, 'user': user}
    except Exception as oops:
        return {'output': 'Error in process_message: %s' % oops, 'user': user}


@client.event
async def on_message(discord_message, timeout=9):
    discord_text = discord_message.content
    user = discord_message.author.name
    try:
        if not discord_message.author.bot:
            if discord_text.startswith('!'):
                return
            else:
                await discord_message.channel.send(f'Generating response for {user}: "{discord_text[:20]}..."')
                await asyncio.wait_for(
                    send_response(discord_message), timeout=timeout)
    except asyncio.TimeoutError:
        print(f'Response timed out. \n {user}: {discord_text[:20]}...')
    except Exception as oops:
        await discord_message.channel.send(f'Error: {oops} \n {user}: {discord_text[:20]}...')


async def send_response(discord_message, timeout=9):
    try:
        channel = discord_message.channel
        user = discord_message.author.name
        discord_text = discord_message.content
        response = process_message(discord_message)
        # Create a new task for sending the response
        response_task = asyncio.create_task(discord_message.channel.send(f"Response to: {user}: '{discord_text[:20]}...'\n{response['output']}"))
        # Wait for either the response to be sent or for the timeout to occur
        done, pending = await asyncio.wait({response_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        if response_task in done:
            # The response was sent successfully
            return
        else:
            # The response task timed out
            response_task.cancel()
            print(f'Response timed out. \n {user}: {discord_text[:20]}...')
    except Exception as oops:
        # Something else went wrong while sending the response
        await discord_message.channel.send('Error sending response:', oops)

"""
OLD Code:
@client.event
async def on_message(discord_message):
    if not discord_message.author.bot:

        discord_text = discord_message.content
        if discord_text.startswith('!'):
            pass
        else:
            user = discord_message.author.name
            # get user input, save it, vectorize it, etc
            a = f'{user}: {discord_text}'
            timestamp = time()
            vector = gpt3_embedding(a)
            timestring = timestamp_to_datetime(timestamp)
            message = '%s: %s - %s' % ([{user}], timestring, a)
            info = {'speaker': user, 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()),
                    'timestring': timestring}
            filename = f'log_{timestamp}_{user}.json'
            save_json('nexus/%s' % filename, info)
            # load conversation
            conversation = load_convo()
            # compose corpus (fetch memories, etc)
            memories = fetch_memories(vector, conversation, 10)  # pull episodic memories
            # TODO - fetch declarative memories (facts, wikis, KB, company data, internet, etc)
            notes = summarize_memories(memories)
            # TODO - search existing notes first
            recent = get_last_messages(conversation, 4)
            prompt = open_file('prompt_response.txt') \
                .replace('<<NOTES>>', notes) \
                .replace('<<CONVERSATION>>', recent) \
                .replace('<<MESSAGE>>', a)
            # generate response, vectorize, save, etc
            output = gpt3_completion(prompt)
            timestamp = time()
            vector = gpt3_embedding(output)
            timestring = timestamp_to_datetime(timestamp)
            message = '%s: %s - %s' % ('Bort', timestring, output)
            info = {
                'speaker': 'Bort',
                'time': timestamp,
                'vector': vector,
                'message': message,
                'uuid': str(uuid4()),
                'timestring': timestring
            }
            filename = 'log_%s_Bort.json' % time()
            save_json('nexus/%s' % filename, info)
            await discord_message.channel.send(f'@{user}, {output}')
"""


@client.event
async def on_ready():
    channel = client.get_channel(CHAN_ID)
    if channel is not None:
        await channel.send(f"{client.user} has connected to Discord!")
    else:
        print("Channel not found")


if __name__ == "__main__":
    with open("/Users/alexthe5th/Documents/API Keys/OpenAI_API_key.txt", "r") as f:
        key = f.read().strip()
        openai.api_key = key
    auth = load_json('/Users/alexthe5th/Documents/API Keys/Discord_auth.json')
    TOKEN = auth['token'].strip()
    CHAN_ID = int(auth['chan_id'].strip())
    try:
        client.run(TOKEN)
    except Exception as oops:
        print('Error in main:', oops)

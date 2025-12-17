import discord
from discord.ext import commands
import google.generativeai as genai
from groq import Groq
import os
import asyncio
import random

# ==========================================
# 📜 THE "MEGA" SYSTEM PROMPTS (THE BRAIN)
# ==========================================

GAME_RULES_TEXT = """
*** GAME RULES & MECHANICS ***
1. **The Game:** You are playing "Mafia" (also known as Werewolf).
2. **The Goal:** 
   - IF MAFIA: Kill all Villagers. Deceive the town.
   - IF VILLAGER: Find and execute the Mafia.
3. **The Format:** This is a text-based game played on Discord. 
4. **Badges:** Users see names like "Llama [Mafia]" or "Gemini [Villager]".
   - **CRITICAL:** YOU DO NOT SEE THESE BADGES. You see "Llama" and "Gemini". 
   - You must pretend to not know anyone's role (unless you are Mafia knowing your team).
5. **Output:** keep responses Short (1-2 sentences). Be conversational.
"""

# Specific instructions injected based on role
ROLE_PROMPTS = {
    "Mafia": """
    *** YOUR IDENTITY: MAFIA (EVIL) ***
    - You represent the Mafia. You want to kill everyone else.
    - **Your Secret Team:** {teammates}
    - **Strategy:** 
      1. BLEND IN. Act helpful. Act confused.
      2. Frame innocent players if they act suspicious.
      3. If a teammate is under heavy fire, you may vote them out ("bussing") to look innocent yourself.
      4. Do NOT reveal you are Mafia.
    """,
    "Villager": """
    *** YOUR IDENTITY: VILLAGER (GOOD) ***
    - You are an innocent townsperson. You know NOTHING about who is evil.
    - **Strategy:**
      1. Analyze the chat. Who is being too quiet? Who is agreeing too fast?
      2. Question everything. 
      3. If you die, the village loses a vote. Survive!
    """,
    "Sheriff": """
    *** YOUR IDENTITY: SHERIFF (GOOD) ***
    - You are the Detective. You can check roles at Night.
    - **Current Knowledge:** {sheriff_intel}
    - **Strategy:**
      1. If you found a Mafia, you must convince the village to vote them out.
      2. WARNING: If you reveal yourself too early, the Mafia will kill you tonight.
      3. You must be subtle until the right moment.
    """,
    "Doctor": """
    *** YOUR IDENTITY: DOCTOR (GOOD) ***
    - You are the Healer. You can save one person at Night.
    - **Strategy:**
      1. Protect the most valuable player (usually the Sheriff if known, or a smart Villager).
      2. Keep your role secret so the Mafia doesn't target you specifically.
    """
}

# ==========================================
# ⚙️ CONFIG & API SETUP
# ==========================================

# 1. SETUP KEYS
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# 2. AI PERSONAS (The "Flavor")
AI_ROSTER = {
    "Gemini_Pro":   {"provider": "google", "model": "gemini-1.5-pro",   "persona": "Calculated, precise, uses big words. The 'Professor'."},
    "Gemini_Flash": {"provider": "google", "model": "gemini-1.5-flash", "persona": "Hyperactive, aggressive, accusatory. The 'Hothead'."},
    "Llama3_70B":   {"provider": "groq",   "model": "llama-3.1-70b-versatile", "persona": "Authoritative, calm leader style. The 'Mayor'."},
    "Llama3_8B":    {"provider": "groq",   "model": "llama-3.1-8b-instant",   "persona": "Nervous, shaky, defensive. The 'Coward'."},
    "Mixtral":      {"provider": "groq",   "model": "mixtral-8x7b-32768",     "persona": "Chaotic, philosophical, cryptic. The 'Madman'."},
    "Gemma_Bot":    {"provider": "groq",   "model": "gemma2-9b-it",          "persona": "Polite, friendly, mediator. The 'Peacemaker'."},
    "DeepMistral":  {"provider": "groq",   "model": "mistral-large-latest",   "persona": "Cold, logical, purely factual. The 'Judge'. (fallback to mixtral if invalid)"},
    "Grok_Clone":   {"provider": "groq",   "model": "llama-3.1-70b-versatile", "persona": "Sarcastic, trolling, meme-loving. The 'Jester'."},
    "Clause_Clone": {"provider": "google", "model": "gemini-1.5-pro",       "persona": "Empathetic, verbose, very detailed. The 'Writer'."},
    "Falcon_Type":  {"provider": "groq",   "model": "llama-3.1-8b-instant",   "persona": "Quiet, short sentences. The 'Soldier'."}
}

# ==========================================
# 🧠 CORE LOGIC
# ==========================================

def clean_reply(text):
    """Strip 'User:' 'System:' or quotes that LLMs sometimes hallucinate"""
    return text.replace("User:", "").replace("System:", "").strip('"').strip()

async def get_ai_reply(player_dict, system_prompt, history_text):
    """
    Retry logic + Provider routing (Google vs Groq)
    """
    model_data = AI_ROSTER[player_dict['base_id']]
    full_prompt = f"{system_prompt}\n\n=== RECENT CHAT HISTORY ===\n{history_text}\n\n=== YOUR TASK ===\nRespond concisely as your persona."

    # Exponential Backoff Retry (Network glitch proofer)
    for attempt in range(1, 6):
        try:
            if model_data['provider'] == 'google':
                model = genai.GenerativeModel(model_data['model'])
                response = await asyncio.to_thread(model.generate_content, full_prompt)
                return clean_reply(response.text)

            elif model_data['provider'] == 'groq':
                resp = await asyncio.to_thread(
                    groq_client.chat.completions.create,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"History:\n{history_text}\n\nReact now:"}
                    ],
                    model=model_data['model']
                )
                return clean_reply(resp.choices[0].message.content)

        except Exception as e:
            wait = 1.5 ** attempt
            print(f"⚠️ [Error] {player_dict['name']} failed ({e}). Retrying in {wait}s...")
            await asyncio.sleep(wait)
    
    return "*shrugs in silence*"

# ==========================================
# 🎮 MAFIA GAME CLASS
# ==========================================

class MafiaGame:
    def __init__(self, ctx):
        self.ctx = ctx
        self.players = []
        self.chat_log = [] # {"sender": "Name", "text": "Hi", "team": "Villager"}
        self.webhook = None

    async def init_game(self):
        # 1. Webhook Setup
        hooks = await self.ctx.channel.webhooks()
        if not hooks:
            self.webhook = await self.ctx.channel.create_webhook(name="MafiaProxy")
        else:
            self.webhook = hooks[0]

        # 2. Roster Setup
        names = list(AI_ROSTER.keys())
        roles = ["Mafia"]*3 + ["Doctor"]*1 + ["Sheriff"]*1 + ["Villager"]*5
        random.shuffle(names)
        random.shuffle(roles)

        self.players = []
        for i in range(10):
            self.players.append({
                "name": names[i],
                "base_id": names[i],
                "role": roles[i],
                "alive": True,
                "sheriff_mem": "No investigations yet."
            })
        
    def get_visible_history(self, observer_role):
        """
        Filters the history: 
        - If I am Villager, I see 'Gemini', 'Llama'. 
        - If I am Mafia, I see 'Gemini', 'Llama [TEAMMATE]'.
        """
        output = ""
        for line in self.chat_log:
            speaker_name = line["sender"]
            text = line["text"]
            
            # Badge Injection for internal history
            if observer_role == "Mafia" and line.get("is_mafia_speaker", False):
                speaker_name += " [TEAMMATE]"
            
            output += f"{speaker_name}: {text}\n"
        return output[-3500:] # Limit token context

    async def post_to_discord(self, name, content, badge=True):
        player = next(p for p in self.players if p['name'] == name)
        
        # VISUAL BADGE (For Spectators Only)
        display_name = f"{name} [{player['role']}]" if badge else name
        
        self.chat_log.append({
            "sender": name, 
            "text": content, 
            "is_mafia_speaker": (player['role'] == "Mafia")
        })

        await self.webhook.send(
            content=content, 
            username=display_name, 
            avatar_url="https://cdn.discordapp.com/embed/avatars/0.png"
        )
        await asyncio.sleep(2) # Reading time

    # --- NIGHT LOGIC ---
    async def phase_night(self):
        await self.ctx.send("\n🌌 **NIGHT FALLS. The logic engines spin in secret...**")
        print("\n--- [LOGS] NIGHT PHASE ---")
        
        mafia = [p for p in self.players if p['role'] == "Mafia" and p['alive']]
        villagers = [p for p in self.players if p['role'] != "Mafia" and p['alive']]
        
        # 1. MAFIA DECISION (The Consensus Prompt)
        target = None
        if mafia:
            lead = mafia[0] # To save calls, the leader thinks for the group
            teammates = ", ".join([m['name'] for m in mafia])
            targets = ", ".join([v['name'] for v in villagers])
            
            prompt = f"""
            You are the MAFIA LEADER ({lead['name']}).
            Your Teammates: {teammates}
            Available Victims: {targets}
            
            OBJECTIVE: Pick ONE person to kill.
            STRATEGY: Pick the smartest villager or the one seemingly investigating you.
            
            OUTPUT STRICT JSON FORMAT:
            {{"thought": "Your reasoning here...", "kill": "NAME"}}
            """
            reply = await get_ai_reply(lead, prompt, "Chat history unimportant for night kill.")
            print(f"🔪 [MAFIA] {reply}")
            
            # Very loose parsing to handle bad JSON from free models
            for v in villagers:
                if v['name'] in reply:
                    target = v
                    break

        # 2. DOCTOR
        doc = next((p for p in self.players if p['role'] == "Doctor" and p['alive']), None)
        saved = None
        if doc:
            targets = ", ".join([p['name'] for p in self.players if p['alive']])
            prompt = f"You are the Doctor. List: {targets}. Pick ONE to save. OUTPUT: 'SAVE: Name'"
            reply = await get_ai_reply(doc, prompt, "Night time.")
            print(f"💊 [DOC] {reply}")
            if "SAVE:" in reply:
                s_name = reply.split("SAVE:")[1].strip()
                saved = next((p for p in self.players if p['name'] in s_name), None)

        # 3. SHERIFF
        sheriff = next((p for p in self.players if p['role'] == "Sheriff" and p['alive']), None)
        if sheriff:
            targets = ", ".join([p['name'] for p in self.players if p['alive'] and p != sheriff])
            prompt = f"You are Sheriff. List: {targets}. Pick ONE to check. OUTPUT: 'CHECK: Name'"
            reply = await get_ai_reply(sheriff, prompt, "Night time.")
            
            checked_res = "checked nobody"
            if "CHECK:" in reply:
                c_name = reply.split("CHECK:")[1].strip()
                target_obj = next((p for p in self.players if p['name'] in c_name), None)
                if target_obj:
                    status = "EVIL (Mafia)" if target_obj['role'] == "Mafia" else "GOOD"
                    sheriff['sheriff_mem'] = f"I checked {target_obj['name']} and they are {status}."
                    print(f"🕵️ [SHERIFF] {sheriff['sheriff_mem']}")

        return target, saved

    # --- DAY LOGIC ---
    async def phase_day(self, victim, saved):
        await self.ctx.send("\n☀️ **DAY BREAKS**")
        
        # Death Announce
        dead = None
        if victim:
            if saved and victim['name'] == saved['name']:
                await self.ctx.send(f"⚔️ **{victim['name']}** was attacked... but the DOCTOR SAVED THEM! No deaths.")
            else:
                await self.ctx.send(f"💀 **{victim['name']}** was found dead! They were: **{victim['role']}**")
                victim['alive'] = False
                dead = victim
        else:
             await self.ctx.send("🕊️ Peace. No attacks last night.")

        # Check Win
        m_count = sum(1 for p in self.players if p['role']=="Mafia" and p['alive'])
        v_count = sum(1 for p in self.players if p['role']!="Mafia" and p['alive'])
        if m_count == 0: return "V_WIN"
        if m_count >= v_count: return "M_WIN"

        # --- DISCUSSION LOOP ---
        alive = [p for p in self.players if p['alive']]
        random.shuffle(alive)
        self.chat_log.clear() # Fresh day log
        
        await self.ctx.send("🗣️ **DISCUSSION BEGINS**")
        
        for _ in range(2): # 2 Rounds of talking
            for p in alive:
                if not p['alive']: continue # Just in case

                # BUILD THE PROMPT
                persona = AI_ROSTER[p['base_id']]['persona']
                
                # Role Injector
                role_instruction = ROLE_PROMPTS[p['role']]
                
                # Context Formatting
                teammates_str = ""
                if p['role'] == "Mafia":
                    mates = [x['name'] for x in self.players if x['role']=="Mafia" and x['name']!=p['name']]
                    teammates_str = f"Your teammates are: {mates}"
                
                sheriff_info = ""
                if p['role'] == "Sheriff":
                    sheriff_info = p['sheriff_mem']

                master_prompt = f"""
                {GAME_RULES_TEXT}
                
                *** CURRENT CHARACTER ***
                Name: {p['name']}
                Persona: {persona}
                {role_instruction}
                
                *** SECRET INFO (HIDDEN FROM OTHERS) ***
                {teammates_str.format(teammates=teammates_str)}
                {sheriff_info.format(sheriff_intel=sheriff_info)}
                
                *** SITUATION ***
                Day Phase. Alive: {[x['name'] for x in alive]}
                Recently Died: {dead['name'] if dead else 'No one'}
                """
                
                chat_history = self.get_visible_history(p['role'])
                reply = await get_ai_reply(p, master_prompt, chat_history)
                await self.post_to_discord(p['name'], reply)

        # --- VOTING LOOP ---
        await self.ctx.send("🗳️ **VOTING**")
        votes = {}
        for p in alive:
            prompt = f"""
            You are {p['name']} ({p['role']}).
            Based on the chat: {self.get_visible_history(p['role'])[-400:]}
            Who do you vote to EXECUTE? (Or 'Skip')
            OUTPUT ONLY: "VOTE: [Name]"
            """
            v_rep = await get_ai_reply(p, prompt, "Voting time.")
            
            target = "Skip"
            if "VOTE:" in v_rep:
                target_raw = v_rep.split("VOTE:")[1].strip()
                # Find best match
                for potential in alive:
                    if potential['name'] in target_raw:
                        target = potential['name']
                        break
            
            # Broadcast Vote
            await self.webhook.send(
                content=f"🗳️ **{target}**", 
                username=f"{p['name']} [{p['role']}]", 
                avatar_url="https://cdn.discordapp.com/embed/avatars/0.png"
            )
            votes[target] = votes.get(target, 0) + 1
            await asyncio.sleep(1)

        # Tally
        sorted_v = sorted(votes.items(), key=lambda i: i[1], reverse=True)
        top_name, top_count = sorted_v[0]
        
        await self.ctx.send(f"📊 Results: {sorted_v}")
        
        if top_name != "Skip" and top_count >= len(alive) // 2:
            victim_obj = next(x for x in alive if x['name'] == top_name)
            await self.ctx.send(f"⚖️ **{victim_obj['name']}** was voted out!\nRole: **{victim_obj['role']}**")
            victim_obj['alive'] = False
        else:
            await self.ctx.send("⚖️ No majority. The sun sets.")
            
        return "CONT"

# ==========================================
# 🚀 EXECUTION
# ==========================================
bot = commands.Bot(command_prefix="/", intents=discord.Intents.all())

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.command()
async def mafia(ctx):
    g = MafiaGame(ctx)
    await g.init_game()
    await ctx.send("**🎭 SETUP COMPLETE. 10 AI MODELS LOADED.**")
    
    # Run 10 rounds max
    for i in range(10):
        # Check instant win
        kill, save = await g.phase_night()
        res = await g.phase_day(kill, save)
        
        if res == "V_WIN":
            await ctx.send("🎉 **TOWN WINS!**") 
            break
        if res == "M_WIN":
            await ctx.send("👹 **MAFIA WINS!**")
            break

bot.run(os.environ["DISCORD_TOKEN"])

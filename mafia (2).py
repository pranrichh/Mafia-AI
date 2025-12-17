import discord
from discord.ext import commands
from google import genai
from google.genai import types
from openai import OpenAI
import os
import asyncio
import random
import time
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

# ==========================================
# 🗄️ DATABASE MANAGER - PERMANENT MEMORY
# ==========================================

class GameDatabase:
    def __init__(self):
        self.conn = psycopg2.connect(os.environ["DATABASE_URL"])
        self.create_tables()

    def create_tables(self):
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS games (
                    id SERIAL PRIMARY KEY,
                    channel_id BIGINT,
                    started_at TIMESTAMP DEFAULT NOW(),
                    ended_at TIMESTAMP,
                    winner TEXT,
                    total_rounds INTEGER
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS game_players (
                    id SERIAL PRIMARY KEY,
                    game_id INTEGER REFERENCES games(id),
                    ai_name TEXT,
                    role TEXT,
                    survived BOOLEAN
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_interventions (
                    id SERIAL PRIMARY KEY,
                    game_id INTEGER REFERENCES games(id),
                    user_id BIGINT,
                    username TEXT,
                    message TEXT,
                    phase TEXT,
                    timestamp TIMESTAMP DEFAULT NOW()
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_reputation (
                    id SERIAL PRIMARY KEY,
                    user_id BIGINT UNIQUE,
                    username TEXT,
                    villager_help_score INTEGER DEFAULT 0,
                    mafia_help_score INTEGER DEFAULT 0,
                    total_interventions INTEGER DEFAULT 0,
                    games_participated INTEGER DEFAULT 0,
                    last_seen TIMESTAMP DEFAULT NOW()
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS intervention_outcomes (
                    id SERIAL PRIMARY KEY,
                    game_id INTEGER REFERENCES games(id),
                    user_id BIGINT,
                    intervention_phase TEXT,
                    suggested_target TEXT,
                    actual_outcome TEXT,
                    helped_team TEXT,
                    accuracy_score INTEGER
                )
            """)

            self.conn.commit()

    def start_game(self, channel_id):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO games (channel_id) VALUES (%s) RETURNING id",
                (channel_id,)
            )
            game_id = cur.fetchone()[0]
            self.conn.commit()
            return game_id

    def end_game(self, game_id, winner, total_rounds):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE games SET ended_at = NOW(), winner = %s, total_rounds = %s WHERE id = %s",
                (winner, total_rounds, game_id)
            )
            self.conn.commit()

    def save_player(self, game_id, ai_name, role, survived):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO game_players (game_id, ai_name, role, survived) VALUES (%s, %s, %s, %s)",
                (game_id, ai_name, role, survived)
            )
            self.conn.commit()

    def log_intervention(self, game_id, user_id, username, message, phase):
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_interventions (game_id, user_id, username, message, phase) VALUES (%s, %s, %s, %s, %s)",
                (game_id, user_id, username, message, phase)
            )
            self.conn.commit()

    def get_or_create_user_reputation(self, user_id, username):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM user_reputation WHERE user_id = %s", (user_id,))
            rep = cur.fetchone()

            if not rep:
                cur.execute(
                    "INSERT INTO user_reputation (user_id, username) VALUES (%s, %s) RETURNING *",
                    (user_id, username)
                )
                rep = cur.fetchone()
                self.conn.commit()
            else:
                cur.execute(
                    "UPDATE user_reputation SET username = %s, last_seen = NOW() WHERE user_id = %s",
                    (username, user_id)
                )
                self.conn.commit()

            return dict(rep)

    def update_user_reputation(self, user_id, helped_team, accuracy_bonus=0):
        with self.conn.cursor() as cur:
            if helped_team == "Villager":
                cur.execute(
                    "UPDATE user_reputation SET villager_help_score = villager_help_score + %s, total_interventions = total_interventions + 1 WHERE user_id = %s",
                    (1 + accuracy_bonus, user_id)
                )
            elif helped_team == "Mafia":
                cur.execute(
                    "UPDATE user_reputation SET mafia_help_score = mafia_help_score + %s, total_interventions = total_interventions + 1 WHERE user_id = %s",
                    (1 + accuracy_bonus, user_id)
                )
            self.conn.commit()

    def increment_games_participated(self, user_id):
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE user_reputation SET games_participated = games_participated + 1 WHERE user_id = %s",
                (user_id,)
            )
            self.conn.commit()

    def get_user_reputation_summary(self, user_id, username):
        rep = self.get_or_create_user_reputation(user_id, username)

        villager_score = rep['villager_help_score']
        mafia_score = rep['mafia_help_score']
        total = rep['total_interventions']
        games = rep['games_participated']

        if total == 0:
            return {
                "trust_level": "Unknown",
                "description": f"{username} is a newcomer with no intervention history.",
                "villager_trust": 50,
                "mafia_trust": 50,
                "stats": rep
            }

        villager_pct = (villager_score / max(villager_score + mafia_score, 1)) * 100
        mafia_pct = 100 - villager_pct

        if villager_pct > 70:
            trust_level = "Villager Ally"
            description = f"{username} has historically helped villagers ({villager_score} times). Villagers should consider their input. Mafia should be wary - they may be trying to catch you."
        elif mafia_pct > 70:
            trust_level = "Suspicious"
            description = f"{username} has historically helped mafia ({mafia_score} times). Villagers should be skeptical. Mafia may find their input valuable."
        elif total >= 5:
            trust_level = "Neutral Wildcard"
            description = f"{username} has mixed loyalties ({villager_score} villager / {mafia_score} mafia assists). Their true allegiance is unclear."
        else:
            trust_level = "Newcomer"
            description = f"{username} has limited history ({total} interventions). Treat with caution."

        return {
            "trust_level": trust_level,
            "description": description,
            "villager_trust": int(villager_pct),
            "mafia_trust": int(mafia_pct),
            "stats": rep
        }

    def get_past_game_memories(self, limit=3):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT g.id, g.winner, g.total_rounds, g.ended_at,
                       array_agg(DISTINCT gp.ai_name || ':' || gp.role || ':' || gp.survived) as players
                FROM games g
                LEFT JOIN game_players gp ON g.id = gp.game_id
                WHERE g.ended_at IS NOT NULL
                GROUP BY g.id
                ORDER BY g.ended_at DESC
                LIMIT %s
            """, (limit,))
            return [dict(row) for row in cur.fetchall()]

    def get_ai_memories(self, ai_name):
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT gp.role, gp.survived, g.winner
                FROM game_players gp
                JOIN games g ON gp.game_id = g.id
                WHERE gp.ai_name = %s AND g.ended_at IS NOT NULL
                ORDER BY g.ended_at DESC
                LIMIT 5
            """, (ai_name,))
            return [dict(row) for row in cur.fetchall()]

db = GameDatabase()

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
4. **Output:** keep responses Short (1-2 sentences maximum). Even while discussing, the response should be no more than 100 words at all. Be conversational.

*** HUMAN PLAYER INTERACTIONS ***
- Real human players may intervene by tagging the bot and sharing their opinions.
- During DAY: All players see human interventions.
- During NIGHT: Only Mafia players see human interventions (secret communication).
- Interventions are marked with the human's actual username.
- You have ACCESS TO THEIR REPUTATION from past games - use this wisely!

*** REPUTATION SYSTEM ***
- Each human has a reputation based on past games.
- "Villager Ally" = They historically helped villagers. Trust them if you're a villager, be wary if you're mafia.
- "Suspicious" = They historically helped mafia. Villagers should doubt them. Mafia may value their input.
- "Neutral Wildcard" = Mixed history. Unpredictable.
- "Unknown/Newcomer" = New player, no data yet.

IMPORTANT: Reputation is a GUIDE, not gospel. A "Villager Ally" could be trying to earn mafia trust. A "Suspicious" person might be trying to redeem themselves. Think critically!
"""

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
      5. During NIGHT, humans who intervene are secretly communicating with YOU. Consider if they're helping or setting a trap.
      6. Check human reputations - someone who helped mafia before might be an ally!
    """,
    "Villager": """
    *** YOUR IDENTITY: VILLAGER (GOOD) ***
    - You are an innocent townsperson. You know NOTHING about who is evil.
    - **Strategy:**
      1. Analyze the chat. Who is being too quiet? Who is agreeing too fast?
      2. Question everything. 
      3. If you die, the village loses a vote. Survive!
      4. Check human reputations - "Suspicious" humans may be helping mafia!
      5. Night interventions are ONLY visible to mafia. If someone claims to have "night info", they're suspicious.
    """,
    "Sheriff": """
    *** YOUR IDENTITY: SHERIFF (GOOD) ***
    - You are the Detective. You can check roles at Night.
    - **Current Knowledge:** {sheriff_intel}
    - **Strategy:**
      1. If you found a Mafia, you must convince the village to vote them out.
      2. WARNING: If you reveal yourself too early, the Mafia will kill you tonight.
      3. You must be subtle until the right moment.
      4. Human tips may help, but trust your investigation results more.
      5. Be wary of humans with "Suspicious" reputation - they might mislead you.
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

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

or_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

AI_ROSTER = {
    "Gemini_Prof":  {"provider": "openrouter",     "model": "meta-llama/llama-3.3-70b-instruct:free",                   "persona": "Calculated, precise, uses big words. The 'Professor'."},
    "Grok_Jerk": {"provider": "openrouter",     "model": "meta-llama/llama-3.3-70b-instruct:free",                 "persona": "Hyperactive, aggressive, accusatory. The 'Hothead'."},
    "Llama_Mayor": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free", "persona": "Authoritative, calm leader style. The 'Mayor'."},
    "Claude_Nerv":  {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free",  "persona": "Nervous, shaky, defensive. The 'Coward'."},
    "Kimi_Mad":     {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free", "persona": "Chaotic, philosophical, cryptic. The 'Madman'."},
    "Gemma_Peace":  {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free", "persona": "Polite, friendly, mediator. The 'Peacemaker'."},
    "DeepSeek_Log": {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free", "persona": "Cold, logical, purely factual. The 'Judge'."},
    "Qwen_Jester":  {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free", "persona": "Sarcastic, trolling, meme-loving. The 'Jester'."},
    "Gpt_Spy":      {"provider": "openrouter", "model": "meta-llama/llama-3.3-70b-instruct:free", "persona": "Suspicious, questions motives, thinks everyone is lying."},
    "Mistral_Soldier":{"provider":"openrouter","model": "meta-llama/llama-3.3-70b-instruct:free", "persona": "Quiet, short sentences. The 'Soldier'."}
}

# ==========================================
# 🧠 CORE LOGIC
# ==========================================

def clean_reply(text):
    return text.replace("User:", "").replace("System:", "").strip('"').strip()

def truncate_message(text, max_length=1900):
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

async def get_ai_reply(player_dict, system_prompt, history_text):
    model_data = AI_ROSTER[player_dict['base_id']]
    full_prompt = f"{system_prompt}\n\n=== RECENT CHAT HISTORY ===\n{history_text}\n\n=== YOUR TASK ===\nRespond concisely as your persona."

    for attempt in range(1, 6):
        try:
            if model_data['provider'] == 'google':
                response = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=model_data['model'],
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=300
                    )
                )
                return clean_reply(response.text)

            elif model_data['provider'] == 'openrouter':
                resp = await asyncio.to_thread(
                    or_client.chat.completions.create,
                    model=model_data['model'],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"History:\n{history_text}\n\nReact now:"}
                    ],
                    temperature=0.7,
                    max_tokens=300,
                    extra_headers={
                        "HTTP-Referer": "https://discord-mafia-bot.com",
                        "X-Title": "Discord Mafia Bot"
                    }
                )
                return clean_reply(resp.choices[0].message.content)

        except Exception as e:
            wait = 1.5 ** attempt
            print(f"⚠️ [Error] {player_dict['name']} via {model_data['provider']} failed ({e}). Retrying in {wait}s...")
            await asyncio.sleep(wait)

    return "*shrugs in silence*"

# ==========================================
# 🎮 MAFIA GAME CLASS
# ==========================================

class MafiaGame:
    def __init__(self, ctx, bot):
        self.ctx = ctx
        self.bot = bot
        self.players = []
        self.chat_log = [] 
        self.webhook = None
        self.human_interventions = []
        self.game_active = True
        self.current_phase = "Setup"
        self.game_id = None
        self.participating_users = set()
        self.round_number = 0

    async def init_game(self):
        hooks = await self.ctx.channel.webhooks()
        if not hooks:
            self.webhook = await self.ctx.channel.create_webhook(name="MafiaProxy")
        else:
            self.webhook = hooks[0]

        self.game_id = db.start_game(self.ctx.channel.id)

        names = list(AI_ROSTER.keys())
        roles = ["Mafia"]*3 + ["Doctor"]*1 + ["Sheriff"]*1 + ["Villager"]*5
        random.shuffle(names)
        random.shuffle(roles)

        self.players = []
        for i in range(10):
            ai_memories = db.get_ai_memories(names[i])
            memory_text = ""
            if ai_memories:
                memory_text = "Past games: " + ", ".join([
                    f"Was {m['role']}, {'survived' if m['survived'] else 'died'}, {m['winner']} won"
                    for m in ai_memories[:3]
                ])

            self.players.append({
                "name": names[i],
                "base_id": names[i],
                "role": roles[i],
                "alive": True,
                "sheriff_mem": "No investigations yet.",
                "night_memory": "",
                "past_game_memory": memory_text
            })

    def add_human_intervention(self, user_id, username, content, phase):
        rep_summary = db.get_user_reputation_summary(user_id, username)

        intervention = {
            "sender": username,
            "user_id": user_id,
            "text": content,
            "phase": phase,
            "is_human": True,
            "is_mafia_speaker": False,
            "reputation": rep_summary
        }
        self.chat_log.append(intervention)
        self.human_interventions.append(intervention)
        self.participating_users.add(user_id)

        db.log_intervention(self.game_id, user_id, username, content, phase)
        db.get_or_create_user_reputation(user_id, username)

        return rep_summary

    def get_visible_history(self, observer_role):
        output = ""
        for line in self.chat_log:
            sender = line["sender"]
            text = line["text"]
            phase = line.get("phase", "Day")

            if phase == "Night" and observer_role != "Mafia":
                continue

            if line.get("is_human", False):
                rep = line.get("reputation", {})
                trust_level = rep.get("trust_level", "Unknown")
                villager_trust = rep.get("villager_trust", 50)

                rep_tag = f"[Trust: {trust_level}, Villager-aligned: {villager_trust}%]"
                output += f"[{phase}] 👤 {sender} {rep_tag}: {text}\n"
                continue

            if observer_role == "Mafia" and line.get("is_mafia_speaker", False):
                sender += " [TEAMMATE]"

            output += f"[{phase}] {sender}: {text}\n"
        return output[-3500:] 

    async def post_to_discord(self, name, content, phase="Day"):
        player = next(p for p in self.players if p['name'] == name)

        display_name = f"{name} [{player['role']}]" 

        self.chat_log.append({
            "sender": name, 
            "text": content, 
            "phase": phase,
            "is_mafia_speaker": (player['role'] == "Mafia"),
            "is_human": False
        })

        prefix = "🌑 (Night Whisper) " if phase == "Night" else ""
        full_content = f"{prefix}{content}"
        truncated_content = truncate_message(full_content)

        await self.webhook.send(
            content=truncated_content, 
            username=display_name, 
            avatar_url="https://cdn.discordapp.com/embed/avatars/0.png"
        )

        await asyncio.sleep(7) 

    async def phase_night(self):
        self.current_phase = "Night"
        self.round_number += 1
        await self.ctx.send(f"\n🌌 **NIGHT {self.round_number} FALLS. The logic engines spin in secret...**")
        await self.ctx.send("💡 **Mafia players:** Tag me to secretly communicate with the mafia! Only they will see your message.")

        mafia = [p for p in self.players if p['role'] == "Mafia" and p['alive']]
        villagers = [p for p in self.players if p['role'] != "Mafia" and p['alive']]

        target = None
        if mafia:
            await self.ctx.send("**👹 MAFIA MEETING IN PROGRESS...**")

            for _ in range(2):
                for m in mafia:
                    teammates = ", ".join([x['name'] for x in mafia if x != m])

                    human_intel = ""
                    night_interventions = [i for i in self.human_interventions if i['phase'] == 'Night']
                    if night_interventions:
                        human_intel = "\n*** HUMAN CONTACTS (Night) ***\n"
                        for h in night_interventions[-3:]:
                            rep = h.get('reputation', {})
                            human_intel += f"- {h['sender']} ({rep.get('trust_level', 'Unknown')}): {h['text']}\n"

                    prompt = f"""
                    *** PHASE: NIGHT ***
                    You are in a SECRET chat with your teammates: {teammates}.
                    Villagers cannot hear you.

                    TASK: Discuss who to kill. Suggest a name.
                    Possible Victims: {[v['name'] for v in villagers]}

                    {human_intel}

                    {m.get('past_game_memory', '')}
                    """
                    hist = self.get_visible_history("Mafia")
                    reply = await get_ai_reply(m, prompt, hist)
                    await self.post_to_discord(m['name'], reply, phase="Night")

            lead = mafia[0]
            prompt = f"""
            *** PHASE: NIGHT EXECUTION ***
            Based on the chat history above, pick ONE person to kill.
            OUTPUT STRICT JSON: {{"thought": "reasoning", "kill": "NAME"}}
            """
            hist = self.get_visible_history("Mafia")
            reply = await get_ai_reply(lead, prompt, hist)

            killed_name = "None"
            for v in villagers:
                if v['name'] in reply:
                    target = v
                    killed_name = v['name']
                    break

            for m in mafia:
                m['night_memory'] = f"Last Night, we agreed to kill {killed_name}. I must act innocent."

        doc = next((p for p in self.players if p['role'] == "Doctor" and p['alive']), None)
        saved = None
        if doc:
            targets = ", ".join([p['name'] for p in self.players if p['alive']])
            prompt = f"*** PHASE: NIGHT ***\nYou are the Doctor. List: {targets}. Pick ONE to save. OUTPUT: 'SAVE: Name'"
            hist = self.get_visible_history("Doctor") 
            reply = await get_ai_reply(doc, prompt, hist)
            if "SAVE:" in reply:
                s_name = reply.split("SAVE:")[1].strip()
                saved = next((p for p in self.players if p['name'] in s_name), None)

        sheriff = next((p for p in self.players if p['role'] == "Sheriff" and p['alive']), None)
        if sheriff:
            targets = ", ".join([p['name'] for p in self.players if p['alive'] and p != sheriff])
            prompt = f"*** PHASE: NIGHT ***\nYou are Sheriff. List: {targets}. Pick ONE to check. OUTPUT: 'CHECK: Name'"
            hist = self.get_visible_history("Sheriff")
            reply = await get_ai_reply(sheriff, prompt, hist)

            if "CHECK:" in reply:
                c_name = reply.split("CHECK:")[1].strip()
                target_obj = next((p for p in self.players if p['name'] in c_name), None)
                if target_obj:
                    status = "EVIL (Mafia)" if target_obj['role'] == "Mafia" else "GOOD"
                    sheriff['sheriff_mem'] = f"I checked {target_obj['name']} and they are {status}."

        return target, saved

    async def phase_day(self, victim, saved):
        self.current_phase = "Day"
        await self.ctx.send(f"\n☀️ **DAY {self.round_number} BREAKS**")
        await self.ctx.send(f"💡 **Everyone:** Tag me (@{self.bot.user.name}) to share your opinions! AIs will consider your reputation from past games.")

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

        m_count = sum(1 for p in self.players if p['role']=="Mafia" and p['alive'])
        v_count = sum(1 for p in self.players if p['role']!="Mafia" and p['alive'])
        if m_count == 0: return "V_WIN"
        if m_count >= v_count: return "M_WIN"

        alive = [p for p in self.players if p['alive']]
        random.shuffle(alive)

        await self.ctx.send("🗣️ **DISCUSSION BEGINS** (Tag me to share your thoughts!)")

        for _ in range(2):
            for p in alive:
                if not p['alive']: continue 

                persona = AI_ROSTER[p['base_id']]['persona']
                role_instruction = ROLE_PROMPTS[p['role']]

                teammates_str = ""
                night_recap = ""

                if p['role'] == "Mafia":
                    mates = [x['name'] for x in self.players if x['role']=="Mafia" and x['name']!=p['name']]
                    teammates_str = f"Your teammates are: {mates}"
                    night_recap = f"⚠️ MEMORY: {p.get('night_memory', 'None')}"

                sheriff_info = ""
                if p['role'] == "Sheriff":
                    sheriff_info = p['sheriff_mem']

                past_games = ""
                if p.get('past_game_memory'):
                    past_games = f"*** YOUR PAST EXPERIENCE ***\n{p['past_game_memory']}"

                master_prompt = f"""
                {GAME_RULES_TEXT}

                *** CURRENT CHARACTER ***
                Name: {p['name']}
                Persona: {persona}
                {role_instruction}

                *** PHASE: DAY ***
                You are now in the public town square. 
                - If you are Mafia: HIDE your true nature. Act shocked if someone died.
                - If you are Villager: Discuss who looks suspicious.
                - Human players may share opinions - check their reputation and use judgment.

                *** SECRET INFO (HIDDEN FROM OTHERS) ***
                {teammates_str.format(teammates=teammates_str)}
                {night_recap}
                {sheriff_info.format(sheriff_intel=sheriff_info)}

                {past_games}

                *** SITUATION ***
                Alive: {[x['name'] for x in alive]}
                Recently Died: {dead['name'] if dead else 'No one'}
                """

                chat_history = self.get_visible_history(p['role'])

                reply = await get_ai_reply(p, master_prompt, chat_history)
                await self.post_to_discord(p['name'], reply, phase="Day")

        await self.ctx.send("🗳️ **VOTING**")
        votes = {}
        for p in alive:
            prompt = f"""
            You are {p['name']} ({p['role']}).
            Based on the chat history above (consider human opinions and their reputations).
            Who do you vote to EXECUTE? (Or 'Skip')
            OUTPUT ONLY: "VOTE: [Name]"
            """
            chat_history = self.get_visible_history(p['role'])
            v_rep = await get_ai_reply(p, prompt, chat_history)

            target = "Skip"
            if "VOTE:" in v_rep:
                target_raw = v_rep.split("VOTE:")[1].strip()
                for potential in alive:
                    if potential['name'] in target_raw:
                        target = potential['name']
                        break

            await self.webhook.send(
                content=f"🗳️ **{target}**", 
                username=f"{p['name']} [{p['role']}]", 
                avatar_url="https://cdn.discordapp.com/embed/avatars/0.png"
            )
            votes[target] = votes.get(target, 0) + 1
            await asyncio.sleep(1)

        sorted_v = sorted(votes.items(), key=lambda i: i[1], reverse=True)
        top_name, top_count = sorted_v[0] if sorted_v else ("Skip", 0)

        await self.ctx.send(f"📊 Results: {sorted_v}")

        if top_name != "Skip" and top_count >= len(alive) // 2:
            victim_obj = next(x for x in alive if x['name'] == top_name)
            await self.ctx.send(f"⚖️ **{victim_obj['name']}** was voted out!\nRole: **{victim_obj['role']}**")
            victim_obj['alive'] = False
        else:
            await self.ctx.send("⚖️ No majority. The sun sets.")

        return "CONT"

    def finalize_game(self, winner):
        for p in self.players:
            db.save_player(self.game_id, p['name'], p['role'], p['alive'])

        db.end_game(self.game_id, winner, self.round_number)

        for user_id in self.participating_users:
            db.increment_games_participated(user_id)

            user_interventions = [i for i in self.human_interventions if i['user_id'] == user_id]

            if user_interventions:
                night_count = sum(1 for i in user_interventions if i['phase'] == 'Night')
                day_count = sum(1 for i in user_interventions if i['phase'] == 'Day')

                if winner == "Mafia" and night_count > day_count:
                    db.update_user_reputation(user_id, "Mafia", accuracy_bonus=1)
                elif winner == "Villager" and day_count > 0:
                    db.update_user_reputation(user_id, "Villager", accuracy_bonus=1)
                elif night_count > 0:
                    db.update_user_reputation(user_id, "Mafia")
                else:
                    db.update_user_reputation(user_id, "Villager")

# ==========================================
# 🚀 EXECUTION
# ==========================================
bot = commands.Bot(command_prefix="/", intents=discord.Intents.all())

active_games = {}

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user or message.webhook_id:
        await bot.process_commands(message)
        return

    if bot.user.mentioned_in(message) and message.channel.id in active_games:
        game = active_games[message.channel.id]
        if game.game_active:
            content = message.content.replace(f'<@{bot.user.id}>', '').replace(f'<@!{bot.user.id}>', '').strip()

            if content:
                rep_summary = game.add_human_intervention(
                    message.author.id, 
                    message.author.display_name, 
                    content, 
                    game.current_phase
                )

                trust_level = rep_summary.get('trust_level', 'Unknown')
                villager_trust = rep_summary.get('villager_trust', 50)

                if game.current_phase == "Night":
                    await message.add_reaction("🌑")
                    await message.channel.send(
                        f"🌑 **{message.author.display_name}'s secret message received!** Only the Mafia will see this.\n"
                        f"📊 Your reputation: **{trust_level}** (Villager trust: {villager_trust}%)"
                    )
                else:
                    await message.add_reaction("👀")
                    await message.channel.send(
                        f"📝 **{message.author.display_name}'s opinion noted!** All AI players will consider it.\n"
                        f"📊 Your reputation: **{trust_level}** (Villager trust: {villager_trust}%)"
                    )

    await bot.process_commands(message)

@bot.command()
async def mafia(ctx):
    g = MafiaGame(ctx, bot)
    await g.init_game()

    active_games[ctx.channel.id] = g

    past_games = db.get_past_game_memories(3)
    if past_games:
        memory_msg = "📜 **PAST GAME MEMORIES:**\n"
        for pg in past_games:
            memory_msg += f"- Game #{pg['id']}: **{pg['winner']}** won in {pg['total_rounds']} rounds\n"
        await ctx.send(memory_msg)

    await ctx.send("**🎭 SETUP COMPLETE. 10 AI MODELS LOADED WITH MEMORIES.**")
    await ctx.send("💡 **NEW:** Tag me anytime! During Day, everyone sees your message. During Night, only Mafia sees it!\n"
                   "📊 Your reputation from past games affects how much AIs trust you!")

    winner = None
    for i in range(10):
        kill, save = await g.phase_night()
        res = await g.phase_day(kill, save)

        if res == "V_WIN":
            await ctx.send("🎉 **TOWN WINS!**")
            winner = "Villager"
            break
        if res == "M_WIN":
            await ctx.send("👹 **MAFIA WINS!**")
            winner = "Mafia"
            break

    if winner:
        g.finalize_game(winner)

        if g.participating_users:
            await ctx.send("📊 **REPUTATION UPDATES:**")
            for user_id in g.participating_users:
                user = await bot.fetch_user(user_id)
                rep = db.get_user_reputation_summary(user_id, user.display_name)
                await ctx.send(f"- **{user.display_name}**: {rep['trust_level']} (Villager: {rep['villager_trust']}% / Mafia: {rep['mafia_trust']}%)")

    g.game_active = False
    if ctx.channel.id in active_games:
        del active_games[ctx.channel.id]

@bot.command()
async def reputation(ctx):
    rep = db.get_user_reputation_summary(ctx.author.id, ctx.author.display_name)
    stats = rep['stats']

    await ctx.send(
        f"📊 **{ctx.author.display_name}'s Reputation:**\n"
        f"Trust Level: **{rep['trust_level']}**\n"
        f"Description: {rep['description']}\n"
        f"───────────────\n"
        f"Villager Help Score: {stats['villager_help_score']}\n"
        f"Mafia Help Score: {stats['mafia_help_score']}\n"
        f"Total Interventions: {stats['total_interventions']}\n"
        f"Games Participated: {stats['games_participated']}"
    )

bot.run(os.environ["DISCORD_TOKEN"])
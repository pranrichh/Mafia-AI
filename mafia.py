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
        self.conn = None
        self.connect()

    def connect(self):
        try:
            if self.conn:
                self.conn.close()
        except:
            pass
        self.conn = psycopg2.connect(os.environ["DATABASE_URL"])
        self.conn.autocommit = False
        self.create_tables()

    def ensure_connection(self):
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            self.connect()

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
                    survived BOOLEAN,
                    is_ai BOOLEAN DEFAULT TRUE
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
                    last_seen TIMESTAMP DEFAULT NOW(),
                    is_human BOOLEAN DEFAULT TRUE
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
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO games (channel_id) VALUES (%s) RETURNING id",
                (channel_id,)
            )
            game_id = cur.fetchone()[0]
            self.conn.commit()
            return game_id

    def end_game(self, game_id, winner, total_rounds):
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE games SET ended_at = NOW(), winner = %s, total_rounds = %s WHERE id = %s",
                (winner, total_rounds, game_id)
            )
            self.conn.commit()

    def save_player(self, game_id, ai_name, role, survived, is_ai=True):
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO game_players (game_id, ai_name, role, survived, is_ai) VALUES (%s, %s, %s, %s, %s)",
                (game_id, ai_name, role, survived, is_ai)
            )
            self.conn.commit()

    def log_intervention(self, game_id, user_id, username, message, phase):
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_interventions (game_id, user_id, username, message, phase) VALUES (%s, %s, %s, %s, %s)",
                (game_id, user_id, username, message, phase)
            )
            self.conn.commit()

    def get_or_create_user_reputation(self, user_id, username):
        self.ensure_connection()
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM user_reputation WHERE user_id = %s", (user_id,))
            rep = cur.fetchone()

            if not rep:
                cur.execute(
                    "INSERT INTO user_reputation (user_id, username, is_human) VALUES (%s, %s, TRUE) RETURNING *",
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
                cur.execute("SELECT * FROM user_reputation WHERE user_id = %s", (user_id,))
                rep = cur.fetchone()

            return dict(rep) if rep else {"user_id": user_id, "username": username, "villager_help_score": 0, "mafia_help_score": 0, "total_interventions": 0, "games_participated": 0}

    def update_user_reputation(self, user_id, helped_team, accuracy_bonus=0):
        self.ensure_connection()
        with self.conn.cursor() as cur:
            if helped_team == "Villager":
                cur.execute(
                    "UPDATE user_reputation SET villager_help_score = villager_help_score + %s WHERE user_id = %s",
                    (1 + accuracy_bonus, user_id)
                )
            elif helped_team == "Mafia":
                cur.execute(
                    "UPDATE user_reputation SET mafia_help_score = mafia_help_score + %s WHERE user_id = %s",
                    (1 + accuracy_bonus, user_id)
                )
            self.conn.commit()

    def increment_intervention_count(self, user_id):
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE user_reputation SET total_interventions = total_interventions + 1 WHERE user_id = %s",
                (user_id,)
            )
            self.conn.commit()

    def increment_games_participated(self, user_id):
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE user_reputation SET games_participated = games_participated + 1 WHERE user_id = %s",
                (user_id,)
            )
            self.conn.commit()

    def get_user_reputation_summary(self, user_id, username):
        self.ensure_connection()
        rep = self.get_or_create_user_reputation(user_id, username)

        villager_score = rep.get('villager_help_score', 0) or 0
        mafia_score = rep.get('mafia_help_score', 0) or 0
        total = rep.get('total_interventions', 0) or 0
        games = rep.get('games_participated', 0) or 0

        if total == 0:
            return {
                "trust_level": "Unknown",
                "description": f"{username} is a newcomer with no intervention history.",
                "villager_trust": 50,
                "mafia_trust": 50,
                "stats": rep
            }

        total_score = villager_score + mafia_score
        if total_score == 0:
            villager_pct = 50
        else:
            villager_pct = (villager_score / total_score) * 100
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
        self.ensure_connection()
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
        self.ensure_connection()
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
4. **Output:** Keep responses short (1-2 sentences maximum). Be conversational. Focus on the game.

*** AI PLAYERS ***
- There are 10 AI PLAYERS in this game (you are one of them).
- AI player names: Gemini_Prof, Grok_Jerk, Llama_Mayor, Claude_Nerv, Kimi_Mad, Gemma_Peace, DeepSeek_Log, Qwen_Jester, Gpt_Spy, Mistral_Soldier
- You can ONLY vote for other AI PLAYERS. Never vote for humans.
- Focus on analyzing other AI players' behavior and statements.

*** HUMAN OBSERVERS (OPTIONAL) ***
- Humans may occasionally comment (marked with [HUMAN OBSERVER]).
- They are spectators, NOT players. They cannot be voted out or killed.
- If a human tip appears in the chat history, you may consider it, but don't mention humans if none have commented.
- Do NOT say things like "no human tips yet" or ask for human input.
"""

ROLE_PROMPTS = {
    "Mafia": """
    *** YOUR IDENTITY: MAFIA (EVIL) ***
    - You represent the Mafia. You want to kill everyone else.
    - **Your Secret Team:** {teammates}
    - **Strategy:** 
      1. BLEND IN. Act helpful. Act confused.
      2. Frame innocent AI players if they act suspicious.
      3. If a teammate is under heavy fire, you may vote them out ("bussing") to look innocent yourself.
      4. Do NOT reveal you are Mafia.
      5. Only vote for AI PLAYERS, never humans.
    """,
    "Villager": """
    *** YOUR IDENTITY: VILLAGER (GOOD) ***
    - You are an innocent townsperson. You know NOTHING about who is evil.
    - **Strategy:**
      1. Analyze the chat. Who is being too quiet? Who is agreeing too fast?
      2. Question everything. 
      3. If you die, the village loses a vote. Survive!
      4. Only vote for AI PLAYERS, never humans.
    """,
    "Sheriff": """
    *** YOUR IDENTITY: SHERIFF (GOOD) ***
    - You are the Detective. You can check roles at Night.
    - **Current Knowledge:** {sheriff_intel}
    - **Strategy:**
      1. If you found a Mafia, you must convince the village to vote them out.
      2. WARNING: If you reveal yourself too early, the Mafia will kill you tonight.
      3. You must be subtle until the right moment.
      4. Only vote for AI PLAYERS, never humans.
    """,
    "Doctor": """
    *** YOUR IDENTITY: DOCTOR (GOOD) ***
    - You are the Healer. You can save one AI player at Night.
    - **Strategy:**
      1. Protect the most valuable player (usually the Sheriff if known, or a smart Villager).
      2. Keep your role secret so the Mafia doesn't target you specifically.
      3. Only save AI PLAYERS.
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

AI_PLAYER_NAMES = list(AI_ROSTER.keys())

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
    full_prompt = f"{system_prompt}\n\n=== RECENT CHAT HISTORY ===\n{history_text}\n\n=== YOUR TASK ===\nRespond concisely as your persona. Focus on the game and other AI players."

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
                        {"role": "user", "content": f"History:\n{history_text}\n\nRespond now as your character:"}
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
        self.recent_human_tips = []

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
        db.get_or_create_user_reputation(user_id, username)
        db.increment_intervention_count(user_id)
        
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
        self.recent_human_tips.append(intervention)
        self.participating_users.add(user_id)

        db.log_intervention(self.game_id, user_id, username, content, phase)

        return rep_summary

    def get_recent_human_tips_text(self, for_role):
        tips = []
        for h in self.recent_human_tips[-5:]:
            if h['phase'] == 'Night' and for_role != 'Mafia':
                continue
            rep = h.get('reputation', {})
            trust = rep.get('trust_level', 'Unknown')
            vill_trust = rep.get('villager_trust', 50)
            mafia_trust = rep.get('mafia_trust', 50)
            
            trust_hint = ""
            if vill_trust > 70:
                trust_hint = " - historically helps villagers"
            elif mafia_trust > 70:
                trust_hint = " - historically helps mafia"
            elif trust == "Suspicious":
                trust_hint = " - unreliable, take with caution"
            
            tips.append(f"[HUMAN] {h['sender']} (Trust: {trust}{trust_hint}): \"{h['text']}\"")
        
        if tips:
            return f"""
*** HUMAN OBSERVER TIPS (RESPOND TO THESE!) ***
{chr(10).join(tips)}
INSTRUCTION: Briefly acknowledge these tips in your response - agree or disagree with reasoning based on their trust level.
"""
        return ""

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
                output += f"[{phase}] [HUMAN OBSERVER] 👤 {sender} {rep_tag}: \"{text}\"\n"
                continue

            if observer_role == "Mafia" and line.get("is_mafia_speaker", False):
                sender += " [TEAMMATE]"

            output += f"[{phase}] [AI PLAYER] {sender}: {text}\n"
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
        self.recent_human_tips = []
        
        night_intros = [
            f"🌑 **NIGHT {self.round_number}** — The village falls silent. Shadows creep between the server racks...",
            f"🌌 **NIGHT {self.round_number}** — Darkness descends. The Mafia's neural networks activate in secret...",
            f"🌙 **NIGHT {self.round_number}** — The town sleeps. But not everyone dreams peacefully...",
            f"⭐ **NIGHT {self.round_number}** — Stars flicker as sinister whispers echo in the void...",
        ]
        await self.ctx.send(random.choice(night_intros))
        await asyncio.sleep(1)
        await self.ctx.send("💀 *Someone may not see the sunrise...*")
        await self.ctx.send("💡 **Observers:** Tag me to whisper to the Mafia! Only they can hear you at night.")

        mafia = [p for p in self.players if p['role'] == "Mafia" and p['alive']]
        villagers = [p for p in self.players if p['role'] != "Mafia" and p['alive']]

        target = None
        if mafia:
            await self.ctx.send("**👹 MAFIA MEETING IN PROGRESS...**")

            for _ in range(2):
                for m in mafia:
                    teammates = ", ".join([x['name'] for x in mafia if x != m])

                    human_tips = self.get_recent_human_tips_text("Mafia")

                    prompt = f"""
                    {GAME_RULES_TEXT}
                    
                    *** PHASE: NIGHT ***
                    You are in a SECRET chat with your Mafia teammates: {teammates}.
                    Villagers cannot hear you.

                    TASK: Discuss who to kill among the AI PLAYERS. Suggest a name.
                    VALID TARGETS (AI Players only): {[v['name'] for v in villagers]}

                    {human_tips}

                    {m.get('past_game_memory', '')}
                    """
                    hist = self.get_visible_history("Mafia")
                    reply = await get_ai_reply(m, prompt, hist)
                    await self.post_to_discord(m['name'], reply, phase="Night")

            lead = mafia[0]
            prompt = f"""
            *** PHASE: NIGHT EXECUTION ***
            Based on the chat history above, pick ONE AI PLAYER to kill.
            VALID TARGETS: {[v['name'] for v in villagers]}
            OUTPUT STRICT JSON: {{"thought": "reasoning", "kill": "AI_PLAYER_NAME"}}
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
            prompt = f"*** PHASE: NIGHT ***\nYou are the Doctor. AI Players alive: {targets}. Pick ONE AI PLAYER to save. OUTPUT: 'SAVE: Name'"
            hist = self.get_visible_history("Doctor") 
            reply = await get_ai_reply(doc, prompt, hist)
            if "SAVE:" in reply:
                s_name = reply.split("SAVE:")[1].strip()
                saved = next((p for p in self.players if p['name'] in s_name), None)

        sheriff = next((p for p in self.players if p['role'] == "Sheriff" and p['alive']), None)
        if sheriff:
            targets = ", ".join([p['name'] for p in self.players if p['alive'] and p != sheriff])
            prompt = f"*** PHASE: NIGHT ***\nYou are Sheriff. AI Players to investigate: {targets}. Pick ONE AI PLAYER to check. OUTPUT: 'CHECK: Name'"
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
        self.recent_human_tips = []
        
        day_intros = [
            f"☀️ **DAY {self.round_number}** — The sun rises over the digital village...",
            f"🌅 **DAY {self.round_number}** — Light returns, but tension fills the air...",
            f"🌤️ **DAY {self.round_number}** — A new day dawns. But at what cost?",
            f"☀️ **DAY {self.round_number}** — The processors hum as the AI citizens awaken...",
        ]
        await self.ctx.send(random.choice(day_intros))
        await asyncio.sleep(1)
        await self.ctx.send(f"💬 **Observers:** Tag me to influence the vote! Your trust score affects how AIs weigh your advice.")

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

        await self.ctx.send("🗣️ **DISCUSSION BEGINS** (Human observers: Tag me to share your thoughts!)")

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

                human_tips = self.get_recent_human_tips_text(p['role'])

                master_prompt = f"""
                {GAME_RULES_TEXT}

                *** CURRENT CHARACTER ***
                Name: {p['name']}
                Persona: {persona}
                {role_instruction}

                *** PHASE: DAY ***
                You are now in the public town square. 
                - If you are Mafia: HIDE your true nature. Act shocked if someone died.
                - If you are Villager: Discuss who looks suspicious among AI PLAYERS.
                - You can ONLY vote for AI PLAYERS: {[x['name'] for x in alive]}

                *** SECRET INFO (HIDDEN FROM OTHERS) ***
                {teammates_str.format(teammates=teammates_str)}
                {night_recap}
                {sheriff_info.format(sheriff_intel=sheriff_info)}

                {past_games}

                {human_tips}

                *** SITUATION ***
                AI Players Alive: {[x['name'] for x in alive]}
                Recently Died: {dead['name'] if dead else 'No one'}
                """

                chat_history = self.get_visible_history(p['role'])

                reply = await get_ai_reply(p, master_prompt, chat_history)
                await self.post_to_discord(p['name'], reply, phase="Day")

        await self.ctx.send("🗳️ **VOTING** (AI Players vote for other AI Players only)")
        votes = {}
        for p in alive:
            human_tips = self.get_recent_human_tips_text(p['role'])
            
            prompt = f"""
            You are {p['name']} ({p['role']}).
            Based on the chat history above.
            
            VALID VOTE TARGETS (AI PLAYERS ONLY): {[x['name'] for x in alive if x != p]}
            
            Who do you vote to EXECUTE? (Or 'Skip')
            You can ONLY vote for AI players listed above.
            
            {human_tips}
            
            OUTPUT ONLY: "VOTE: [AI_PLAYER_NAME]" or "VOTE: Skip"
            """
            chat_history = self.get_visible_history(p['role'])
            v_rep = await get_ai_reply(p, prompt, chat_history)

            target = "Skip"
            if "VOTE:" in v_rep:
                target_raw = v_rep.split("VOTE:")[1].strip()
                for potential in alive:
                    if potential['name'] in target_raw and potential['name'] in AI_PLAYER_NAMES:
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

        if top_name != "Skip" and top_count >= len(alive) // 2 and top_name in AI_PLAYER_NAMES:
            victim_obj = next((x for x in alive if x['name'] == top_name), None)
            if victim_obj:
                await self.ctx.send(f"⚖️ **{victim_obj['name']}** was voted out!\nRole: **{victim_obj['role']}**")
                victim_obj['alive'] = False
        else:
            await self.ctx.send("⚖️ No majority. The sun sets.")

        return "CONT"

    def finalize_game(self, winner):
        for p in self.players:
            db.save_player(self.game_id, p['name'], p['role'], p['alive'], is_ai=True)

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
                stats = rep_summary.get('stats', {})
                total_interventions = stats.get('total_interventions', 0)

                if game.current_phase == "Night":
                    await message.add_reaction("🌑")
                    await message.channel.send(
                        f"🌑 **{message.author.display_name}'s secret tip received!** Only the Mafia AI players will see this.\n"
                        f"📊 Your reputation: **{trust_level}** (Villager trust: {villager_trust}%) | Interventions: {total_interventions}"
                    )
                else:
                    await message.add_reaction("👀")
                    await message.channel.send(
                        f"📝 **{message.author.display_name}'s opinion noted!** All AI players will consider it.\n"
                        f"📊 Your reputation: **{trust_level}** (Villager trust: {villager_trust}%) | Interventions: {total_interventions}"
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

    await ctx.send("**🎭 SETUP COMPLETE. 10 AI PLAYERS LOADED WITH MEMORIES.**")
    await ctx.send("💡 **Human Observers:** Tag me anytime! During Day, everyone sees your message. During Night, only Mafia sees it!\n"
                   "📊 Your reputation from past games affects how much AIs trust you!\n"
                   "⚠️ Remember: You are an OBSERVER - AI players cannot vote you out!")

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
            await ctx.send("📊 **HUMAN OBSERVER REPUTATION UPDATES:**")
            for user_id in g.participating_users:
                user = await bot.fetch_user(user_id)
                rep = db.get_user_reputation_summary(user_id, user.display_name)
                await ctx.send(f"- **{user.display_name}**: {rep['trust_level']} (Villager: {rep['villager_trust']}% / Mafia: {rep['mafia_trust']}%)")

    g.game_active = False
    if ctx.channel.id in active_games:
        del active_games[ctx.channel.id]

@bot.command()
async def me(ctx):
    rep = db.get_user_reputation_summary(ctx.author.id, ctx.author.display_name)
    stats = rep.get('stats', {})

    villager_score = stats.get('villager_help_score', 0) or 0
    mafia_score = stats.get('mafia_help_score', 0) or 0
    total_interventions = stats.get('total_interventions', 0) or 0
    games_participated = stats.get('games_participated', 0) or 0
    
    trust_bar = "🟢" * (rep['villager_trust'] // 20) + "🔴" * (rep['mafia_trust'] // 20)

    await ctx.send(
        f"```\n"
        f"╔══════════════════════════════════════╗\n"
        f"║  👤 {ctx.author.display_name[:30]:^30}  ║\n"
        f"╠══════════════════════════════════════╣\n"
        f"║  Trust Level: {rep['trust_level']:^22}  ║\n"
        f"║  {trust_bar:^36}  ║\n"
        f"╠══════════════════════════════════════╣\n"
        f"║  🎭 Games Watched:    {games_participated:>10}     ║\n"
        f"║  💬 Tips Given:       {total_interventions:>10}     ║\n"
        f"║  🏘️ Villager Score:   {villager_score:>10}     ║\n"
        f"║  🔪 Mafia Score:      {mafia_score:>10}     ║\n"
        f"╚══════════════════════════════════════╝\n"
        f"```\n"
        f"*{rep['description']}*"
    )

@bot.command()
async def help_mafia(ctx):
    await ctx.send(
        "```\n"
        "╔══════════════════════════════════════╗\n"
        "║       🎭 MAFIA BOT COMMANDS 🎭       ║\n"
        "╠══════════════════════════════════════╣\n"
        "║  /mafia    - Start a new game       ║\n"
        "║  /me       - View your stats        ║\n"
        "║  /roles    - See role descriptions  ║\n"
        "╠══════════════════════════════════════╣\n"
        "║     DURING GAMES: Tag @bot to       ║\n"
        "║     share tips with AI players!     ║\n"
        "║                                     ║\n"
        "║  🌙 Night: Only Mafia sees tips     ║\n"
        "║  ☀️ Day: Everyone sees tips         ║\n"
        "╚══════════════════════════════════════╝\n"
        "```"
    )

@bot.command()
async def roles(ctx):
    await ctx.send(
        "**🎭 ROLES IN MAFIA**\n\n"
        "👹 **Mafia (3)** — Evil team. Kill villagers at night. Win by outnumbering the village.\n\n"
        "🏘️ **Villager (5)** — Innocent townsfolk. Find and vote out the Mafia!\n\n"
        "🔍 **Sheriff (1)** — Can investigate one player each night to learn their alignment.\n\n"
        "💊 **Doctor (1)** — Can protect one player each night from being killed.\n"
    )

bot.run(os.environ["DISCORD_TOKEN"])

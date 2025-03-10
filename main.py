import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, ConversationHandler, MessageHandler, Filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import itertools
import warnings
import time
import os
import json
from stable_baselines3.common.utils import get_schedule_fn
from flask import Flask
from threading import Thread
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask keep-alive server
app = Flask(__name__)
@app.route('/healthz')
def health_check():
    return "OK", 200

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="telegram.ext.conversationhandler")

# Lock file and users file
LOCK_FILE = "bot.lock"
USERS_FILE = "users.json"

# Check if another instance is running
if os.path.exists(LOCK_FILE):
    logger.error(f"Another instance of the bot is running. Remove {LOCK_FILE} to force start.")
    exit(1)
else:
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    logger.info(f"Lock file created with PID {os.getpid()}")

# Load or initialize users database
if os.path.exists(USERS_FILE):
    try:
        with open(USERS_FILE, 'r') as f:
            users_db = json.load(f)
            if not isinstance(users_db, dict):  # Ensure valid dictionary format
                raise ValueError("Invalid users.json format")
    except (json.JSONDecodeError, ValueError):
        users_db = {}  # Reset to empty dictionary if corrupted
        with open(USERS_FILE, 'w') as f:
            json.dump(users_db, f)
else:
    users_db = {}
    with open(USERS_FILE, 'w') as f:
        json.dump(users_db, f)


# Hardcoded admin credentials
ADMIN_USERNAME = "admin123"
ADMIN_PASSWORD = "secret456"

# Load the trained model
logger.info("Starting to load DQN model...")
try:
    custom_objects = {
        'lr_schedule': get_schedule_fn(0.001),
        'exploration_schedule': get_schedule_fn(0.1)
    }
    dqn_model = DQN.load('dqn_betting_model', custom_objects=custom_objects)
    logger.info("Model loaded successfully.")
    logger.info(f"DQN model observation space: {dqn_model.observation_space}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    os.remove(LOCK_FILE)
    exit(1)

# States for conversation
LOGIN, LOGIN_USERNAME, LOGIN_PASSWORD, SIGNUP_USERNAME, SIGNUP_PASSWORD, ADMIN_MENU, GET_OUTCOMES, PREDICT, GET_FEEDBACK, SKIP_FEEDBACK = range(10)

# Global messages
APOLOGY_MESSAGE = (
    "😔 Maafi! \n Hum continuously 6 baar haar chuke hain, aur yeh dil se bura lag raha hai. 💔\n\n"
    "Mujhe pata hai ki yeh phase tough hai, lekin tension mat lo! Main full effort dal raha hoon recovery ke liye. 🔥\n"
    "Bas thodi si patience rakho, shayad agla turn humare favor mein ho! 🍀\n\n"
    "Agar aap chahein toh hum aage continue kar sakte hain 🔄, ya phir reset dabake ek naya start le sakte hain. 🔃\n"
)

INACTIVITY_WARNING = (
    "⚠️ **Warning:** You have been inactive for 90 seconds. Please interact with the bot, or it will clear your session in 30 more seconds.\n"
    "You can restart anytime with `/start`."
)

INACTIVITY_CLEANUP = (
    "⏹️ **Session Cleared:** You have been inactive for 120 seconds. All data related to this session has been cleared.\n"
    "Restart the bot with `/start` to begin anew."
)

# Helper functions
def prepare_live_input(recent_outcomes, lookback=20):
    start_time = time.time()
    if not recent_outcomes:
        raise ValueError("recent_outcomes is empty. Cannot prepare input for prediction.")
    df = pd.DataFrame({'Outcome': recent_outcomes[-10:]})
    df['Outcome'] = df['Outcome'].map({'Big': 1, 'Small': 0})
    df['Streak'] = df['Outcome'].rolling(window=5, min_periods=1).sum().fillna(0)
    df['Trap_Risk'] = df['Outcome'].diff().abs().rolling(window=5, min_periods=1).mean().fillna(0)
    features = df[['Outcome', 'Streak', 'Trap_Risk']].values
    if len(features) < lookback:
        padding = np.zeros((lookback - len(features), 3), dtype=np.float32)
        features = np.vstack((padding, features))
    end_time = time.time()
    logger.debug(f"prepare_live_input took {end_time - start_time:.4f} seconds for {len(recent_outcomes)} outcomes")
    return features.astype(np.float32)

def predict_next_outcome(recent_outcomes):
    start_time = time.time()
    try:
        obs = prepare_live_input(recent_outcomes, lookback=20)
        assert obs.shape == (20, 3), f"Expected shape (20, 3), got {obs.shape}"
        if len(dqn_model.observation_space.shape) == 3 and dqn_model.observation_space.shape[1:] == (20, 3):
            obs = obs.reshape(1, 20, 3)
            logger.debug(f"Reshaped obs to {obs.shape} for (n_env, 20, 3) environment")
        action, _ = dqn_model.predict(obs, deterministic=True)
        predicted = 'Big' if action == 1 else 'Small'
        end_time = time.time()
        logger.debug(f"Model prediction time: {end_time - start_time:.4f} seconds for history {recent_outcomes[-10:]}")
        logger.info(f"Predicted outcome: {predicted}")
        return predicted
    except Exception as e:
        logger.error(f"Error in predict_next_outcome: {str(e)}")
        raise

def get_outcome_keyboard(step):
    keyboard = [
        [
            InlineKeyboardButton("Big", callback_data=f"outcome_{step}_Big"),
            InlineKeyboardButton("Small", callback_data=f"outcome_{step}_Small")
        ],
        [
            InlineKeyboardButton("🔄 Reset", callback_data="reset"),
            InlineKeyboardButton("🗑️ Clear History", callback_data="clear_history")
        ],
        [
            InlineKeyboardButton("🚪 Exit", callback_data="exit")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_feedback_keyboard(disabled=False):
    keyboard = [
        [
            InlineKeyboardButton("✅ Yes", callback_data="feedback_yes" if not disabled else "disabled"),
            InlineKeyboardButton("❌ No", callback_data="feedback_no" if not disabled else "disabled"),
        ],
        [
            InlineKeyboardButton("🔄 Reset", callback_data="reset"),
            InlineKeyboardButton("🗑️ Clear History", callback_data="clear_history")
        ],
        [
            InlineKeyboardButton("🚪 Exit", callback_data="exit")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_skip_feedback_keyboard(step):
    keyboard = [
        [
            InlineKeyboardButton("✅ Yes", callback_data=f"skip_{step}_yes"),
            InlineKeyboardButton("❌ No", callback_data=f"skip_{step}_no")
        ],
        [
            InlineKeyboardButton("🔄 Reset", callback_data="reset"),
            InlineKeyboardButton("🗑️ Clear History", callback_data="clear_history")
        ],
        [
            InlineKeyboardButton("🚪 Exit", callback_data="exit")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_login_keyboard():
    keyboard = [
        [InlineKeyboardButton("Login", callback_data="login")],
        [InlineKeyboardButton("Signup", callback_data="signup")],
        [InlineKeyboardButton("Exit", callback_data="exit")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_admin_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("Prediction", callback_data="admin_predict")],
        [InlineKeyboardButton("User Logs", callback_data="admin_logs")],
        [InlineKeyboardButton("Exit", callback_data="exit")]
    ]
    return InlineKeyboardMarkup(keyboard)

def format_prediction_message(history, predictions, wins, predicted, loss_freq_6, current_streak_length, max_streak_length):
    last_bet_result = f"<b>Last Bet:</b> {predictions[-1]} {'✅' if wins[-1] else '❌'}\n" if predictions and wins else ""
    history_display = "<b>Bet History (Last 10):</b>\n\n"
    if len(predictions) == 0:
        history_display += "No bets yet.\n"
    else:
        start_idx = max(0, len(predictions) - 10)
        for i in range(start_idx, len(predictions)):
            bet = predictions[i]
            result = "💸" * (7 if wins[i] else 0)
            history_display += f"{bet} {result}\n"
    streak_info = f"<b>Current Streak:</b> {current_streak_length} wins | <b>Max Streak:</b> {max_streak_length} wins\n"
    loss_freq_display = f"<b>6 Consecutive Losses Frequency:</b> {loss_freq_6}\n"
    prediction_display = (
        f"\n"
        f"═══════════════════════\n"
        f"**NEXT PREDICTION: 🎯{predicted.upper()}**\n"
        f"═══════════════════════\n"
    )
    message = (
        "🎰 <b>WIN GO 1 MIN</b> 🎰\n"
        "🌟 <b>MAINTAIN LEVEL 7</b> 🌟\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{last_bet_result}"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{history_display}"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{streak_info}"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{loss_freq_display}"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n{prediction_display}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        "💰 <b>Keep Earning!</b> 💰\n"
        "━━━━━━━━━━━━━━━━━━━━━━━"
    )
    return message

def calculate_streak_frequencies(losses, wins):
    loss_streaks = [sum(1 for _ in g) for k, g in itertools.groupby(losses) if k == 1]
    win_streaks = [sum(1 for _ in g) for k, g in itertools.groupby(wins) if k == 1]
    max_losses = max(loss_streaks) if loss_streaks else 0
    max_wins = max(win_streaks) if win_streaks else 0
    loss_freq = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    win_freq = {3: 0, 4: 0, 5: 0, 6: 0}
    for streak in loss_streaks:
        if streak in loss_freq:
            loss_freq[streak] += 1
        elif streak > 7:
            loss_freq[7] += 1
    for streak in win_streaks:
        if streak in win_freq:
            win_freq[streak] += 1
        elif streak > 6:
            win_freq[6] += 1
    return loss_freq, win_freq, max_losses, max_wins

def reset_state(context):
    chat_id = context.user_data.get('chat_id', 'unknown')
    context.user_data.clear()
    context.user_data.update({
        'outcomes': [],
        'step': 0,
        'history': [],
        'predictions': [],
        'correct': 0,
        'total_bets': 0,
        'reward': 0,
        'losses': [],
        'wins': [],
        'initial_outcomes_collected': False,
        'current_streak_length': 0,
        'max_streak_length': 0,
        'loss_streak_length': 0,
        'skip_count': 0,
        'play_with_streak': False,
        'wait_count': 0,
        'message_ids': [],
        'all_message_ids': [],
        'apology_triggered': False,
        'last_feedback_time': 0,
        'last_active_time': time.time(),
        'chat_id': chat_id
    })
    logger.info(f"Reset state for user {chat_id} at {time.ctime()}")

def exit_user(update, context):
    if update.callback_query:
        query = update.callback_query
        query.answer()
        chat_id = query.message.chat_id
    else:
        chat_id = update.message.chat_id
    logger.info(f"Exiting user session for chat_id: {chat_id}")
    context.user_data.clear()
    context.bot.send_message(chat_id=chat_id, text="🚪 Session exited. All processes cleared. Restart with `/start`.")
    return ConversationHandler.END

def check_inactivity(update, context):
    chat_id = update.effective_chat.id
    current_time = time.time()
    last_active = context.user_data.get('last_active_time', current_time)

    inactivity_duration = current_time - last_active

    if inactivity_duration >= 120:
        logger.info(f"Automatic cleanup for chat_id {chat_id} due to 120 seconds inactivity")
        context.user_data.clear()
        context.bot.send_message(chat_id=chat_id, text=INACTIVITY_CLEANUP)
        return ConversationHandler.END
    elif inactivity_duration >= 90:
        if not context.user_data.get('inactivity_warning_sent', False):
            logger.info(f"Sending warning for chat_id {chat_id} due to 90 seconds inactivity")
            sent_message = context.bot.send_message(chat_id=chat_id, text=INACTIVITY_WARNING)
            context.user_data['inactivity_warning_sent'] = True
            context.user_data['message_ids'].append(sent_message.message_id)
            context.user_data['all_message_ids'].append(sent_message.message_id)

    context.user_data['last_active_time'] = current_time
    context.user_data['inactivity_warning_sent'] = False
    return None

def send_apology_message(update, context):
    logger.debug("Sending apology message")
    if update.callback_query:
        query = update.callback_query
        query.answer()
        sent_message = query.message.reply_text(
            APOLOGY_MESSAGE,
            reply_markup=get_outcome_keyboard(context.user_data['step'] + 1)
        )
    else:
        sent_message = update.message.reply_text(
            APOLOGY_MESSAGE,
            reply_markup=get_outcome_keyboard(context.user_data['step'] + 1)
        )
    context.user_data['message_ids'].append(sent_message.message_id)
    context.user_data['all_message_ids'].append(sent_message.message_id)
    context.user_data['apology_triggered'] = True
    context.user_data['last_active_time'] = time.time()
    logger.debug("Transitioning to GET_OUTCOMES after apology")
    return GET_OUTCOMES

# Handle /start command
def start(update, context):
    user = update.message.from_user
    chat_id = update.message.chat_id
    context.user_data['chat_id'] = chat_id
    reset_state(context)
    
    welcome_message = f"🎉 Hello {user.first_name}! Welcome to **Shitty Predicts** 🎉\nPlease login or signup to continue."
    sent_welcome = update.message.reply_text(
        welcome_message,
        parse_mode='Markdown',
        reply_markup=get_login_keyboard()
    )
    context.user_data['message_ids'].append(sent_welcome.message_id)
    context.user_data['all_message_ids'].extend([update.message.message_id, sent_welcome.message_id])
    context.user_data['last_active_time'] = time.time()
    logger.debug(f"Transitioning to LOGIN for user {chat_id}")
    return LOGIN

# Handle login/signup selection
def login(update, context):
    query = update.callback_query
    query.answer()
    chat_id = query.message.chat_id

    if query.data == "login":
        sent_message = query.edit_message_text("🔹✨ Please Enter Your Username ✨🔹  ")
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug(f"User {chat_id} chose login, transitioning to LOGIN_USERNAME")
        return LOGIN_USERNAME
    elif query.data == "signup":
        sent_message = query.edit_message_text("Please Choose a Username: ")
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug(f"User {chat_id} chose signup, transitioning to SIGNUP_USERNAME")
        return SIGNUP_USERNAME
    elif query.data == "exit":
        return exit_user(update, context)

    return LOGIN

def handle_login_username(update, context):
    username = update.message.text.strip()
    context.user_data['temp_username'] = username
    chat_id = update.message.chat_id
    sent_message = update.message.reply_text("🔑🔒 Please Enter Your Password 🔒🔑 ")
    context.user_data['message_ids'].append(sent_message.message_id)
    context.user_data['all_message_ids'].extend([update.message.message_id, sent_message.message_id])
    logger.debug(f"User {chat_id} entered username: {username}, transitioning to LOGIN_PASSWORD")
    return LOGIN_PASSWORD

def handle_login_password(update, context):
    password = update.message.text.strip()
    username = context.user_data.get('temp_username')
    chat_id = update.message.chat_id

    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        logger.info(f"Admin login successful for {username} at {time.ctime()}")
        users_db[username] = users_db.get(username, {"password": password, "logins": []})
        users_db[username]["logins"].append(time.ctime())
        with open(USERS_FILE, 'w') as f:
            json.dump(users_db, f)
        context.user_data['is_admin'] = True
        sent_message = update.message.reply_text(
            "✅ Admin login successful! Choose an option:",
            reply_markup=get_admin_menu_keyboard()
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].extend([update.message.message_id, sent_message.message_id])
        logger.debug(f"Admin {chat_id} logged in, transitioning to ADMIN_MENU")
        return ADMIN_MENU
    elif username in users_db and users_db[username]["password"] == password:
        logger.info(f"User login successful for {username} at {time.ctime()}")
        users_db[username]["logins"].append(time.ctime())
        with open(USERS_FILE, 'w') as f:
            json.dump(users_db, f)
        context.user_data['username'] = username
        sent_message = update.message.reply_text(
            f"✅ Login successful, {username}! What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].extend([update.message.message_id, sent_message.message_id])
        logger.debug(f"User {chat_id} logged in, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    else:
        sent_message = update.message.reply_text(
            "❌ Invalid username or password. Please try again or signup.",
            reply_markup=get_login_keyboard()
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].extend([update.message.message_id, sent_message.message_id])
        logger.debug(f"Login failed for {username}, returning to LOGIN")
        return LOGIN

def signup_username(update, context):
    username = update.message.text.strip()
    chat_id = update.message.chat_id
    if username in users_db or username == ADMIN_USERNAME:
        sent_message = update.message.reply_text("❌ Username already taken. Please choose another:")
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].extend([update.message.message_id, sent_message.message_id])
        logger.debug(f"Username {username} taken for {chat_id}")
        return SIGNUP_USERNAME
    context.user_data['temp_username'] = username
    sent_message = update.message.reply_text("Please enter a password:")
    context.user_data['message_ids'].append(sent_message.message_id)
    context.user_data['all_message_ids'].extend([update.message.message_id, sent_message.message_id])
    logger.debug(f"User {chat_id} chose username: {username}")
    return SIGNUP_PASSWORD

def signup_password(update, context):
    password = update.message.text.strip()
    username = context.user_data['temp_username']
    chat_id = update.message.chat_id
    users_db[username] = {"password": password, "logins": [time.ctime()]}
    with open(USERS_FILE, 'w') as f:
        json.dump(users_db, f)
    context.user_data['username'] = username
    sent_message = update.message.reply_text(
        f"✅ Signup successful, {username}! What was outcome 1?",
        reply_markup=get_outcome_keyboard(1)
    )
    context.user_data['message_ids'].append(sent_message.message_id)
    context.user_data['all_message_ids'].extend([update.message.message_id, sent_message.message_id])
    logger.debug(f"User {chat_id} signed up, transitioning to GET_OUTCOMES")
    return GET_OUTCOMES

def admin_menu(update, context):
    query = update.callback_query
    query.answer()
    chat_id = query.message.chat_id

    if query.data == "admin_predict":
        sent_message = query.edit_message_text(
            "✅ Entering prediction mode. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug(f"Admin {chat_id} chose prediction, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "admin_logs":
        logs = "📋 **User Logs** 📋\n━━━━━━━━━━━━━━━━━━━━━━━\n"
        for username, data in users_db.items():
            if username != ADMIN_USERNAME:
                logs += f"**Username:** {username}\n"
                logs += f"**Login Count:** {len(data['logins'])}\n"
                logs += f"**Last Login:** {data['logins'][-1] if data['logins'] else 'N/A'}\n"
                logs += "━━━━━━━━━━━━━━━━━━━━━━━\n"
        sent_message = query.edit_message_text(logs, parse_mode='Markdown')
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug(f"Admin {chat_id} viewed user logs")
        return ADMIN_MENU
    elif query.data == "exit":
        return exit_user(update, context)

    return ADMIN_MENU

def get_outcomes(update, context):
    check_inactivity(update, context)
    query = update.callback_query
    query.answer()
    step = context.user_data['step']
    outcomes = context.user_data['outcomes']

    if query.data.startswith('outcome_'):
        _, step_str, outcome = query.data.split('_')
        step = int(step_str)
        outcomes.append(outcome)
        context.user_data['step'] = step + 1
        context.user_data['outcomes'] = outcomes

        if step < 10:
            query.edit_message_text(
                f"📝 Outcome {step} recorded: {outcome}. What was outcome {step + 1}?",
                reply_markup=get_outcome_keyboard(step + 1)
            )
            logger.debug(f"Staying in GET_OUTCOMES, step {step + 1}")
            return GET_OUTCOMES
        else:
            context.user_data['history'] = outcomes[-10:]
            context.user_data['initial_outcomes_collected'] = True
            context.user_data['step'] = step
            logger.debug(f"Transitioning to PREDICT with history = {context.user_data['history']}")
            query.edit_message_text("✅ Thanks! All 10 outcomes collected. Predicting the next outcome...")
            return predict_next(update, context)
    elif query.data == "reset":
        reset_state(context)
        sent_message = query.message.reply_text(
            f"🔄 Reset successful, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Reset triggered, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "clear_history":
        bot = context.bot
        chat_id = query.message.chat_id
        for msg_id in context.user_data['all_message_ids']:
            try:
                bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception as e:
                logger.error(f"Failed to delete message {msg_id}: {e}")
        context.user_data['all_message_ids'] = []
        context.user_data['message_ids'] = []
        sent_message = query.message.reply_text(
            f"🗑️ Chat history cleared, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Cleared history, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "exit":
        return exit_user(update, context)

    logger.debug("Staying in GET_OUTCOMES (default case)")
    return GET_OUTCOMES

def predict_next(update, context):
    check_inactivity(update, context)
    query = update.callback_query
    if query:
        query.answer()

    logger.debug("Entered predict_next")
    history = context.user_data['history'][-10:]
    if len(history) < 10:
        logger.error(f"history length is {len(history)}, expected at least 10. History: {history}")
        error_msg = "⚠️ Internal error: Not enough history to make a prediction. Please provide more outcomes or reset."
        reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reset", callback_data="reset"), InlineKeyboardButton("🚪 Exit", callback_data="exit")]])
        sent_message = (update.callback_query.message.reply_text if update.callback_query else update.message.reply_text)(
            error_msg, reply_markup=reply_markup
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Transitioning to GET_OUTCOMES due to insufficient history")
        return GET_OUTCOMES

    predictions = context.user_data['predictions']
    wins = context.user_data['wins']
    losses = context.user_data['losses']
    loss_freq, win_freq, max_losses, max_wins = calculate_streak_frequencies(losses, wins)
    loss_streak_length = context.user_data['loss_streak_length']
    logger.debug(f"Current loss streak length = {loss_streak_length}")

    start_time = time.time()
    try:
        if loss_streak_length >= 6 and not context.user_data.get('apology_triggered', False):
            logger.debug("Triggering apology for 6 consecutive losses")
            return send_apology_message(update, context)

        if loss_streak_length >= 5:
            logger.debug("Triggering waiting phase for 5 consecutive losses")
            wait_bets = min(4, 10 - len(history))
            if context.user_data.get('wait_count', 0) < wait_bets:
                context.user_data['wait_count'] = context.user_data.get('wait_count', 0) + 1
                msg = f"⏳ **Loss Streak ({loss_streak_length}) Detected!** 🔴\nWaiting and analyzing... (Bet {context.user_data['wait_count']}/{wait_bets})\nPlease provide the next outcome."
                sent_message = (update.callback_query.message.reply_text if update.callback_query else update.message.reply_text)(
                    msg, reply_markup=get_outcome_keyboard(context.user_data['step'] + 1)
                )
                context.user_data['message_ids'].append(sent_message.message_id)
                context.user_data['all_message_ids'].append(sent_message.message_id)
                logger.debug("Transitioning to GET_OUTCOMES for waiting phase")
                return GET_OUTCOMES
            else:
                context.user_data['wait_count'] = 0
                logger.debug("Wait phase complete, making strong prediction")
                predicted = predict_next_outcome(history)
                context.user_data['play_with_streak'] = False
        else:
            logger.debug("Checking for artificial streak")
            if len(predictions) >= 4 and all(p == predictions[-4] for p in predictions[-4:]):
                logger.debug(f"Artificial streak detected with predictions: {predictions[-4:]}")
                msg = "⚠️ **Artificial Streak Detected!** 🚨\nIt seems the streak (4-5 same outcomes) might be manipulated. Should we skip 1-2 bets to observe?\nPlease confirm the outcome of the next bet..."
                sent_message = (update.callback_query.message.reply_text if update.callback_query else update.message.reply_text)(
                    msg, reply_markup=get_skip_feedback_keyboard(1)
                )
                context.user_data['message_ids'].append(sent_message.message_id)
                context.user_data['all_message_ids'].append(sent_message.message_id)
                context.user_data['skip_count'] = 1
                logger.debug("Transitioning to SKIP_FEEDBACK due to artificial streak")
                return SKIP_FEEDBACK

            logger.debug("Checking play_with_streak")
            if context.user_data.get('play_with_streak', False) and len(predictions) >= 1:
                predicted = predictions[-1]
                logger.debug(f"Continuing streak with prediction = {predicted}")
            else:
                logger.debug("Calling predict_next_outcome")
                predicted = predict_next_outcome(history)

        end_time = time.time()
        logger.debug(f"Total prediction time: {end_time - start_time:.4f} seconds for history {history}")
        context.user_data['last_predicted'] = predicted

        logger.debug("Formatting prediction message")
        prediction_message = format_prediction_message(
            history, predictions, wins, predicted, loss_freq[6],
            context.user_data.get('current_streak_length', 0),
            context.user_data.get('max_streak_length', 0)
        )
        logger.debug("Sending prediction message")
        send_start = time.time()
        sent_message = (update.callback_query.message.reply_text if update.callback_query else update.message.reply_text)(
            prediction_message, parse_mode='HTML', reply_markup=get_feedback_keyboard()
        )
        send_end = time.time()
        logger.debug(f"Sending message took {send_end - send_start:.4f} seconds")
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)

        context.user_data['predictions'].append(predicted)
        context.user_data['total_bets'] += 1
        logger.debug("Transitioning to GET_FEEDBACK after prediction")
        return GET_FEEDBACK

    except Exception as e:
        logger.error(f"Error in predict_next: {str(e)}")
        error_msg = f"⚠️ An error occurred during prediction: {str(e)}. Please reset and start over."
        reply_markup = InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reset", callback_data="reset"), InlineKeyboardButton("🚪 Exit", callback_data="exit")]])
        sent_message = (update.callback_query.message.reply_text if update.callback_query else update.message.reply_text)(
            error_msg, reply_markup=reply_markup
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Transitioning to GET_OUTCOMES due to error in predict_next")
        return GET_OUTCOMES

def get_feedback(update, context):
    check_inactivity(update, context)
    query = update.callback_query
    if not query:
        return GET_FEEDBACK

    current_time = time.time()
    last_feedback_time = context.user_data.get('last_feedback_time', 0)
    if current_time - last_feedback_time < 1.0:
        query.answer("Please wait a moment before clicking again.")
        logger.debug(f"Debounced feedback click at {current_time}, last at {last_feedback_time}")
        return GET_FEEDBACK

    query.answer()
    logger.debug(f"Entered get_feedback at {current_time}")

    if 'last_predicted' not in context.user_data:
        reset_state(context)
        sent_message = query.message.reply_text(
            "⚠️ No prediction found. Session may have been reset. Starting over.\nWhat was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Transitioning to GET_OUTCOMES due to missing last_predicted")
        return GET_OUTCOMES

    predicted = context.user_data['last_predicted']
    actual = None
    was_correct = False

    if query.data == "feedback_yes":
        context.user_data['correct'] += 1
        context.user_data['reward'] += 1
        context.user_data['wins'].append(1)
        context.user_data['losses'].append(0)
        actual = predicted
        context.user_data['current_streak_length'] += 1
        context.user_data['max_streak_length'] = max(context.user_data['max_streak_length'], context.user_data['current_streak_length'])
        context.user_data['loss_streak_length'] = 0
        context.user_data['apology_triggered'] = False
        was_correct = True
        logger.debug("Feedback yes, win recorded")
    elif query.data == "feedback_no":
        context.user_data['reward'] -= 1
        context.user_data['wins'].append(0)
        context.user_data['losses'].append(1)
        actual = 'Small' if predicted == 'Big' else 'Big'
        context.user_data['current_streak_length'] = 0
        context.user_data['loss_streak_length'] += 1
        was_correct = False
        logger.debug("Feedback no, loss recorded")
    elif query.data == "reset":
        reset_state(context)
        sent_message = query.message.reply_text(
            f"🔄 Reset successful, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Reset triggered, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "clear_history":
        bot = context.bot
        chat_id = query.message.chat_id
        for msg_id in context.user_data['all_message_ids']:
            try:
                bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception as e:
                logger.error(f"Failed to delete message {msg_id}: {e}")
        context.user_data['all_message_ids'] = []
        context.user_data['message_ids'] = []
        sent_message = query.message.reply_text(
            f"🗑️ Chat history cleared, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Cleared history, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "exit":
        return exit_user(update, context)
    elif query.data == "disabled":
        query.answer("Button disabled, please wait.")
        return GET_FEEDBACK
    else:
        sent_message = query.message.reply_text("⚠️ Please select 'Yes' or 'No'.", reply_markup=get_feedback_keyboard())
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Invalid feedback, staying in GET_FEEDBACK")
        return GET_FEEDBACK

    context.user_data['history'].append(actual)
    if len(context.user_data['history']) > 10:
        context.user_data['history'] = context.user_data['history'][-10:]
    context.user_data['step'] = len(context.user_data['history'])

    if context.user_data['correct'] == 10:
        bot = context.bot
        chat_id = query.message.chat_id
        for msg_id in context.user_data['message_ids']:
            try:
                bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception as e:
                logger.error(f"Failed to delete message {msg_id}: {e}")
        sent_message = query.message.reply_text("🎉 **10 Wins Complete!** 🏆\nStarting a new prediction cycle...", parse_mode='Markdown')
        context.user_data['message_ids'] = []
        context.user_data['all_message_ids'].append(sent_message.message_id)
        history = context.user_data['history']
        predictions = context.user_data['predictions']
        wins = context.user_data['wins']
        losses = context.user_data['losses']
        loss_freq, win_freq, max_losses, max_wins = calculate_streak_frequencies(losses, wins)
        predicted = predict_next_outcome(history)
        context.user_data['last_predicted'] = predicted
        prediction_message = format_prediction_message(
            history, predictions, wins, predicted, loss_freq[6],
            context.user_data.get('current_streak_length', 0),
            context.user_data.get('max_streak_length', 0)
        )
        sent_message = query.message.reply_text(prediction_message, parse_mode='HTML', reply_markup=get_feedback_keyboard())
        context.user_data['message_ids'] = [sent_message.message_id]
        context.user_data['all_message_ids'].append(sent_message.message_id)
        context.user_data['predictions'].append(predicted)
        context.user_data['total_bets'] += 1
        logger.debug("10 wins reached, transitioning to GET_FEEDBACK")
        return GET_FEEDBACK
    else:
        if context.user_data['play_with_streak'] and not was_correct:
            context.user_data['play_with_streak'] = False
        elif len(context.user_data['predictions']) >= 4 and all(p == context.user_data['predictions'][-4] for p in context.user_data['predictions'][-4:]) and was_correct:
            context.user_data['play_with_streak'] = True

        query.edit_message_reply_markup(reply_markup=get_feedback_keyboard(disabled=True))
        query.message.reply_text(f"✅ Feedback recorded: {actual}. Predicting next outcome...", reply_markup=None)
        context.user_data['last_feedback_time'] = current_time
        logger.debug(f"Transitioning to PREDICT after feedback, time: {current_time}")
        return predict_next(update, context)

def skip_feedback(update, context):
    check_inactivity(update, context)
    query = update.callback_query
    query.answer()
    skip_count = context.user_data['skip_count']

    if query.data.startswith('skip_'):
        _, step_str, outcome = query.data.split('_')
        step = int(step_str)
        actual = 'Big' if outcome == 'yes' else 'Small'
        context.user_data['history'].append(actual)
        if len(context.user_data['history']) > 10:
            context.user_data['history'] = context.user_data['history'][-10:]
        context.user_data['step'] += 1
        context.user_data['skip_count'] += 1

        if context.user_data['skip_count'] < 2:
            query.edit_message_text(
                f"📝 Skipped bet {step} recorded: {actual}. What was the outcome of skipped bet {step + 1}?",
                reply_markup=get_skip_feedback_keyboard(step + 1)
            )
            logger.debug(f"Staying in SKIP_FEEDBACK, skip_count = {skip_count + 1}")
            return SKIP_FEEDBACK
        else:
            context.user_data['skip_count'] = 0
            context.user_data['predictions'] = context.user_data['predictions'][:-2]
            query.edit_message_text("✅ Skipping complete! Predicting the next outcome...")
            logger.debug(f"Skipping complete, predictions reset to {context.user_data['predictions']}")
            logger.debug(f"Current history after skipping: {context.user_data['history']}")
            return predict_next(update, context)
    elif query.data == "reset":
        reset_state(context)
        sent_message = query.message.reply_text(
            f"🔄 Reset successful, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Reset triggered, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "clear_history":
        bot = context.bot
        chat_id = query.message.chat_id
        for msg_id in context.user_data['all_message_ids']:
            try:
                bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception as e:
                logger.error(f"Failed to delete message {msg_id}: {e}")
        context.user_data['all_message_ids'] = []
        context.user_data['message_ids'] = []
        sent_message = query.message.reply_text(
            f"🗑️ Chat history cleared, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        logger.debug("Cleared history, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "exit":
        return exit_user(update, context)

    logger.debug("Staying in SKIP_FEEDBACK (default case)")
    return SKIP_FEEDBACK

def stats(update, context):
    check_inactivity(update, context)
    correct = context.user_data.get('correct', 0)
    total_bets = context.user_data.get('total_bets', 0)
    reward = context.user_data.get('reward', 0)
    bet_amount = 0.10
    losses = context.user_data.get('losses', [])
    wins = context.user_data.get('wins', [])
    loss_freq, win_freq, max_losses, max_wins = calculate_streak_frequencies(losses, wins)

    stats_message = (
        f"📊 **Betting Stats** 📊\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎲 **Bets Placed:** {total_bets}\n"
        f"📈 **Accuracy:** {correct / total_bets if total_bets > 0 else 0:.4f} ({correct / total_bets * 100 if total_bets > 0 else 0:.2f}%)\n"
        f"💸 **Total Reward:** {reward} (${reward * bet_amount:.2f})\n"
        f"🔴 **Max Consecutive Losses:** {max_losses}\n"
        f"🟢 **Max Consecutive Wins:** {max_wins}\n\n"
        f"**Loss Streak Frequencies:**\n"
        f"3 Losses: {loss_freq[3]}\n"
        f"4 Losses: {loss_freq[4]}\n"
        f"5 Losses: {loss_freq[5]}\n"
        f"6 Losses: {loss_freq[6]}\n"
        f"7+ Losses: {loss_freq[7]}\n\n"
        f"**Win Streak Frequencies:**\n"
        f"3 Wins: {win_freq[3]}\n"
        f"4 Wins: {win_freq[4]}\n"
        f"5 Wins: {win_freq[5]}\n"
        f"6+ Wins: {win_freq[6]}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    sent_message = update.message.reply_text(stats_message, parse_mode='Markdown',
                                            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reset", callback_data="reset"), InlineKeyboardButton("🚪 Exit", callback_data="exit")]]))
    context.user_data['message_ids'].append(sent_message.message_id)
    context.user_data['all_message_ids'].extend([update.message.message_id, sent_message.message_id])
    return GET_OUTCOMES

def stop(update, context):
    update.message.reply_text("Stopping the bot gracefully...")
    context.bot.stop_polling()
    cleanup()
    exit(0)

def error_handler(update, context):
    error = context.error
    logger.error(f"Error occurred: {str(error)}")
    if update and update.message:
        sent_message = update.message.reply_text(
            f"⚠️ An error occurred: {str(error)}. Please reset and start over.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reset", callback_data="reset"), InlineKeyboardButton("🚪 Exit", callback_data="exit")]])
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
    if "Conflict: terminated by other getUpdates request" in str(error):
        logger.error("Conflict detected. Please ensure only one bot instance is running.")

def cleanup():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        logger.info(f"Lock file {LOCK_FILE} removed.")

def main():
    logger.info("Bot process starting with PID %s", os.getpid())
    logger.info("Starting Flask thread...")
    flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=8080))
    flask_thread.daemon = True
    flask_thread.start()
    logger.info("Flask server started on port 8080 for keep-alive")

    attempt = 0
    max_attempts = 5
    while attempt < max_attempts:
        try:
            logger.info("Attempt %d/%d: Initializing Telegram bot...", attempt + 1, max_attempts)
            updater = Updater("7942589435:AAHGGR7dKjfOl491rQ4Bmsz8RmujC4ZZvUk", use_context=True)
            dp = updater.dispatcher

            conv_handler = ConversationHandler(
                entry_points=[CommandHandler('start', start)],
                states={
                    LOGIN: [
                        CallbackQueryHandler(login, pattern='^login$|^signup$|^exit$')
                    ],
                    LOGIN_USERNAME: [
                        MessageHandler(Filters.text & ~Filters.command, handle_login_username)
                    ],
                    LOGIN_PASSWORD: [
                        MessageHandler(Filters.text & ~Filters.command, handle_login_password)
                    ],
                    SIGNUP_USERNAME: [
                        MessageHandler(Filters.text & ~Filters.command, signup_username)
                    ],
                    SIGNUP_PASSWORD: [
                        MessageHandler(Filters.text & ~Filters.command, signup_password)
                    ],
                    ADMIN_MENU: [
                        CallbackQueryHandler(admin_menu, pattern='^admin_predict$|^admin_logs$|^exit$')
                    ],
                    GET_OUTCOMES: [
                        CallbackQueryHandler(get_outcomes, pattern='^outcome_.*$|^reset$|^clear_history$|^exit$')
                    ],
                    PREDICT: [
                        CallbackQueryHandler(predict_next)
                    ],
                    GET_FEEDBACK: [
                        CallbackQueryHandler(get_feedback, pattern='^feedback_yes$|^feedback_no$|^reset$|^clear_history$|^exit$|^disabled$')
                    ],
                    SKIP_FEEDBACK: [
                        CallbackQueryHandler(skip_feedback, pattern='^skip_.*$|^reset$|^clear_history$|^exit$')
                    ]
                },
                fallbacks=[CommandHandler('start', start), CommandHandler('stats', stats)]
            )

            dp.add_handler(conv_handler)
            dp.add_handler(CommandHandler("stats", stats))
            dp.add_handler(CommandHandler("stop", stop))
            dp.add_error_handler(error_handler)

            logger.info("Bot handlers registered, beginning polling...")
            updater.start_polling(timeout=15, drop_pending_updates=True)
            logger.info("Polling started successfully")
            updater.idle()
            logger.info("Bot stopped normally via idle")
            break
        except Exception as e:
            attempt += 1
            logger.error("Error in main loop (attempt %d/%d): %s", attempt, max_attempts, str(e))
            if "Conflict" in str(e):
                logger.warning("Telegram conflict detected, retrying in 5 seconds...")
                time.sleep(5)
                try:
                    updater.stop()
                except:
                    pass
            elif attempt >= max_attempts:
                logger.error("Max attempts reached, giving up.")
                cleanup()
                raise e
            else:
                cleanup()
                raise e
        finally:
            cleanup()

if __name__ == '__main__':
    main()

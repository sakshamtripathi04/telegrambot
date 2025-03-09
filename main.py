import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, ConversationHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import itertools
import warnings
import time
import os
from stable_baselines3.common.utils import get_schedule_fn
from flask import Flask
from threading import Thread
import logging  # Added for logging

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

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="telegram.ext.conversationhandler")

# Lock file to prevent multiple instances
LOCK_FILE = "bot.lock"

# Check if another instance is running
if os.path.exists(LOCK_FILE):
    logger.error(f"Another instance of the bot is running. Remove {LOCK_FILE} to force start.")
    exit(1)
else:
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))
    logger.info(f"Lock file created with PID {os.getpid()}")

# Load the trained model with custom_objects
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
GET_OUTCOMES, PREDICT, GET_FEEDBACK, SKIP_FEEDBACK = range(4)

# Global messages
APOLOGY_MESSAGE = (  
    "😔 **Maafi!** \n Hum continuously 6 baar haar chuke hain, aur yeh dil se bura lag raha hai. 💔\n\n"
    "Mujhe pata hai ki yeh phase tough hai, lekin tension mat lo! Main full effort dal raha hoon recovery ke liye. 🔥\n"
    "Bas thodi si patience rakho, shayad agla turn humare favor mein ho! 🍀\n\n"
    "Agar aap chahein toh hum **aage continue kar sakte hain 🔄,** ya phir reset dabake ek naya start le sakte hain. 🔃\n"  
)

INACTIVITY_WARNING = (
    "⚠️ **Warning:** You have been inactive for 90 seconds. Please interact with the bot, or it will clear your session in 30 more seconds.\n"
    "You can restart anytime with `/start`."
)

INACTIVITY_CLEANUP = (
    "⏹️ **Session Cleared:** You have been inactive for 120 seconds. All data related to this session has been cleared.\n"
    "Restart the bot with `/start` to begin anew."
)

# Function to preprocess 10 recent outcomes and pad to 20 timesteps
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

# Function to predict next outcome using DQN with timing
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

# Create inline keyboard for outcomes
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

# Create inline keyboard for feedback with disabled state
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

# Create inline keyboard for skip feedback
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

# Format the prediction message with exact spacing
def format_prediction_message(history, predictions, wins, predicted, loss_freq_6, current_streak_length, max_streak_length):
    last_bet_result = f"**Last Bet:** {predictions[-1]} {'✅' if wins[-1] else '❌'}\n" if predictions and wins else ""
    history_display = "**Bet History (Last 10):**\n\n"
    if len(predictions) == 0:
        history_display += "No bets yet.\n"
    else:
        start_idx = max(0, len(predictions) - 10)
        for i in range(start_idx, len(predictions)):
            bet = predictions[i]
            result = "💸" * (7 if wins[i] else 0)
            history_display += f"{bet} {result}\n"
    streak_info = f"**Current Streak:** {current_streak_length} wins | **Max Streak:** {max_streak_length} wins\n"
    loss_freq_display = f"**6 Consecutive Losses Frequency:** {loss_freq_6}\n"
    prediction_display = f"🎯 **Next Prediction:** {predicted.upper()} 🎯\n"
    message = (
        "🎰 **WIN GO 1 MIN** 🎰\n"
        "🌟 **MAINTAIN LEVEL 7** 🌟\n"
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
        "💰 **Keep Earning!** 💰\n"
        "━━━━━━━━━━━━━━━━━━━━━━━"
    )
    return message

# Calculate streak frequencies
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

# Reset function to clear state and restart
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

# Function to exit user session
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

# Function to check inactivity
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

# Function to send apology message
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
    
    welcome_message = f"🎉 Hello {user.first_name}! Welcome to **Shitty Predicts** 🎉\nWhat was outcome 1?"
    sent_welcome = update.message.reply_text(
        welcome_message,
        parse_mode='Markdown',
        reply_markup=get_outcome_keyboard(1)
    )
    context.user_data['message_ids'].append(sent_welcome.message_id)
    context.user_data['all_message_ids'].extend([update.message.message_id, sent_welcome.message_id])
    context.user_data['last_active_time'] = time.time()
    logger.debug(f"Transitioning to GET_OUTCOMES for user {chat_id}")
    return GET_OUTCOMES

# Handle outcome selection
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

# Predict the next outcome
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
        if update.callback_query:
            sent_message = update.callback_query.message.reply_text(error_msg, reply_markup=reply_markup)
        else:
            sent_message = update.message.reply_text(error_msg, reply_markup=reply_markup)
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
            prediction_message, parse_mode='Markdown', reply_markup=get_feedback_keyboard()
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

# Handle feedback
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
        sent_message = query.message.reply_text(prediction_message, parse_mode='Markdown', reply_markup=get_feedback_keyboard())
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

# Handle skip feedback
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

# Handle /stats command
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

# Handle /stop command
def stop(update, context):
    update.message.reply_text("Stopping the bot gracefully...")
    context.bot.stop_polling()
    cleanup()
    exit(0)

# Error handler
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

# Cleanup function to remove lock file on exit
def cleanup():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        logger.info(f"Lock file {LOCK_FILE} removed.")

# Main function to start the bot
def main():
    logger.info("Starting Flask thread...")
    flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=8080))
    flask_thread.daemon = True
    flask_thread.start()
    logger.info("Flask server started on port 8080 for keep-alive")

    try:
        logger.info("Initializing Telegram bot...")
        updater = Updater("7942589435:AAFPSKeu-9DXcEw2x7lLKHkur2K8po0Y2eU", use_context=True)
        dp = updater.dispatcher

        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', start)],
            states={
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

        logger.info("Bot started, beginning polling...")
        updater.start_polling()
        updater.idle()
    except KeyboardInterrupt:
        logger.info("Bot stopped via KeyboardInterrupt")
        cleanup()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        cleanup()
        raise e
    finally:
        cleanup()

if __name__ == '__main__':
    main()

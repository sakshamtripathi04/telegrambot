import pandas as pd
import numpy as np
from stable_baselines3 import DQN
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, ConversationHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import itertools
import warnings
import time
import os

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="telegram.ext.conversationhandler")

# Lock file to prevent multiple instances
LOCK_FILE = "bot.lock"

# Check if another instance is running
if os.path.exists(LOCK_FILE):
    print(f"Error: Another instance of the bot is running. Remove {LOCK_FILE} to force start.")
    exit(1)
else:
    with open(LOCK_FILE, "w") as f:
        f.write(str(os.getpid()))  # Write the process ID to the lock file
    print(f"Lock file created with PID {os.getpid()}")

# Load the trained model
try:
    dqn_model = DQN.load('dqn_betting_model')
    print("Model loaded successfully.")
    print(f"DQN model observation space: {dqn_model.observation_space}")  # Debug print to confirm shape
except Exception as e:
    print(f"Error loading model: {e}")
    os.remove(LOCK_FILE)  # Clean up lock file on error
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
    # Use the last 10 outcomes (or fewer if not enough)
    df = pd.DataFrame({'Outcome': recent_outcomes[-10:]})  # Limit to last 10 outcomes
    df['Outcome'] = df['Outcome'].map({'Big': 1, 'Small': 0})
    df['Streak'] = df['Outcome'].rolling(window=5, min_periods=1).sum().fillna(0)
    df['Trap_Risk'] = df['Outcome'].diff().abs().rolling(window=5, min_periods=1).mean().fillna(0)
    features = df[['Outcome', 'Streak', 'Trap_Risk']].values
    # Pad with zeros to ensure 20 timesteps
    if len(features) < lookback:
        padding = np.zeros((lookback - len(features), 3), dtype=np.float32)
        features = np.vstack((padding, features))
    end_time = time.time()
    print(f"prepare_live_input took {end_time - start_time:.4f} seconds for {len(recent_outcomes)} outcomes")
    return features.astype(np.float32)

# Function to predict next outcome using DQN with timing
def predict_next_outcome(recent_outcomes):
    start_time = time.time()
    try:
        obs = prepare_live_input(recent_outcomes, lookback=20)
        assert obs.shape == (20, 3), f"Expected shape (20, 3), got {obs.shape}"
        # Reshape to (1, 20, 3) if the model expects (n_env, 20, 3)
        if len(dqn_model.observation_space.shape) == 3 and dqn_model.observation_space.shape[1:] == (20, 3):
            obs = obs.reshape(1, 20, 3)
            print(f"Reshaped obs to {obs.shape} for (n_env, 20, 3) environment")
        action, _ = dqn_model.predict(obs, deterministic=True)
        predicted = 'Big' if action == 1 else 'Small'
        end_time = time.time()
        print(f"Model prediction time: {end_time - start_time:.4f} seconds for history {recent_outcomes[-10:]}")
        print(f"Predicted outcome: {predicted}")
        return predicted
    except Exception as e:
        print(f"Error in predict_next_outcome: {str(e)}")
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
            result = "💸" * (7 if wins[i] else 0)  # 7 money icons for wins
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
    context.user_data.clear()
    context.user_data['outcomes'] = []
    context.user_data['step'] = 0
    context.user_data['history'] = []
    context.user_data['predictions'] = []
    context.user_data['correct'] = 0
    context.user_data['total_bets'] = 0
    context.user_data['reward'] = 0
    context.user_data['losses'] = []
    context.user_data['wins'] = []
    context.user_data['initial_outcomes_collected'] = False
    context.user_data['current_streak_length'] = 0
    context.user_data['max_streak_length'] = 0
    context.user_data['loss_streak_length'] = 0
    context.user_data['skip_count'] = 0
    context.user_data['play_with_streak'] = False
    context.user_data['wait_count'] = 0
    context.user_data['message_ids'] = []  # Store bot message IDs for 10-win deletion
    context.user_data['all_message_ids'] = []  # Store all message IDs for clearing history
    context.user_data['apology_triggered'] = False  # Track if apology has been sent
    context.user_data['last_feedback_time'] = 0  # Track last feedback time for debouncing
    context.user_data['last_active_time'] = time.time()  # Track last active time
    chat_id = context.user_data.get('chat_id', 'unknown')
    print(f"Reset state for user {chat_id} at {time.ctime()}")

# Function to exit user session
def exit_user(update, context):
    if update.callback_query:
        query = update.callback_query
        query.answer()
        chat_id = query.message.chat_id
    else:
        chat_id = update.message.chat_id
    print(f"Exiting user session for chat_id: {chat_id}")
    context.user_data.clear()  # Clear all user data
    context.bot.send_message(chat_id=chat_id, text="🚪 Session exited. All processes cleared. Restart with `/start`.")
    return ConversationHandler.END

# Function to check inactivity
def check_inactivity(update, context):
    chat_id = update.effective_chat.id
    current_time = time.time()
    last_active = context.user_data.get('last_active_time', current_time)

    inactivity_duration = current_time - last_active

    if inactivity_duration >= 120:  # 120 seconds inactivity
        print(f"Automatic cleanup for chat_id {chat_id} due to 120 seconds inactivity")
        context.user_data.clear()
        context.bot.send_message(chat_id=chat_id, text=INACTIVITY_CLEANUP)
        return ConversationHandler.END
    elif inactivity_duration >= 90:  # 90 seconds inactivity
        if not context.user_data.get('inactivity_warning_sent', False):
            print(f"Sending warning for chat_id {chat_id} due to 90 seconds inactivity")
            sent_message = context.bot.send_message(chat_id=chat_id, text=INACTIVITY_WARNING)
            context.user_data['inactivity_warning_sent'] = True
            context.user_data['message_ids'].append(sent_message.message_id)
            context.user_data['all_message_ids'].append(sent_message.message_id)

    context.user_data['last_active_time'] = current_time  # Update last active time
    context.user_data['inactivity_warning_sent'] = False  # Reset warning flag on activity
    return None

# Function to send apology message
def send_apology_message(update, context):
    print("Debug: Sending apology message")
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
    context.user_data['apology_triggered'] = True  # Mark apology as sent
    context.user_data['last_active_time'] = time.time()  # Update last active time
    print("Debug: Transitioning to GET_OUTCOMES after apology")
    return GET_OUTCOMES

# Handle /start command
def start(update, context):
    user = update.message.from_user
    chat_id = update.message.chat_id
    context.user_data['chat_id'] = chat_id  # Store chat ID for debugging
    reset_state(context)  # Ensure a fresh start every time /start is called
    
    # Send welcome message with keyboard directly (Disclaimer removed)
    welcome_message = f"🎉 Hello {user.first_name}! Welcome to **Shitty Predicts** 🎉\nWhat was outcome 1?"
    sent_welcome = update.message.reply_text(
        welcome_message,
        parse_mode='Markdown',
        reply_markup=get_outcome_keyboard(1)
    )
    context.user_data['message_ids'].append(sent_welcome.message_id)
    context.user_data['all_message_ids'].append(update.message.message_id)  # User's /start
    context.user_data['all_message_ids'].append(sent_welcome.message_id)  # Bot's welcome

    context.user_data['last_active_time'] = time.time()  # Set initial active time
    print(f"Debug: Transitioning to GET_OUTCOMES for user {chat_id}")
    return GET_OUTCOMES

# Handle outcome selection
def get_outcomes(update, context):
    check_inactivity(update, context)  # Check inactivity before processing
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

        if step < 10:  # Collect only 10 outcomes
            query.edit_message_text(
                f"📝 Outcome {step} recorded: {outcome}. What was outcome {step + 1}?",
                reply_markup=get_outcome_keyboard(step + 1)
            )
            print(f"Debug: Staying in GET_OUTCOMES, step {step + 1}")
            return GET_OUTCOMES
        else:
            context.user_data['history'] = outcomes[-10:]  # Limit to last 10 outcomes
            context.user_data['initial_outcomes_collected'] = True
            context.user_data['step'] = step
            print(f"Debug: Transitioning to PREDICT with history = {context.user_data['history']}")
            query.edit_message_text("✅ Thanks! All 10 outcomes collected. Predicting the next outcome...")

            # Explicitly call predict_next instead of relying on state transition
            return predict_next(update, context)
    elif query.data == "reset":
        reset_state(context)
        sent_message = query.message.reply_text(
            f"🔄 Reset successful, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        print("Debug: Reset triggered, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "clear_history":
        bot = context.bot
        chat_id = query.message.chat_id
        for msg_id in context.user_data['all_message_ids']:
            try:
                bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception as e:
                print(f"Failed to delete message {msg_id}: {e}")
                continue
        context.user_data['all_message_ids'] = []
        context.user_data['message_ids'] = []
        sent_message = query.message.reply_text(
            f"🗑️ Chat history cleared, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        print("Debug: Cleared history, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "exit":
        return exit_user(update, context)

    print("Debug: Staying in GET_OUTCOMES (default case)")
    return GET_OUTCOMES

# Predict the next outcome
def predict_next(update, context):
    check_inactivity(update, context)  # Check inactivity before processing
    query = update.callback_query
    if query:
        query.answer()

    print("Debug: Entered predict_next")
    history = context.user_data['history'][-10:]  # Use only last 10 outcomes
    if len(history) < 10:  # Ensure we have at least 10 outcomes
        print(f"Error: history length is {len(history)}, expected at least 10. History: {history}")
        if update.callback_query:
            update.callback_query.message.reply_text(
                "⚠️ Internal error: Not enough history to make a prediction. Please provide more outcomes or reset.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reset", callback_data="reset"), InlineKeyboardButton("🚪 Exit", callback_data="exit")]])
            )
        else:
            update.message.reply_text(
                "⚠️ Internal error: Not enough history to make a prediction. Please provide more outcomes or reset.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reset", callback_data="reset"), InlineKeyboardButton("🚪 Exit", callback_data="exit")]])
            )
        print("Debug: Transitioning to GET_OUTCOMES due to insufficient history")
        return GET_OUTCOMES

    predictions = context.user_data['predictions']
    wins = context.user_data['wins']
    losses = context.user_data['losses']
    loss_freq, win_freq, max_losses, max_wins = calculate_streak_frequencies(losses, wins)
    loss_streak_length = context.user_data['loss_streak_length']
    print(f"Debug: Current loss streak length = {loss_streak_length}")

    start_time = time.time()
    try:
        # Check for apology after 6 consecutive losses
        if loss_streak_length >= 6 and not context.user_data.get('apology_triggered', False):
            print("Debug: Triggering apology for 6 consecutive losses")
            return send_apology_message(update, context)  # Send apology and pause for next outcome

        # Check loss streak and wait if >= 5
        if loss_streak_length >= 5:
            print("Debug: Triggering waiting phase for 5 consecutive losses")
            wait_bets = min(4, 10 - len(history))  # Limit to 10 outcomes
            if context.user_data.get('wait_count', 0) < wait_bets:
                context.user_data['wait_count'] = context.user_data.get('wait_count', 0) + 1
                if update.callback_query:
                    sent_message = update.callback_query.message.reply_text(
                        f"⏳ **Loss Streak ({loss_streak_length}) Detected!** 🔴\n"
                        f"Waiting and analyzing... (Bet {context.user_data['wait_count']}/{wait_bets})\n"
                        "Please provide the next outcome.",
                        reply_markup=get_outcome_keyboard(context.user_data['step'] + 1)
                    )
                else:
                    sent_message = update.message.reply_text(
                        f"⏳ **Loss Streak ({loss_streak_length}) Detected!** 🔴\n"
                        f"Waiting and analyzing... (Bet {context.user_data['wait_count']}/{wait_bets})\n"
                        "Please provide the next outcome.",
                        reply_markup=get_outcome_keyboard(context.user_data['step'] + 1)
                    )
                context.user_data['message_ids'].append(sent_message.message_id)
                context.user_data['all_message_ids'].append(sent_message.message_id)
                print("Debug: Transitioning to GET_OUTCOMES for waiting phase")
                return GET_OUTCOMES
            else:
                context.user_data['wait_count'] = 0
                print("Debug: Wait phase complete, making strong prediction")
                predicted = predict_next_outcome(history)
                context.user_data['play_with_streak'] = False
        else:
            print("Debug: Checking for artificial streak")
            # Check for artificial streak (4-5 consecutive same predictions)
            if len(predictions) >= 4:
                last_four = predictions[-4:]
                if all(p == last_four[0] for p in last_four):
                    print(f"Debug: Artificial streak detected with predictions: {predictions[-4:]}")
                    if update.callback_query:
                        sent_message = update.callback_query.message.reply_text(
                            "⚠️ **Artificial Streak Detected!** 🚨\n"
                            "It seems the streak (4-5 same outcomes) might be manipulated. Should we skip 1-2 bets to observe?\n"
                            "Please confirm the outcome of the next bet...",
                            reply_markup=get_skip_feedback_keyboard(1)
                        )
                    else:
                        sent_message = update.message.reply_text(
                            "⚠️ **Artificial Streak Detected!** 🚨\n"
                            "It seems the streak (4-5 same outcomes) might be manipulated. Should we skip 1-2 bets to observe?\n"
                            "Please confirm the outcome of the next bet...",
                            reply_markup=get_skip_feedback_keyboard(1)
                        )
                    context.user_data['message_ids'].append(sent_message.message_id)
                    context.user_data['all_message_ids'].append(sent_message.message_id)
                    context.user_data['skip_count'] = 1
                    print("Debug: Transitioning to SKIP_FEEDBACK due to artificial streak")
                    return SKIP_FEEDBACK

            print("Debug: Checking play_with_streak")
            # Play with streak if active
            if context.user_data.get('play_with_streak', False) and len(predictions) >= 1:
                predicted = predictions[-1]  # Continue the streak
                print("Debug: Continuing streak with prediction =", predicted)
            else:
                print("Debug: Calling predict_next_outcome")
                predicted = predict_next_outcome(history)

        end_time = time.time()
        print(f"Total prediction time: {end_time - start_time:.4f} seconds for history {history}")
        context.user_data['last_predicted'] = predicted

        print("Debug: Formatting prediction message")
        # Format and send the prediction message, store the message ID
        prediction_message = format_prediction_message(
            history, predictions, wins, predicted, loss_freq[6],
            context.user_data.get('current_streak_length', 0),
            context.user_data.get('max_streak_length', 0)
        )
        print("Debug: Sending prediction message")
        send_start = time.time()
        if update.callback_query:
            sent_message = update.callback_query.message.reply_text(
                prediction_message,
                parse_mode='Markdown',
                reply_markup=get_feedback_keyboard()
            )
        else:
            sent_message = update.message.reply_text(
                prediction_message,
                parse_mode='Markdown',
                reply_markup=get_feedback_keyboard()
            )
        send_end = time.time()
        print(f"Sending message took {send_end - send_start:.4f} seconds")
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)

        context.user_data['predictions'].append(predicted)
        context.user_data['total_bets'] += 1
        print("Debug: Transitioning to GET_FEEDBACK after prediction")
        return GET_FEEDBACK

    except Exception as e:
        print(f"Error in predict_next: {str(e)}")
        if update.callback_query:
            update.callback_query.message.reply_text(
                f"⚠️ An error occurred during prediction: {str(e)}. Please reset and start over.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reset", callback_data="reset"), InlineKeyboardButton("🚪 Exit", callback_data="exit")]])
            )
        else:
            update.message.reply_text(
                f"⚠️ An error occurred during prediction: {str(e)}. Please reset and start over.",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reset", callback_data="reset"), InlineKeyboardButton("🚪 Exit", callback_data="exit")]])
            )
        print("Debug: Transitioning to GET_OUTCOMES due to error in predict_next")
        return GET_OUTCOMES

# Handle feedback
def get_feedback(update, context):
    check_inactivity(update, context)  # Check inactivity before processing
    query = update.callback_query
    if not query:
        return GET_FEEDBACK  # Ignore if not a callback query

    current_time = time.time()
    last_feedback_time = context.user_data.get('last_feedback_time', 0)
    if current_time - last_feedback_time < 1.0:  # Debounce: Ignore clicks within 1 second
        query.answer("Please wait a moment before clicking again.")
        print(f"Debug: Debounced feedback click at {current_time}, last at {last_feedback_time}")
        return GET_FEEDBACK

    query.answer()  # Acknowledge the click immediately
    print(f"Debug: Entered get_feedback at {current_time}")

    if 'last_predicted' not in context.user_data:
        # Safeguard: If no prediction exists, reset and start over
        reset_state(context)
        sent_message = query.message.reply_text(
            "⚠️ No prediction found. Session may have been reset. Starting over.\nWhat was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        print("Debug: Transitioning to GET_OUTCOMES due to missing last_predicted")
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
        context.user_data['loss_streak_length'] = 0  # Reset loss streak on win
        context.user_data['apology_triggered'] = False  # Reset apology flag on win
        was_correct = True
        print("Debug: Feedback yes, win recorded")
    elif query.data == "feedback_no":
        context.user_data['reward'] -= 1
        context.user_data['wins'].append(0)
        context.user_data['losses'].append(1)
        actual = 'Small' if predicted == 'Big' else 'Big'
        context.user_data['current_streak_length'] = 0
        context.user_data['loss_streak_length'] += 1  # Increment loss streak on loss
        was_correct = False
        print("Debug: Feedback no, loss recorded")
    elif query.data == "reset":
        reset_state(context)
        sent_message = query.message.reply_text(
            f"🔄 Reset successful, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        print("Debug: Reset triggered, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "clear_history":
        bot = context.bot
        chat_id = query.message.chat_id
        for msg_id in context.user_data['all_message_ids']:
            try:
                bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception as e:
                print(f"Failed to delete message {msg_id}: {e}")
                continue
        context.user_data['all_message_ids'] = []
        context.user_data['message_ids'] = []
        sent_message = query.message.reply_text(
            f"🗑️ Chat history cleared, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        print("Debug: Cleared history, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "exit":
        return exit_user(update, context)
    elif query.data == "disabled":
        query.answer("Button disabled, please wait.")
        return GET_FEEDBACK
    else:
        sent_message = query.message.reply_text("⚠️ Please select 'Yes' or 'No'.",
                                               reply_markup=get_feedback_keyboard())
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        print("Debug: Invalid feedback, staying in GET_FEEDBACK")
        return GET_FEEDBACK

    # Update history
    context.user_data['history'].append(actual)
    if len(context.user_data['history']) > 10:  # Limit to last 10 outcomes
        context.user_data['history'] = context.user_data['history'][-10:]
    context.user_data['step'] = len(context.user_data['history'])

    # Check for 10 wins and handle accordingly
    if context.user_data['correct'] == 10:
        # Delete all stored message IDs
        bot = context.bot
        chat_id = query.message.chat_id
        for msg_id in context.user_data['message_ids']:
            try:
                bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception as e:
                print(f"Failed to delete message {msg_id}: {e}")
                continue

        sent_message = query.message.reply_text("🎉 **10 Wins Complete!** 🏆\nStarting a new prediction cycle...",
                                               parse_mode='Markdown')
        context.user_data['message_ids'] = []
        context.user_data['all_message_ids'].append(sent_message.message_id)
        # Generate new prediction
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
        sent_message = query.message.reply_text(prediction_message, parse_mode='Markdown',
                                               reply_markup=get_feedback_keyboard())
        context.user_data['message_ids'] = [sent_message.message_id]  # Reset with new message ID
        context.user_data['all_message_ids'].append(sent_message.message_id)
        context.user_data['predictions'].append(predicted)
        context.user_data['total_bets'] += 1
        print("Debug: 10 wins reached, transitioning to GET_FEEDBACK")
        return GET_FEEDBACK
    else:
        # Adjust strategy based on streak and loss
        if context.user_data['play_with_streak'] and not was_correct:
            context.user_data['play_with_streak'] = False  # Streak broken, revert to DQN
        elif len(context.user_data['predictions']) >= 4:
            last_four = context.user_data['predictions'][-4:]
            if all(p == last_four[0] for p in last_four) and was_correct:
                context.user_data['play_with_streak'] = True  # Continue playing with streak

        # Disable buttons after first click and confirm
        query.edit_message_reply_markup(reply_markup=get_feedback_keyboard(disabled=True))
        query.message.reply_text(f"✅ Feedback recorded: {actual}. Predicting next outcome...", reply_markup=None)
        context.user_data['last_feedback_time'] = current_time
        print(f"Debug: Transitioning to PREDICT after feedback, time: {current_time}")
        return predict_next(update, context)  # Transition to PREDICT state immediately

# Handle skip feedback
def skip_feedback(update, context):
    check_inactivity(update, context)  # Check inactivity before processing
    query = update.callback_query
    query.answer()
    skip_count = context.user_data['skip_count']

    if query.data.startswith('skip_'):
        _, step_str, outcome = query.data.split('_')
        step = int(step_str)
        actual = 'Big' if outcome == 'yes' else 'Small'
        context.user_data['history'].append(actual)
        if len(context.user_data['history']) > 10:  # Limit to last 10 outcomes
            context.user_data['history'] = context.user_data['history'][-10:]
        context.user_data['step'] += 1
        context.user_data['skip_count'] += 1

        if context.user_data['skip_count'] < 2:
            query.edit_message_text(
                f"📝 Skipped bet {step} recorded: {actual}. What was the outcome of skipped bet {step + 1}?",
                reply_markup=get_skip_feedback_keyboard(step + 1)
            )
            print(f"Debug: Staying in SKIP_FEEDBACK, skip_count = {skip_count + 1}")
            return SKIP_FEEDBACK
        else:
            context.user_data['skip_count'] = 0
            context.user_data['predictions'] = context.user_data['predictions'][:-2]  # Remove last 2 predictions to break streak
            query.edit_message_text("✅ Skipping complete! Predicting the next outcome...")
            print(f"Debug: Skipping complete, predictions reset to {context.user_data['predictions']}")
            print(f"Debug: Current history after skipping: {context.user_data['history']}")
            # Force a new prediction immediately
            return predict_next(update, context)  # Directly call predict_next instead of just transitioning
    elif query.data == "reset":
        reset_state(context)
        sent_message = query.message.reply_text(
            f"🔄 Reset successful, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        print("Debug: Reset triggered, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "clear_history":
        bot = context.bot
        chat_id = query.message.chat_id
        for msg_id in context.user_data['all_message_ids']:
            try:
                bot.delete_message(chat_id=chat_id, message_id=msg_id)
            except Exception as e:
                print(f"Failed to delete message {msg_id}: {e}")
                continue
        context.user_data['all_message_ids'] = []
        context.user_data['message_ids'] = []
        sent_message = query.message.reply_text(
            f"🗑️ Chat history cleared, {query.from_user.first_name}! Starting over. What was outcome 1?",
            reply_markup=get_outcome_keyboard(1)
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
        print("Debug: Cleared history, transitioning to GET_OUTCOMES")
        return GET_OUTCOMES
    elif query.data == "exit":
        return exit_user(update, context)

    print("Debug: Staying in SKIP_FEEDBACK (default case)")
    return SKIP_FEEDBACK

# Handle /stats command
def stats(update, context):
    check_inactivity(update, context)  # Check inactivity before processing
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
    context.user_data['all_message_ids'].append(update.message.message_id)
    context.user_data['all_message_ids'].append(sent_message.message_id)
    return GET_OUTCOMES  # Stay in GET_OUTCOMES to allow further interaction

# Handle /stop command
def stop(update, context):
    update.message.reply_text("Stopping the bot gracefully...")
    context.bot.stop_polling()
    cleanup()
    exit(0)

# Error handler
def error_handler(update, context):
    error = context.error
    if update and update.message:
        sent_message = update.message.reply_text(
            f"⚠️ An error occurred: {str(error)}. Please reset and start over.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Reset", callback_data="reset"), InlineKeyboardButton("🚪 Exit", callback_data="exit")]])
        )
        context.user_data['message_ids'].append(sent_message.message_id)
        context.user_data['all_message_ids'].append(sent_message.message_id)
    else:
        print(f"Error occurred without update: {str(error)}")
    if "Conflict: terminated by other getUpdates request" in str(error):
        print("Conflict detected. Please ensure only one bot instance is running.")

# Cleanup function to remove lock file on exit
def cleanup():
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        print(f"Lock file {LOCK_FILE} removed.")

# Main function to start the bot
def main():
    try:
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
            fallbacks=[CommandHandler('start', start), CommandHandler('stats', stats)]  # Allow /start and /stats to restart or check stats
        )

        dp.add_handler(conv_handler)
        dp.add_handler(CommandHandler("stats", stats))
        dp.add_handler(CommandHandler("stop", stop))
        dp.add_error_handler(error_handler)

        updater.start_polling()
        updater.idle()
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        cleanup()
        raise e
    finally:
        cleanup()

if __name__ == '__main__':
    main()
    print('Bot started')
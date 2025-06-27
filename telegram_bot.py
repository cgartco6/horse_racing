# telegram_bot.py
import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
import data_simulator
import model
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the model
racing_model = model.HorseRacingModel()

# For simulating the Hollywoodbets and Betway predictions, we'll create a function that slightly perturbs our model's predictions.
def simulate_external_predictions(probs):
    # Let's assume Hollywoodbets and Betway have their own models which are slightly different
    hollywoodbets_probs = np.array(probs) * np.random.uniform(0.9, 1.1, len(probs))
    hollywoodbets_probs = hollywoodbets_probs / hollywoodbets_probs.sum()
    
    betway_probs = np.array(probs) * np.random.uniform(0.8, 1.2, len(probs))
    betway_probs = betway_probs / betway_probs.sum()
    
    return hollywoodbets_probs, betway_probs

def predict_race(race_data):
    # Convert the race data to a DataFrame for prediction
    df = pd.DataFrame(race_data['horses'])
    # We need to preprocess: one-hot encode track and weather, and standardize if needed.
    # For simplicity, we'll assume our model was trained on numeric features only and we ignore categoricals in this simulation.
    # In reality, we would one-hot encode.
    features = ['horse_age', 'jockey_win_rate', 'trainer_win_rate', 'weight', 'recent_form', 'days_since_last_run', 'injury', 'distance']
    X = df[features]
    # Predict
    probas = racing_model.predict(X)
    # We are predicting the probability of winning for each horse
    win_probs = probas[:, 1]  # assuming binary classification: 1 is win, 0 is not win
    df['win_prob'] = win_probs
    df['rank'] = df['win_prob'].rank(ascending=False)
    df = df.sort_values('rank')
    
    # Simulate external predictions
    hollywoodbets, betway = simulate_external_predictions(win_probs)
    df['hollywoodbets'] = hollywoodbets
    df['betway'] = betway
    
    return df

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Welcome to Horse Racing Prediction Bot! Use /today to get today\'s predictions.')

def today(update: Update, context: CallbackContext) -> None:
    # Simulate today's races
    races = data_simulator.simulate_race_data(num_races=3, num_horses_per_race=5)
    
    for race in races:
        track = race['track']
        time = race['race_time']
        distance = race['distance']
        weather = race['weather']
        
        # Predict the race
        result_df = predict_race(race)
        top_horse = result_df.iloc[0]
        
        # Format the message
        message = f"🏇 *Race at {track}* 🏇\n"
        message += f"Time: {time}, Distance: {distance}m, Weather: {weather}\n\n"
        message += "*Top Pick to Win:*\n"
        message += f"🐴 {top_horse['horse_name']} (Win Probability: {top_horse['win_prob']*100:.1f}%)\n\n"
        message += "*Full Race Prediction (Win Probability):*\n"
        for idx, row in result_df.iterrows():
            message += f"{int(row['rank'])}. {row['horse_name']}: {row['win_prob']*100:.1f}% (Hollywoodbets: {row['hollywoodbets']*100:.1f}%, Betway: {row['betway']*100:.1f}%)\n"
        
        # Send the message
        context.bot.send_message(chat_id=update.effective_chat.id, text=message, parse_mode='Markdown')

def main() -> None:
    # Create the Updater and pass it your bot's token.
    # TODO: Replace with your Telegram bot token
    updater = Updater("YOUR_TELEGRAM_BOT_TOKEN")

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("today", today))

    # Start the Bot
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

use teloxide::prelude::*;
use teloxide::types::{KeyboardButton, KeyboardMarkup};
use teloxide::utils::command::BotCommands;
use dotenv::dotenv;
use std::collections::HashMap;
use tokio::sync::Mutex;
use std::sync::Arc;

type UserStates = Arc<Mutex<HashMap<ChatId, UserState>>>;

#[derive(Clone, Debug)]
enum UserState {
    Idle,
    WaitingFirstNumber,
    WaitingSecondNumber(u32),
}

#[tokio::main]
async fn main() {
    dotenv().ok();
    println!("Starting bot...");
    let bot = Bot::from_env();
    let user_states: UserStates = Arc::new(Mutex::new(HashMap::new()));
    println!("Bot started. Waiting for commands...");
    
    let handler = dptree::entry()
        .branch(Update::filter_message().filter_command::<Command>().endpoint(handle_commands))
        .branch(Update::filter_message().endpoint(handle_text));
    
    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![user_states])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;
}

#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase")]
enum Command {
    #[command(description = "start the bot")]
    Start,
    #[command(description = "show help")]
    Help,
    #[command(description = "greet user with info")]
    Greeting,
    #[command(description = "sum two numbers")]
    Sum,
}

fn main_menu() -> KeyboardMarkup {
    KeyboardMarkup::new(vec![
        vec![
            KeyboardButton::new("👋 Greeting"),
            KeyboardButton::new("➕ Sum"),
        ],
        vec![KeyboardButton::new("❓ Help")],
    ])
    .resize_keyboard(true)
}

fn back_button() -> KeyboardMarkup {
    KeyboardMarkup::new(vec![
        vec![KeyboardButton::new("⬅️ Back to Menu")],
    ])
    .resize_keyboard(true)
}

async fn handle_commands(bot: Bot, msg: Message, cmd: Command, user_states: UserStates) -> ResponseResult<()> {
    let chat_id = msg.chat.id;
    
    match cmd {
        Command::Start => {
            reset_state(chat_id, &user_states).await;
            bot.send_message(chat_id, "🤖 Welcome! Choose what you want to do:")
                .reply_markup(main_menu())
                .await?;
        }
        Command::Help => {
            reset_state(chat_id, &user_states).await;
            bot.send_message(chat_id, "📋 Available commands:\n/start - Start bot\n/help - Show help\n/greeting - Show your info\n/sum - Add numbers")
                .reply_markup(main_menu())
                .await?;
        }
        Command::Greeting => {
            reset_state(chat_id, &user_states).await;
            show_user_info(&bot, &msg).await?;
        }
        Command::Sum => {
            start_sum(&bot, chat_id, &user_states).await?;
        }
    }
    Ok(())
}

async fn handle_text(bot: Bot, msg: Message, user_states: UserStates) -> ResponseResult<()> {
    if let Some(text) = msg.text() {
        let chat_id = msg.chat.id;
        
        // Handle button presses first
        match text {
            "👋 Greeting" => {
                reset_state(chat_id, &user_states).await;
                show_user_info(&bot, &msg).await?;
                return Ok(());
            }
            "➕ Sum" => {
                start_sum(&bot, chat_id, &user_states).await?;
                return Ok(());
            }
            "❓ Help" => {
                reset_state(chat_id, &user_states).await;
                bot.send_message(chat_id, "📋 Available commands:\n/start - Start bot\n/help - Show help\n/greeting - Show your info\n/sum - Add numbers")
                    .reply_markup(main_menu())
                    .await?;
                return Ok(());
            }
            "⬅️ Back to Menu" => {
                reset_state(chat_id, &user_states).await;
                bot.send_message(chat_id, "🏠 Back to main menu!")
                    .reply_markup(main_menu())
                    .await?;
                return Ok(());
            }
            _ => {}
        }
        
        // Handle number input for sum
        let state = get_state(chat_id, &user_states).await;
        match state {
            UserState::WaitingFirstNumber => {
                if let Ok(num) = text.parse::<u32>() {
                    set_state(chat_id, UserState::WaitingSecondNumber(num), &user_states).await;
                    bot.send_message(chat_id, format!("✅ First number: {}\n\n➕ Now enter the second number:", num))
                        .reply_markup(back_button())
                        .await?;
                } else {
                    bot.send_message(chat_id, "❌ Please enter a valid number:")
                        .reply_markup(back_button())
                        .await?;
                }
            }
            UserState::WaitingSecondNumber(first) => {
                if let Ok(second) = text.parse::<u32>() {
                    let result = first + second;
                    reset_state(chat_id, &user_states).await;
                    bot.send_message(chat_id, format!("🧮 Calculation Complete!\n\n{} + {} = {}", first, second, result))
                        .reply_markup(main_menu())
                        .await?;
                } else {
                    bot.send_message(chat_id, "❌ Please enter a valid number:")
                        .reply_markup(back_button())
                        .await?;
                }
            }
            UserState::Idle => {
                bot.send_message(chat_id, "🤖 Use /start to see the menu or click the buttons below!")
                    .reply_markup(main_menu())
                    .await?;
            }
        }
    }
    Ok(())
}

async fn show_user_info(bot: &Bot, msg: &Message) -> ResponseResult<()> {
    let user = msg.from().unwrap();
    let info = format!(
        "👋 Hello {}!\n\n\
        📊 Your Information:\n\
        🆔 User ID: {}\n\
        👤 First Name: {}\n\
        👥 Last Name: {}\n\
        📧 Username: @{}\n\
        🌐 Language: {}\n\
        💬 Chat ID: {}",
        user.first_name,
        user.id,
        user.first_name,
        user.last_name.as_deref().unwrap_or("Not set"),
        user.username.as_deref().unwrap_or("Not set"),
        user.language_code.as_deref().unwrap_or("Not set"),
        msg.chat.id
    );
    
    bot.send_message(msg.chat.id, info)
        .reply_markup(main_menu())
        .await?;
    Ok(())
}

async fn start_sum(bot: &Bot, chat_id: ChatId, user_states: &UserStates) -> ResponseResult<()> {
    set_state(chat_id, UserState::WaitingFirstNumber, user_states).await;
    bot.send_message(chat_id, "🔢 Sum Calculator\n\nEnter the first number:")
        .reply_markup(back_button())
        .await?;
    Ok(())
}

async fn get_state(chat_id: ChatId, user_states: &UserStates) -> UserState {
    let states = user_states.lock().await;
    states.get(&chat_id).cloned().unwrap_or(UserState::Idle)
}

async fn set_state(chat_id: ChatId, state: UserState, user_states: &UserStates) {
    let mut states = user_states.lock().await;
    states.insert(chat_id, state);
}

async fn reset_state(chat_id: ChatId, user_states: &UserStates) {
    let mut states = user_states.lock().await;
    states.insert(chat_id, UserState::Idle);
}

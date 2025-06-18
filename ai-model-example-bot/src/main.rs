use teloxide::prelude::*;
use teloxide::types::ChatAction;
use reqwest::Proxy;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::time::Duration;
use dotenv::dotenv;

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
}

async fn call_openai(text: &str) -> Result<String, Box<dyn Error + Send + Sync>> {
    let api_key = std::env::var("OPENAI_API_KEY")?;
    let proxy_url = std::env::var("OPENAI_PROXY")?;
    
    let client = reqwest::Client::builder()
        .proxy(Proxy::all(&proxy_url)?)
        .timeout(Duration::from_secs(30))
        .build()?;
    
    let request = OpenAIRequest {
        model: "gpt-4o".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: text.to_string(),
        }],
    };
    
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;
    
    if !response.status().is_success() {
        let error_text = response.text().await?;
        return Err(format!("OpenAI API error: {}", error_text).into());
    }
    
    let openai_response: OpenAIResponse = response.json().await?;
    
    Ok(openai_response
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .unwrap_or_else(|| "No response from OpenAI".to_string()))
}

#[tokio::main]
async fn main() {
    pretty_env_logger::init();
    dotenv().ok();
    
    let bot = Bot::from_env();
    
    teloxide::repl(bot, |bot: Bot, msg: Message| async move {
        if let Some(text) = msg.text() {
            bot.send_chat_action(msg.chat.id, ChatAction::Typing).await?;
            
            match call_openai(text).await {
                Ok(response) => {
                    bot.send_message(msg.chat.id, response).await?;
                }
                Err(e) => {
                    bot.send_message(msg.chat.id, format!("Error: {}", e)).await?;
                }
            }
        }
        
        Ok(())
    })
    .await;
}

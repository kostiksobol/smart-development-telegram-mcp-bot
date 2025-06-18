use teloxide::prelude::*;
use teloxide::types::{KeyboardButton, KeyboardMarkup, ChatAction, ParseMode};
use teloxide::utils::command::BotCommands;
use dotenv::dotenv;
use std::collections::HashMap;
use tokio::sync::Mutex;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use tokio::process::{Command as ProcessCommand, Child};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use anyhow::{Result, anyhow};
use log::{info, debug};
use std::path::PathBuf;
use reqwest::Proxy;
use std::time::Duration;
use indexmap::IndexMap;

type UserStates = Arc<Mutex<HashMap<ChatId, UserState>>>;
type McpServers = Arc<Mutex<HashMap<String, McpServerHandle>>>;

#[derive(Clone, Debug)]
enum UserState {
    Idle,
    SelectingServer,
    SelectingCommand(String), // server name
    CollectingArguments {
        server: String,
        command: String,
        params: Vec<ToolParam>,
        param_names: Vec<String>,
        collected: Vec<(String, serde_json::Value)>,
        current_index: usize,
    },
}

#[derive(Debug)]
struct McpServerHandle {
    _process: Child,
    stdin: tokio::process::ChildStdin,
    reader: Arc<Mutex<BufReader<tokio::process::ChildStdout>>>,
    tools: Vec<Tool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct McpConfig {
    #[serde(rename = "mcpServers")]
    mcp_servers: HashMap<String, ServerConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ServerConfig {
    command: String,
    args: Vec<String>,
    #[serde(default)]
    env: HashMap<String, String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct Tool {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(rename = "inputSchema", alias = "input_schema", default)]
    input_schema: Option<InputSchema>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct InputSchema {
    #[serde(rename = "type")]
    schema_type: String,
    #[serde(default)]
    properties: IndexMap<String, ToolParam>,
    #[serde(default)]
    required: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ToolParam {
    #[serde(rename = "type")]
    param_type: String,
    #[serde(default)]
    description: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: serde_json::Value,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<serde_json::Value>,
}

// OpenAI API structures
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

#[tokio::main]
async fn main() {
    dotenv().ok();
    env_logger::init();
    
    info!("Starting Enhanced MCP Telegram bot with AI...");
    let bot = Bot::from_env();
    let user_states: UserStates = Arc::new(Mutex::new(HashMap::new()));
    let mcp_servers: McpServers = Arc::new(Mutex::new(HashMap::new()));
    
    info!("Bot started. Waiting for commands...");
    
    let handler = dptree::entry()
        .branch(Update::filter_message().filter_command::<Command>().endpoint(handle_commands))
        .branch(Update::filter_message().endpoint(handle_text));
    
    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![user_states, mcp_servers])
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
}

fn main_menu() -> KeyboardMarkup {
    KeyboardMarkup::new(vec![
        vec![KeyboardButton::new("🔌 MCP Servers")],
        vec![KeyboardButton::new("❓ Help")],
    ])
    .resize_keyboard(true)
}

fn server_menu(servers: Vec<String>) -> KeyboardMarkup {
    let mut keyboard = vec![];
    
    for server in servers.chunks(2) {
        let row: Vec<KeyboardButton> = server
            .iter()
            .map(|s| KeyboardButton::new(format!("📡 {}", s)))
            .collect();
        keyboard.push(row);
    }
    
    keyboard.push(vec![KeyboardButton::new("⬅️ Back to Menu")]);
    
    KeyboardMarkup::new(keyboard).resize_keyboard(true)
}

fn command_menu(commands: Vec<String>) -> KeyboardMarkup {
    let mut keyboard = vec![];
    
    for cmd in commands.chunks(2) {
        let row: Vec<KeyboardButton> = cmd
            .iter()
            .map(|c| KeyboardButton::new(format!("⚡ {}", c)))
            .collect();
        keyboard.push(row);
    }
    
    keyboard.push(vec![KeyboardButton::new("⬅️ Back to Servers")]);
    
    KeyboardMarkup::new(keyboard).resize_keyboard(true)
}

fn back_button() -> KeyboardMarkup {
    KeyboardMarkup::new(vec![
        vec![KeyboardButton::new("⬅️ Back to Commands")],
    ])
    .resize_keyboard(true)
}

fn escape_markdown_v2(text: &str) -> String {
    let chars_to_escape = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!'];
    let mut result = String::with_capacity(text.len() * 2);
    
    for ch in text.chars() {
        if chars_to_escape.contains(&ch) {
            result.push('\\');
        }
        result.push(ch);
    }
    
    result
}

async fn handle_commands(bot: Bot, msg: Message, cmd: Command, user_states: UserStates, _mcp_servers: McpServers) -> ResponseResult<()> {
    let chat_id = msg.chat.id;
    
    match cmd {
        Command::Start => {
            reset_state(chat_id, &user_states).await;
            bot.send_message(chat_id, "🤖 Welcome to Enhanced MCP Bot with AI!\n\nI can help you interact with Model Context Protocol servers and provide AI analysis of commands and responses.")
                .reply_markup(main_menu())
                .await?;
        }
        Command::Help => {
            reset_state(chat_id, &user_states).await;
            bot.send_message(
                chat_id, 
                "📋 Enhanced MCP Bot Help\n\n\
                This bot allows you to interact with MCP servers and get AI-powered insights.\n\n\
                Features:\n\
                • Execute MCP commands with visual input/output\n\
                • AI analysis of command results\n\
                • JSON request/response visualization\n\n\
                How to use:\n\
                1. Click '🔌 MCP Servers' to see available servers\n\
                2. Select a server to connect and see its tools\n\
                3. Choose a tool to execute\n\
                4. Provide required arguments when prompted\n\
                5. View the results and AI analysis\n\n\
                Commands:\n\
                /start - Start the bot\n\
                /help - Show this help message"
            )
                .reply_markup(main_menu())
                .await?;
        }
    }
    Ok(())
}

async fn handle_text(bot: Bot, msg: Message, user_states: UserStates, mcp_servers: McpServers) -> ResponseResult<()> {
    if let Some(text) = msg.text() {
        let chat_id = msg.chat.id;
        
        // Handle main menu buttons
        match text {
            "🔌 MCP Servers" => {
                show_servers(&bot, chat_id, &user_states).await?;
                return Ok(());
            }
            "❓ Help" => {
                reset_state(chat_id, &user_states).await;
                bot.send_message(
                    chat_id, 
                    "📋 Enhanced MCP Bot Help\n\n\
                    This bot allows you to interact with MCP servers and get AI-powered insights.\n\n\
                    Features:\n\
                    • Execute MCP commands with visual input/output\n\
                    • AI analysis of command results\n\
                    • JSON request/response visualization\n\n\
                    How to use:\n\
                    1. Click '🔌 MCP Servers' to see available servers\n\
                    2. Select a server to connect and see its tools\n\
                    3. Choose a tool to execute\n\
                    4. Provide required arguments when prompted\n\
                    5. View the results and AI analysis\n\n\
                    Commands:\n\
                    /start - Start the bot\n\
                    /help - Show this help message"
                )
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
            "⬅️ Back to Servers" => {
                show_servers(&bot, chat_id, &user_states).await?;
                return Ok(());
            }
            "⬅️ Back to Commands" => {
                // Check if we're in argument collection state
                let state = get_state(chat_id, &user_states).await;
                if let UserState::CollectingArguments { server, .. } = state {
                    // Go back to command selection for this server
                    let servers = mcp_servers.lock().await;
                    if let Some(server_handle) = servers.get(&server) {
                        let commands: Vec<String> = server_handle.tools.iter().map(|t| t.name.clone()).collect();
                        
                        let command_list = server_handle.tools.iter()
                            .map(|t| format!("• {} {}", t.name, 
                                t.description.as_deref()
                                    .map(|d| format!("- {}", d))
                                    .unwrap_or_default()
                            ))
                            .collect::<Vec<_>>()
                            .join("\n");
                        
                        set_state(chat_id, UserState::SelectingCommand(server.clone()), &user_states).await;
                        bot.send_message(
                            chat_id,
                            format!("⚡ Available commands for '{}' ({}):\n\n{}\n\nSelect a command to execute:", server, commands.len(), command_list),
                        )
                            .reply_markup(command_menu(commands))
                            .await?;
                    }
                } else {
                    // If not in argument collection, just go back to menu
                    reset_state(chat_id, &user_states).await;
                    bot.send_message(chat_id, "🏠 Back to main menu!")
                        .reply_markup(main_menu())
                        .await?;
                }
                return Ok(());
            }
            _ => {}
        }
        
        // Handle server selection
        if text.starts_with("📡 ") {
            let server_name = text.trim_start_matches("📡 ");
            connect_to_server(&bot, chat_id, server_name, &user_states, &mcp_servers).await?;
            return Ok(());
        }
        
        // Handle command selection
        if text.starts_with("⚡ ") {
            let command_name = text.trim_start_matches("⚡ ");
            if let UserState::SelectingCommand(server) = get_state(chat_id, &user_states).await {
                start_command_collection(&bot, chat_id, &server, command_name, &user_states, &mcp_servers).await?;
            }
            return Ok(());
        }
        
        // Handle argument collection
        let state = get_state(chat_id, &user_states).await;
        if let UserState::CollectingArguments { server, command, params, param_names, mut collected, current_index } = state {
            if current_index < params.len() {
                let param = &params[current_index];
                let param_name = param_names[current_index].clone();
                
                // Parse the value based on type
                let value = match param.param_type.as_str() {
                    "number" => {
                        if let Ok(num) = text.parse::<f64>() {
                            serde_json::Value::Number(serde_json::Number::from_f64(num).unwrap())
                        } else {
                            bot.send_message(chat_id, "❌ Please enter a valid number:")
                                .reply_markup(back_button())
                                .await?;
                            return Ok(());
                        }
                    }
                    "boolean" => {
                        match text.to_lowercase().as_str() {
                            "true" | "yes" | "1" => serde_json::Value::Bool(true),
                            "false" | "no" | "0" => serde_json::Value::Bool(false),
                            _ => {
                                bot.send_message(chat_id, "❌ Please enter true/false, yes/no, or 1/0:")
                                    .reply_markup(back_button())
                                    .await?;
                                return Ok(());
                            }
                        }
                    }
                    _ => serde_json::Value::String(text.to_string()),
                };
                
                collected.push((param_name, value));
                
                // Check if we need more arguments
                if current_index + 1 < params.len() {
                    let next_param = &params[current_index + 1];
                    let next_param_name = param_names[current_index + 1].clone();
                    
                    set_state(
                        chat_id,
                        UserState::CollectingArguments {
                            server: server.clone(),
                            command: command.clone(),
                            params: params.clone(),
                            param_names: param_names.clone(),
                            collected,
                            current_index: current_index + 1,
                        },
                        &user_states,
                    ).await;
                    
                    bot.send_message(
                        chat_id,
                        format!(
                            "✅ Got it!\n\n📝 Now enter {} ({}):\n{}",
                            next_param_name,
                            next_param.param_type,
                            if !next_param.description.is_empty() { &next_param.description } else { "No description" }
                        ),
                    )
                        .reply_markup(back_button())
                        .await?;
                } else {
                    // All arguments collected, execute the command
                    execute_command_with_ai(&bot, chat_id, &server, &command, collected, &user_states, &mcp_servers).await?;
                }
            }
            return Ok(());
        }
        
        // Default response
        bot.send_message(chat_id, "🤖 Use /start to see the menu or click the buttons below!")
            .reply_markup(main_menu())
            .await?;
    }
    Ok(())
}

async fn show_servers(bot: &Bot, chat_id: ChatId, user_states: &UserStates) -> ResponseResult<()> {
    // Read mcp.json from project directory
    let mcp_path = PathBuf::from("mcp.json");
    
    match tokio::fs::read_to_string(&mcp_path).await {
        Ok(content) => {
            match serde_json::from_str::<McpConfig>(&content) {
                Ok(config) => {
                    let servers: Vec<String> = config.mcp_servers.keys().cloned().collect();
                    
                    if servers.is_empty() {
                        bot.send_message(chat_id, "❌ No MCP servers found in mcp.json")
                            .reply_markup(main_menu())
                            .await?;
                    } else {
                        set_state(chat_id, UserState::SelectingServer, user_states).await;
                        let server_list = servers.iter()
                            .map(|s| format!("• {}", s))
                            .collect::<Vec<_>>()
                            .join("\n");
                        bot.send_message(
                            chat_id,
                            format!("🔌 Available MCP Servers ({})\n\n{}\n\nSelect a server to connect:", servers.len(), server_list),
                        )
                            .reply_markup(server_menu(servers))
                            .await?;
                    }
                }
                Err(e) => {
                    bot.send_message(chat_id, format!("❌ Error parsing mcp.json: {}", e))
                        .reply_markup(main_menu())
                        .await?;
                }
            }
        }
        Err(e) => {
            bot.send_message(chat_id, format!("❌ Error reading mcp.json: {}\n\nMake sure the file exists at: {}", e, mcp_path.display()))
                .reply_markup(main_menu())
                .await?;
        }
    }
    Ok(())
}

async fn connect_to_server(
    bot: &Bot,
    chat_id: ChatId,
    server_name: &str,
    user_states: &UserStates,
    mcp_servers: &McpServers,
) -> ResponseResult<()> {
    bot.send_message(chat_id, format!("🔄 Connecting to '{}'...", server_name))
        .await?;
    
    // Check if already connected
    {
        let servers = mcp_servers.lock().await;
        if servers.contains_key(server_name) {
            // Already connected, just show commands
            if let Some(server) = servers.get(server_name) {
                let commands: Vec<String> = server.tools.iter().map(|t| t.name.clone()).collect();
                
                if commands.is_empty() {
                    bot.send_message(
                        chat_id,
                        format!("⚠️ Server '{}' has no available commands.\n\nThis might happen if:\n• The server doesn't expose any tools\n• The server failed to initialize properly\n• The tools/list method is not implemented", server_name),
                    )
                        .reply_markup(main_menu())
                        .await?;
                    return Ok(());
                }
                
                let command_list = server.tools.iter()
                    .map(|t| format!("• {} {}", t.name, 
                        t.description.as_deref()
                            .map(|d| format!("- {}", d))
                            .unwrap_or_default()
                    ))
                    .collect::<Vec<_>>()
                    .join("\n");
                
                set_state(chat_id, UserState::SelectingCommand(server_name.to_string()), user_states).await;
                bot.send_message(
                    chat_id,
                    format!("⚡ Available commands for '{}' ({}):\n\n{}\n\nSelect a command to execute:", server_name, commands.len(), command_list),
                )
                    .reply_markup(command_menu(commands))
                    .await?;
                return Ok(());
            }
        }
    }
    
    // Read config and spawn server
    let mcp_path = PathBuf::from("mcp.json");
    let config_content = match tokio::fs::read_to_string(&mcp_path).await {
        Ok(content) => content,
        Err(e) => {
            bot.send_message(chat_id, format!("❌ Failed to read mcp.json: {}", e))
                .reply_markup(main_menu())
                .await?;
            return Ok(());
        }
    };
    
    let config: McpConfig = match serde_json::from_str(&config_content) {
        Ok(config) => config,
        Err(e) => {
            bot.send_message(chat_id, format!("❌ Failed to parse mcp.json: {}", e))
                .reply_markup(main_menu())
                .await?;
            return Ok(());
        }
    };
    
    if let Some(server_config) = config.mcp_servers.get(server_name) {
        match spawn_mcp_server(server_config).await {
            Ok(mut handle) => {
                // Initialize and list tools
                if let Err(e) = initialize_mcp(&mut handle).await {
                    bot.send_message(chat_id, format!("❌ Failed to initialize server: {}", e))
                        .reply_markup(main_menu())
                        .await?;
                    return Ok(());
                }
                
                if let Err(e) = list_tools(&mut handle).await {
                    bot.send_message(chat_id, format!("❌ Failed to list tools: {}", e))
                        .reply_markup(main_menu())
                        .await?;
                    return Ok(());
                }
                
                let commands: Vec<String> = handle.tools.iter().map(|t| t.name.clone()).collect();
                
                if commands.is_empty() {
                    bot.send_message(
                        chat_id,
                        format!("⚠️ Server '{}' has no available commands.\n\nThis might happen if:\n• The server doesn't expose any tools\n• The server failed to initialize properly\n• The tools/list method is not implemented", server_name),
                    )
                        .reply_markup(main_menu())
                        .await?;
                    return Ok(());
                }
                
                let command_list = handle.tools.iter()
                    .map(|t| format!("• {} {}", t.name, 
                        t.description.as_deref()
                            .map(|d| format!("- {}", d))
                            .unwrap_or_default()
                    ))
                    .collect::<Vec<_>>()
                    .join("\n");
                
                // Store server handle
                mcp_servers.lock().await.insert(server_name.to_string(), handle);
                
                set_state(chat_id, UserState::SelectingCommand(server_name.to_string()), user_states).await;
                bot.send_message(
                    chat_id,
                    format!("✅ Connected to '{}'!\n\n⚡ Available commands ({}):\n\n{}\n\nSelect a command to execute:", server_name, commands.len(), command_list),
                )
                    .reply_markup(command_menu(commands))
                    .await?;
            }
            Err(e) => {
                bot.send_message(chat_id, format!("❌ Failed to connect to server: {}", e))
                    .reply_markup(main_menu())
                    .await?;
            }
        }
    } else {
        bot.send_message(chat_id, format!("❌ Server '{}' not found in mcp.json", server_name))
            .reply_markup(main_menu())
            .await?;
    }
    
    Ok(())
}

async fn spawn_mcp_server(config: &ServerConfig) -> Result<McpServerHandle> {
    let mut cmd = ProcessCommand::new(&config.command);
    cmd.args(&config.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    
    for (key, value) in &config.env {
        cmd.env(key, value);
    }
    
    let mut process = cmd.spawn()
        .map_err(|e| anyhow!("Failed to spawn process: {}", e))?;
    
    let stdin = process.stdin.take()
        .ok_or_else(|| anyhow!("Failed to get stdin"))?;
    
    let stdout = process.stdout.take()
        .ok_or_else(|| anyhow!("Failed to get stdout"))?;
    
    let reader = Arc::new(Mutex::new(BufReader::new(stdout)));
    
    Ok(McpServerHandle {
        _process: process,
        stdin,
        reader,
        tools: vec![],
    })
}

async fn initialize_mcp(handle: &mut McpServerHandle) -> Result<()> {
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::json!(1),
        method: "initialize".to_string(),
        params: Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "telegram-mcp-bot",
                "version": "0.1.0"
            }
        })),
    };
    
    send_request(handle, &request).await?;
    let response = read_response_with_id(handle, Some(serde_json::json!(1))).await?;
    
    if response.error.is_some() {
        return Err(anyhow!("Initialize error: {:?}", response.error));
    }
    
    // Send initialized notification
    let notification = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::Value::Null,
        method: "notifications/initialized".to_string(),
        params: None,
    };
    
    send_request(handle, &notification).await?;
    
    Ok(())
}

async fn list_tools(handle: &mut McpServerHandle) -> Result<Vec<Tool>> {
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::json!(2),
        method: "tools/list".to_string(),
        params: None,
    };
    
    send_request(handle, &request).await?;
    let response = read_response_with_id(handle, Some(serde_json::json!(2))).await?;
    
    if let Some(error) = response.error {
        return Err(anyhow!("tools/list error: {:?}", error));
    }
    
    if let Some(result) = response.result {
        if let Some(tools_value) = result.get("tools") {
            let tools: Vec<Tool> = serde_json::from_value(tools_value.clone())?;
            handle.tools = tools.clone();
            return Ok(tools);
        }
    }
    
    Err(anyhow!("Invalid tools/list response"))
}

async fn send_request(handle: &mut McpServerHandle, request: &JsonRpcRequest) -> Result<()> {
    let json = serde_json::to_string(request)?;
    debug!("Sending request: {}", json);
    
    handle.stdin.write_all(json.as_bytes()).await?;
    handle.stdin.write_all(b"\n").await?;
    handle.stdin.flush().await?;
    
    Ok(())
}

async fn read_response_with_id(handle: &mut McpServerHandle, expected_id: Option<serde_json::Value>) -> Result<JsonRpcResponse> {
    let reader = Arc::clone(&handle.reader);
    let max_attempts = 10;
    let mut attempts = 0;
    
    loop {
        attempts += 1;
        if attempts > max_attempts {
            return Err(anyhow!("Failed to find response with expected id after {} attempts", max_attempts));
        }
        
        let mut reader_lock = reader.lock().await;
        let mut line = String::new();
        
        match reader_lock.read_line(&mut line).await {
            Ok(0) => return Err(anyhow!("Server closed connection")),
            Ok(_) => {
                debug!("Received line: {}", line.trim());
                
                if let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&line) {
                    if let Some(expected) = &expected_id {
                        if response.id == *expected {
                            return Ok(response);
                        } else {
                            debug!("Skipping response with id {:?}, expecting {:?}", response.id, expected);
                            continue;
                        }
                    } else {
                        return Ok(response);
                    }
                }
            }
            Err(e) => return Err(anyhow!("Failed to read response: {}", e)),
        }
    }
}

async fn start_command_collection(
    bot: &Bot,
    chat_id: ChatId,
    server_name: &str,
    command_name: &str,
    user_states: &UserStates,
    mcp_servers: &McpServers,
) -> ResponseResult<()> {
    let servers = mcp_servers.lock().await;
    if let Some(server_handle) = servers.get(server_name) {
        if let Some(tool) = server_handle.tools.iter().find(|t| t.name == command_name) {
            // Check if command has parameters
            if let Some(input_schema) = &tool.input_schema {
                if !input_schema.properties.is_empty() {
                    // Collect parameters with their names to maintain order
                    let params_with_names: Vec<(String, ToolParam)> = input_schema.properties
                        .iter()
                        .map(|(name, param)| (name.clone(), param.clone()))
                        .collect();
                    
                    if let Some((first_param_name, first_param)) = params_with_names.first() {
                        
                        set_state(
                            chat_id,
                            UserState::CollectingArguments {
                                server: server_name.to_string(),
                                command: command_name.to_string(),
                                params: params_with_names.iter().map(|(_, p)| p.clone()).collect(),
                                param_names: params_with_names.iter().map(|(n, _)| n.clone()).collect(),
                                collected: vec![],
                                current_index: 0,
                            },
                            user_states,
                        ).await;
                        
                        bot.send_message(
                            chat_id,
                            format!(
                                "📝 Command: {}\n{}\n\n🔧 Enter {} ({}):\n{}",
                                command_name,
                                tool.description.as_deref().unwrap_or("No description"),
                                first_param_name,
                                first_param.param_type,
                                if !first_param.description.is_empty() { &first_param.description } else { "No description" }
                            ),
                        )
                            .reply_markup(back_button())
                            .await?;
                    }
                } else {
                    // No parameters, execute immediately
                    execute_command_with_ai(bot, chat_id, server_name, command_name, vec![], user_states, mcp_servers).await?;
                }
            } else {
                // No input schema, execute immediately
                execute_command_with_ai(bot, chat_id, server_name, command_name, vec![], user_states, mcp_servers).await?;
            }
        }
    }
    Ok(())
}

async fn execute_command_with_ai(
    bot: &Bot,
    chat_id: ChatId,
    server_name: &str,
    command_name: &str,
    args: Vec<(String, serde_json::Value)>,
    user_states: &UserStates,
    mcp_servers: &McpServers,
) -> ResponseResult<()> {
    // Send typing indicator
    bot.send_chat_action(chat_id, ChatAction::Typing).await?;
    
    // Prepare the request
    let params_obj: serde_json::Map<String, serde_json::Value> = args.into_iter().collect();
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::json!(command_name),
        method: "tools/call".to_string(),
        params: Some(serde_json::json!({
            "name": command_name,
            "arguments": params_obj
        })),
    };
    
    // Format input JSON for display
    let input_json = serde_json::to_string_pretty(&request)
        .unwrap_or_else(|_| "Error formatting input".to_string());
    
    bot.send_message(chat_id, format!("📤 *Input JSON:*\n```json\n{}\n```", input_json))
        .parse_mode(ParseMode::MarkdownV2)
        .await?;
    
    // Execute the command
    let mut servers = mcp_servers.lock().await;
    if let Some(server_handle) = servers.get_mut(server_name) {
        // Get commands list for the current server
        let commands: Vec<String> = server_handle.tools.iter().map(|t| t.name.clone()).collect();
        
        match send_request(server_handle, &request).await {
            Ok(_) => {
                match read_response_with_id(server_handle, Some(serde_json::json!(command_name))).await {
                    Ok(response) => {
                        // Format output JSON for display
                        let output_json = serde_json::to_string_pretty(&response)
                            .unwrap_or_else(|_| "Error formatting output".to_string());
                        
                        bot.send_message(chat_id, format!("📥 *Output JSON:*\n```json\n{}\n```", output_json))
                            .parse_mode(ParseMode::MarkdownV2)
                            .await?;
                        
                        // Get command description for AI context
                        let command_description = server_handle.tools.iter()
                            .find(|t| t.name == command_name)
                            .and_then(|t| t.description.as_ref())
                            .unwrap_or(&"No description available".to_string())
                            .clone();
                        
                        // Prepare context for AI
                        let ai_prompt = format!(
                            "An MCP command was just run. Here's the data:\n\
                            - Command: {command_name}\n\
                            - Description: {description}\n\
                            - Input: {input_json}\n\
                            - Output: {output_json}\n\n\
                            Your task is to provide a very brief, conversational, one-paragraph summary of what happened. \
                            Be friendly and concise. Do not use markdown lists or formal headings.",
                            command_name = command_name,
                            description = command_description,
                            input_json = input_json,
                            output_json = output_json
                        );
                        
                        // Send typing indicator for AI analysis
                        bot.send_chat_action(chat_id, ChatAction::Typing).await?;
                        
                        // Call AI for analysis
                        match call_openai(&ai_prompt).await {
                            Ok(ai_response) => {
                                let escaped_response = escape_markdown_v2(&ai_response);
                                bot.send_message(chat_id, format!("🤖 *AI Analysis:*\n\n{}", escaped_response))
                                    .parse_mode(ParseMode::MarkdownV2)
                                    .reply_markup(command_menu(commands.clone()))
                                    .await?;
                            }
                            Err(e) => {
                                bot.send_message(chat_id, format!("⚠️ AI analysis failed: {}", e))
                                    .reply_markup(command_menu(commands.clone()))
                                    .await?;
                            }
                        }
                        
                        // Show result summary
                        if let Some(error) = response.error {
                            bot.send_message(chat_id, format!("❌ Command failed with error: {:?}", error))
                                .reply_markup(command_menu(commands.clone()))
                                .await?;
                        } else if let Some(result) = response.result {
                            // Check if result has content field (common pattern)
                            if let Some(content) = result.get("content") {
                                if let Some(content_array) = content.as_array() {
                                    if let Some(first_item) = content_array.first() {
                                        if let Some(text) = first_item.get("text").and_then(|t| t.as_str()) {
                                            let escaped_text = escape_markdown_v2(text);
                                            bot.send_message(chat_id, format!("✅ *Command Result:*\n{}", escaped_text))
                                                .parse_mode(ParseMode::MarkdownV2)
                                                .reply_markup(command_menu(commands.clone()))
                                                .await?;
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Set state back to selecting command for this server
                        set_state(chat_id, UserState::SelectingCommand(server_name.to_string()), user_states).await;
                    }
                    Err(e) => {
                        bot.send_message(chat_id, format!("❌ Failed to read response: {}", e))
                            .reply_markup(command_menu(commands.clone()))
                            .await?;
                    }
                }
            }
            Err(e) => {
                bot.send_message(chat_id, format!("❌ Failed to send request: {}", e))
                    .reply_markup(command_menu(commands))
                    .await?;
            }
        }
    } else {
        bot.send_message(chat_id, "❌ Server connection lost!")
            .reply_markup(main_menu())
            .await?;
    }
    
    Ok(())
}

async fn call_openai(prompt: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
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
            content: prompt.to_string(),
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
        .unwrap_or_else(|| "No response from AI".to_string()))
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
    states.remove(&chat_id);
}

use teloxide::prelude::*;
use teloxide::types::{KeyboardButton, KeyboardMarkup};
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
use log::{info, error, debug, warn};
use std::path::PathBuf;

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
    properties: HashMap<String, ToolParam>,
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

#[tokio::main]
async fn main() {
    dotenv().ok();
    env_logger::init();
    
    info!("Starting MCP Telegram bot...");
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

async fn handle_commands(bot: Bot, msg: Message, cmd: Command, user_states: UserStates, _mcp_servers: McpServers) -> ResponseResult<()> {
    let chat_id = msg.chat.id;
    
    match cmd {
        Command::Start => {
            reset_state(chat_id, &user_states).await;
            bot.send_message(chat_id, "🤖 Welcome to MCP Bot!\n\nI can help you interact with Model Context Protocol servers.")
                .reply_markup(main_menu())
                .await?;
        }
        Command::Help => {
            reset_state(chat_id, &user_states).await;
            bot.send_message(
                chat_id, 
                "📋 MCP Bot Help\n\n\
                This bot allows you to interact with MCP servers defined in your mcp.json file.\n\n\
                How to use:\n\
                1. Click '🔌 MCP Servers' to see available servers\n\
                2. Select a server to connect and see its tools\n\
                3. Choose a tool to execute\n\
                4. Provide required arguments when prompted\n\n\
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
                    "📋 MCP Bot Help\n\n\
                    This bot allows you to interact with MCP servers defined in your mcp.json file.\n\n\
                    How to use:\n\
                    1. Click '🔌 MCP Servers' to see available servers\n\
                    2. Select a server to connect and see its tools\n\
                    3. Choose a tool to execute\n\
                    4. Provide required arguments when prompted\n\n\
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
        if let UserState::CollectingArguments { server, command, params, mut collected, current_index } = state {
            if current_index < params.len() {
                let param = &params[current_index];
                let param_name = {
                    // Find param name from the original tool schema
                    let servers = mcp_servers.lock().await;
                    servers.get(&server)
                        .and_then(|s| s.tools.iter().find(|t| t.name == command))
                        .and_then(|t| t.input_schema.as_ref())
                        .and_then(|schema| {
                            schema.properties.keys()
                                .nth(current_index)
                                .cloned()
                        })
                        .unwrap_or_else(|| format!("param{}", current_index))
                };
                
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
                    let next_param_name = {
                        let servers = mcp_servers.lock().await;
                        servers.get(&server)
                            .and_then(|s| s.tools.iter().find(|t| t.name == command))
                            .and_then(|t| t.input_schema.as_ref())
                            .and_then(|schema| {
                                schema.properties.keys()
                                    .nth(current_index + 1)
                                    .cloned()
                            })
                            .unwrap_or_else(|| format!("param{}", current_index + 1))
                    };
                    
                    set_state(
                        chat_id,
                        UserState::CollectingArguments {
                            server: server.clone(),
                            command: command.clone(),
                            params: params.clone(),
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
                    execute_command(&bot, chat_id, &server, &command, collected, &user_states, &mcp_servers).await?;
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
    
    // Read mcp.json to get server config
    let mcp_path = PathBuf::from("mcp.json");
    
    let config = match tokio::fs::read_to_string(&mcp_path).await {
        Ok(content) => match serde_json::from_str::<McpConfig>(&content) {
            Ok(cfg) => cfg,
            Err(e) => {
                bot.send_message(chat_id, format!("❌ Error parsing mcp.json: {}", e))
                    .reply_markup(main_menu())
                    .await?;
                return Ok(());
            }
        },
        Err(e) => {
            bot.send_message(chat_id, format!("❌ Error reading mcp.json: {}", e))
                .reply_markup(main_menu())
                .await?;
            return Ok(());
        }
    };
    
    let server_config = match config.mcp_servers.get(server_name) {
        Some(cfg) => cfg,
        None => {
            bot.send_message(chat_id, format!("❌ Server '{}' not found in mcp.json", server_name))
                .reply_markup(main_menu())
                .await?;
            return Ok(());
        }
    };
    
    // Spawn the MCP server
    match spawn_mcp_server(server_config).await {
        Ok(mut handle) => {
            // Initialize the MCP connection
            match initialize_mcp(&mut handle).await {
                Ok(_) => {
                    // List available tools
                    match list_tools(&mut handle).await {
                        Ok(tools) => {
                            info!("Server '{}' returned {} tools", server_name, tools.len());
                            let commands: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
                            
                            if commands.is_empty() {
                                bot.send_message(
                                    chat_id,
                                    format!("⚠️ Connected to '{}' but it has no available commands.\n\nThis might happen if:\n• The server doesn't expose any tools\n• The tools/list method returned an empty list", server_name),
                                )
                                    .reply_markup(main_menu())
                                    .await?;
                                return Ok(());
                            }
                            
                            let command_list = tools.iter()
                                .map(|t| format!("• {} {}", t.name, 
                                    t.description.as_deref()
                                        .map(|d| format!("- {}", d))
                                        .unwrap_or_default()
                                ))
                                .collect::<Vec<_>>()
                                .join("\n");
                            
                            handle.tools = tools;
                            
                            // Store the server handle
                            {
                                let mut servers = mcp_servers.lock().await;
                                servers.insert(server_name.to_string(), handle);
                            }
                            
                            set_state(chat_id, UserState::SelectingCommand(server_name.to_string()), user_states).await;
                            bot.send_message(
                                chat_id,
                                format!("✅ Connected to '{}'!\n\n⚡ Available commands ({}):\n\n{}\n\nSelect a command to execute:", server_name, commands.len(), command_list),
                            )
                                .reply_markup(command_menu(commands))
                                .await?;
                        }
                        Err(e) => {
                            bot.send_message(chat_id, format!("❌ Failed to list tools: {}", e))
                                .reply_markup(main_menu())
                                .await?;
                        }
                    }
                }
                Err(e) => {
                    bot.send_message(chat_id, format!("❌ Failed to initialize MCP connection: {}", e))
                        .reply_markup(main_menu())
                        .await?;
                }
            }
        }
        Err(e) => {
            bot.send_message(chat_id, format!("❌ Failed to spawn MCP server: {}", e))
                .reply_markup(main_menu())
                .await?;
        }
    }
    
    Ok(())
}

async fn spawn_mcp_server(config: &ServerConfig) -> Result<McpServerHandle> {
    let mut cmd = ProcessCommand::new(&config.command);
    
    // Add arguments
    for arg in &config.args {
        cmd.arg(arg);
    }
    
    // Set environment variables
    for (key, value) in &config.env {
        cmd.env(key, value);
    }
    
    // Set up pipes
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    
    let mut process = cmd.spawn()?;
    
    let stdin = process.stdin.take().ok_or_else(|| anyhow!("Failed to get stdin"))?;
    let stdout = process.stdout.take().ok_or_else(|| anyhow!("Failed to get stdout"))?;
    let reader = Arc::new(Mutex::new(BufReader::new(stdout)));
    
    Ok(McpServerHandle {
        _process: process,
        stdin,
        reader,
        tools: vec![],
    })
}

async fn initialize_mcp(handle: &mut McpServerHandle) -> Result<()> {
    // Send initialize request
    let init_id = serde_json::json!(1);
    let init_request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: init_id.clone(),
        method: "initialize".to_string(),
        params: Some(serde_json::json!({
            "protocolVersion": "0.1.0",
            "capabilities": {
                "tools": {}
            },
            "clientInfo": {
                "name": "telegram-mcp-bot",
                "version": "0.1.0"
            }
        })),
    };
    
    info!("Sending initialize request");
    send_request(handle, &init_request).await?;
    let response = read_response_with_id(handle, Some(init_id)).await?;
    
    if let Some(error) = &response.error {
        error!("Initialize failed with error: {:?}", error);
        return Err(anyhow!("Initialize failed: {:?}", error));
    }
    
    info!("Initialize successful, sending initialized notification");
    
    // Send initialized notification (no response expected)
    let initialized = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: serde_json::Value::Null,
        method: "initialized".to_string(),
        params: None,
    };
    
    send_request(handle, &initialized).await?;
    // Don't wait for response on notifications
    
    // Give the server a moment to process the notification
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    Ok(())
}

async fn list_tools(handle: &mut McpServerHandle) -> Result<Vec<Tool>> {
    let request_id = serde_json::json!(2);
    let request = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: request_id.clone(),
        method: "tools/list".to_string(),
        params: None,
    };
    
    debug!("Sending tools/list request");
    send_request(handle, &request).await?;
    let response = read_response_with_id(handle, Some(request_id)).await?;
    
    if let Some(error) = &response.error {
        error!("tools/list returned error: {:?}", error);
        return Err(anyhow!("Server returned error: {:?}", error));
    }
    
    if let Some(result) = response.result {
        debug!("tools/list response: {:?}", result);
        if let Some(tools_value) = result.get("tools") {
            let tools: Vec<Tool> = serde_json::from_value(tools_value.clone())?;
            info!("Parsed {} tools from response", tools.len());
            return Ok(tools);
        } else {
            warn!("tools/list response missing 'tools' field");
        }
    } else {
        warn!("tools/list response missing result");
    }
    
    Ok(vec![])
}

async fn send_request(handle: &mut McpServerHandle, request: &JsonRpcRequest) -> Result<()> {
    let json = serde_json::to_string(request)?;
    debug!("Sending: {}", json);
    handle.stdin.write_all(json.as_bytes()).await?;
    handle.stdin.write_all(b"\n").await?;
    handle.stdin.flush().await?;
    Ok(())
}



async fn read_response_with_id(handle: &mut McpServerHandle, expected_id: Option<serde_json::Value>) -> Result<JsonRpcResponse> {
    let mut reader = handle.reader.lock().await;
    let mut line = String::new();
    
    loop {
        line.clear();
        reader.read_line(&mut line).await?;
        
        if line.trim().is_empty() {
            continue;
        }
        
        debug!("Received: {}", line.trim());
        
        match serde_json::from_str::<JsonRpcResponse>(&line) {
            Ok(response) => {
                // Check if this is an error response to a notification (which we should ignore)
                if response.id == serde_json::Value::Null && response.error.is_some() {
                    debug!("Ignoring error response to notification: {:?}", response.error);
                    continue;
                }
                
                // If we're expecting a specific ID, check for it
                if let Some(ref id) = expected_id {
                    if response.id != *id {
                        debug!("Ignoring response with unexpected id: {} (expected: {})", response.id, id);
                        continue;
                    }
                }
                
                return Ok(response);
            }
            Err(e) => {
                // Try to parse as a notification (which we can ignore)
                if let Ok(_notification) = serde_json::from_str::<JsonRpcRequest>(&line) {
                    debug!("Ignoring incoming notification");
                    continue;
                }
                error!("Failed to parse response: {} - Line: {}", e, line);
            }
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
    if let Some(server) = servers.get(server_name) {
        if let Some(tool) = server.tools.iter().find(|t| t.name == command_name) {
            if let Some(schema) = &tool.input_schema {
                if !schema.properties.is_empty() {
                    // Collect parameter names and types in order
                    let mut params: Vec<(String, ToolParam)> = vec![];
                    
                    // First add required parameters
                    for req in &schema.required {
                        if let Some(param) = schema.properties.get(req) {
                            params.push((req.clone(), param.clone()));
                        }
                    }
                    
                    // Then add optional parameters
                    for (name, param) in &schema.properties {
                        if !schema.required.contains(name) {
                            params.push((name.clone(), param.clone()));
                        }
                    }
                    
                    if !params.is_empty() {
                        let first_param = &params[0];
                        
                        set_state(
                            chat_id,
                            UserState::CollectingArguments {
                                server: server_name.to_string(),
                                command: command_name.to_string(),
                                params: params.iter().map(|(_, p)| p.clone()).collect(),
                                collected: vec![],
                                current_index: 0,
                            },
                            user_states,
                        ).await;
                        
                        bot.send_message(
                            chat_id,
                            format!(
                                "⚡ Executing: {}\n{}\n\n📝 Enter {} ({}):\n{}",
                                command_name,
                                tool.description.as_deref().unwrap_or("No description"),
                                first_param.0,
                                first_param.1.param_type,
                                if !first_param.1.description.is_empty() { &first_param.1.description } else { "No description" }
                            ),
                        )
                            .reply_markup(back_button())
                            .await?;
                        
                        return Ok(());
                    }
                }
            }
            
            // No parameters needed, execute directly
            execute_command(bot, chat_id, server_name, command_name, vec![], user_states, mcp_servers).await?;
        }
    }
    
    Ok(())
}

async fn execute_command(
    bot: &Bot,
    chat_id: ChatId,
    server_name: &str,
    command_name: &str,
    args: Vec<(String, serde_json::Value)>,
    user_states: &UserStates,
    mcp_servers: &McpServers,
) -> ResponseResult<()> {
    bot.send_message(chat_id, "🔄 Executing command...")
        .await?;
    
    let mut servers = mcp_servers.lock().await;
    if let Some(handle) = servers.get_mut(server_name) {
        // Build arguments object
        let mut arguments = serde_json::Map::new();
        for (name, value) in args {
            arguments.insert(name, value);
        }
        
        let request_id = serde_json::json!(3);
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: request_id.clone(),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": command_name,
                "arguments": arguments
            })),
        };
        
        match send_request(handle, &request).await {
            Ok(_) => {
                match read_response_with_id(handle, Some(request_id)).await {
                    Ok(response) => {
                        reset_state(chat_id, user_states).await;
                        
                        // Prepare input JSON for display
                        let input_json = serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": 3,
                            "method": "tools/call",
                            "params": {
                                "name": command_name,
                                "arguments": arguments
                            }
                        });
                        
                        let input_pretty = serde_json::to_string_pretty(&input_json)
                            .unwrap_or_else(|_| "Failed to format input".to_string());
                        
                        // Prepare output JSON for display
                        let output_json = serde_json::json!({
                            "jsonrpc": response.jsonrpc,
                            "id": response.id,
                            "result": response.result,
                            "error": response.error
                        });
                        
                        let output_pretty = serde_json::to_string_pretty(&output_json)
                            .unwrap_or_else(|_| "Failed to format output".to_string());
                        
                        // Escape special characters for Telegram Markdown
                        let escape_markdown = |text: &str| -> String {
                            text.replace('_', "\\_")
                                .replace('*', "\\*")
                                .replace('[', "\\[")
                                .replace(']', "\\]")
                                .replace('(', "\\(")
                                .replace(')', "\\)")
                                .replace('~', "\\~")
                                .replace('`', "\\`")
                                .replace('>', "\\>")
                                .replace('#', "\\#")
                                .replace('+', "\\+")
                                .replace('-', "\\-")
                                .replace('=', "\\=")
                                .replace('|', "\\|")
                                .replace('{', "\\{")
                                .replace('}', "\\}")
                                .replace('.', "\\.")
                                .replace('!', "\\!")
                        };
                        
                        let status = if response.error.is_some() { "❌ Error" } else { "✅ Success" };
                        
                        let message = format!(
                            "{} Command executed\\!\n\n📥 __Input \\(JSON\\-RPC Request\\):__\n```json\n{}\n```\n\n📤 __Output \\(JSON\\-RPC Response\\):__\n```json\n{}\n```",
                            status,
                            escape_markdown(&input_pretty),
                            escape_markdown(&output_pretty)
                        );
                        
                        bot.send_message(chat_id, message)
                            .parse_mode(teloxide::types::ParseMode::MarkdownV2)
                            .reply_markup(main_menu())
                            .await?;
                    }
                    Err(e) => {
                        reset_state(chat_id, user_states).await;
                        bot.send_message(chat_id, format!("❌ Error reading response: {}", e))
                            .reply_markup(main_menu())
                            .await?;
                    }
                }
            }
            Err(e) => {
                reset_state(chat_id, user_states).await;
                bot.send_message(chat_id, format!("❌ Error sending request: {}", e))
                    .reply_markup(main_menu())
                    .await?;
            }
        }
    } else {
        reset_state(chat_id, user_states).await;
        bot.send_message(chat_id, "❌ Server not connected")
            .reply_markup(main_menu())
            .await?;
    }
    
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

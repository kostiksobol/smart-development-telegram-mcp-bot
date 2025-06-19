use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use async_openai::types::{
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessage, ChatCompletionRequestToolMessage,
    ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent,
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType,
    FunctionObject, Role,
};
use async_openai::{config::OpenAIConfig, Client};
use futures::future::join_all;
use log::{error, info};
use mcp::McpServerHandle;
use reqwest::Proxy;
use teloxide::prelude::*;
use teloxide::types::{ChatAction, KeyboardButton, KeyboardMarkup, KeyboardRemove};
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use uuid::Uuid;
use lazy_static::lazy_static;
use regex::Regex;
use teloxide::utils::command::BotCommands;

mod mcp {
    use super::*;
    use anyhow::{anyhow, Result};
    use log::{debug, error, warn};
    use serde::{Deserialize, Serialize};
    use std::process::Stdio;
    use std::sync::Arc;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::{Child, Command as ProcessCommand};
    use tokio::sync::Mutex;

    #[derive(Debug)]
    pub struct McpServerHandle {
        _process: Child,
        stdin: tokio::process::ChildStdin,
        reader: Arc<Mutex<BufReader<tokio::process::ChildStdout>>>,
        pub tools: Vec<Tool>,
        pub name: String,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct McpConfig {
        #[serde(rename = "mcpServers")]
        pub mcp_servers: HashMap<String, ServerConfig>,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct ServerConfig {
        pub command: String,
        pub args: Vec<String>,
        #[serde(default)]
        pub env: HashMap<String, String>,
    }

    #[derive(Debug, Clone, Deserialize, Serialize)]
    pub struct Tool {
        pub name: String,
        #[serde(default)]
        pub description: Option<String>,
        #[serde(rename = "inputSchema", alias = "input_schema", default)]
        pub input_schema: Option<serde_json::Value>,
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

    pub async fn spawn_mcp_server(name: &str, config: &ServerConfig) -> Result<McpServerHandle> {
        let mut cmd = ProcessCommand::new(&config.command);
        cmd.args(&config.args);
        cmd.envs(&config.env);
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
            name: name.to_string(),
        })
    }

    async fn send_request(handle: &mut McpServerHandle, request: &JsonRpcRequest) -> Result<()> {
        let json = serde_json::to_string(request)?;
        debug!("(to srv {}) > {}", handle.name, json);
        handle.stdin.write_all(json.as_bytes()).await?;
        handle.stdin.write_all(b"\n").await?;
        handle.stdin.flush().await?;
        Ok(())
    }

    async fn read_response(handle: &mut McpServerHandle) -> Result<JsonRpcResponse> {
        let mut reader = handle.reader.lock().await;
        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).await?;
            if bytes_read == 0 {
                return Err(anyhow!("Server {} exited unexpectedly", handle.name));
            }

            if line.trim().is_empty() {
                continue;
            }

            debug!("(from srv {}) < {}", handle.name, line.trim());
            match serde_json::from_str::<JsonRpcResponse>(&line) {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if let Ok(request) = serde_json::from_str::<JsonRpcRequest>(&line) {
                         warn!("Got a request from server, ignoring: {:?}", request);
                         continue;
                    }
                    error!("Failed to parse response from {}: {}. Line: {}", handle.name, e, line);
                    return Err(e.into());
                }
            }
        }
    }

    pub async fn initialize_mcp(handle: &mut McpServerHandle) -> Result<()> {
        let init_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: serde_json::json!(Uuid::new_v4().to_string()),
            method: "initialize".to_string(),
            params: Some(serde_json::json!({
                "protocolVersion": "0.1.0",
                "capabilities": { "tools": {} },
                "clientInfo": { "name": "telegram-prompt-mcp-bot", "version": "0.1.0" }
            })),
        };
        send_request(handle, &init_request).await?;
        let response = read_response(handle).await?;
        if let Some(error) = response.error {
            return Err(anyhow!("MCP initialize failed: {:?}", error));
        }

        // Skip the "initialized" notification as some servers don't support it
        Ok(())
    }

    pub async fn list_tools(handle: &mut McpServerHandle) -> Result<Vec<Tool>> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: serde_json::json!(Uuid::new_v4().to_string()),
            method: "tools/list".to_string(),
            params: None,
        };
        send_request(handle, &request).await?;
        let response = read_response(handle).await?;
        if let Some(error) = response.error {
            return Err(anyhow!("tools/list failed: {:?}", error));
        }
        let result = response.result.ok_or_else(|| anyhow!("tools/list response missing result"))?;
        let tools_value = result.get("tools").ok_or_else(|| anyhow!("tools/list result missing 'tools' field"))?;
        let tools: Vec<Tool> = serde_json::from_value(tools_value.clone())?;
        Ok(tools)
    }
    
    pub async fn call_tool(handle: &mut McpServerHandle, name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: serde_json::json!(Uuid::new_v4().to_string()),
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": name,
                "arguments": args
            })),
        };
        send_request(handle, &request).await?;
        let response = read_response(handle).await?;

        if let Some(error) = response.error {
            return Err(anyhow!("Tool call '{}' failed: {:?}", name, error));
        }
        
        response.result.ok_or_else(|| anyhow!("Tool call response for '{}' missing result", name))
    }
}

// Represents the entire application state
struct AppState {
    openai_client: Client<OpenAIConfig>,
    global_mcp_servers: Arc<Mutex<HashMap<String, McpServerHandle>>>,
    user_sessions: Arc<Mutex<HashMap<ChatId, UserSession>>>,
    pending_tool_calls: Arc<Mutex<HashMap<ChatId, PendingToolExecution>>>,
}

// Holds the state for a single user's session
#[derive(Debug, Clone, Default)]
struct UserSession {
    // server_name -> set of enabled command_names
    active_servers: HashMap<String, HashSet<String>>,
}

#[derive(Debug, Clone)]
struct PendingToolExecution {
    messages: Vec<ChatCompletionRequestMessage>,
    tool_calls: Vec<async_openai::types::ChatCompletionMessageToolCall>,
}

lazy_static! {
    static ref BOLD_REGEX: Regex = Regex::new(r"\*\*(.*?)\*\*").unwrap();
    static ref ITALIC_REGEX: Regex = Regex::new(r"\*(.*?)\*").unwrap();
    static ref H3_REGEX: Regex = Regex::new(r"### (.*)").unwrap();
    static ref LIST_ITEM_REGEX: Regex = Regex::new(r"^\s*[\d-]+\.\s*").unwrap();
}

fn format_for_telegram(text: &str) -> String {
    let mut result = text.to_string();
    // Convert ### to bold
    result = H3_REGEX.replace_all(&result, "<b>$1</b>").to_string();
    // Convert **bold** to <b>bold</b>
    result = BOLD_REGEX.replace_all(&result, "<b>$1</b>").to_string();
    // Convert *italic* to <i>italic</i> - must be careful not to match bold
    result = ITALIC_REGEX.replace_all(&result, "<i>$1</i>").to_string();
    // Convert numbered/dashed list items to bullet points
    result = LIST_ITEM_REGEX.replace_all(&result, "• ").to_string();
    result
}

// Bot Commands Definition
#[derive(BotCommands, Clone)]
#[command(rename_rule = "lowercase", description = "These commands manage your available tools:")]
enum Command {
    #[command(description = "Show this help message.")]
    Help,
    #[command(description = "List all servers you can add.")]
    Available,
    #[command(description = "Add a server to your active list. e.g., /add shit")]
    Add(String),
    #[command(description = "Remove a server from your list. e.g., /remove shit")]
    Remove(String),
    #[command(description = "List your active servers and their tools.")]
    MyServers,
    #[command(description = "Enable a tool for a server. e.g., /enable shit greet_user")]
    Enable(String),
    #[command(description = "Disable a tool for a server. e.g., /disable shit add_numbers")]
    Disable(String),
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    env_logger::init();

    info!("Starting bot...");

    let global_mcp_servers = Arc::new(Mutex::new(load_mcp_servers().await?));

    // Create OpenAI client with proxy support
    let openai_client = if let Ok(proxy_url) = std::env::var("OPENAI_PROXY") {
        info!("Using proxy: {}", proxy_url);
        let http_client = reqwest::Client::builder()
            .proxy(Proxy::all(&proxy_url)?)
            .timeout(Duration::from_secs(30))
            .build()?;
        
        let config = OpenAIConfig::new().with_api_key(std::env::var("OPENAI_API_KEY")?);
        Client::with_config(config).with_http_client(http_client)
    } else {
        info!("No proxy configured, using direct connection");
        Client::new()
    };

    let app_state = Arc::new(AppState {
        openai_client,
        global_mcp_servers,
        user_sessions: Arc::new(Mutex::new(HashMap::new())),
        pending_tool_calls: Arc::new(Mutex::new(HashMap::new())),
    });
    
    let bot = Bot::from_env();

    let handler = dptree::entry()
        .branch(Update::filter_message().filter_command::<Command>().endpoint(command_handler))
        .branch(Update::filter_message().endpoint(text_handler));

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![app_state])
        .enable_ctrlc_handler()
        .build()
        .dispatch()
        .await;

    Ok(())
}

async fn load_mcp_servers() -> Result<HashMap<String, McpServerHandle>> {
    let mcp_path = PathBuf::from("mcp.json");
    info!("Loading MCP servers from {}", mcp_path.display());
    let content = tokio::fs::read_to_string(&mcp_path)
        .await
        .with_context(|| format!("Failed to read mcp.json from {}", mcp_path.display()))?;
    let config: mcp::McpConfig = serde_json::from_str(&content)?;

    let mut handles = HashMap::new();
    let mut futures: Vec<JoinHandle<Result<(String, McpServerHandle)>>> = vec![];

    for (name, server_config) in config.mcp_servers {
        info!("Spawning server: {}", name);
        let future = async move {
            let mut handle = mcp::spawn_mcp_server(&name, &server_config).await?;
            mcp::initialize_mcp(&mut handle).await?;
            let tools = mcp::list_tools(&mut handle).await?;
            handle.tools = tools;
            info!("Server '{}' connected with {} tools.", name, handle.tools.len());
            Ok((name, handle))
        };
        futures.push(tokio::spawn(future));
    }
    
    let results = join_all(futures).await;

    for result in results {
        match result {
            Ok(Ok((name, handle))) => {
                handles.insert(name, handle);
            }
            Ok(Err(e)) => error!("Failed to initialize a server: {}", e),
            Err(e) => error!("A server task panicked: {}", e),
        }
    }

    info!("Finished loading {} MCP servers.", handles.len());
    Ok(handles)
}

// Creates the tool list for the AI based ONLY on the user's active servers/commands
fn create_openai_tools(
    global_servers: &HashMap<String, McpServerHandle>,
    user_session: &UserSession,
) -> Vec<ChatCompletionTool> {
    let mut tools = Vec::new();
    
    // Always include system tools (they are always active)
    tools.extend(get_system_tools());
    
    for (server_name, enabled_commands) in &user_session.active_servers {
        if server_name == "system" {
            continue; // Already handled above
        }
        if let Some(handle) = global_servers.get(server_name) {
            for tool in &handle.tools {
                if enabled_commands.contains(&tool.name) {
                    let function = FunctionObject {
                        name: format!("{}_{}", server_name, tool.name),
                        description: tool.description.clone(),
                        parameters: tool.input_schema.clone(),
                    };
                    tools.push(ChatCompletionTool {
                        r#type: ChatCompletionToolType::Function,
                        function,
                    });
                }
            }
        }
    }
    tools
}

// THIS IS THE MISSING FUNCTION
fn get_system_tools() -> Vec<ChatCompletionTool> {
    vec![
        // --- Global Informational Tools ---
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_list_available_servers".to_string(),
                description: Some("List all servers available for adding. ONLY use this when user asks 'what servers are available' or 'list servers'. Do NOT use this when user wants to add a specific server.".to_string()),
                parameters: Some(serde_json::json!({"type": "object", "properties": {}})),
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_list_available_servers_with_tools".to_string(),
                description: Some("List all available servers and all of their respective tools.".to_string()),
                parameters: Some(serde_json::json!({"type": "object", "properties": {}})),
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_list_tools_for_available_server".to_string(),
                description: Some("List all tools for a specific available server.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "server_name": { "type": "string", "description": "The name of the server to inspect." }},
                    "required": ["server_name"]
                })),
            },
        },

        // --- User's Personal Server Management Tools ---
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_add_server_to_my_list".to_string(),
                description: Some("IMMEDIATELY add a server to the user's personal list when they say 'add [server]'. Use this directly without checking availability first. Examples: 'add shit', 'add server shit', 'add rust server'.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "server_name": { "type": "string", "description": "The exact name of the server to add (e.g., 'shit', 'rust-mcp-from-scratch')" }},
                    "required": ["server_name"]
                })),
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_remove_server_from_my_list".to_string(),
                description: Some("Remove a server from your personal list (cannot remove 'system').".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "server_name": { "type": "string", "description": "The name of the server to remove from your list." }},
                    "required": ["server_name"]
                })),
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_list_my_servers".to_string(),
                description: Some("List only the servers on your personal active list.".to_string()),
                parameters: Some(serde_json::json!({"type": "object", "properties": {}})),
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_list_my_servers_with_active_tools".to_string(),
                description: Some("List your active servers and their currently active tools.".to_string()),
                parameters: Some(serde_json::json!({"type": "object", "properties": {}})),
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_list_active_tools_for_my_server".to_string(),
                description: Some("List the active tools for a specific server on your personal list.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": { "server_name": { "type": "string", "description": "The name of the server on your list to inspect." }},
                    "required": ["server_name"]
                })),
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_activate_tool_for_my_server".to_string(),
                description: Some("Activate a tool for one of your personal servers.".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object", "properties": {
                        "server_name": { "type": "string", "description": "The name of the server on your list." },
                        "command_name": { "type": "string", "description": "The name of the tool to activate." }
                    }, "required": ["server_name", "command_name"]
                })),
            },
        },
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: "system_deactivate_tool_for_my_server".to_string(),
                description: Some("Deactivate a tool for one of your personal servers (cannot deactivate system tools).".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object", "properties": {
                        "server_name": { "type": "string", "description": "The name of the server on your list." },
                        "command_name": { "type": "string", "description": "The name of the tool to deactivate." }
                    }, "required": ["server_name", "command_name"]
                })),
            },
        },
    ]
}

// Handles built-in /commands
async fn command_handler(bot: Bot, msg: Message, command: Command, state: Arc<AppState>) -> Result<()> {
    let chat_id = msg.chat.id;
    let mut sessions = state.user_sessions.lock().await;
    let session = sessions.entry(chat_id).or_default();
    let global_servers = state.global_mcp_servers.lock().await;

    let response_text = match command {
        Command::Help => Command::descriptions().to_string(),
        Command::Available => {
            let mut server_names: Vec<String> = global_servers.keys().cloned().collect();
            server_names.push("system".to_string());
            server_names.sort();
            format!("Here are all the servers you can add:\n- {}", server_names.join("\n- "))
        }
        Command::Add(server_name) => {
            if server_name == "system" {
                format!("The 'system' server is always available and cannot be added manually. 😉")
            } else if !global_servers.contains_key(&server_name) {
                format!("Sorry, '{}' is not a valid server name. 🤷‍♂️", server_name)
            } else if session.active_servers.contains_key(&server_name) {
                format!("You've already added the '{}' server. 😉", server_name)
            } else {
                let all_commands = global_servers[&server_name].tools.iter().map(|t| t.name.clone()).collect();
                session.active_servers.insert(server_name.clone(), all_commands);
                format!("Awesome! Added '{}' to your list. All its tools are now active. ✅", server_name)
            }
        }
        Command::Remove(server_name) => {
            if server_name == "system" {
                format!("The 'system' server is permanent and cannot be removed. 🔒")
            } else if session.active_servers.remove(&server_name).is_some() {
                format!("Okay, removed '{}' from your list. 👍", server_name)
            } else {
                format!("'{}' isn't on your list, so I can't remove it.", server_name)
            }
        }
        Command::MyServers => {
            let mut response = "Here are your active servers and their tools:\n\n".to_string();
            
            // Always show system server first
            response += "<b>system</b>:\n";
            for tool in get_system_tools() {
                response += &format!("  ✅ {}\n", tool.function.name);
            }
            response += "\n";
            
            // Show user's added servers
            for (name, enabled_cmds) in &session.active_servers {
                if name == "system" {
                    continue; // Already handled above
                }
                response += &format!("<b>{}</b>:\n", name);
                if let Some(handle) = global_servers.get(name) {
                    for tool in &handle.tools {
                        let status = if enabled_cmds.contains(&tool.name) { "✅" } else { "❌" };
                        response += &format!("  {} {}\n", status, tool.name);
                    }
                }
                response += "\n";
            }
            response
        }
        Command::Enable(ref args) | Command::Disable(ref args) => {
            let parts: Vec<&str> = args.split_whitespace().collect();
            if parts.len() != 2 {
                return bot.send_message(chat_id, "Please provide the server and command name, like `/enable server_name command_name`.").await.map(|_| ()).map_err(Into::into);
            }
            let server_name = parts[0];
            let command_name = parts[1];
            
            if server_name == "system" {
                format!("System tools are always active and cannot be changed. 🔒")
            } else if !session.active_servers.contains_key(server_name) {
                format!("'{}' isn't on your list, so you can't change its tools.", server_name)
            } else if !global_servers[server_name].tools.iter().any(|t| t.name == command_name) {
                format!("The command '{}' doesn't exist on the '{}' server.", command_name, server_name)
            } else {
                let enabled_commands = session.active_servers.get_mut(server_name).unwrap();
                let is_enable = matches!(command, Command::Enable(_));
                if is_enable {
                    if enabled_commands.insert(command_name.to_string()) {
                        format!("✅ Enabled '{}' for server '{}'.", command_name, server_name)
                    } else {
                        format!("'{}' was already enabled for '{}'. 😉", command_name, server_name)
                    }
                } else { // Disable
                    if enabled_commands.remove(command_name) {
                        format!("❌ Disabled '{}' for server '{}'.", command_name, server_name)
                    } else {
                        format!("'{}' was already disabled for '{}'.", command_name, server_name)
                    }
                }
            }
        }
    };

    bot.send_message(chat_id, response_text).parse_mode(teloxide::types::ParseMode::Html).await?;
    Ok(())
}

// Handles general text messages sent to the AI
async fn text_handler(bot: Bot, msg: Message, state: Arc<AppState>) -> Result<()> {
    let text = if let Some(text) = msg.text() {
        text
    } else {
        return Ok(());
    };
    
    let chat_id = msg.chat.id;

    // Handle confirmation button presses
    if text == "✅ Yes, execute tools" || text == "❌ No, cancel" {
        return handle_tool_execution(bot, chat_id, state, text == "✅ Yes, execute tools").await;
    }

    bot.send_chat_action(chat_id, ChatAction::Typing).await?;

    // Generate tools dynamically based on the user's session
    let tools = {
        let mut user_sessions = state.user_sessions.lock().await;
        let user_session = user_sessions.entry(chat_id).or_default();
        let global_servers = state.global_mcp_servers.lock().await;
        create_openai_tools(&global_servers, user_session)
    };
    
    // Tools should never be empty since system tools are always available

    let mut messages: Vec<ChatCompletionRequestMessage> = vec![
        ChatCompletionRequestSystemMessage {
            content: "You are a friendly Telegram assistant with MCP server management tools. 

IMPORTANT: When users say 'add [server_name]' or 'add server [name]' or similar, ALWAYS use system_add_server_to_my_list immediately with the server name. Do NOT list servers first.

Tool usage:
- ADD/ENABLE a server (e.g. 'add shit', 'add server shit'): use system_add_server_to_my_list with server_name
- REMOVE/DISABLE a server: use system_remove_server_from_my_list  
- LIST available servers: use system_list_available_servers
- LIST their active servers: use system_list_my_servers
- ACTIVATE/ENABLE a tool: use system_activate_tool_for_my_server
- DEACTIVATE/DISABLE a tool: use system_deactivate_tool_for_my_server

Be casual and use emojis.".to_string(),
            role: Role::System,
            name: None,
        }.into(),
        ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Text(text.to_string()),
            role: Role::User,
            name: None,
        }.into()
    ];

    loop {
        let request = async_openai::types::CreateChatCompletionRequestArgs::default()
            .model("gpt-4o")
            .messages(messages.clone())
            .tools(tools.clone())
            .tool_choice(ChatCompletionToolChoiceOption::Auto)
            .build()?;

        info!("Sending request to OpenAI...");
        let response = state.openai_client.chat().create(request).await?;
        let choice = response.choices.get(0).context("No choice from OpenAI")?;
        let response_message = &choice.message;

        messages.push(
            ChatCompletionRequestAssistantMessage {
                content: response_message.content.clone(),
                role: Role::Assistant,
                name: None,
                tool_calls: response_message.tool_calls.clone(),
                function_call: None,
            }
            .into(),
        );

        let tool_calls = if let Some(tc) = &response_message.tool_calls {
            tc.clone()
        } else {
            // No tool calls, we have a final answer
            if let Some(content) = &response_message.content {
                let formatted_content = format_for_telegram(content);
                bot.send_message(chat_id, formatted_content)
                    .reply_markup(KeyboardRemove::new())
                    .parse_mode(teloxide::types::ParseMode::Html)
                    .await?;
            } else {
                bot.send_message(chat_id, "I received a response with no content.")
                    .reply_markup(KeyboardRemove::new())
                    .await?;
            }
            break; // Exit loop
        };

        // Ask user for confirmation before executing tools
        let tool_descriptions: Vec<String> = tool_calls.iter().map(|tc| {
            let (server_name, tool_name) = if tc.function.name.starts_with("system_") {
                ("system", &tc.function.name[7..]) // Remove "system_" prefix
            } else {
                let parts: Vec<&str> = tc.function.name.splitn(2, '_').collect();
                (*parts.get(0).unwrap_or(&"unknown"), *parts.get(1).unwrap_or(&"unknown"))
            };
            
            // Parse and format the arguments nicely
            let args_display = match serde_json::from_str::<serde_json::Value>(&tc.function.arguments) {
                Ok(args) => {
                    match serde_json::to_string_pretty(&args) {
                        Ok(pretty_json) => format!("\n<pre><code>{}</code></pre>", pretty_json),
                        Err(_) => format!("\n<code>{}</code>", tc.function.arguments)
                    }
                },
                Err(_) => format!("\n<code>{}</code>", tc.function.arguments)
            };
            
            format!("🔧 <b>{}</b> from <i>{}</i> server{}", tool_name, server_name, args_display)
        }).collect();

        let confirmation_text = format!(
            "I'd like to execute the following tools to help you:\n\n{}\n\nWould you like me to proceed?",
            tool_descriptions.join("\n\n")
        );

        let keyboard = KeyboardMarkup::new(vec![
            vec![
                KeyboardButton::new("✅ Yes, execute tools"),
                KeyboardButton::new("❌ No, cancel"),
            ]
        ])
        .resize_keyboard(true)
        .one_time_keyboard(true);

        // Store the pending tool calls
        {
            let mut pending = state.pending_tool_calls.lock().await;
            pending.insert(chat_id, PendingToolExecution {
                messages: messages.clone(),
                tool_calls: tool_calls.clone(),
            });
        }

        bot.send_message(chat_id, confirmation_text)
            .reply_markup(keyboard)
            .parse_mode(teloxide::types::ParseMode::Html)
            .await?;

        break; // Wait for user response
    }

    Ok(())
}

async fn handle_tool_execution(bot: Bot, chat_id: ChatId, state: Arc<AppState>, execute: bool) -> Result<()> {
    let pending = {
        let mut pending_map = state.pending_tool_calls.lock().await;
        pending_map.remove(&chat_id)
    };

    if let Some(pending_execution) = pending {
        if execute {
            bot.send_message(chat_id, "🔄 Executing tools...")
                .reply_markup(KeyboardRemove::new())
                .await?;
            bot.send_chat_action(chat_id, ChatAction::Typing).await?;

            let mut messages = pending_execution.messages;

            let mut tool_futures = Vec::new();
            for tool_call in pending_execution.tool_calls {
                let function = tool_call.function.clone();
                info!(
                    "Model wants to call tool: {}({})",
                    function.name, function.arguments
                );

                let state_clone = state.clone();
                let chat_id_clone = chat_id;
                let future = async move {
                    let args: serde_json::Value = match serde_json::from_str(&function.arguments) {
                        Ok(args) => args,
                        Err(e) => return (tool_call.id, Err(e.into())),
                    };

                    // Handle system tools
                    if function.name.starts_with("system_") {
                        let result = match function.name.as_str() {
                            "system_list_available_servers" => {
                                execute_list_available_servers(&state_clone.global_mcp_servers).await
                            },
                            "system_list_available_servers_with_tools" => {
                                execute_list_available_servers_with_tools(&state_clone.global_mcp_servers).await
                            },
                            "system_list_tools_for_available_server" => {
                                execute_list_tools_for_available_server(&state_clone.global_mcp_servers, args).await
                            },
                            "system_add_server_to_my_list" => {
                                execute_add_server_to_my_list(chat_id_clone, &state_clone.user_sessions, &state_clone.global_mcp_servers, args).await
                            },
                            "system_remove_server_from_my_list" => {
                                execute_remove_server_from_my_list(chat_id_clone, &state_clone.user_sessions, args).await
                            },
                            "system_list_my_servers" => {
                                execute_list_my_servers(chat_id_clone, &state_clone.user_sessions).await
                            },
                            "system_list_my_servers_with_active_tools" => {
                                execute_list_my_servers_with_active_tools(chat_id_clone, &state_clone.user_sessions, &state_clone.global_mcp_servers).await
                            },
                            "system_list_active_tools_for_my_server" => {
                                execute_list_active_tools_for_my_server(chat_id_clone, &state_clone.user_sessions, args).await
                            },
                            "system_activate_tool_for_my_server" => {
                                execute_activate_tool_for_my_server(chat_id_clone, &state_clone.user_sessions, &state_clone.global_mcp_servers, args).await
                            },
                            "system_deactivate_tool_for_my_server" => {
                                execute_deactivate_tool_for_my_server(chat_id_clone, &state_clone.user_sessions, args).await
                            },
                            _ => Err(anyhow!("Unknown system tool: {}", function.name))
                        };
                        return (tool_call.id, result);
                    }

                    // Handle MCP tools
                    let parts: Vec<&str> = function.name.splitn(2, '_').collect();
                    if parts.len() != 2 {
                        return (
                            tool_call.id,
                            Err(anyhow!("Invalid tool name format for '{}'", function.name)),
                        );
                    }
                    let server_name = parts[0];
                    let tool_name = parts[1];

                    let mut servers = state_clone.global_mcp_servers.lock().await;
                    if let Some(handle) = servers.get_mut(server_name) {
                        let result = mcp::call_tool(handle, tool_name, args).await;
                        (tool_call.id, result)
                    } else {
                        (
                            tool_call.id,
                            Err(anyhow!("Server '{}' not found", server_name)),
                        )
                    }
                };
                tool_futures.push(tokio::spawn(future));
            }

            let tool_results = join_all(tool_futures).await;

            for result in tool_results {
                let (tool_call_id, tool_execution_result) = result?;

                let content = match tool_execution_result {
                    Ok(value) => serde_json::to_string(&value)?,
                    Err(e) => {
                        error!("Tool execution failed: {}", e);
                        format!("Error: {}", e)
                    }
                };

                messages.push(
                    ChatCompletionRequestToolMessage {
                        tool_call_id,
                        content,
                        role: Role::Tool,
                    }
                    .into(),
                );
            }

            // Get final response from OpenAI
            let request = async_openai::types::CreateChatCompletionRequestArgs::default()
                .model("gpt-4o")
                .messages(messages)
                .build()?;

            let response = state.openai_client.chat().create(request).await?;
            let choice = response.choices.get(0).context("No choice from OpenAI")?;

            if let Some(content) = &choice.message.content {
                let formatted_content = format_for_telegram(content);
                bot.send_message(chat_id, formatted_content)
                    .parse_mode(teloxide::types::ParseMode::Html)
                    .await?;
            } else {
                bot.send_message(chat_id, "✅ Tools executed successfully!").await?;
            }
        } else {
            bot.send_message(chat_id, "❌ Tool execution cancelled. How else can I help you?")
                .reply_markup(KeyboardRemove::new())
                .await?;
        }
    } else {
        bot.send_message(chat_id, "❌ No pending tool calls found. Please try again.")
            .reply_markup(KeyboardRemove::new())
            .await?;
    }

    Ok(())
}

// --- System Tool Execution Functions ---

async fn execute_list_available_servers(
    global_mcp_servers: &Arc<Mutex<HashMap<String, McpServerHandle>>>,
) -> Result<serde_json::Value> {
    let servers = global_mcp_servers.lock().await;
    let mut server_list: Vec<_> = servers.keys().map(|name| serde_json::json!({ "name": name })).collect();
    server_list.push(serde_json::json!({"name": "system"}));
    
    Ok(serde_json::json!({
        "available_servers": server_list,
        "message": format!("Found {} servers you can add! 🚀", server_list.len()),
    }))
}

async fn execute_list_available_servers_with_tools(
    global_mcp_servers: &Arc<Mutex<HashMap<String, McpServerHandle>>>,
) -> Result<serde_json::Value> {
    let servers = global_mcp_servers.lock().await;
    let mut server_details = vec![];

    for (name, handle) in servers.iter() {
        let tools: Vec<_> = handle.tools.iter().map(|t| serde_json::json!({"name": t.name, "description": t.description})).collect();
        server_details.push(serde_json::json!({ "server_name": name, "tools": tools }));
    }
    
    let system_tools: Vec<_> = get_system_tools().into_iter().map(|t| serde_json::json!({"name": t.function.name, "description": t.function.description})).collect();
    server_details.push(serde_json::json!({ "server_name": "system", "tools": system_tools }));

    Ok(serde_json::json!({ "servers": server_details }))
}

async fn execute_list_tools_for_available_server(
    global_mcp_servers: &Arc<Mutex<HashMap<String, McpServerHandle>>>,
    args: serde_json::Value,
) -> Result<serde_json::Value> {
    let server_name = args["server_name"].as_str().context("Missing server_name")?;
    
    if server_name == "system" {
        let tools: Vec<_> = get_system_tools().into_iter().map(|t| serde_json::json!({"name": t.function.name, "description": t.function.description})).collect();
        return Ok(serde_json::json!({ "server_name": "system", "tools": tools }));
    }

    let servers = global_mcp_servers.lock().await;
    if let Some(handle) = servers.get(server_name) {
        let tools: Vec<_> = handle.tools.iter().map(|t| serde_json::json!({"name": t.name, "description": t.description})).collect();
        Ok(serde_json::json!({ "server_name": server_name, "tools": tools }))
    } else {
        Err(anyhow!("Server '{}' not found 🤷‍♂️", server_name))
    }
}

async fn execute_add_server_to_my_list(
    chat_id: ChatId,
    user_sessions: &Arc<Mutex<HashMap<ChatId, UserSession>>>,
    global_mcp_servers: &Arc<Mutex<HashMap<String, McpServerHandle>>>,
    args: serde_json::Value,
) -> Result<serde_json::Value> {
    let server_name = args["server_name"].as_str().context("Missing server_name")?;
    if server_name == "system" {
        return Ok(serde_json::json!({"message": "The 'system' server is always on your list. 😉"}));
    }

    let global_servers = global_mcp_servers.lock().await;
    if !global_servers.contains_key(server_name) {
        return Err(anyhow!("Server '{}' isn't available to be added.", server_name));
    }

    let mut sessions = user_sessions.lock().await;
    let session = sessions.entry(chat_id).or_default();
    
    if session.active_servers.contains_key(server_name) {
        return Ok(serde_json::json!({ "message": format!("You've already added '{}'.", server_name)}));
    }

    let all_commands = global_servers[server_name].tools.iter().map(|t| t.name.clone()).collect();
    session.active_servers.insert(server_name.to_string(), all_commands);

    Ok(serde_json::json!({ "message": format!("Awesome! Added '{}' to your list. All its tools are active. ✅", server_name)}))
}

async fn execute_remove_server_from_my_list(
    chat_id: ChatId,
    user_sessions: &Arc<Mutex<HashMap<ChatId, UserSession>>>,
    args: serde_json::Value,
) -> Result<serde_json::Value> {
    let server_name = args["server_name"].as_str().context("Missing server_name")?;
    if server_name == "system" {
        return Err(anyhow!("The 'system' server is permanent and cannot be removed."));
    }
    
    let mut sessions = user_sessions.lock().await;
    if let Some(session) = sessions.get_mut(&chat_id) {
        if session.active_servers.remove(server_name).is_some() {
            return Ok(serde_json::json!({ "message": format!("Okay, removed '{}' from your list. 👍", server_name)}));
        }
    }
    Err(anyhow!("Server '{}' isn't on your list.", server_name))
}

async fn execute_list_my_servers(
    chat_id: ChatId,
    user_sessions: &Arc<Mutex<HashMap<ChatId, UserSession>>>,
) -> Result<serde_json::Value> {
    let sessions = user_sessions.lock().await;
    let mut server_names = vec!["system".to_string()];
    
    if let Some(session) = sessions.get(&chat_id) {
        for name in session.active_servers.keys() {
            if name != "system" {
                server_names.push(name.clone());
            }
        }
    }
    
    Ok(serde_json::json!({ "my_servers": server_names }))
}

async fn execute_list_my_servers_with_active_tools(
    chat_id: ChatId,
    user_sessions: &Arc<Mutex<HashMap<ChatId, UserSession>>>,
    global_mcp_servers: &Arc<Mutex<HashMap<String, McpServerHandle>>>,
) -> Result<serde_json::Value> {
    let sessions = user_sessions.lock().await;
    let global_servers = global_mcp_servers.lock().await;
    let mut server_details = vec![];

    // Always include system server with all its tools
    let system_tools: Vec<_> = get_system_tools().iter().map(|t| serde_json::json!({"name": &t.function.name})).collect();
    server_details.push(serde_json::json!({ "server_name": "system", "active_tools": system_tools }));

    if let Some(session) = sessions.get(&chat_id) {
        for (name, enabled_cmds) in &session.active_servers {
            if name == "system" {
                continue; // Already handled above
            }
            let tools: Vec<_> = global_servers.get(name).unwrap().tools.iter()
                .filter(|t| enabled_cmds.contains(&t.name))
                .map(|t| serde_json::json!({"name": &t.name}))
                .collect();
            server_details.push(serde_json::json!({ "server_name": name, "active_tools": tools }));
        }
    }
    
    Ok(serde_json::json!({ "my_servers": server_details }))
}

async fn execute_list_active_tools_for_my_server(
    chat_id: ChatId,
    user_sessions: &Arc<Mutex<HashMap<ChatId, UserSession>>>,
    args: serde_json::Value,
) -> Result<serde_json::Value> {
    let server_name = args["server_name"].as_str().context("Missing server_name")?;
    
    if server_name == "system" {
        let system_tools: Vec<_> = get_system_tools().iter().map(|t| t.function.name.clone()).collect();
        return Ok(serde_json::json!({ "server_name": "system", "active_tools": system_tools }));
    }
    
    let sessions = user_sessions.lock().await;
    if let Some(session) = sessions.get(&chat_id) {
        if let Some(enabled_cmds) = session.active_servers.get(server_name) {
            return Ok(serde_json::json!({ "server_name": server_name, "active_tools": enabled_cmds }));
        }
    }
    Err(anyhow!("Server '{}' isn't on your list.", server_name))
}

async fn execute_activate_tool_for_my_server(
    chat_id: ChatId,
    user_sessions: &Arc<Mutex<HashMap<ChatId, UserSession>>>,
    global_mcp_servers: &Arc<Mutex<HashMap<String, McpServerHandle>>>,
    args: serde_json::Value,
) -> Result<serde_json::Value> {
    let server_name = args["server_name"].as_str().context("Missing server_name")?;
    let command_name = args["command_name"].as_str().context("Missing command_name")?;

    if server_name == "system" {
        return Err(anyhow!("System commands are always active and cannot be changed."));
    }

    let mut sessions = user_sessions.lock().await;
    let session = sessions.entry(chat_id).or_default();

    if !session.active_servers.contains_key(server_name) {
        return Err(anyhow!("Server '{}' isn't on your list.", server_name));
    }
    
    let global_servers = global_mcp_servers.lock().await;
    if !global_servers[server_name].tools.iter().any(|t| t.name == command_name) {
         return Err(anyhow!("Command '{}' doesn't exist on the '{}' server.", command_name, server_name));
    }

    let enabled_commands = session.active_servers.get_mut(server_name).unwrap();
    if enabled_commands.insert(command_name.to_string()) {
        Ok(serde_json::json!({"message": format!("✅ Activated '{}' for server '{}'.", command_name, server_name)}))
    } else {
        Ok(serde_json::json!({"message": format!("'{}' was already active for '{}'.", command_name, server_name)}))
    }
}

async fn execute_deactivate_tool_for_my_server(
    chat_id: ChatId,
    user_sessions: &Arc<Mutex<HashMap<ChatId, UserSession>>>,
    args: serde_json::Value,
) -> Result<serde_json::Value> {
    let server_name = args["server_name"].as_str().context("Missing server_name")?;
    let command_name = args["command_name"].as_str().context("Missing command_name")?;
    
    if server_name == "system" {
        return Err(anyhow!("System commands cannot be deactivated."));
    }

    let mut sessions = user_sessions.lock().await;
    let session = sessions.entry(chat_id).or_default();
    
    if !session.active_servers.contains_key(server_name) {
        return Err(anyhow!("Server '{}' isn't on your list.", server_name));
    }
    
    let enabled_commands = session.active_servers.get_mut(server_name).unwrap();
    if enabled_commands.remove(command_name) {
        Ok(serde_json::json!({"message": format!("❌ Deactivated '{}' for server '{}'.", command_name, server_name)}))
    } else {
        Ok(serde_json::json!({"message": format!("'{}' was already inactive for '{}'.", command_name, server_name)}))
    }
}

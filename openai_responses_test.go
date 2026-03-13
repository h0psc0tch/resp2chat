package resp2chat_test

import (
	"encoding/json"
	"testing"

	. "resp2chat"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnmarshalOpenAIResponsesRequest_MinimalRequest(t *testing.T) {
	data := `{"model": "gpt-4o", "input": "Hello"}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", req.Model)
	assert.Equal(t, "Hello", req.Input.Text)
	assert.Empty(t, req.Input.Items)
}

func TestUnmarshalOpenAIResponsesRequest_StringInput(t *testing.T) {
	data := `{"model": "gpt-4o", "input": "Tell me a story"}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	assert.Equal(t, "Tell me a story", req.Input.Text)
	assert.Empty(t, req.Input.Items)
}

func TestUnmarshalOpenAIResponsesRequest_ArrayInputSingleMessage(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "user", "content": "Hello"}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.Len(t, req.Input.Items, 1)
	item := req.Input.Items[0]
	assert.Equal(t, "message", item.Type)
	assert.Equal(t, "user", item.Role)
	require.NotNil(t, item.Content)
	assert.Equal(t, "Hello", item.Content.Text)
}

func TestUnmarshalOpenAIResponsesRequest_ArrayInputMultipleRoles(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "system", "content": "You are a helpful assistant"},
			{"type": "message", "role": "user", "content": "Hello"},
			{"type": "message", "role": "assistant", "content": "Hi there!"},
			{"type": "message", "role": "developer", "content": "Internal note"}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.Len(t, req.Input.Items, 4)
	assert.Equal(t, "system", req.Input.Items[0].Role)
	assert.Equal(t, "user", req.Input.Items[1].Role)
	assert.Equal(t, "assistant", req.Input.Items[2].Role)
	assert.Equal(t, "developer", req.Input.Items[3].Role)
}

func TestUnmarshalOpenAIResponsesRequest_ContentPartsText(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_text", "text": "What is this?"}
				]
			}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	item := req.Input.Items[0]
	require.NotNil(t, item.Content)
	require.Len(t, item.Content.Parts, 1)
	assert.Equal(t, "input_text", item.Content.Parts[0].Type)
	assert.Equal(t, "What is this?", item.Content.Parts[0].Text)
}

func TestUnmarshalOpenAIResponsesRequest_ContentPartsImage(t *testing.T) {
	imageURL := "https://example.com/image.jpg"
	detail := "high"
	data := `{
		"model": "gpt-4o",
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_text", "text": "What is in this image?"},
					{"type": "input_image", "image_url": "https://example.com/image.jpg", "detail": "high"}
				]
			}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	parts := req.Input.Items[0].Content.Parts
	require.Len(t, parts, 2)
	assert.Equal(t, "input_text", parts[0].Type)
	assert.Equal(t, "input_image", parts[1].Type)
	assert.Equal(t, &imageURL, parts[1].ImageURL)
	assert.Equal(t, &detail, parts[1].Detail)
}

func TestUnmarshalOpenAIResponsesRequest_ContentPartsImageFileID(t *testing.T) {
	fileID := "file-abc123"
	data := `{
		"model": "gpt-4o",
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_image", "file_id": "file-abc123", "detail": "auto"}
				]
			}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	part := req.Input.Items[0].Content.Parts[0]
	assert.Equal(t, "input_image", part.Type)
	assert.Equal(t, &fileID, part.FileID)
}

func TestUnmarshalOpenAIResponsesRequest_ContentPartsFile(t *testing.T) {
	fileURL := "https://example.com/doc.pdf"
	data := `{
		"model": "gpt-4o",
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_text", "text": "Summarise this file"},
					{"type": "input_file", "file_url": "https://example.com/doc.pdf"}
				]
			}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	parts := req.Input.Items[0].Content.Parts
	assert.Equal(t, "input_file", parts[1].Type)
	assert.Equal(t, &fileURL, parts[1].FileURL)
}

func TestUnmarshalOpenAIResponsesRequest_ItemReference(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": [
			{"type": "item_reference", "id": "msg_abc123"},
			{"type": "message", "role": "user", "content": "Follow up"}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.Len(t, req.Input.Items, 2)
	assert.Equal(t, "item_reference", req.Input.Items[0].Type)
	assert.Equal(t, "msg_abc123", req.Input.Items[0].ID)
}

func TestUnmarshalOpenAIResponsesRequest_FunctionTool(t *testing.T) {
	strict := true
	data := `{
		"model": "gpt-4o",
		"input": "Hello",
		"tools": [
			{
				"type": "function",
				"name": "get_weather",
				"description": "Get current weather",
				"parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
				"strict": true
			}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.Len(t, req.Tools, 1)
	tool := req.Tools[0]
	assert.Equal(t, "function", tool.Type)
	require.NotNil(t, tool.Function)
	assert.Equal(t, "get_weather", tool.Function.Name)
	assert.NotNil(t, tool.Function.Description)
	assert.Equal(t, "Get current weather", *tool.Function.Description)
	assert.Equal(t, &strict, tool.Function.Strict)
	assert.NotEmpty(t, tool.Function.Parameters)
}

func TestUnmarshalOpenAIResponsesRequest_FileSearchTool(t *testing.T) {
	maxResults := 10
	data := `{
		"model": "gpt-4o",
		"input": "Search docs",
		"tools": [
			{
				"type": "file_search",
				"vector_store_ids": ["vs_abc123", "vs_def456"],
				"max_num_results": 10
			}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	tool := req.Tools[0]
	assert.Equal(t, "file_search", tool.Type)
	require.NotNil(t, tool.FileSearch)
	assert.Equal(t, []string{"vs_abc123", "vs_def456"}, tool.FileSearch.VectorStoreIDs)
	assert.Equal(t, &maxResults, tool.FileSearch.MaxNumResults)
}

func TestUnmarshalOpenAIResponsesRequest_WebSearchTool(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Search the web",
		"tools": [{"type": "web_search", "search_context_size": "medium"}]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	tool := req.Tools[0]
	assert.Equal(t, "web_search", tool.Type)
	require.NotNil(t, tool.WebSearch)
	assert.Equal(t, "medium", *tool.WebSearch.SearchContextSize)
}

func TestUnmarshalOpenAIResponsesRequest_WebSearchPreviewTool(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Search",
		"tools": [{"type": "web_search_preview"}]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	assert.Equal(t, "web_search_preview", req.Tools[0].Type)
	assert.NotNil(t, req.Tools[0].WebSearch)
}

func TestUnmarshalOpenAIResponsesRequest_CodeInterpreterTool(t *testing.T) {
	data := `{"model": "gpt-4o", "input": "Run code", "tools": [{"type": "code_interpreter"}]}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	tool := req.Tools[0]
	assert.Equal(t, "code_interpreter", tool.Type)
	require.NotNil(t, tool.CodeInterpreter)
}

func TestUnmarshalOpenAIResponsesRequest_MCPTool(t *testing.T) {
	serverURL := "https://mcp.example.com"
	data := `{
		"model": "gpt-4o",
		"input": "Use MCP",
		"tools": [
			{
				"type": "mcp",
				"server_label": "my-server",
				"server_url": "https://mcp.example.com",
				"allowed_tools": ["tool_a", "tool_b"],
				"headers": {"Authorization": "Bearer token"}
			}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	tool := req.Tools[0]
	assert.Equal(t, "mcp", tool.Type)
	require.NotNil(t, tool.MCP)
	assert.Equal(t, "my-server", tool.MCP.ServerLabel)
	assert.Equal(t, &serverURL, tool.MCP.ServerURL)
	assert.Equal(t, "Bearer token", tool.MCP.Headers["Authorization"])
}

func TestUnmarshalOpenAIResponsesRequest_UnknownToolPreservesRaw(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Test",
		"tools": [{"type": "future_tool", "some_field": "value"}]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	tool := req.Tools[0]
	assert.Equal(t, "future_tool", tool.Type)
	assert.NotEmpty(t, tool.Raw)
}

func TestUnmarshalOpenAIResponsesRequest_ToolChoiceStringModes(t *testing.T) {
	for _, mode := range []string{"none", "auto", "required"} {
		t.Run(mode, func(t *testing.T) {
			data := `{"model": "gpt-4o", "input": "Hi", "tool_choice": "` + mode + `"}`
			req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
			require.NoError(t, err)
			require.NotNil(t, req.ToolChoice)
			assert.Equal(t, mode, req.ToolChoice.Mode)
		})
	}
}

func TestUnmarshalOpenAIResponsesRequest_ToolChoiceFunction(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Hi",
		"tool_choice": {"type": "function", "name": "get_weather"}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.ToolChoice)
	require.NotNil(t, req.ToolChoice.Fn)
	assert.Equal(t, "function", req.ToolChoice.Fn.Type)
	assert.Equal(t, "get_weather", req.ToolChoice.Fn.Name)
}

func TestUnmarshalOpenAIResponsesRequest_ToolChoiceHosted(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Search",
		"tool_choice": {"type": "file_search"}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.ToolChoice)
	require.NotNil(t, req.ToolChoice.Hosted)
	assert.Equal(t, "file_search", req.ToolChoice.Hosted.Type)
}

func TestUnmarshalOpenAIResponsesRequest_ToolChoiceMCP(t *testing.T) {
	toolName := "my_tool"
	data := `{
		"model": "gpt-4o",
		"input": "Use MCP",
		"tool_choice": {"type": "mcp", "server_label": "my-server", "name": "my_tool"}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.ToolChoice.MCP)
	assert.Equal(t, "my-server", req.ToolChoice.MCP.ServerLabel)
	assert.Equal(t, &toolName, req.ToolChoice.MCP.Name)
}

func TestUnmarshalOpenAIResponsesRequest_ToolChoiceCustom(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Hi",
		"tool_choice": {"type": "custom", "name": "my_custom_tool"}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.ToolChoice.Custom)
	assert.Equal(t, "my_custom_tool", req.ToolChoice.Custom.Name)
}

func TestUnmarshalOpenAIResponsesRequest_ToolChoiceAllowed(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Hi",
		"tool_choice": {
			"type": "allowed_tools",
			"mode": "auto",
			"tools": [{"type": "function", "name": "get_weather"}]
		}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.ToolChoice.Allowed)
	assert.Equal(t, "auto", req.ToolChoice.Allowed.Mode)
	assert.Len(t, req.ToolChoice.Allowed.Tools, 1)
}

func TestUnmarshalOpenAIResponsesRequest_ConversationAsString(t *testing.T) {
	data := `{"model": "gpt-4o", "input": "Hello", "conversation": "conv_abc123"}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.Conversation)
	assert.Equal(t, "conv_abc123", req.Conversation.ID)
}

func TestUnmarshalOpenAIResponsesRequest_ConversationAsObject(t *testing.T) {
	data := `{"model": "gpt-4o", "input": "Hello", "conversation": {"id": "conv_def456"}}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.Conversation)
	assert.Equal(t, "conv_def456", req.Conversation.ID)
}

func TestUnmarshalOpenAIResponsesRequest_Reasoning(t *testing.T) {
	effort := ReasoningEffort("high")
	summary := "detailed"
	data := `{
		"model": "o3",
		"input": "Solve this",
		"reasoning": {"effort": "high", "summary": "detailed"}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.Reasoning)
	assert.Equal(t, &effort, req.Reasoning.Effort)
	assert.Equal(t, &summary, req.Reasoning.Summary)
}

func TestUnmarshalOpenAIResponsesRequest_TextFormatText(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Hi",
		"text": {"format": {"type": "text"}}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.Text)
	require.NotNil(t, req.Text.Format)
	assert.Equal(t, "text", req.Text.Format.Type)
}

func TestUnmarshalOpenAIResponsesRequest_TextFormatJsonSchema(t *testing.T) {
	strict := true
	data := `{
		"model": "gpt-4o",
		"input": "Return structured data",
		"text": {
			"format": {
				"type": "json_schema",
				"name": "my_schema",
				"description": "A structured response",
				"schema": {"type": "object", "properties": {"name": {"type": "string"}}},
				"strict": true
			}
		}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	f := req.Text.Format
	assert.Equal(t, "json_schema", f.Type)
	assert.Equal(t, "my_schema", f.Name)
	assert.Equal(t, "A structured response", f.Description)
	assert.NotEmpty(t, f.Schema)
	assert.Equal(t, &strict, f.Strict)
}

func TestUnmarshalOpenAIResponsesRequest_TextFormatJsonObject(t *testing.T) {
	data := `{"model": "gpt-4o", "input": "Hi", "text": {"format": {"type": "json_object"}}}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	assert.Equal(t, "json_object", req.Text.Format.Type)
}

func TestUnmarshalOpenAIResponsesRequest_StreamingRequest(t *testing.T) {
	stream := true
	includeObfuscation := false
	data := `{
		"model": "gpt-4o",
		"input": "Stream this",
		"stream": true,
		"stream_options": {"include_obfuscation": false}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	assert.Equal(t, &stream, req.Stream)
	require.NotNil(t, req.StreamOptions)
	assert.Equal(t, &includeObfuscation, req.StreamOptions.IncludeObfuscation)
}

func TestUnmarshalOpenAIResponsesRequest_ContextManagement(t *testing.T) {
	threshold := 2000
	data := `{
		"model": "gpt-4o",
		"input": "Long conversation",
		"context_management": [{"type": "compaction", "compact_threshold": 2000}]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.Len(t, req.ContextManagement, 1)
	cm := req.ContextManagement[0]
	assert.Equal(t, "compaction", cm.Type)
	assert.Equal(t, &threshold, cm.CompactThreshold)
}

func TestUnmarshalOpenAIResponsesRequest_Prompt(t *testing.T) {
	version := "2"
	data := `{
		"model": "gpt-4o",
		"input": "Hi",
		"prompt": {
			"id": "prompt_abc123",
			"version": "2",
			"variables": {"name": "Alice"}
		}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.Prompt)
	assert.Equal(t, "prompt_abc123", req.Prompt.ID)
	assert.Equal(t, &version, req.Prompt.Version)
	assert.NotEmpty(t, req.Prompt.Variables)
}

func TestUnmarshalOpenAIResponsesRequest_AllOptionalScalarFields(t *testing.T) {
	temperature := 0.7
	topP := 0.9
	topLogProbs := 5
	maxOutputTokens := 1024
	maxToolCalls := 10
	parallelToolCalls := true
	store := false
	background := false
	user := "user-123"
	instructions := "Be concise"
	prevID := "resp_prev123"
	truncation := "auto"
	serviceTier := "default"
	promptCacheRetention := "24h"
	safetyID := "safety-abc"
	cacheKey := "cache-key-abc"

	data := `{
		"model": "gpt-4o",
		"input": "Hi",
		"temperature": 0.7,
		"top_p": 0.9,
		"top_logprobs": 5,
		"max_output_tokens": 1024,
		"max_tool_calls": 10,
		"parallel_tool_calls": true,
		"store": false,
		"background": false,
		"user": "user-123",
		"instructions": "Be concise",
		"previous_response_id": "resp_prev123",
		"truncation": "auto",
		"service_tier": "default",
		"prompt_cache_retention": "24h",
		"safety_identifier": "safety-abc",
		"prompt_cache_key": "cache-key-abc",
		"include": ["message.output_text.logprobs"],
		"metadata": {"env": "prod", "version": "1"}
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	assert.Equal(t, &temperature, req.Temperature)
	assert.Equal(t, &topP, req.TopP)
	assert.Equal(t, &topLogProbs, req.TopLogProbs)
	assert.Equal(t, &maxOutputTokens, req.MaxOutputTokens)
	assert.Equal(t, &maxToolCalls, req.MaxToolCalls)
	assert.Equal(t, &parallelToolCalls, req.ParallelToolCalls)
	assert.Equal(t, &store, req.Store)
	assert.Equal(t, &background, req.Background)
	assert.Equal(t, &user, req.User)
	assert.Equal(t, &instructions, req.Instructions)
	assert.Equal(t, &prevID, req.PreviousResponseID)
	assert.Equal(t, &truncation, req.Truncation)
	assert.Equal(t, &serviceTier, req.ServiceTier)
	assert.Equal(t, &promptCacheRetention, req.PromptCacheRetention)
	assert.Equal(t, &safetyID, req.SafetyIdentifier)
	assert.Equal(t, &cacheKey, req.PromptCacheKey)
	assert.Equal(t, []string{"message.output_text.logprobs"}, req.Include)
	assert.Equal(t, map[string]string{"env": "prod", "version": "1"}, req.Metadata)
}

func TestUnmarshalOpenAIResponsesRequest_NullOptionalFields(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Hi",
		"temperature": null,
		"reasoning": null,
		"conversation": null,
		"tool_choice": null
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	assert.Nil(t, req.Temperature)
	assert.Nil(t, req.Reasoning)
	assert.Nil(t, req.Conversation)
	assert.Nil(t, req.ToolChoice)
}

func TestUnmarshalOpenAIResponsesRequest_InvalidJSON(t *testing.T) {
	_, err := UnmarshalOpenAIResponsesRequest([]byte(`{invalid json`))
	assert.Error(t, err)
}

func TestUnmarshalOpenAIResponsesRequest_InvalidInputType(t *testing.T) {
	_, err := UnmarshalOpenAIResponsesRequest([]byte(`{"model": "gpt-4o", "input": 42}`))
	assert.Error(t, err)
}

func TestUnmarshalOpenAIResponsesRequest_MultipleToolTypes(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Hi",
		"tools": [
			{"type": "function", "name": "fn_one", "parameters": null, "strict": null},
			{"type": "file_search", "vector_store_ids": ["vs_1"]},
			{"type": "web_search"},
			{"type": "code_interpreter"},
			{"type": "image_generation"}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.Len(t, req.Tools, 5)
	assert.Equal(t, "function", req.Tools[0].Type)
	assert.Equal(t, "file_search", req.Tools[1].Type)
	assert.Equal(t, "web_search", req.Tools[2].Type)
	assert.Equal(t, "code_interpreter", req.Tools[3].Type)
	assert.Equal(t, "image_generation", req.Tools[4].Type)
}

func TestUnmarshalOpenAIResponsesRequest_RoundTripPreservesToolJSON(t *testing.T) {
	toolJSON := `{"type":"function","name":"get_weather","description":"Get weather","parameters":{"type":"object"},"strict":true}`
	data := `{"model":"gpt-4o","input":"Hi","tools":[` + toolJSON + `]}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	marshaled, err := json.Marshal(req.Tools[0])
	require.NoError(t, err)
	assert.JSONEq(t, toolJSON, string(marshaled))
}

func TestUnmarshalOpenAIResponsesRequest_PreviousResponseIDMultiTurn(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": "Follow up question",
		"previous_response_id": "resp_xyz789"
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	require.NotNil(t, req.PreviousResponseID)
	assert.Equal(t, "resp_xyz789", *req.PreviousResponseID)
}

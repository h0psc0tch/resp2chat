package resp2chat

import (
	"encoding/json"
	"fmt"
)

// ReasoningEffort controls the depth of reasoning for o-series and gpt-5 models.
type ReasoningEffort string

const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
)

// Reasoning configures reasoning behaviour for o-series and gpt-5 models.
type Reasoning struct {
	Effort  *ReasoningEffort `json:"effort,omitempty"`
	Summary *string          `json:"summary,omitempty"` // "auto" | "concise" | "detailed"
}

// TextFormat specifies the output format. Use type "text", "json_schema", or "json_object".
type TextFormat struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// ResponseTextConfig configures text output, including format.
type ResponseTextConfig struct {
	Format    *TextFormat `json:"format,omitempty"`
	Verbosity *string     `json:"verbosity,omitempty"`
}

// FunctionTool defines a callable function exposed to the model.
type FunctionTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description *string         `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

// FileSearchTool searches uploaded files via vector store.
type FileSearchTool struct {
	Type           string   `json:"type"`
	VectorStoreIDs []string `json:"vector_store_ids"`
	MaxNumResults  *int     `json:"max_num_results,omitempty"`
}

// WebSearchTool enables web search. Type is "web_search" or "web_search_preview" family.
type WebSearchTool struct {
	Type              string          `json:"type"`
	SearchContextSize *string         `json:"search_context_size,omitempty"` // "low" | "medium" | "high"
	Filters           json.RawMessage `json:"filters,omitempty"`
	UserLocation      json.RawMessage `json:"user_location,omitempty"`
}

// CodeInterpreterTool enables Python code execution.
type CodeInterpreterTool struct {
	Type string `json:"type"`
}

// ComputerTool enables computer-use interaction.
type ComputerTool struct {
	Type          string `json:"type"`
	DisplayWidth  int    `json:"display_width,omitempty"`
	DisplayHeight int    `json:"display_height,omitempty"`
	Environment   string `json:"environment,omitempty"`
}

// ImageGenTool enables image generation.
type ImageGenTool struct {
	Type string `json:"type"`
}

// MCPTool provides access to tools on a remote MCP server.
type MCPTool struct {
	Type              string            `json:"type"`
	ServerLabel       string            `json:"server_label"`
	ServerURL         *string           `json:"server_url,omitempty"`
	ConnectorID       *string           `json:"connector_id,omitempty"`
	Authorization     *string           `json:"authorization,omitempty"`
	ServerDescription *string           `json:"server_description,omitempty"`
	Headers           map[string]string `json:"headers,omitempty"`
	AllowedTools      json.RawMessage   `json:"allowed_tools,omitempty"`
	RequireApproval   json.RawMessage   `json:"require_approval,omitempty"`
	DeferLoading      *bool             `json:"defer_loading,omitempty"`
}

// Tool is a discriminated union of all tool types. After unmarshalling, exactly one
// typed field (Function, FileSearch, etc.) will be non-nil, and Raw holds the
// original JSON.
type Tool struct {
	Type            string               `json:"-"`
	Function        *FunctionTool        `json:"-"`
	FileSearch      *FileSearchTool      `json:"-"`
	WebSearch       *WebSearchTool       `json:"-"`
	CodeInterpreter *CodeInterpreterTool `json:"-"`
	Computer        *ComputerTool        `json:"-"`
	ImageGen        *ImageGenTool        `json:"-"`
	MCP             *MCPTool             `json:"-"`
	Raw             json.RawMessage      `json:"-"`
}

func (t *Tool) UnmarshalJSON(data []byte) error {
	t.Raw = data
	var typed struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &typed); err != nil {
		return fmt.Errorf("tool: %w", err)
	}
	t.Type = typed.Type
	switch typed.Type {
	case "function":
		t.Function = &FunctionTool{}
		return json.Unmarshal(data, t.Function)
	case "file_search":
		t.FileSearch = &FileSearchTool{}
		return json.Unmarshal(data, t.FileSearch)
	case "web_search", "web_search_2025_08_26",
		"web_search_preview", "web_search_preview_2025_03_11":
		t.WebSearch = &WebSearchTool{}
		return json.Unmarshal(data, t.WebSearch)
	case "code_interpreter":
		t.CodeInterpreter = &CodeInterpreterTool{Type: typed.Type}
	case "computer", "computer_use_preview", "computer_use":
		t.Computer = &ComputerTool{}
		return json.Unmarshal(data, t.Computer)
	case "image_generation":
		t.ImageGen = &ImageGenTool{Type: typed.Type}
	case "mcp":
		t.MCP = &MCPTool{}
		return json.Unmarshal(data, t.MCP)
	}
	return nil
}

func (t Tool) MarshalJSON() ([]byte, error) {
	if len(t.Raw) > 0 {
		return t.Raw, nil
	}
	return []byte("null"), nil
}

// ToolChoiceFunction forces the model to call a specific function.
type ToolChoiceFunction struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

// ToolChoiceHosted forces use of a specific built-in hosted tool.
type ToolChoiceHosted struct {
	Type string `json:"type"`
}

// ToolChoiceMCP forces use of a specific MCP server tool.
type ToolChoiceMCP struct {
	Type        string  `json:"type"`
	ServerLabel string  `json:"server_label"`
	Name        *string `json:"name,omitempty"`
}

// ToolChoiceCustom forces use of a named custom tool.
type ToolChoiceCustom struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

// ToolChoiceAllowed constrains the model to a set of allowed tools.
type ToolChoiceAllowed struct {
	Type  string            `json:"type"`
	Mode  string            `json:"mode"` // "auto" | "required"
	Tools []json.RawMessage `json:"tools"`
}

// ToolChoice is a union type. It is either a mode string ("none", "auto", "required")
// or a specific tool choice object. After unmarshalling, either Mode is set or exactly
// one object field (Fn, Hosted, MCP, Custom, Allowed) is non-nil.
type ToolChoice struct {
	Mode    string
	Fn      *ToolChoiceFunction
	Hosted  *ToolChoiceHosted
	MCP     *ToolChoiceMCP
	Custom  *ToolChoiceCustom
	Allowed *ToolChoiceAllowed
}

var hostedToolChoiceTypes = map[string]bool{
	"file_search": true, "web_search_preview": true,
	"web_search_preview_2025_03_11": true, "computer": true,
	"computer_use_preview": true, "computer_use": true,
	"code_interpreter": true, "image_generation": true,
}

func (tc *ToolChoice) UnmarshalJSON(data []byte) error {
	var mode string
	if err := json.Unmarshal(data, &mode); err == nil {
		tc.Mode = mode
		return nil
	}
	var typed struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &typed); err != nil {
		return fmt.Errorf("tool_choice: %w", err)
	}
	switch typed.Type {
	case "function":
		tc.Fn = &ToolChoiceFunction{}
		return json.Unmarshal(data, tc.Fn)
	case "mcp":
		tc.MCP = &ToolChoiceMCP{}
		return json.Unmarshal(data, tc.MCP)
	case "custom":
		tc.Custom = &ToolChoiceCustom{}
		return json.Unmarshal(data, tc.Custom)
	case "allowed_tools":
		tc.Allowed = &ToolChoiceAllowed{}
		return json.Unmarshal(data, tc.Allowed)
	default:
		if hostedToolChoiceTypes[typed.Type] {
			tc.Hosted = &ToolChoiceHosted{Type: typed.Type}
		}
	}
	return nil
}

func (tc ToolChoice) MarshalJSON() ([]byte, error) {
	if tc.Mode != "" {
		return json.Marshal(tc.Mode)
	}
	if tc.Fn != nil {
		return json.Marshal(tc.Fn)
	}
	if tc.Hosted != nil {
		return json.Marshal(tc.Hosted)
	}
	if tc.MCP != nil {
		return json.Marshal(tc.MCP)
	}
	if tc.Custom != nil {
		return json.Marshal(tc.Custom)
	}
	if tc.Allowed != nil {
		return json.Marshal(tc.Allowed)
	}
	return []byte("null"), nil
}

// InputContentPart is a single part within a multi-part message content array.
// Type is one of "input_text", "input_image", or "input_file".
type InputContentPart struct {
	Type     string  `json:"type"`
	Text     string  `json:"text,omitempty"`
	ImageURL *string `json:"image_url,omitempty"`
	FileID   *string `json:"file_id,omitempty"`
	Detail   *string `json:"detail,omitempty"`
	Filename *string `json:"filename,omitempty"`
	FileData *string `json:"file_data,omitempty"`
	FileURL  *string `json:"file_url,omitempty"`
}

// ResponseInputItemContent handles the content field of an input item, which can be
// either a plain string or an array of typed content parts.
type ResponseInputItemContent struct {
	Text  string
	Parts []InputContentPart
}

func (c *ResponseInputItemContent) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		c.Text = s
		return nil
	}
	var parts []InputContentPart
	if err := json.Unmarshal(data, &parts); err != nil {
		return fmt.Errorf("input content: %w", err)
	}
	c.Parts = parts
	return nil
}

func (c ResponseInputItemContent) MarshalJSON() ([]byte, error) {
	if c.Text != "" {
		return json.Marshal(c.Text)
	}
	return json.Marshal(c.Parts)
}

// ResponseInputItem represents one item in the input array. It handles the discriminated
// union of EasyInputMessage (type="message"), ItemReferenceParam (type="item_reference"),
// and other item types. Raw preserves the original JSON for non-message types.
type ResponseInputItem struct {
	Type    string                    `json:"type,omitempty"`
	Role    string                    `json:"role,omitempty"`
	Content *ResponseInputItemContent `json:"-"`
	ID      string                    `json:"id,omitempty"`
	Raw     json.RawMessage           `json:"-"`
}

func (r *ResponseInputItem) UnmarshalJSON(data []byte) error {
	r.Raw = data
	var raw struct {
		Type    string          `json:"type"`
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
		ID      string          `json:"id"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	r.Type = raw.Type
	r.Role = raw.Role
	r.ID = raw.ID
	if len(raw.Content) > 0 && string(raw.Content) != "null" {
		r.Content = &ResponseInputItemContent{}
		if err := json.Unmarshal(raw.Content, r.Content); err != nil {
			return fmt.Errorf("input item content: %w", err)
		}
	}
	return nil
}

func (r ResponseInputItem) MarshalJSON() ([]byte, error) {
	if len(r.Raw) > 0 {
		return r.Raw, nil
	}
	type alias struct {
		Type    string          `json:"type,omitempty"`
		Role    string          `json:"role,omitempty"`
		Content json.RawMessage `json:"content,omitempty"`
		ID      string          `json:"id,omitempty"`
	}
	a := alias{Type: r.Type, Role: r.Role, ID: r.ID}
	if r.Content != nil {
		var err error
		a.Content, err = json.Marshal(r.Content)
		if err != nil {
			return nil, err
		}
	}
	return json.Marshal(a)
}

// ResponseInput handles the polymorphic input field which is either a plain string
// (treated as a user text message) or an array of input items.
// After unmarshalling, either Text or Items is populated, never both.
type ResponseInput struct {
	Text  string
	Items []ResponseInputItem
}

func (r *ResponseInput) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		r.Text = s
		return nil
	}
	var items []ResponseInputItem
	if err := json.Unmarshal(data, &items); err != nil {
		return fmt.Errorf("input: %w", err)
	}
	r.Items = items
	return nil
}

func (r ResponseInput) MarshalJSON() ([]byte, error) {
	if r.Text != "" {
		return json.Marshal(r.Text)
	}
	return json.Marshal(r.Items)
}

// ConversationParam identifies a conversation, either by ID string or by an object
// carrying the ID. After unmarshalling, ID is always populated.
type ConversationParam struct {
	ID string
}

func (c *ConversationParam) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		c.ID = s
		return nil
	}
	var obj struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(data, &obj); err != nil {
		return fmt.Errorf("conversation: %w", err)
	}
	c.ID = obj.ID
	return nil
}

func (c ConversationParam) MarshalJSON() ([]byte, error) {
	return json.Marshal(c.ID)
}

// ResponsePrompt references a stored prompt template.
type ResponsePrompt struct {
	ID        string          `json:"id"`
	Version   *string         `json:"version,omitempty"`
	Variables json.RawMessage `json:"variables,omitempty"`
}

// StreamOptions configures streaming behaviour. Only relevant when Stream is true.
type StreamOptions struct {
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
}

// ContextManagementParam configures automatic context compaction.
type ContextManagementParam struct {
	Type             string `json:"type"`
	CompactThreshold *int   `json:"compact_threshold,omitempty"`
}

// OpenAIResponsesRequest is the request body for POST /v1/responses.
type OpenAIResponsesRequest struct {
	Model string        `json:"model"`
	Input ResponseInput `json:"input"`

	Instructions      *string             `json:"instructions,omitempty"`
	PreviousResponseID *string            `json:"previous_response_id,omitempty"`
	Reasoning         *Reasoning          `json:"reasoning,omitempty"`
	Background        *bool               `json:"background,omitempty"`
	MaxOutputTokens   *int                `json:"max_output_tokens,omitempty"`
	MaxToolCalls      *int                `json:"max_tool_calls,omitempty"`
	Text              *ResponseTextConfig `json:"text,omitempty"`
	Tools             []Tool              `json:"tools,omitempty"`
	ToolChoice        *ToolChoice         `json:"tool_choice,omitempty"`
	Prompt            *ResponsePrompt     `json:"prompt,omitempty"`
	Truncation        *string             `json:"truncation,omitempty"`

	Temperature          *float64          `json:"temperature,omitempty"`
	TopP                 *float64          `json:"top_p,omitempty"`
	TopLogProbs          *int              `json:"top_logprobs,omitempty"`
	Metadata             map[string]string `json:"metadata,omitempty"`
	User                 *string           `json:"user,omitempty"`
	SafetyIdentifier     *string           `json:"safety_identifier,omitempty"`
	PromptCacheKey       *string           `json:"prompt_cache_key,omitempty"`
	ServiceTier          *string           `json:"service_tier,omitempty"`
	PromptCacheRetention *string           `json:"prompt_cache_retention,omitempty"`

	Include           []string                 `json:"include,omitempty"`
	ParallelToolCalls *bool                    `json:"parallel_tool_calls,omitempty"`
	Store             *bool                    `json:"store,omitempty"`
	Stream            *bool                    `json:"stream,omitempty"`
	StreamOptions     *StreamOptions           `json:"stream_options,omitempty"`
	Conversation      *ConversationParam       `json:"conversation,omitempty"`
	ContextManagement []ContextManagementParam `json:"context_management,omitempty"`
}


package res2lak

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ── helper ───────────────────────────────────────────────────────────────────

func strPtr(s string) *string { return &s }

func messageItem(role, text string) ResponseInputItem {
	return ResponseInputItem{
		Type: "message",
		Role: role,
		Content: &ResponseInputItemContent{
			Text: text,
		},
	}
}

func partsItem(role string, parts []InputContentPart) ResponseInputItem {
	return ResponseInputItem{
		Type:    "message",
		Role:    role,
		Content: &ResponseInputItemContent{Parts: parts},
	}
}

// ── string input ─────────────────────────────────────────────────────────────

func TestTranslate_StringInput(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Text: "Hello world"},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	require.Len(t, result.Messages.Content, 1)
	assert.Equal(t, "text", result.Messages.Content[0].Type)
	assert.Equal(t, "Hello world", result.Messages.Content[0].Text)
}

func TestTranslate_StringInput_PreservesFullText(t *testing.T) {
	long := "Explain quantum computing in detail, covering superposition, entanglement, and error correction."
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Text: long},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, long, result.Messages.Content[0].Text)
}

// ── array input – single message ─────────────────────────────────────────────

func TestTranslate_ArrayInput_SingleUserMessage_StringContent(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			messageItem("user", "What is the capital of France?"),
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	require.Len(t, result.Messages.Content, 1)
	assert.Equal(t, "text", result.Messages.Content[0].Type)
	assert.Equal(t, "What is the capital of France?", result.Messages.Content[0].Text)
}

func TestTranslate_ArrayInput_SingleUserMessage_InputTextPart(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			partsItem("user", []InputContentPart{
				{Type: "input_text", Text: "Describe this concept"},
			}),
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	require.Len(t, result.Messages.Content, 1)
	assert.Equal(t, "text", result.Messages.Content[0].Type)
	assert.Equal(t, "Describe this concept", result.Messages.Content[0].Text)
}

func TestTranslate_ArrayInput_MultipleInputTextParts(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			partsItem("user", []InputContentPart{
				{Type: "input_text", Text: "First part."},
				{Type: "input_text", Text: "Second part."},
			}),
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	require.Len(t, result.Messages.Content, 2)
	assert.Equal(t, "First part.", result.Messages.Content[0].Text)
	assert.Equal(t, "Second part.", result.Messages.Content[1].Text)
}

// ── role preservation ─────────────────────────────────────────────────────────

func TestTranslate_ArrayInput_SystemRole(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			messageItem("system", "You are a helpful assistant."),
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "system", result.Messages.Role)
}

func TestTranslate_ArrayInput_DeveloperRole(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			messageItem("developer", "Internal instruction."),
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "developer", result.Messages.Role)
}

func TestTranslate_ArrayInput_AssistantRole(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			messageItem("assistant", "I can help with that."),
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "assistant", result.Messages.Role)
}

// ── multi-message arrays – uses last translatable item ───────────────────────

func TestTranslate_ArrayInput_MultipleMessages_UsesLast(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			messageItem("system", "You are helpful."),
			messageItem("user", "First question"),
			messageItem("assistant", "First answer"),
			messageItem("user", "Follow-up question"),
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	assert.Equal(t, "Follow-up question", result.Messages.Content[0].Text)
}

func TestTranslate_ArrayInput_SkipsItemReferences_UsesLastMessage(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			messageItem("user", "Original question"),
			{Type: "item_reference", ID: "msg_abc123"},
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	assert.Equal(t, "Original question", result.Messages.Content[0].Text)
}

func TestTranslate_ArrayInput_SkipsNonMessageItems_FindsEarlierMessage(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			messageItem("user", "What is 2+2?"),
			// tool call output – no role, raw JSON only
			{Type: "function_call_output", Raw: []byte(`{"type":"function_call_output","output":"4"}`)},
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	assert.Equal(t, "What is 2+2?", result.Messages.Content[0].Text)
}

// ── image and file content parts ─────────────────────────────────────────────

func TestTranslate_ImagePartsDropped_TextPartsPreserved(t *testing.T) {
	imageURL := "https://example.com/image.jpg"
	detail := "high"
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			partsItem("user", []InputContentPart{
				{Type: "input_text", Text: "What is in this image?"},
				{Type: "input_image", ImageURL: &imageURL, Detail: &detail},
			}),
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	// image part is not representable in OpenAIContent – only text survives
	require.Len(t, result.Messages.Content, 1)
	assert.Equal(t, "text", result.Messages.Content[0].Type)
	assert.Equal(t, "What is in this image?", result.Messages.Content[0].Text)
}

func TestTranslate_FilePartsDropped_TextPartsPreserved(t *testing.T) {
	fileURL := "https://example.com/doc.pdf"
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			partsItem("user", []InputContentPart{
				{Type: "input_text", Text: "Summarise the attached file."},
				{Type: "input_file", FileURL: &fileURL},
			}),
		}},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	require.Len(t, result.Messages.Content, 1)
	assert.Equal(t, "Summarise the attached file.", result.Messages.Content[0].Text)
}

func TestTranslate_OnlyImageParts_SkipsToEarlierItem(t *testing.T) {
	imageURL := "https://example.com/img.jpg"
	detail := "auto"
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			messageItem("user", "Earlier text message"),
			partsItem("user", []InputContentPart{
				{Type: "input_image", ImageURL: &imageURL, Detail: &detail},
			}),
		}},
	}
	// last item has only image content → not representable → fall back to previous item
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "Earlier text message", result.Messages.Content[0].Text)
}

// ── instructions fallback ────────────────────────────────────────────────────

func TestTranslate_InstructionsOnly_NoInput(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model:        "gpt-4o",
		Input:        ResponseInput{},
		Instructions: strPtr("You are a helpful assistant."),
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "system", result.Messages.Role)
	require.Len(t, result.Messages.Content, 1)
	assert.Equal(t, "text", result.Messages.Content[0].Type)
	assert.Equal(t, "You are a helpful assistant.", result.Messages.Content[0].Text)
}

func TestTranslate_InstructionsIgnored_WhenInputPresent(t *testing.T) {
	// When both are set, input takes precedence; instructions cannot be
	// represented alongside the user input in the single Messages field.
	req := OpenAIResponsesRequest{
		Model:        "gpt-4o",
		Input:        ResponseInput{Text: "Hello"},
		Instructions: strPtr("You are a helpful assistant."),
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	assert.Equal(t, "Hello", result.Messages.Content[0].Text)
}

func TestTranslate_InstructionsIgnored_WhenArrayInputPresent(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			messageItem("user", "Tell me a joke"),
		}},
		Instructions: strPtr("Be funny."),
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	assert.Equal(t, "Tell me a joke", result.Messages.Content[0].Text)
}

// ── error cases ───────────────────────────────────────────────────────────────

func TestTranslate_EmptyInput_NoInstructions_ReturnsError(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{},
	}
	_, err := Translate(req)
	assert.Error(t, err)
}

func TestTranslate_OnlyItemReferences_ReturnsError(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			{Type: "item_reference", ID: "msg_1"},
			{Type: "item_reference", ID: "msg_2"},
		}},
	}
	_, err := Translate(req)
	assert.Error(t, err)
}

func TestTranslate_ArrayWithOnlyImageContent_ReturnsError(t *testing.T) {
	imageURL := "https://example.com/img.jpg"
	detail := "auto"
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			partsItem("user", []InputContentPart{
				{Type: "input_image", ImageURL: &imageURL, Detail: &detail},
			}),
		}},
	}
	_, err := Translate(req)
	assert.Error(t, err)
}

func TestTranslate_ArrayWithOnlyFileContent_ReturnsError(t *testing.T) {
	fileURL := "https://example.com/file.pdf"
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Items: []ResponseInputItem{
			partsItem("user", []InputContentPart{
				{Type: "input_file", FileURL: &fileURL},
			}),
		}},
	}
	_, err := Translate(req)
	assert.Error(t, err)
}

func TestTranslate_EmptyInstructions_ReturnsError(t *testing.T) {
	empty := ""
	req := OpenAIResponsesRequest{
		Model:        "gpt-4o",
		Input:        ResponseInput{},
		Instructions: &empty,
	}
	_, err := Translate(req)
	assert.Error(t, err)
}

// ── output structure ──────────────────────────────────────────────────────────

func TestTranslate_ContentTypeIsAlwaysText(t *testing.T) {
	req := OpenAIResponsesRequest{
		Model: "gpt-4o",
		Input: ResponseInput{Text: "Hi"},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	for _, part := range result.Messages.Content {
		assert.Equal(t, "text", part.Type)
	}
}

func TestTranslate_OtherRequestFieldsDoNotAffectOutput(t *testing.T) {
	// Properties with no corresponding OpenAI field must not alter the result.
	temperature := 0.5
	store := true
	req := OpenAIResponsesRequest{
		Model:       "gpt-4o",
		Input:       ResponseInput{Text: "Hello"},
		Temperature: &temperature,
		Store:       &store,
		Tools: []Tool{
			{Type: "function", Function: &FunctionTool{Type: "function", Name: "get_time"}},
		},
	}
	result, err := Translate(req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	assert.Equal(t, "Hello", result.Messages.Content[0].Text)
}

// ── round-trip via JSON unmarshal ─────────────────────────────────────────────

func TestTranslate_ViaUnmarshal_StringInput(t *testing.T) {
	data := `{"model":"gpt-4o","input":"What is the speed of light?"}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	result, err := Translate(*req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	assert.Equal(t, "What is the speed of light?", result.Messages.Content[0].Text)
}

func TestTranslate_ViaUnmarshal_ArrayInput(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "system", "content": "Be concise."},
			{"type": "message", "role": "user", "content": "What is 2+2?"}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	result, err := Translate(*req)
	require.NoError(t, err)
	assert.Equal(t, "user", result.Messages.Role)
	assert.Equal(t, "What is 2+2?", result.Messages.Content[0].Text)
}

func TestTranslate_ViaUnmarshal_InputWithContentParts(t *testing.T) {
	data := `{
		"model": "gpt-4o",
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_text", "text": "Describe the image."},
					{"type": "input_image", "image_url": "https://example.com/img.jpg", "detail": "auto"}
				]
			}
		]
	}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	result, err := Translate(*req)
	require.NoError(t, err)
	// image part dropped; only the text part survives
	require.Len(t, result.Messages.Content, 1)
	assert.Equal(t, "Describe the image.", result.Messages.Content[0].Text)
}

func TestTranslate_ViaUnmarshal_InstructionsOnly(t *testing.T) {
	data := `{"model":"gpt-4o","input":"","instructions":"You are a pirate."}`
	req, err := UnmarshalOpenAIResponsesRequest([]byte(data))
	require.NoError(t, err)
	result, err := Translate(*req)
	require.NoError(t, err)
	assert.Equal(t, "system", result.Messages.Role)
	assert.Equal(t, "You are a pirate.", result.Messages.Content[0].Text)
}

package resp2chat_test

import (
	"encoding/json"
	"testing"

	. "resp2chat"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToOpenAI_StringInput(t *testing.T) {
	req := unmarshalRequest(t, `{"model": "gpt-4o", "input": "Hello"}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	assert.Equal(t, "user", result.Messages[0].Role)
	require.Len(t, result.Messages[0].Content, 1)
	assert.Equal(t, "text", result.Messages[0].Content[0].Type)
	assert.Equal(t, "Hello", result.Messages[0].Content[0].Text)
}

func TestToOpenAI_StringInputWithInstructions(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": "Hello",
		"instructions": "Be concise"
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 2)
	assert.Equal(t, "system", result.Messages[0].Role)
	assert.Equal(t, "Be concise", result.Messages[0].Content[0].Text)
	assert.Equal(t, "user", result.Messages[1].Role)
	assert.Equal(t, "Hello", result.Messages[1].Content[0].Text)
}

func TestToOpenAI_ArrayInputSingleMessage(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "user", "content": "Hello"}
		]
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	assert.Equal(t, "user", result.Messages[0].Role)
	assert.Equal(t, "Hello", result.Messages[0].Content[0].Text)
}

func TestToOpenAI_ArrayInputMultipleRoles(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "system", "content": "You are helpful"},
			{"type": "message", "role": "user", "content": "Hello"},
			{"type": "message", "role": "assistant", "content": "Hi there!"}
		]
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 3)
	assert.Equal(t, "system", result.Messages[0].Role)
	assert.Equal(t, "You are helpful", result.Messages[0].Content[0].Text)
	assert.Equal(t, "user", result.Messages[1].Role)
	assert.Equal(t, "Hello", result.Messages[1].Content[0].Text)
	assert.Equal(t, "assistant", result.Messages[2].Role)
	assert.Equal(t, "Hi there!", result.Messages[2].Content[0].Text)
}

func TestToOpenAI_InstructionsPrependedBeforeArrayInput(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "user", "content": "Hello"}
		],
		"instructions": "System prompt"
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 2)
	assert.Equal(t, "system", result.Messages[0].Role)
	assert.Equal(t, "System prompt", result.Messages[0].Content[0].Text)
	assert.Equal(t, "user", result.Messages[1].Role)
}

func TestToOpenAI_ContentPartsTextOnly(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_text", "text": "First part"},
					{"type": "input_text", "text": "Second part"}
				]
			}
		]
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	require.Len(t, result.Messages[0].Content, 2)
	assert.Equal(t, "First part", result.Messages[0].Content[0].Text)
	assert.Equal(t, "Second part", result.Messages[0].Content[1].Text)
}

func TestToOpenAI_ContentPartsNonTextIgnored(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_text", "text": "Describe this"},
					{"type": "input_image", "image_url": "https://example.com/img.jpg"}
				]
			}
		]
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages[0].Content, 1)
	assert.Equal(t, "Describe this", result.Messages[0].Content[0].Text)
}

func TestToOpenAI_ItemReferenceSkipped(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "item_reference", "id": "msg_abc123"},
			{"type": "message", "role": "user", "content": "Follow up"}
		]
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	assert.Equal(t, "user", result.Messages[0].Role)
	assert.Equal(t, "Follow up", result.Messages[0].Content[0].Text)
}

func TestToOpenAI_DeveloperRole(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "developer", "content": "Internal note"}
		]
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	assert.Equal(t, "developer", result.Messages[0].Role)
}

func TestToOpenAI_EmptyInputError(t *testing.T) {
	req := unmarshalRequest(t, `{"model": "gpt-4o", "input": []}`)
	_, err := req.ToOpenAI()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no messages")
}

func TestToOpenAI_OnlyNonMessageItemsError(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "item_reference", "id": "msg_abc123"}
		]
	}`)
	_, err := req.ToOpenAI()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no messages")
}

func TestToOpenAI_MissingRoleError(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "content": "No role"}
		]
	}`)
	_, err := req.ToOpenAI()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "missing role")
}

func TestToOpenAI_EmptyInstructionsIgnored(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": "Hello",
		"instructions": ""
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	assert.Equal(t, "user", result.Messages[0].Role)
}

func TestToOpenAI_MessageWithNilContent(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "assistant"}
		]
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	assert.Equal(t, "assistant", result.Messages[0].Role)
	assert.Nil(t, result.Messages[0].Content)
}

func TestToOpenAI_ContentPartsAllNonTextYieldsEmptyContent(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_image", "image_url": "https://example.com/img.jpg"}
				]
			}
		]
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	assert.Empty(t, result.Messages[0].Content)
}

func TestToOpenAI_FullConversationWithInstructions(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "message", "role": "system", "content": "Context info"},
			{"type": "message", "role": "user", "content": "Question 1"},
			{"type": "message", "role": "assistant", "content": "Answer 1"},
			{"type": "message", "role": "user", "content": "Question 2"}
		],
		"instructions": "Be helpful"
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 5)
	assert.Equal(t, "system", result.Messages[0].Role)
	assert.Equal(t, "Be helpful", result.Messages[0].Content[0].Text)
	assert.Equal(t, "system", result.Messages[1].Role)
	assert.Equal(t, "Context info", result.Messages[1].Content[0].Text)
	assert.Equal(t, "user", result.Messages[2].Role)
	assert.Equal(t, "assistant", result.Messages[3].Role)
	assert.Equal(t, "user", result.Messages[4].Role)
}

func TestToOpenAI_MixedItemTypesFiltered(t *testing.T) {
	req := unmarshalRequest(t, `{
		"model": "gpt-4o",
		"input": [
			{"type": "item_reference", "id": "ref1"},
			{"type": "message", "role": "user", "content": "Hello"},
			{"type": "item_reference", "id": "ref2"},
			{"type": "message", "role": "assistant", "content": "Hi"}
		]
	}`)
	result, err := req.ToOpenAI()
	require.NoError(t, err)
	require.Len(t, result.Messages, 2)
	assert.Equal(t, "user", result.Messages[0].Role)
	assert.Equal(t, "assistant", result.Messages[1].Role)
}

func unmarshalRequest(t *testing.T, data string) OpenAIResponsesRequest {
	t.Helper()
	var req OpenAIResponsesRequest
	err := json.Unmarshal([]byte(data), &req)
	require.NoError(t, err)
	return req
}

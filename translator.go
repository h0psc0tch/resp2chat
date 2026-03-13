package resp2chat

import "errors"

// Translate converts the receiver into the simplified OpenAI chat completions
// representation.
//
// Mapping rules:
//   - A string input becomes a user message with a single text content part.
//   - An array input uses the last item that carries a role and translatable
//     text content. Items without a role (e.g. item_reference, tool-call
//     outputs) and items whose content consists only of image or file parts
//     (which have no equivalent in OpenAIContent) are skipped.
//   - When the input array yields no usable message but Instructions is set,
//     Instructions is used as a system message.
//   - An error is returned when neither the input nor instructions provides any
//     translatable content.
func (req OpenAIResponsesRequest) Translate() (OpenAI, error) {
	msg, err := buildMessage(req)
	if err != nil {
		return OpenAI{}, err
	}
	return OpenAI{Messages: msg}, nil
}

func buildMessage(req OpenAIResponsesRequest) (OpenAIMessage, error) {
	if req.Input.Text != "" {
		return OpenAIMessage{
			Role:    "user",
			Content: []OpenAIContent{{Type: "text", Text: req.Input.Text}},
		}, nil
	}

	if len(req.Input.Items) > 0 {
		for i := len(req.Input.Items) - 1; i >= 0; i-- {
			item := req.Input.Items[i]
			if item.Role == "" {
				continue
			}
			content := contentPartsToOpenAI(item.Content)
			if len(content) > 0 {
				return OpenAIMessage{Role: item.Role, Content: content}, nil
			}
		}
		return OpenAIMessage{}, errors.New("input contains no translatable text content")
	}

	if req.Instructions != nil && *req.Instructions != "" {
		return OpenAIMessage{
			Role:    "system",
			Content: []OpenAIContent{{Type: "text", Text: *req.Instructions}},
		}, nil
	}

	return OpenAIMessage{}, errors.New("input is empty")
}

// contentPartsToOpenAI converts a ResponseInputItemContent to OpenAIContent
// parts. Only text content (string content or input_text parts) can be
// represented; input_image and input_file parts are silently dropped because
// OpenAIContent has no image or file fields.
func contentPartsToOpenAI(c *ResponseInputItemContent) []OpenAIContent {
	if c == nil {
		return nil
	}
	if c.Text != "" {
		return []OpenAIContent{{Type: "text", Text: c.Text}}
	}
	var out []OpenAIContent
	for _, part := range c.Parts {
		if part.Type == "input_text" {
			out = append(out, OpenAIContent{Type: "text", Text: part.Text})
		}
	}
	return out
}

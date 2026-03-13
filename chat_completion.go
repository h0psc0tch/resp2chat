package resp2chat

type OpenAI struct {
	Messages OpenAIMessage `json:"messages"`
}

type OpenAIMessage struct {
	Role    string          `json:"role"`
	Content []OpenAIContent `json:"content"`
}

type OpenAIContent struct {
	Type  string `json:"type"`
	Text  string `josn:"text,omitempty"`
	Audio string `json:"audio,omitempty"`
}

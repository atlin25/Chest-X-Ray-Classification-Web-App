export function useHandleStreamResponse({ onChunk, onFinish }) {
  return async function handle(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let text = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      onChunk(text);
    }

    onFinish(text);
  };
}


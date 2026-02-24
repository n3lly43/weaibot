import warnings
import gradio as gr

from agents import support_agent

warnings.filterwarnings('ignore')

with gr.Blocks() as demo:
    with gr.Sidebar():
        gr.LoginButton()   
    gr.Markdown("# WEAI-bot")
    chatbot = gr.Chatbot(type='messages', 
                         allow_tags=True)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def handle_undo(history, undo_data: gr.UndoData):
        return history[:undo_data.index], history[undo_data.index]['content'][0]["text"]

    def handle_retry(history, retry_data: gr.RetryData):
        new_history = history[:retry_data.index]
        previous_prompt = history[retry_data.index]['content'][0]["text"]
        yield from support_agent_fn(previous_prompt, new_history)

    def support_agent_fn(message, history):
        result = support_agent.invoke({"messages": [{"role": "user", "content": message}]})

        response = result['messages'][-1].content#.split('<|start_header_id|>assistant<|end_header_id|>')[-1].strip()
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        return "", history
        
    def handle_like(data: gr.LikeData):
        if data.liked:
            print("You upvoted this response: ", data.value)
        else:
            print("You downvoted this response: ", data.value)

    def handle_edit(history, edit_data: gr.EditData):
        new_history = history[:edit_data.index]
        new_history[-1]['content'] = [{"text": edit_data.value, "type": "text"}]
        return new_history

    msg.submit(support_agent_fn, [msg, chatbot], [msg, chatbot])

    chatbot.undo(handle_undo, chatbot, [chatbot, msg])
    chatbot.retry(handle_retry, chatbot, chatbot)
    chatbot.like(handle_like, None, None)
    chatbot.edit(handle_edit, chatbot, chatbot)

if __name__ == "__main__":
    demo.launch()
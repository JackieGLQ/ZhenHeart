import gradio as gr
import time
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import torch
from torch import nn
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList
from transformers.utils import logging

from modelscope import AutoTokenizer, AutoModelForCausalLM  # isort: skip
import torch

system_1 = """
"""

system_2 = """
"""

def load_model(model_version):
    global model, tokenizer
    if model or tokenizer:
        del model, tokenizer
    if model_version == "禅心·明镜V1.0":
        model_path = "JakcieGao/ZhenHeart"
        system = system_1
   # elif model_version == "Internlm2-chat-7b":
   #     model_path = "/root/model/internlm2-chat-7b"
   #     system = system_2
    else:
        model_path = "JakcieGao/ZhenHeart"
        system = system_1
    model = (
        AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("=================init success=====================")
    return gr.update(label=f"ChatBot Using {model_version}"), gr.update(value=system)


@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors="pt")
    input_length = len(inputs["input_ids"][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs["input_ids"]
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = (
        kwargs.get("max_length") is None and generation_config.max_length is not None
    )
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            "This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we"
            " recommend using `max_new_tokens` to control the maximum \
                length of the generation.",
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = (
            generation_config.max_new_tokens + input_ids_seq_length
        )
        if not has_default_max_length:
            print(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/"
                "en/main_classes/text_generation)",
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = "input_ids"
        print(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            "This can lead to unexpected behavior. You should consider"
            " increasing 'max_new_tokens'."
        )

    # 2. Set generation parameters if not already defined
    logits_processor = (
        logits_processor if logits_processor is not None else LogitsProcessorList()
    )
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False
        )
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long()
        )

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            break


def grtodict(chat_history):
    messages = []
    for i in chat_history[:-1]:
        try:
            messages.append({"role": "user", "content": i[0]})
            messages.append({"role": "robot", "content": i[1]})
        except:
            pass
    return messages


def combine_prompt(prompt, system, messages):
    total_prompt = f"<s><|im_start|>system\n{system}<|im_end|>\n"
    for message in messages:
        cur_content = message["content"]
        if message["role"] == "user":
            cur_prompt = user_prompt.format(user=cur_content)
        elif message["role"] == "robot":
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def user(user_message, history):
    return "", history + [[user_message, None]]


def get_respond(
    chat_history, max_new_tokens, temperature, repetition_penalty, top_p, system
):

    user_message = chat_history[-1][0]
    messages = grtodict(chat_history)
    prompt = combine_prompt(user_message, system, messages)

    generation_config = GenerationConfig(
        max_length=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
    )

    bot_message_gen = generate_interactive(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        additional_eos_token_id=92542,
        **asdict(generation_config),
    )

    bot_message = ""

    for i in bot_message_gen:
        bot_message = i

    # messages.append({"role": "user", "content": bot_message})

    chat_history[-1][1] = ""
    for character in bot_message:
        chat_history[-1][1] += character
        time.sleep(0.05)
        yield chat_history


def clear_respond():
    # global messages
    # messages.clear()
    return "", ""


def withdraw_last_respond(chat_history):
    # global messages
    # messages = messages[:-2]
    chat_history = chat_history[:-1]
    return chat_history


def regenerate_respond(
    chat_history, max_new_tokens, temperature, repetition_penalty, top_p, system
):
    # 删除生成的最近的内容
    chat_history[-1][1] = ""
    messages = grtodict(chat_history)
    user_message = chat_history[-1][0]
    prompt = combine_prompt(user_message, system, messages)

    generation_config = GenerationConfig(
        max_length=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        top_p=top_p,
    )

    bot_message_gen = generate_interactive(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        additional_eos_token_id=92542,
        **asdict(generation_config),
    )

    bot_message = ""

    for i in bot_message_gen:
        bot_message = i

    # messages.append({"role": "user", "content": bot_message})

    chat_history[-1][1] = ""
    for character in bot_message:
        chat_history[-1][1] += character
        time.sleep(0.05)
        yield chat_history


user_prompt = "<|im_start|>user\n{user}<|im_end|>\n"
robot_prompt = "<|im_start|>assistant\n{robot}<|im_end|>\n"
cur_query_prompt = "<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n"

messages = []

model, tokenizer = None, None
load_model("禅心·明镜V1.0")

with gr.Blocks(title="禅心·明镜") as demo:
    #gr.HTML("<h1 ><center>禅心·明镜</h1>")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            system_image = gr.Image(
                label="悟",
                value="./image/fozi.png",
                scale=1,
                interactive=False, # 设置为 False 禁止用户交互修改图片
            ) 
            system = gr.Textbox(
                label="境",
                value="菩提本无树，明镜亦非台\n本来无一物，何处惹尘埃！",
                scale=3,
                interactive=True,
            )
 
            max_new_tokens = gr.Slider(
                minimum=0,
                maximum=1024,
                value=512,
                step=64,
                interactive=True,
                label="最多生成tokens数量",
           )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.1,
                step=0.1,
                interactive=True,
                label="温度",
                #info="控制文本生成的随机性，数值越高，模型生成的文本越随机，可能会更创新但也更不可预测。数值较低时，模型生成的文本通常更加确定、连贯，但可能会缺乏多样性。温度通常设定在 0 到 1 之间",
            )
            repetition_penalty = gr.Slider(
                minimum=0.0,
                maximum=5.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="重复惩罚参数",
                #info="重复惩罚参数用来减少生成文本中的重复内容。当模型在生成文本时重复使用特定的单词或短语，可以通过增加 repetition_penalty 的值来抑制这种重复。通常，这个值设定为 1.0（即没有惩罚），如果你希望减少重复，可以将其设置得更高。",
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.75,
                step=0.1,
                interactive=True,
                label="Top P",
                #info="通过调整 top_p 的值，可以影响生成文本的随机性和创造性。当 top_p 设置为 1.0 时，核采样等同于传统的随机采样！",
            )


        with gr.Column(scale=4):
            with gr.Row():
                model_version = gr.Dropdown(
                    label="模型版本",
                    #choices=["禅心·明镜", "Internlm2-chat-7b"],
                    choices=["禅心·明镜V1.0"],
                    value="禅心·明镜V1.0",
                    interactive=True,
                    info="选择不同的模型版本",
                )
                init_but = gr.Button("选择模型版本")

            with gr.Group() as chat_board:
                chatbot = gr.Chatbot(label="ChatBot Using 禅心·明镜V1.0")
                history = gr.State([])
                msg = gr.Textbox(label="施主，有何疑惑，不妨道来！",placeholder="贫僧乃六祖慧能是也。今日与施主相遇，实乃缘分。若施主有何困惑，贫僧定当竭尽所能，为施主排忧解难！")
                with gr.Row():
                    ask = gr.Button("🚀 发送信息")
                    clear = gr.Button("🧹 清除记录")
                    #withdraw = gr.Button("↩️ Recall last message")
                    regenerate = gr.Button("🔁 重新生成")

    init_but.click(load_model, [model_version], [chatbot, system])

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        get_respond,
        [chatbot, max_new_tokens, temperature, repetition_penalty, top_p, system],
        chatbot,
    )

    ask.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        get_respond,
        [chatbot, max_new_tokens, temperature, repetition_penalty, top_p, system],
        chatbot,
    )

    clear.click(clear_respond, outputs=[msg, chatbot])

    #withdraw.click(withdraw_last_respond, inputs=[chatbot], outputs=[chatbot])

    regenerate.click(
        regenerate_respond,
        inputs=[
            chatbot,
            max_new_tokens,
            temperature,
            repetition_penalty,
            top_p,
            system,
        ],
        outputs=[chatbot],
    )


demo.queue()
demo.launch(server_name="0.0.0.0")

import time, re, os
import streamlit as st
from openai import OpenAI
from copy import deepcopy
from streamlit_pills import pills
import streamlit_antd_components as sac
from streamlit_option_menu import option_menu
from rag.retrieve import retrieve, init_models
from utils import print_colorful, Fore, random_icon

st.set_page_config(page_title="小Ai同学", page_icon="🐧", layout="wide")

from configs import llm_configs, rag_configs, SYSTEM_PROMPTS, ROBOT_AVATAR
from page.database_page import db_page


st.markdown(
    """<style>
.stDeployButton {
    visibility: hidden;
}
.block-container {
    padding: 3rem 4rem 2rem 4rem;
}
.st-emotion-cache-kgpedg {
    padding: 0.2rem 1.5rem;
}
</style>""",
    unsafe_allow_html=True,
)
state = st.session_state


def siderbar_title(title: str):
    with st.sidebar:
        st.markdown(
            f"<div align='center'><font size=6>{title}</font></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div align='center'><font size=2></font></div>",
            unsafe_allow_html=True,
        )


def init_st():
    params = dict(
        # 当前会话的所有对话数据
        messages=[],
        # dialogue_history
        dialogue_history=[[]],
        # 初始化
        session_last_change_key=0,
        # 处于数据库处理流程
        IS_DB_MODE=False,
        # 当前对话检索到的doc
        now_search_docs=[],
    )
    for k, v in params.items():
        if k not in state:
            state[k] = v

    # 创建数据库
    os.makedirs(rag_configs["database"]["db_docs_path"], exist_ok=True)
    os.makedirs(rag_configs["database"]["db_vector_path"], exist_ok=True)


def clear_chat_history():
    st.toast("我们再聊聊吧🌿~", icon=ROBOT_AVATAR)
    if "messages" in state:
        del state.messages


def callback_session_change():
    """切换对话"""
    dialogue_history = state.dialogue_history
    change_key = state.session_change_key
    last_change_key = state.session_last_change_key
    if last_change_key == 0 and change_key != 0:
        state.dialogue_history[0] = {
            "messages": state.messages,
            "time": time.strftime("%y-%m-%d %H%M%S"),
            "configs": {},
        }
    state.messages = dialogue_history[change_key]["messages"]
    state.session_last_change_key = change_key


def callback_db_setting():
    state["IS_DB_MODE"] = True


def callback_db_setting_finish():
    state["IS_DB_MODE"] = False


def show_related_docs(related_docs):
    # 只显示信息源
    # if not state.now_search_docs:
    #     _tr = lambda x: x[:6] + "..." + x[-6:] if len(x) > 12 else x
    #     sources = [i.metadata for i in related_docs]
    #     sources = list(
    #         set([_tr(os.path.basename(i["source"])) for i in sources])
    #     )
    #     sources = [f"{idx}.{s}" for idx, s in enumerate(sources, 1)]
    #     state.now_search_docs = sources
    # else:
    #     sources = state.now_search_docs
    # st.markdown(f"")
    # sac.tags(sources, size="sm", color="blue")

    # 显示相关的文本和得分
    cols = st.columns([0.9, 0.1])
    with cols[0].expander("🔻知识库索引", expanded=False):
        for doc in related_docs:
            score = doc.metadata.get("score", None)
            score = f"{score:.3f}" if score else ""
            s = f':green[**{os.path.basename(doc.metadata["source"])}**] `{score}`\n'  #
            d = re.sub(r"\s+", "\n>", doc.page_content.strip())
            s += f"> {d}"
            st.markdown(s)


def init_chat_history():
    with st.chat_message("assistant", avatar="assets/app-indicator.svg"):
        st.markdown("我是你的小助手，快开始跟我对话吧💭💭", unsafe_allow_html=True)
    if "messages" in state:
        for message in state.messages:
            if (
                message["role"] not in ["system", "tool"]
                and "tool_calls" not in message
            ):
                avatar = "🧑‍💻" if message["role"] == "user" else ROBOT_AVATAR
                with st.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"], unsafe_allow_html=True)
                    if related_docs := message.get("related_docs"):
                        show_related_docs(related_docs)
    else:
        state.messages = []

    return state.messages


def run_chat(cfgs):
    messages = init_chat_history()
    client = OpenAI(
        api_key=cfgs["api_key"],
        base_url=cfgs["base_url"],
    )

    def llm_generate(messages, **kargs):
        msgs = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        return client.chat.completions.create(
            **{
                "model": cfgs["model_name"],
                "messages": msgs,
                "temperature": cfgs["temperature"],
                **kargs,
            }
        )

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt, unsafe_allow_html=True)
        messages.append({"role": "user", "content": prompt})
        print_colorful(f"[user] {prompt}")

        with st.chat_message("assistant", avatar=ROBOT_AVATAR):
            placeholder = st.empty()

            # RAG 检索相关性信息
            related_docs = []
            ori_question = messages[-1]["content"]
            if cfgs["RAG"]:
                print_colorful(
                    f'使用的知识库名{cfgs["index_name"]}', text_color=Fore.RED
                )
                rag_configs["database"]["index_name"] = cfgs["index_name"]
                retriever, reranker = init_models(configs=rag_configs)
                with placeholder.status("正在检索知识库..."):
                    configs = {
                        "sample_top": cfgs["rag_topk"],
                        "sample_threshold": cfgs.get("rag_threshold", -100),
                    }
                    print(f"state.enable_rewrite_key:{state.enable_rewrite_key}")
                    messages, related_docs = retrieve(
                        llm_generate,
                        messages,
                        retriever,
                        reranker,
                        rewrite=state.enable_rewrite_key != "关闭",
                        configs=configs,
                        show_info=True,
                    )
                placeholder.empty()

            if not cfgs.get("stream", True):
                with placeholder.status("让我思考一下吧🙇...", expanded=False):
                    completion = llm_generate(messages=messages, stream=False)
                    response = completion.choices[0].message.content
                placeholder.markdown(response, unsafe_allow_html=True)
            else:
                msgs = []
                for chunk_message in llm_generate(messages=messages, stream=True):
                    tmp = chunk_message.choices[0].delta.content
                    if not tmp:
                        continue
                    msgs.append(tmp)
                    placeholder.markdown("".join(msgs), unsafe_allow_html=True)
                response = "".join(msgs)
            print_colorful(f"[assistant] {response.strip()}")

            # 知识库索引
            if cfgs["RAG"]:
                show_related_docs(related_docs)

            # 恢复原问题
            messages[-1]["content"] = ori_question
            messages.append(
                {"role": "assistant", "content": response, "related_docs": related_docs}
            )

    if len(state.messages) > 1:
        # 优化去掉背景和label占位 anaconda3\Lib\site-packages\streamlit_pills\frontend\build
        # 设置：body background-color: rgba(240, 242, 246, 0) !important;
        # 设置：label min-height: 0rem;
        select_m = pills(
            label="",
            label_visibility="hidden",
            options=["请输入..", "清空对话", "新建对话"],
            index=0,
            icons=["🔅", "♻️", "🪄"],  #
            clearable=True,
        )
        if select_m == "新建对话":
            state.dialogue_history.append(
                {
                    "messages": deepcopy(state.messages),
                    "time": time.strftime("%y-%m-%d %H%M%S"),
                    "configs": cfgs,
                }
            )
            clear_chat_history()
            st.rerun()
        elif select_m == "清空对话":
            clear_chat_history()
            st.rerun()


def siderbar_params(configs: dict):
    with st.sidebar:
        with st.expander("⚙️ **知识库配置**", expanded=True):
            if configs["RAG"]:
                if not state["IS_DB_MODE"]:
                    st.button(
                        "🧩 数据库管理",
                        help="✅ 点击打开数据库管理界面",
                        on_click=callback_db_setting,
                        key="into_db_setting_key1",
                        use_container_width=True,
                    )
                else:
                    cols = st.columns([0.6, 0.4])
                    cols[0].button(
                        "数据库管理",
                        help="✅ 点击打开数据库管理",
                        on_click=callback_db_setting,
                        key="into_db_setting_key2",
                        use_container_width=True,
                    )
                    cols[1].button(
                        "✅完成",
                        help="✅ 点击关闭数据库管理",
                        on_click=callback_db_setting_finish,
                        key="into_db_setting_return_key",
                        use_container_width=True,
                    )

                index_names = [
                    i
                    for i in os.listdir(rag_configs["database"]["db_docs_path"])
                    if not i.endswith("json")
                ]
                configs["index_name"] = st.selectbox(
                    "**选择数据库**", tuple(index_names)
                )
                cols = st.columns(2)
                configs["rag_topk"] = cols[0].slider(
                    "**检索数量**", value=3, min_value=1, max_value=10, step=1
                )
                configs["rag_threshold"] = cols[1].slider(
                    "**相似度阈值**", value=0.2, min_value=0.0, max_value=3.0, step=0.01
                )
                enable_rewrite = st.radio(
                    "xxx",
                    ["关闭", "打开检索增强"],
                    horizontal=True,
                    label_visibility="collapsed",
                    help="需要消耗更多的Tokens,但更准确",
                    key="enable_rewrite_key",
                )
                if enable_rewrite != "关闭":
                    st.caption("连续询问时，检索更准确")
                else:
                    st.caption("适合直接检索相关数据")
        # 对话历史
        dialogue_history = state.dialogue_history
        if dialogue_history:
            st.selectbox(
                f"📇 **历史对话**",
                tuple(range(len(dialogue_history))),
                index=0,  # len(dialogue_history) - 1
                on_change=callback_session_change,
                format_func=lambda x: (
                    f"{random_icon(x)}" + f" 历史对话 {x}" if x else "👉 当前对话"
                ),
                key="session_change_key",
                help="对话历史记录",
            )

        with st.expander("**模型设置**", expanded=False):
            # model 选择
            model_name = st.selectbox(
                "🧸 **选择模型**",
                tuple(llm_configs),
                index=(
                    list(llm_configs).index(configs["model_name"])
                    if configs.get("model_name")
                    else 0
                ),
                help="查看configs文件进行修改",
            )
            configs["model_name"] = model_name
            if "[" in model_name:
                configs["model_name"] = model_name[: model_name.rfind("[")]
            print_colorful(f"[model] {model_name}")

            # key
            # api_key = st.text_input(
            #     "🗝️ **API_KEY**",
            #     value=llm_configs[model_name]["api_key"],
            #     help="支持openai格式的key",
            #     type="password",
            # )
            # base_url = st.text_input(
            #     "🔑 **BASE_URL**", value=llm_configs[model_name]["base_url"], help=""
            # )
            configs["api_key"] = llm_configs[model_name]["api_key"]
            configs["base_url"] = llm_configs[model_name]["base_url"]

            # temperature
            st.markdown("⚖️ **Temperature**", help="温度系数")
            cols = st.columns([0.001, 0.998, 0.001])
            temperature = cols[1].slider(
                "xxx",
                value=0.1,
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                label_visibility="collapsed",
            )
            configs["temperature"] = temperature

            # stream
            # stream = pills(
            #     label="➿ 流式对话",
            #     options=["True", "False"],
            #     index=0,
            # )
            # configs["stream"] = stream == "True"
            stream = sac.buttons(
                [
                    sac.ButtonsItem("打开", icon="fast-forward", color="green"),
                    sac.ButtonsItem("关闭", icon="x-square"),
                ],
                label="➿ 流式对话",
                use_container_width=True,
                size="xs",
                variant="link",
                align="center",
            )
            configs["stream"] = stream == "打开"

            # -- 添加系统提示词 --
            if not configs["RAG"]:
                system_prompts = SYSTEM_PROMPTS
                system_prompt_name = st.selectbox(
                    "🔖 **系统提示词**",
                    tuple(system_prompts),
                    index=0,
                    help="让大模型跟随指令进行回答",
                )
                system_prompt = system_prompts[system_prompt_name]
            else:
                system_prompt = "你是一名知识渊博的资深学者，拥有非常强的逻辑理解能力，问题解决能力。"
            system_prompt = st.text_area(
                "🧹 **编辑提示词**",
                system_prompt,
                help="编辑系统提示词",
            )

            if "messages" in state:
                if state.messages and state.messages[0]["role"] == "system":
                    state.messages[0]["content"] = system_prompt
                else:
                    state.messages.insert(
                        0, {"role": "system", "content": system_prompt}
                    )
            else:
                state.messages = [{"role": "system", "content": system_prompt}]


def siderbar_bottom():
    with st.sidebar:
        # st.divider()
        text_contents = "\n".join(
            [
                "\n" * (m["role"] == "user") + f'<{m["role"]}>\t: {m["content"]}'
                for m in state.messages
            ]
        )
        name = time.strftime("%y-%m-%d %H%M%S")
        st.download_button(
            "🌝保存对话信息🐾",
            text_contents,
            file_name=f"messages {name}.txt",
            use_container_width=True,
        )


def main():
    siderbar_title("智能对话系统")
    init_st()

    with st.sidebar:
        selected = option_menu(
            None,
            ["对话模式", "知识库问答"],  # , "文件问答"
            icons=["activity", "database"],  # ,"file-earmark-medical"
            menu_icon="cast",
            default_index=0,
        )
    configs = {
        "RAG": (selected == "知识库问答" or state["IS_DB_MODE"]),
        "SELECTED_MODE": selected,
    }

    siderbar_params(configs)

    if state["IS_DB_MODE"]:
        db_page()
        if selected == "对话模式":
            st.toast(":green[点击>✅完成< 退出数据库管理]", icon="🤗")
    else:
        run_chat(configs)

    siderbar_bottom()


if __name__ == "__main__":
    main()

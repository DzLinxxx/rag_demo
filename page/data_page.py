import os
import time
import torch
import gc
import shutil
import streamlit as st
from PIL import Image
from configs import model_names
import streamlit_antd_components as sac
from utils import load_data, save_data, load_model, SuperMethod


def nowdate():
    return time.strftime("%y/%m/%d %H:%M:%S")


def process(model_name, show_data, images_path):
    model, preprocess = load_model(model_name)
    p = SuperMethod(model, preprocess, show_data)
    img_features = p.get_imgs_features(images_path).tolist()
    show_data["features"][model_name].extend(img_features.copy())
    del model, preprocess, img_features
    gc.collect()
    torch.cuda.empty_cache()


def data_page():
    tabs = sac.tabs(
        [
            sac.TabsItem("图片上传", icon="file-richtext"),
            sac.TabsItem("数据库管理", icon="sliders"),
        ],
        variant="outline",
        color="green",
    )

    if tabs == "图片上传":
        with st.container(border=True):
            upload_files = st.file_uploader(
                "📣 **拖动或点击上传图片**",
                type=["jpg", "png"],
                accept_multiple_files=True,
                label_visibility="visible",
            )
            submit = st.button("保存到数据库", use_container_width=True)
            if submit:
                if upload_files:
                    ph = st.empty()
                    with ph.status("正在保存数据中...", expanded=True) as status:

                        st.write(f":green[{nowdate()}] 保存图片到数据库...")
                        images_path = []
                        for img in upload_files:
                            path = f"database/images/{img.name}"
                            Image.open(img).save(path)
                            images_path.append(path)

                        st.write(f":green[{nowdate()}] 正在读取数据库...")
                        show_data = load_data()
                        show_data["images_path"].extend(images_path)

                        st.write(f":green[{nowdate()}] 开始计算 ViT-B-16 特征...")
                        process("ViT-B-16", show_data, images_path)

                        # st.write(f":green[{nowdate()}] 开始计算 ViT-L-14 特征...")
                        # process("ViT-L-14", show_data, images_path)

                        # st.write(f":green[{nowdate()}] 开始计算 ViT-L-14-336 特征...")
                        # process("ViT-L-14-336", show_data, images_path)

                        # 保存数据并更新缓存
                        save_data(show_data)
                        st.session_state["db_data"] = load_data()

                    status.update(label="处理完毕", state="complete", expanded=False)
                    st.toast(":orange[保存完毕]", icon="✨")

                else:
                    st.toast(":orange[请选择文件]", icon="🔔")
    else:
        state = st.session_state
        inum = len(state["db_data"]["images_path"])
        if not inum:
            st.warning("没有数据", icon="💢")
        else:
            show_data = {
                "图片": [os.path.basename(i) for i in state["db_data"]["images_path"]]
            }
            for name in model_names:
                num = len(state["db_data"]["features"][name])
                show_data[name] = ["✅"] * num + ["❌"] * (inum - num)
            show_data["选择"] = [False] * inum

            st.markdown("🔻 **数据列表**")
            with st.container(border=True, height=380):
                edited_data = st.data_editor(
                    show_data, use_container_width=True, hide_index=False
                )

            cols = st.columns(2)
            if cols[0].button("删除所选", use_container_width=True):
                if any(edited_data["选择"]):

                    for idx, value in zip(
                        range(len(edited_data["选择"]) - 1, -1, -1),
                        edited_data["选择"][::-1],
                    ):
                        if value:
                            imp = state["db_data"]["images_path"][idx]
                            os.remove(imp)
                            del state["db_data"]["images_path"][idx]
                            for name in model_names:
                                if show_data[name][idx] == "✅":
                                    del state["db_data"]["features"][name][idx]

                    save_data(state["db_data"])
                    st.toast(":green[删除成功]", icon="✅")
                    st.rerun()
                else:
                    st.toast(":orange[未选择数据]", icon="🔊")

            if cols[1].button("清空数据库", use_container_width=True):
                shutil.rmtree("database")
                del state["db_data"]
                st.rerun()

            # st.table(edited_data)


def data_page_style2():
    tabs = sac.tabs(
        [
            sac.TabsItem("图片上传", icon="file-richtext"),
            sac.TabsItem("数据库管理", icon="sliders"),
        ],
        variant="outline",
        color="green",
    )

    if tabs == "图片上传":
        with st.container(border=True):
            upload_files = st.file_uploader(
                "📣 **拖动或点击上传图片**",
                type=["jpg", "png"],
                accept_multiple_files=True,
                label_visibility="visible",
            )
            submit = st.button("保存到数据库", use_container_width=True)
            if submit:
                if upload_files:
                    ph = st.empty()
                    with ph.status("正在保存数据中...", expanded=True) as status:

                        st.write(f":green[{nowdate()}] 保存数据到数据库...")
                        images_path = []
                        for img in upload_files:
                            path = f"database/images/{img.name}"
                            Image.open(img).save(path)
                            images_path.append(path)

                        st.write(f":green[{nowdate()}] 读取数据库...")
                        show_data = load_data()
                        show_data["images_path"].extend(images_path)

                        for mname in model_names:
                            st.write(f":green[{nowdate()}] 计算 {mname} 特征...")
                            process(mname, show_data, images_path)

                        # 保存数据并更新缓存
                        save_data(show_data)
                        st.session_state["db_data"] = load_data()

                    status.update(label="处理完毕", state="complete", expanded=False)
                    st.toast(":orange[保存完毕]", icon="✨")

                else:
                    st.toast(":orange[请选择文件]", icon="🔔")
    else:
        state = st.session_state
        inum = len(state["db_data"]["images_path"])
        if not inum:
            st.warning("没有数据", icon="💢")
        else:
            # 处理数据
            show_data = {
                "图片": [os.path.basename(i) for i in state["db_data"]["images_path"]]
            }
            for name in model_names:
                num = len(state["db_data"]["features"][name])
                show_data[name] = ["✅"] * num + ["❌"] * (inum - num)
            show_data["选择"] = [False] * inum

            # 显示数据
            st.markdown("🔻 **数据列表**")
            edited_data = st.data_editor(
                show_data, use_container_width=True, hide_index=False, height=290
            )

            cols = st.columns(2)
            if cols[0].button("删除所选", use_container_width=True):
                if any(edited_data["选择"]):

                    for idx, value in zip(
                        range(len(edited_data["选择"]) - 1, -1, -1),
                        edited_data["选择"][::-1],
                    ):
                        if value:
                            imp = state["db_data"]["images_path"][idx]
                            os.remove(imp)
                            del state["db_data"]["images_path"][idx]
                            for name in model_names:
                                if show_data[name][idx] == "✅":
                                    del state["db_data"]["features"][name][idx]

                    save_data(state["db_data"])
                    st.toast(":green[删除成功]", icon="✅")
                    st.rerun()
                else:
                    st.toast(":orange[未选择数据]", icon="🔊")

            if cols[1].button("清空数据库", use_container_width=True):
                shutil.rmtree("database")
                del state["db_data"]
                st.rerun()

            # st.table(edited_data)


if __name__ == "__main__":
    data_page()

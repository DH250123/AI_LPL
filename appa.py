# %%
import streamlit as st
import pandas as pd
import os

st.title("Teamsファイル操作アプリ")

uploaded_file = st.file_uploader("TeamsからCSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("読み込んだデータ:")
    st.dataframe(df)

    st.write("編集後のデータ:")
    edited_df = st.data_editor(df)
    
    if st.button("保存"):
        edited_df.to_csv("edited_file.csv", index=False)
        st.success("ファイルを保存しました（ローカル）")

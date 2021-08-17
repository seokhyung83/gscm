import streamlit as st
import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

def _call_db_info(db_table):
    return pymysql.connect(
        host = '10.94.77.9',
        port= 13306,
        user = 'dxbot1',
        password='ensol2020!',
        db = db_table,
        charset = 'utf8')

# st.set_page_config(layout="wide")
st.set_page_config(page_title='GSCM', page_icon="chart_with_upwards_trend", layout='wide',
                initial_sidebar_state='auto')

max_width = 1200
padding_top =0
padding_bottom = 0
padding_left =0
padding_right =0
COLOR = '#A50135'
BACKGROUND_COLOR ='#f1f1f1'


st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
)
#image = Image.open('/img/LGES_ver00.jpg')
#st.title('News Summary Report')
st.sidebar.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {max_width}px;
        padding-top: {padding_top}rem;
        padding-right: {padding_right}rem;
        padding-left: {padding_left}rem;
        padding-bottom: {padding_bottom}rem;
    }}
    .reportview-container .main {{
        color: {COLOR};
        background-color: {BACKGROUND_COLOR};
    }}
</style>
""",
        unsafe_allow_html=True,
)

tmp_product = ['E61D', 'P34', 'CBEV']
tmp_work_num = np.arange(202101, 202116)
menu = ["News", "Analysis"]
#st.sidebar.image('./img/LGES_ver00.jpg')
product = st.sidebar.selectbox("Product", tmp_product)
work_num = st.sidebar.selectbox("Work Week", tmp_work_num)
submit = st.sidebar.button("Search")


st.markdown("<h1 style='text-align: center; color: black;'>GSCM Report</h1>", unsafe_allow_html=True)

st.write( """
            ##
            """)


#_, r1c2, r1c3, r1c4 = st.beta_columns((4,2,2,2))
#r1c2.markdown("<h3 style='text-align: center; color: black;'>Date : </h3>", unsafe_allow_html=True)

if submit:
    st.write( """
                ##

                """)
    db_name = "gscm_"+product.lower()
    conn = _call_db_info(db_name)
    read = conn.cursor()
    tmp_rslt_sql = "select * from inven_day where ww=%s"%work_num
    tmp_ref_sql = "select  ww, inven from inven_rslt where ww >= %s"%work_num
    tmp_lt_sql = "select * from lt_day"
    read.execute(tmp_rslt_sql)    
    inven_d = pd.DataFrame(read.fetchall())
    read.execute(tmp_ref_sql)
    inven = pd.DataFrame(read.fetchall())
    read.execute(tmp_lt_sql)
    lt_d = pd.DataFrame(read.fetchall())    
    read.close()

    inven.columns = ['ww', 'inven_true']
    inven = inven.astype({'ww' : 'str'})
    lt_d.columns = ['ww', 'lt_d']
    lt_d = lt_d.astype({'ww' : 'str'})

    tmp_inven_day_rslt = [x.split(":") for x in inven_d[1].values.tolist()[0].split('/')]
    tmp_inven_day_rslt = pd.DataFrame(tmp_inven_day_rslt)
    tmp_inven_day_rslt.columns = ['ww', 'inven', 'inven_pred', 'sale', 'day']
    inven_day_rslt = tmp_inven_day_rslt[['inven', 'inven_pred', 'sale', 'day']]
    inven_day_rslt = inven_day_rslt.astype('float')
    inven_day_rslt['ww'] = [x[:6] for x in tmp_inven_day_rslt['ww']]
    inven_day_rslt['inven'] = inven_day_rslt['inven'].replace(-999, np.nan)
    tmp_s = inven_day_rslt.index[inven_day_rslt['ww'] == str(work_num)][0]
    tmp_e = inven_day_rslt.shape[0]

    inven_merge = pd.merge(inven_day_rslt, inven, how='left', on='ww')#[inven['ww'] >= 202101]
    col1 = np.where(inven_merge.columns.values == 'inven')[0][0]
    col2 = np.where(inven_merge.columns.values == 'inven_true')[0][0]
    inven_merge.iloc[tmp_s-1, col2] =inven_merge.iloc[tmp_s-1, col1]
    inven_merge1 = pd.merge(inven_merge, lt_d, how='left', on='ww')
    inven_merge1['drive_d'] =  [2] * inven_merge1.shape[0]
    inven_merge1['safe_d'] = inven_merge1['day'] - inven_merge1['lt_d'] - inven_merge1['drive_d']
    
    total_day = inven_merge1.loc[tmp_s:tmp_e].day
    period_day = inven_merge1.loc[tmp_s:tmp_e].lt_d
    drive_day = 2
    safe_day = total_day - period_day - drive_day

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=inven_merge1.loc[(tmp_s-7):tmp_e].ww, 
                            y=inven_merge1.loc[tmp_s:tmp_e].day, name="총 재고일수"))  #inven_merge1.loc[tmp_s:tmp_e].day
    fig_bar.add_trace(go.Bar(x=inven_merge1.loc[(tmp_s-7):tmp_e].ww, 
                            y=inven_merge1.loc[tmp_s:tmp_e].drive_d, name="운송재고 일수"))  #inven_merge1.loc[tmp_s:tmp_e].day
    fig_bar.add_trace(go.Bar(x=inven_merge1.loc[(tmp_s-7):tmp_e].ww, 
                            y=inven_merge1.loc[tmp_s:tmp_e].lt_d, name="주기재고 일수"))  #inven_merge1.loc[tmp_s:tmp_e].day
    fig_bar.add_trace(go.Bar(x=inven_merge1.loc[(tmp_s-7):tmp_e].ww, 
                            y=inven_merge1.loc[tmp_s:tmp_e].safe_d, name="안전 재고일수"))  #inven_merge1.loc[tmp_s:tmp_e].day
    fig_bar.update_layout(barmode='stack')

    #fig_bar.add_trace(go.Bar(x=inven_day_rslt.ww, y=inven_day_rslt.day))
    

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=inven_merge1.ww, y=inven_merge1.inven_pred, name='예측 재고량',
                                    line=dict(color='red', dash='dot')))
    fig_line.add_trace(go.Scatter(x=inven_merge1.ww, y=inven_merge1.inven, name='실적 재고량(학습)',
                                    line=dict(color='blue')))    
    fig_line.add_trace(go.Scatter(x=inven_merge1.ww, y=inven_merge1.inven_true, name='실적 재고량(검증)',
                                    line=dict(color='black')))   
    st.title("예측 된 총 재고량") 
    st.plotly_chart(fig_line, use_container_width=True)
    st.title("%s년 %s주 : 총 재고일수"%(str(work_num)[:4], str(work_num)[4:]))
    st.plotly_chart(fig_bar, use_container_width=True)





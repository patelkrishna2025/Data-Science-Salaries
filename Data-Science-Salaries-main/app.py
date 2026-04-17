import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Data Science Salary Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f2027,#203a43,#2c5364);}
[data-testid="stSidebar"] *{color:#e0e0e0!important;}
.main-title{font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:10px 0;}
.metric-card{background:linear-gradient(135deg,#1a1a2e,#16213e);border:1px solid #0f3460;border-radius:16px;padding:20px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.3);}
.metric-card h3{color:#a0a0c0;font-size:0.85rem;margin:0;text-transform:uppercase;letter-spacing:1px;}
.metric-card h1{color:#ffffff;font-size:2rem;margin:8px 0 0 0;font-weight:700;}
.sec{color:#7c83fd;font-size:1.2rem;font-weight:700;border-left:4px solid #7c83fd;padding-left:12px;margin:20px 0 10px 0;}
.cb-user{background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:12px 16px;border-radius:18px 18px 4px 18px;margin:8px 0;max-width:80%;float:right;clear:both;}
.cb-bot{background:#1a1a2e;border:1px solid #0f3460;color:#e0e0e0;padding:12px 16px;border-radius:18px 18px 18px 4px;margin:8px 0;max-width:80%;float:left;clear:both;}
.cf{clear:both;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data_ss.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
    if 'unnamed:_0' in df.columns:
        df.drop(columns=['unnamed:_0'], inplace=True)
    df['experience_level'] = df['experience_level'].map({'SE':'Senior','MI':'Mid-Level','EN':'Entry','EX':'Executive'}).fillna(df['experience_level'])
    df['employment_type'] = df['employment_type'].map({'FT':'Full-Time','PT':'Part-Time','CT':'Contract','FL':'Freelance'}).fillna(df['employment_type'])
    df['company_size'] = df['company_size'].map({'S':'Small','M':'Medium','L':'Large'}).fillna(df['company_size'])
    if 'remote_ratio' in df.columns:
        df['job_type'] = df['remote_ratio'].map({100:'Remote',50:'Hybrid',0:'On-site'}).fillna('Unknown')
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

with st.sidebar:
    st.markdown("## 🔍 Filters")
    exp_opts = sorted(df['experience_level'].dropna().unique())
    sel_exp = st.multiselect("Experience Level", exp_opts, default=exp_opts)
    emp_opts = sorted(df['employment_type'].dropna().unique())
    sel_emp = st.multiselect("Employment Type", emp_opts, default=emp_opts)
    sz_opts = sorted(df['company_size'].dropna().unique())
    sel_sz = st.multiselect("Company Size", sz_opts, default=sz_opts)
    jt_opts = sorted(df['job_type'].dropna().unique()) if 'job_type' in df.columns else []
    sel_jt = st.multiselect("Job Type", jt_opts, default=jt_opts) if jt_opts else []
    yr_min, yr_max = int(df['work_year'].min()), int(df['work_year'].max())
    sel_yr = st.slider("Work Year", yr_min, yr_max, (yr_min, yr_max))
    st.markdown("---")
    st.info(f"**Records:** {len(df)}\n\n**Columns:** {len(df.columns)}")

fdf = df[df['experience_level'].isin(sel_exp) & df['employment_type'].isin(sel_emp) & df['company_size'].isin(sel_sz) & df['work_year'].between(*sel_yr)]
if sel_jt and 'job_type' in df.columns:
    fdf = fdf[fdf['job_type'].isin(sel_jt)]

st.markdown('<div class="main-title">💼 Data Science Job Salaries Dashboard</div>', unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview","🔬 Deep Analysis","🤖 Salary Predictor","🖼️ CV Analysis","💬 Chatbot"])

# ── TAB 1 ──
with tab1:
    avg_sal = int(fdf['salary_in_usd'].fillna(0).mean())
    max_sal = int(fdf['salary_in_usd'].max()) if not fdf.empty else 0
    total_jobs = fdf['job_title'].nunique() if 'job_title' in fdf.columns else 0
    total_cntry = fdf['employee_residence'].nunique() if 'employee_residence' in fdf.columns else 0
    c1,c2,c3,c4 = st.columns(4)
    for col,lab,val,ico in zip([c1,c2,c3,c4],["Avg Salary","Max Salary","Job Titles","Countries"],[f"${avg_sal:,}",f"${max_sal:,}",total_jobs,total_cntry],["💵","🏆","🎯","🌍"]):
        col.markdown(f'<div class="metric-card"><h3>{ico} {lab}</h3><h1>{val}</h1></div>', unsafe_allow_html=True)
    st.markdown("")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec">Salary Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(fdf, x='salary_in_usd', nbins=40, color_discrete_sequence=['#667eea'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="sec">Experience Level Mix</div>', unsafe_allow_html=True)
        vc = fdf['experience_level'].value_counts().reset_index(); vc.columns=['Level','Count']
        fig = px.pie(vc, names='Level', values='Count', color_discrete_sequence=px.colors.sequential.Plasma_r, hole=0.4)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig, use_container_width=True)
    col3,col4 = st.columns(2)
    with col3:
        st.markdown('<div class="sec">Salary by Experience</div>', unsafe_allow_html=True)
        fig = px.box(fdf, x='experience_level', y='salary_in_usd', color='experience_level', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.markdown('<div class="sec">Avg Salary by Company Size</div>', unsafe_allow_html=True)
        cs = fdf.groupby('company_size')['salary_in_usd'].mean().reset_index()
        fig = px.bar(cs, x='company_size', y='salary_in_usd', color='company_size', color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="sec">Top 15 Job Titles by Avg Salary</div>', unsafe_allow_html=True)
    tj = fdf.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(15).reset_index()
    fig = px.bar(tj, x='salary_in_usd', y='job_title', orientation='h', color='salary_in_usd', color_continuous_scale='Viridis')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', height=450)
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2 ──
with tab2:
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec">Salary Trend by Year</div>', unsafe_allow_html=True)
        yt = fdf.groupby('work_year')['salary_in_usd'].mean().reset_index()
        fig = px.line(yt, x='work_year', y='salary_in_usd', markers=True, color_discrete_sequence=['#764ba2'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown('<div class="sec">Employment Type Distribution</div>', unsafe_allow_html=True)
        et = fdf['employment_type'].value_counts().reset_index(); et.columns=['Type','Count']
        fig = px.bar(et, x='Type', y='Count', color='Type', color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    if 'job_type' in fdf.columns:
        col3,col4 = st.columns(2)
        with col3:
            st.markdown('<div class="sec">Salary by Job Type</div>', unsafe_allow_html=True)
            fig = px.violin(fdf, x='job_type', y='salary_in_usd', color='job_type', box=True, color_discrete_sequence=px.colors.qualitative.Prism)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.markdown('<div class="sec">Remote Work Distribution</div>', unsafe_allow_html=True)
            jp = fdf['job_type'].value_counts().reset_index(); jp.columns=['Type','Count']
            fig = px.pie(jp, names='Type', values='Count', hole=0.45, color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="sec">Correlation Heatmap</div>', unsafe_allow_html=True)
    nc = fdf.select_dtypes(include=[np.number]).columns.tolist()
    if len(nc) > 1:
        fig = px.imshow(fdf[nc].corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect='auto', zmin=-1, zmax=1)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig, use_container_width=True)
    if 'employee_residence' in fdf.columns:
        st.markdown('<div class="sec">Global Salary Map</div>', unsafe_allow_html=True)
        cm = fdf.groupby('employee_residence')['salary_in_usd'].mean().reset_index()
        fig = px.choropleth(cm, locations='employee_residence', color='salary_in_usd', locationmode='ISO-3', color_continuous_scale='Plasma')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0', geo_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="sec">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(fdf.head(50), use_container_width=True)

# ── TAB 3 ──
with tab3:
    st.markdown('<div class="sec">🤖 ML Salary Predictor (Random Forest)</div>', unsafe_allow_html=True)
    @st.cache_resource
    def train_model(df_hash):
        features = ['experience_level','employment_type','company_size','job_title']
        if 'job_type' in df.columns: features.append('job_type')
        if 'work_year' in df.columns: features.append('work_year')
        mdf = df[features+['salary_in_usd']].dropna().copy()
        encs = {}
        for col in features:
            if mdf[col].dtype == object:
                le = LabelEncoder()
                mdf[col] = le.fit_transform(mdf[col].astype(str))
                encs[col] = le
        X = mdf[features]; y = mdf['salary_in_usd']
        Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
        m = RandomForestRegressor(n_estimators=100, random_state=42)
        m.fit(Xtr,ytr)
        return m, encs, round(m.score(Xte,yte),3), features

    mdl, encs, score, feat_cols = train_model(len(df))
    st.success(f"✅ Model trained | R² Score: **{score}**")
    pc1,pc2,pc3 = st.columns(3)
    with pc1:
        pred_exp = st.selectbox("Experience Level", sorted(df['experience_level'].dropna().unique()))
        pred_emp = st.selectbox("Employment Type", sorted(df['employment_type'].dropna().unique()))
    with pc2:
        pred_sz = st.selectbox("Company Size", sorted(df['company_size'].dropna().unique()))
        pred_job = st.selectbox("Job Title", sorted(df['job_title'].dropna().unique()))
    with pc3:
        pred_jt = st.selectbox("Job Type", sorted(df['job_type'].dropna().unique())) if 'job_type' in df.columns else None
        pred_yr = st.selectbox("Work Year", sorted(df['work_year'].dropna().unique(), reverse=True))
    if st.button("🔮 Predict Salary", use_container_width=True):
        inp = {'experience_level':pred_exp,'employment_type':pred_emp,'company_size':pred_sz,'job_title':pred_job}
        if 'job_type' in feat_cols and pred_jt: inp['job_type'] = pred_jt
        if 'work_year' in feat_cols: inp['work_year'] = int(pred_yr)
        row = []
        for col in feat_cols:
            v = inp.get(col)
            if col in encs:
                v = encs[col].transform([str(v)])[0] if str(v) in encs[col].classes_ else 0
            row.append(v)
        pred = mdl.predict([row])[0]
        st.markdown(f'<div style="background:linear-gradient(135deg,#667eea,#764ba2);border-radius:16px;padding:30px;text-align:center;margin-top:20px;"><h2 style="color:white;margin:0;">💡 Predicted Salary</h2><h1 style="color:white;font-size:3rem;margin:10px 0;">${pred:,.0f}</h1><p style="color:#e0d0ff;">USD per year</p></div>', unsafe_allow_html=True)
        fi = pd.DataFrame({'Feature':feat_cols,'Importance':mdl.feature_importances_}).sort_values('Importance')
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Plasma', title="Feature Importance")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#e0e0e0')
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 4 ──
with tab4:
    st.markdown('<div class="sec">🖼️ Computer Vision Analysis</div>', unsafe_allow_html=True)
    uploaded_img = st.file_uploader("Upload an Image", type=["png","jpg","jpeg","bmp","webp"])
    if uploaded_img:
        fbytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(fbytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h,w,ch = img_rgb.shape
        cv1,cv2c,cv3 = st.columns(3)
        cv1.metric("Width (px)", w); cv2c.metric("Height (px)", h); cv3.metric("Channels", ch)
        cv_ops = st.multiselect("Select CV Operations:", ["Original","Grayscale","Edge Detection (Canny)","Blur (Gaussian)","Sharpen","Emboss","Threshold (Binary)","Sepia Filter","Invert Colors","Color Histogram"], default=["Original","Grayscale","Edge Detection (Canny)"])
        def sepia(img):
            k = np.array([[0.272,0.534,0.131],[0.349,0.686,0.168],[0.393,0.769,0.189]])
            return np.clip(img @ k.T, 0, 255).astype(np.uint8)
        results = {}
        for op in cv_ops:
            if op=="Original": results[op]=img_rgb
            elif op=="Grayscale": results[op]=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            elif op=="Edge Detection (Canny)":
                g=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY); results[op]=cv2.Canny(g,100,200)
            elif op=="Blur (Gaussian)": results[op]=cv2.GaussianBlur(img_rgb,(15,15),0)
            elif op=="Sharpen":
                k=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]); results[op]=cv2.filter2D(img_rgb,-1,k)
            elif op=="Emboss":
                k=np.array([[-2,-1,0],[-1,1,1],[0,1,2]]); g=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY); results[op]=cv2.filter2D(g,-1,k)
            elif op=="Threshold (Binary)":
                g=cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY); _,results[op]=cv2.threshold(g,127,255,cv2.THRESH_BINARY)
            elif op=="Sepia Filter": results[op]=sepia(img_rgb)
            elif op=="Invert Colors": results[op]=cv2.bitwise_not(img_rgb)
            elif op=="Color Histogram":
                fig,ax=plt.subplots(figsize=(7,3)); fig.patch.set_alpha(0); ax.set_facecolor('#1a1a2e')
                for i,c in enumerate(['red','green','blue']):
                    hist=cv2.calcHist([img_rgb],[i],None,[256],[0,256]); ax.plot(hist,color=c,linewidth=1.5,alpha=0.8)
                ax.set_title("Color Histogram",color='white'); ax.tick_params(colors='white')
                buf=io.BytesIO(); plt.savefig(buf,format='png',bbox_inches='tight',facecolor='#1a1a2e'); buf.seek(0); results[op]=Image.open(buf); plt.close()
        non_hist = {k:v for k,v in results.items() if k!="Color Histogram"}
        if non_hist:
            ncols = min(3,len(non_hist))
            cols = st.columns(ncols)
            for i,(name,img_out) in enumerate(non_hist.items()):
                with cols[i%ncols]:
                    st.markdown(f"**{name}**")
                    st.image(img_out, use_column_width=True)
        if "Color Histogram" in results:
            st.markdown("**Color Histogram**"); st.image(results["Color Histogram"], use_column_width=True)
        st.markdown("### 📊 Pixel Statistics")
        stats_df = pd.DataFrame({'Channel':['Red','Green','Blue'],'Mean':[img_rgb[:,:,i].mean().round(2) for i in range(3)],'Std':[img_rgb[:,:,i].std().round(2) for i in range(3)],'Min':[img_rgb[:,:,i].min() for i in range(3)],'Max':[img_rgb[:,:,i].max() for i in range(3)]})
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("👆 Upload an image to start CV Analysis\n\n**Available:** Grayscale • Edge Detection • Blur • Sharpen • Emboss • Sepia • Invert • Histogram • Binary Threshold")

# ── TAB 5 ──
with tab5:
    st.markdown('<div class="sec">💬 Data Science Salary Chatbot</div>', unsafe_allow_html=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [("bot","👋 Hello! I'm your **Data Science Salary Assistant**.\n\nTry asking:\n- *Average salary for senior roles?*\n- *Top paying job titles?*\n- *How many remote jobs?*\n- *Salary trend by year?*")]
    def bot_reply(q, df):
        ql = q.lower()
        if any(x in ql for x in ['hello','hi','hey','howdy']): return "👋 Hi! Ask me about Data Science salaries!"
        if 'help' in ql: return "I can answer:\n- Avg/Max salaries by level\n- Top paying job titles\n- Remote job stats\n- Company size comparison\n- Salary trends by year\n- Country-wise salary"
        if 'senior' in ql: v=df[df['experience_level']=='Senior']['salary_in_usd'].mean(); return f"📊 Senior avg salary: **${v:,.0f}** USD/year"
        if 'mid' in ql: v=df[df['experience_level']=='Mid-Level']['salary_in_usd'].mean(); return f"📊 Mid-Level avg salary: **${v:,.0f}** USD/year"
        if 'entry' in ql: v=df[df['experience_level']=='Entry']['salary_in_usd'].mean(); return f"📊 Entry Level avg salary: **${v:,.0f}** USD/year"
        if 'executive' in ql or 'exec' in ql: v=df[df['experience_level']=='Executive']['salary_in_usd'].mean(); return f"📊 Executive avg salary: **${v:,.0f}** USD/year"
        if ('top' in ql or 'highest' in ql or 'best' in ql) and ('job' in ql or 'role' in ql or 'title' in ql):
            top=df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(5)
            return "🏆 **Top 5 Highest Paying Roles:**\n" + "\n".join([f"- {t}: **${v:,.0f}**" for t,v in top.items()])
        if 'remote' in ql and 'job_type' in df.columns:
            cnt=df[df['job_type']=='Remote'].shape[0]; pct=cnt/len(df)*100; avg=df[df['job_type']=='Remote']['salary_in_usd'].mean()
            return f"🏠 Remote jobs: **{cnt}** ({pct:.1f}%)\nAvg Remote Salary: **${avg:,.0f}** USD"
        if 'company' in ql and 'size' in ql:
            return "🏢 **Avg Salary by Company Size:**\n" + "\n".join([f"- {s}: **${df[df['company_size']==s]['salary_in_usd'].mean():,.0f}**" for s in ['Small','Medium','Large']])
        if 'average' in ql or 'avg' in ql or 'mean' in ql:
            v=df['salary_in_usd'].mean(); return f"📊 Overall avg salary: **${v:,.0f}** USD/year"
        if 'max' in ql or 'maximum' in ql or 'highest' in ql:
            v=df['salary_in_usd'].max(); r=df.loc[df['salary_in_usd'].idxmax()]
            return f"🏆 Max salary: **${v:,.0f}** USD\nRole: **{r.get('job_title','N/A')}**"
        if 'year' in ql or 'trend' in ql:
            yt=df.groupby('work_year')['salary_in_usd'].mean()
            return "📅 **Salary Trend:**\n" + "\n".join([f"- {yr}: **${v:,.0f}**" for yr,v in yt.items()])
        if 'data scientist' in ql:
            m=df[df['job_title'].str.contains('Data Scientist',case=False,na=False)]
            return f"🔬 Data Scientist: **{len(m)}** roles\nAvg: **${m['salary_in_usd'].mean():,.0f}** USD"
        if 'how many' in ql or 'count' in ql:
            if 'job' in ql: return f"🎯 **{df['job_title'].nunique()}** unique job titles"
            if 'country' in ql: return f"🌍 Employees from **{df['employee_residence'].nunique()}** countries"
            return f"📋 Total records: **{len(df)}**"
        matches = df[df['job_title'].str.contains('|'.join([w for w in ql.split() if len(w)>3]),case=False,na=False)] if any(len(w)>3 for w in ql.split()) else pd.DataFrame()
        if len(matches) > 0: return f"🔍 Found **{len(matches)}** matching roles.\nAvg Salary: **${matches['salary_in_usd'].mean():,.0f}** USD"
        return "🤔 Try: *'average salary senior'*, *'top job titles'*, *'remote jobs'*, *'salary trend'*"
    for role,msg in st.session_state.chat_history:
        css = "cb-user" if role=="user" else "cb-bot"
        st.markdown(f'<div class="{css}">{msg}</div><div class="cf"></div>', unsafe_allow_html=True)
    col_i,col_b = st.columns([5,1])
    with col_i:
        user_q = st.text_input("Ask...", key="chat_input", label_visibility="collapsed", placeholder="e.g. Average salary for senior data scientists?")
    with col_b:
        if st.button("Send 🚀", use_container_width=True) and user_q.strip():
            st.session_state.chat_history.append(("user", user_q))
            st.session_state.chat_history.append(("bot", bot_reply(user_q, df)))
            st.rerun()
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = [("bot","👋 Chat cleared! Ask me anything about DS salaries.")]
        st.rerun()

st.markdown("---")
st.markdown('<div style="text-align:center;color:#666;font-size:0.8rem;">📊 Data Science Salary Dashboard | Streamlit + Plotly + OpenCV + Scikit-learn</div>', unsafe_allow_html=True)

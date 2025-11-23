# -*- coding: utf-8 -*-
from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os

def _download(df, label, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )

# =====================================================
# CONFIG BÁSICO
# =====================================================
st.set_page_config(page_title="PIB per capita — Bayes (Gamma) & EDA", layout="wide")
BASE_DIR = Path("/content/drive/MyDrive/PIB_Forecast")
PLOTLY_TEMPLATE = "plotly_white"

NGROK_URL = os.getenv("NGROK_URL", "")
if NGROK_URL:
    st.success(f"🔗 Link externo do app (ngrok): **{NGROK_URL}**")
else:
    st.info("🔌 Rode a célula no Colab que inicia o túnel do ngrok para ter o link externo.")

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
html, body, [class*="css"]{
    font-family:'Inter',system-ui,-apple-system,Segoe UI,Roboto,sans-serif
}
.section{
    border-radius:18px;
    border:1px solid rgba(0,0,0,.06);
    padding:18px;
    background:#fff;
    box-shadow:0 8px 24px rgba(0,0,0,.04)
}
.metric-card{
    border-radius:16px;
    padding:16px 18px;
    border:1px solid rgba(0,0,0,.06);
    background:linear-gradient(180deg,rgba(0,0,0,.03),rgba(0,0,0,.015));
    box-shadow:0 4px 12px rgba(0,0,0,.05)
}
.small{color:#666;font-size:.9rem}
h1,h2,h3{letter-spacing:.2px}
hr{margin:.6rem 0 1rem 0;border-color:rgba(0,0,0,.08)}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPERS DE LEITURA + SANITIZAÇÃO
# =====================================================

# --- (cole aqui as funções sanitize_columns, normalize_keys, etc.) ---
# elas precisam estar ACIMA da função load_regression_data()

def sanitize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("ç", "c")
        .str.replace("ã", "a")
        .str.replace("á", "a")
        .str.replace("é", "e")
        .str.replace("í", "i")
    )
    return df

def normalize_keys(df):
    df = df.copy()
    if "cod_mun" in df.columns:
        df["cod_mun"] = df["cod_mun"].astype(str).str.zfill(7)
    if "ano_pib" in df.columns:
        df["ano_pib"] = df["ano_pib"].astype(int)
    return df

@st.cache_data(show_spinner=True)
def load_regression_data():
    """
    Carrega e sanitiza:
      - base_regressao_PIBpc_composicao.csv
      - efeitos_municipais_PIBpc_BAYES.csv
      - metrics_bayes_PIBpc_summary.csv
    """

    # =====================================================
    # 1) CARREGAR ARQUIVOS RAW
    # =====================================================
    df_reg_raw = pd.read_csv(BASE_DIR / "base_regressao_PIBpc_composicao.csv")
    df_mun_raw = pd.read_csv(BASE_DIR / "efeitos_municipais_PIBpc_BAYES.csv")
    summary    = pd.read_csv(BASE_DIR / "metrics_bayes_PIBpc_summary.csv", index_col=0)

    # =====================================================
    # 2) SANITIZAÇÃO PROFUNDA
    # =====================================================
    df_reg = sanitize_columns(df_reg_raw)
    df_mun = sanitize_columns(df_mun_raw)

    df_reg = normalize_keys(df_reg)
    df_mun = normalize_keys(df_mun)

    # =====================================================
    # 3) AJUSTES PADRONIZADOS
    # =====================================================
    df_reg["cod_mun"] = df_reg["cod_mun"].astype(str).str.zfill(7)
    df_mun["cod_mun"] = df_mun["cod_mun"].astype(str).str.zfill(7)

    if "nome_municipio" not in df_reg.columns:
        df_reg["nome_municipio"] = df_reg["cod_mun"]
    if "nome_municipio" not in df_mun.columns:
        df_mun["nome_municipio"] = df_mun["cod_mun"]

    # =====================================================
    # 4) MERGE DOS EFEITOS MUNICIPAIS
    # =====================================================
    if "alpha_mun_mean" not in df_mun.columns:
        raise ValueError("Coluna alpha_mun_mean não encontrada em efeitos municipais.")

    df_reg = df_reg.merge(
        df_mun[["cod_mun", "alpha_mun_mean"]],
        how="left",
        on="cod_mun"
    )

    # =====================================================
    # 5) EXTRAÇÃO DOS PARÂMETROS DO SUMMARY
    # =====================================================
    def _p(name):
        if name in summary.index:
            return float(summary.loc[name, "mean"]), float(summary.loc[name, "sd"])
        return np.nan, np.nan

    mu_alpha_mean, mu_alpha_sd  = _p("mu_alpha")
    sigma_alpha_mean, sigma_alpha_sd = _p("sigma_alpha")
    beta_agro_mean, beta_agro_sd = _p("beta_agro")
    beta_ind_mean,  beta_ind_sd  = _p("beta_ind")
    beta_adm_mean,  beta_adm_sd  = _p("beta_adm")
    sigma_mean,     sigma_sd     = _p("sigma")

    # =====================================================
    # 6) GAMMA DE ANO
    # =====================================================
    anos = sorted(df_reg["ano_pib"].dropna().astype(int).unique())

    gamma = summary[summary.index.str.contains(r"gamma_t\[")].copy()
    gamma["idx"] = gamma.index.str.extract(r"(\d+)").astype(int)
    gamma = gamma.sort_values("idx")

    gamma_by_year = {}
    for i, ano in enumerate(anos):
        if i < len(gamma):
            gamma_by_year[ano] = float(gamma.iloc[i]["mean"])

    df_reg["gamma_year"] = df_reg["ano_pib"].map(gamma_by_year)

    # =====================================================
    # 7) PREVISÃO (usa r_agro, r_ind, r_adm se existirem)
    # =====================================================
    for c in ["r_agro", "r_ind", "r_adm"]:
        if c not in df_reg.columns:
            df_reg[c] = 0.0

    df_reg["y_hat_mean"] = (
        df_reg["alpha_mun_mean"] +
        df_reg["gamma_year"].fillna(0) +
        beta_agro_mean * df_reg["r_agro"] +
        beta_ind_mean  * df_reg["r_ind"] +
        beta_adm_mean  * df_reg["r_adm"]
    )

    df_reg["resid"] = df_reg["vl_pib_per_capta"] - df_reg["y_hat_mean"]

    if np.isnan(sigma_mean) or sigma_mean <= 0:
        sigma_mean = df_reg["resid"].std()

    df_reg["y_hat_lo"] = df_reg["y_hat_mean"] - 1.96 * sigma_mean
    df_reg["y_hat_hi"] = df_reg["y_hat_mean"] + 1.96 * sigma_mean

    # =====================================================
    # 8) RETORNO FINAL
    # =====================================================
    params = {
        "mu_alpha": mu_alpha_mean,
        "sigma_alpha": sigma_alpha_mean,
        "beta_agro": beta_agro_mean,
        "beta_ind":  beta_ind_mean,
        "beta_adm":  beta_adm_mean,
        "sigma":     sigma_mean,
        "gamma_by_year": gamma_by_year,
    }

    return df_reg, df_mun, summary, params

# =====================================================
#  CARREGAR BASES
# =====================================================
df_reg, df_mun, summary, params = load_regression_data()

# =====================================================
#  FILTROS LATERAIS
# =====================================================
st.sidebar.header("Filtros")

anos_disp = sorted(df_reg["ano_pib"].dropna().unique())
setores_disp = ["Todos", "Agropecuária", "Indústria", "Serviços", "Administração Pública"]

ano_selecionado = st.sidebar.multiselect(
    "Ano do PIB",
    options=anos_disp,
    default=anos_disp
)

df_filt = df_reg[df_reg["ano_pib"].isin(ano_selecionado)].copy()

if df_filt.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")

# =====================================================
# TABS PRINCIPAIS
# =====================================================
tab_eda, tab_modelo, tab_municipios = st.tabs(
    ["✨ EDA & escolha da distribuição", "🧠 Modelo Bayesiano (betas, prior × posterior)", "🏙️ Efeitos por município"]
)

# =====================================================
# TAB 1 — EDA & ESCOLHA DA DISTRIBUIÇÃO (GAMMA)
# =====================================================
with tab_eda:
    if df_filt.empty:
        st.info("Sem dados para os filtros atuais.")
        st.stop()

    left, right = st.columns(2, gap="large")

    # ---------- Métricas gerais ----------
    with left:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Visão geral do PIB per capita")

        vals = df_filt["vl_pib_per_capta"].dropna()

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(
            f'<div class="metric-card"><div class="small">Observações</div>'
            f'<h3>{len(vals):,}</h3></div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="metric-card"><div class="small">Média</div>'
            f'<h3>{vals.mean():,.0f}</h3></div>'.replace(",", "."),
            unsafe_allow_html=True,
        )
        m3.markdown(
            f'<div class="metric-card"><div class="small">Mediana</div>'
            f'<h3>{vals.median():,.0f}</h3></div>'.replace(",", "."),
            unsafe_allow_html=True,
        )
        m4.markdown(
            f'<div class="metric-card"><div class="small">Desvio-padrão</div>'
            f'<h3>{vals.std(ddof=0):,.0f}</h3></div>'.replace(",", "."),
            unsafe_allow_html=True,
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        # ---------- Histograma + ajuste Gamma ----------
        st.markdown("### Histograma do PIB per capita e ajuste da distribuição Gamma")

        mean_y = vals.mean()
        std_y = vals.std(ddof=0)
        if std_y <= 0 or mean_y <= 0:
            st.warning("Não foi possível ajustar Gamma (média ou desvio = 0).")
        else:
            k = (mean_y / std_y) ** 2
            theta = (std_y ** 2) / mean_y

            x_min = max(vals.min(), 1e-6)
            x_max = vals.max()
            x_grid = np.linspace(x_min, x_max, 300)

            coef = 1.0 / (math.gamma(k) * (theta ** k))
            gamma_pdf = coef * (x_grid ** (k - 1)) * np.exp(-x_grid / theta)

            fig_hist = go.Figure()

            fig_hist.add_trace(
                go.Histogram(
                    x=vals,
                    histnorm="probability density",
                    name="Dados",
                    nbinsx=40,
                    opacity=0.6,
                )
            )
            fig_hist.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=gamma_pdf,
                    mode="lines",
                    name=f"Gamma ajustada (k={k:.2f}, θ={theta:.2f})",
                )
            )
            fig_hist.update_layout(
                template=PLOTLY_TEMPLATE,
                title="PIB per capita — histograma vs. densidade Gamma ajustada",
                xaxis_title="PIB per capita",
                yaxis_title="Densidade",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.caption(
                f"""
                • **Média** = {mean_y:,.0f}, **Desvio** = {std_y:,.0f}  
                • Parâmetros da Gamma (método dos momentos):  **k = {k:.2f}**, **θ = {theta:.2f}**.  

                O formato assimétrico, com longa cauda à direita e concentração de muitos municípios
                em faixas baixas de PIB per capita, é compatível com distribuições da família exponencial,
                como a Gamma. Isso justifica o uso de um modelo Bayesiano com verossimilhança Gamma para 
                o PIB per capita municipal.
                """.replace(",", ".")
            )

        _download(df_filt, "⬇️ Baixar recorte (CSV)", "base_regressao_filtrada.csv")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Relação composição × PIB per capita ----------
    with right:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Composição setorial × PIB per capita")

        eixo_x = st.selectbox(
            "Escolha a razão (em relação a Serviços) para o eixo X:",
            [
                ("r_ind", "Indústria / Serviços"),
                ("r_agro", "Agropecuária / Serviços"),
                ("r_adm", "Administração Pública / Serviços"),
            ],
            format_func=lambda x: x[1],
        )
        col_x = eixo_x[0]
        label_x = eixo_x[1]

        df_plot = df_filt.copy()
        # garantir nome_municipio como string para hover
        df_plot["nome_municipio"] = df_plot["nome_municipio"].astype(str)

        fig_scatter = px.scatter(
            df_plot,
            x=col_x,
            y="vl_pib_per_capta",
            color="ano_pib",
            template=PLOTLY_TEMPLATE,
            labels={
                col_x: label_x,
                "vl_pib_per_capta": "PIB per capita",
                "ano_pib": "Ano",
            },
            title=f"PIB per capita vs {label_x}",
        )
        fig_scatter.update_traces(
            mode="markers",
            marker=dict(size=5, opacity=0.7),
            hovertemplate="Município=%{customdata[0]}<br>"
                          + label_x + "=%{x:.2f}<br>PIBpc=%{y:,.0f}<extra></extra>",
            customdata=np.stack(
                [df_plot["nome_municipio"].values], axis=-1
            ),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Distribuição do PIB per capita por ano")
        fig_box = px.box(
            df_reg,
            x="ano_pib",
            y="vl_pib_per_capta",
            template=PLOTLY_TEMPLATE,
            labels={"ano_pib": "Ano", "vl_pib_per_capta": "PIB per capita"},
            title="Boxplot do PIB per capita por ano",
        )
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("""
        <p style='font-size:14px; color:#6e6e6e; line-height:1.45;'>
        Os três gráficos mostram que o PIB per capita municipal no Brasil segue um padrão 
        altamente assimétrico: a maior parte dos municípios tem baixa produtividade e baixa 
        diversificação econômica, enquanto poucos municípios concentram níveis muito altos de PIB per capita.
        <br><br>
        Quando observamos as razões setoriais em relação a serviços (Indústria/Serviços, Agro/Serviços, 
        Administração/Serviços), é possível perceber que <strong>os pontos dos anos mais antigos</strong> 
        (tons arroxeados) tendem a apresentar valores relativamente mais altos nessas razões. Isso indica que, 
        no passado, setores como indústria, agropecuária e administração pública tinham participação proporcional 
        mais relevante no PIB local.
        <br><br>
        Nos anos mais recentes (tons amarelados), há uma convergência: a maior parte dos municípios passa a ter 
        proporções menores desses setores em relação aos serviços, sugerindo uma <strong>dependência crescente 
        do setor de serviços</strong>. Em termos da pergunta central — como a composição setorial se relaciona 
        com o PIB per capita — isso significa que:
        <br>
        • a estrutura econômica está cada vez mais concentrada em serviços; <br>
        • setores tradicionalmente associados a maior valor agregado (indústria complexa, agro intensivo, etc.) 
          têm peso relativo menor; <br>
        • isso reforça a assimetria observada na distribuição do PIB per capita e a adequação de modelos do 
          tipo Gamma para descrever esses dados.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# TAB 2 — MODELO BAYESIANO: BETAS, PRIOR × POSTERIOR
# =====================================================
with tab_modelo:
    left, right = st.columns(2, gap="large")

    # ---------- Tabela resumo ----------
    with left:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Resumo do modelo Bayesiano (principais parâmetros)")

        focus_index = [
            "mu_alpha",
            "sigma_alpha",
            "beta_agro",
            "beta_ind",
            "beta_adm",
            "sigma",
        ]
        focus_rows = [idx for idx in focus_index if idx in summary.index]
        df_sum_focus = summary.loc[focus_rows].copy()
        df_sum_focus = df_sum_focus[["mean", "sd", "hdi_3%", "hdi_97%"]]
        df_sum_focus = df_sum_focus.rename(
            columns={
                "mean": "média posterior",
                "sd": "desvio posterior",
                "hdi_3%": "HDI 3%",
                "hdi_97%": "HDI 97%",
            }
        )
        st.dataframe(df_sum_focus, use_container_width=True)
        _download(df_sum_focus.reset_index(), "⬇️ Baixar resumo (CSV)", "metrics_bayes_PIBpc_summary_focus.csv")

        st.markdown(
            """
            **Leitura rápida dos betas:**
            - `beta_ind` > 0  → aumentar a participação relativa da **Indústria** (vs Serviços)
              tende a aumentar o PIB per capita, em média.
            - `beta_agro` e `beta_adm` próximos de 0  → o efeito de Agropecuária e Administração Pública,
              isoladamente, é menos claro e vem acompanhado de maior incerteza.
            """,
            unsafe_allow_html=False,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Prior × Posterior ----------
    with right:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("Distribuições priori × posteriori (aproximação Normal)")

        priors = {
            "mu_alpha": {"type": "normal", "mu": 20000, "sigma": 10000},
            "beta_agro": {"type": "normal", "mu": 0, "sigma": 1000},
            "beta_ind":  {"type": "normal", "mu": 0, "sigma": 1000},
            "beta_adm":  {"type": "normal", "mu": 0, "sigma": 1000},
        }

        param_names = [p for p in priors.keys() if p in summary.index]
        if not param_names:
            st.info("Parâmetros para prior × posterior não encontrados.")
        else:
            selecionado = st.selectbox(
                "Escolha o parâmetro:",
                param_names,
                format_func=lambda x: x,
            )

            prior = priors[selecionado]
            post_mean = float(summary.loc[selecionado, "mean"])
            post_sd = float(summary.loc[selecionado, "sd"])

            x_min = post_mean - 4 * post_sd
            x_max = post_mean + 4 * post_sd
            x = np.linspace(x_min, x_max, 400)

            mu0, s0 = prior["mu"], prior["sigma"]
            prior_y = (
                np.exp(-(x - mu0) ** 2 / (2 * s0 ** 2))
                / (np.sqrt(2 * np.pi) * s0)
            )

            post_y = (
                np.exp(-(x - post_mean) ** 2 / (2 * post_sd ** 2))
                / (np.sqrt(2 * np.pi) * post_sd)
            )

            fig_pp = go.Figure()
            fig_pp.add_trace(
                go.Scatter(
                    x=x,
                    y=prior_y,
                    mode="lines",
                    name="Prior (Normal)",
                )
            )
            fig_pp.add_trace(
                go.Scatter(
                    x=x,
                    y=post_y,
                    mode="lines",
                    name="Posterior (aprox. Normal)",
                )
            )
            fig_pp.update_layout(
                template=PLOTLY_TEMPLATE,
                title=f"Prior × Posterior — {selecionado}",
                xaxis_title="valor do parâmetro",
                yaxis_title="densidade",
            )
            st.plotly_chart(fig_pp, use_container_width=True)

            st.caption(
                f"""
                - Prior: Normal({mu0:.0f}, {s0:.0f}²)  
                - Posterior (aprox.): Normal({post_mean:.2f}, {post_sd:.2f}²)  

                Os parâmetros beta associados aos setores (indústria, agropecuária e administração pública) 
                permanecem relativamente baixos no modelo. Isso indica que, quando comparados ao setor de serviços, 
                nenhum setor isoladamente é capaz de explicar grandes variações do PIB per capita municipal.
                Na prática, isso reflete a realidade de que a maior parte dos municípios brasileiros tem uma estrutura
                econômica muito parecida — fortemente dependente de serviços — e que os diferenciais de PIB per capita
                vêm de fatores estruturais mais amplos (produtividade, complexidade econômica, urbanização, clusters produtivos etc.).
                """.replace(",", ".")
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Avaliação do ajuste ----------
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Avaliação do ajuste: observados vs previstos")

    df_av = df_reg.dropna(subset=["vl_pib_per_capta", "y_hat_mean"]).copy()
    y = df_av["vl_pib_per_capta"].values
    y_hat = df_av["y_hat_mean"].values
    resid = df_av["resid"].values

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-card"><div class="small">R² aproximado</div>'
        f'<h3>{r2:.3f}</h3></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="metric-card"><div class="small">MAE</div>'
        f'<h3>{np.mean(np.abs(resid)):,.0f}</h3></div>'.replace(",", "."),
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="metric-card"><div class="small">RMSE</div>'
        f'<h3>{np.sqrt(np.mean(resid**2)):,.0f}</h3></div>'.replace(",", "."),
        unsafe_allow_html=True,
    )

    df_sample = df_av.sample(min(len(df_av), 5000), random_state=42)

    fig_fit = px.scatter(
        df_sample,
        x="vl_pib_per_capta",
        y="y_hat_mean",
        color="ano_pib",
        template=PLOTLY_TEMPLATE,
        labels={
            "vl_pib_per_capta": "PIB per capita observado",
            "y_hat_mean": "PIB per capita previsto (média posterior)",
            "ano_pib": "Ano",
        },
        title="Observado vs previsto (subamostra para visualização)",
    )
    max_val = max(df_av["vl_pib_per_capta"].max(), df_av["y_hat_mean"].max())
    fig_fit.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="linha 45°",
            line=dict(dash="dash", color="black"),
        )
    )
    st.plotly_chart(fig_fit, use_container_width=True)

    st.markdown("### Distribuição dos resíduos (y − ŷ)")
    fig_res = px.histogram(
        df_av,
        x="resid",
        nbins=40,
        template=PLOTLY_TEMPLATE,
        labels={"resid": "Resíduo"},
        title="Histograma dos resíduos",
    )
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("""
    <p style='font-size:14px; color:#6e6e6e; line-height:1.45;'>
    O modelo apresenta um ajuste consistente, com um R² razoável e erro médio (MAE) relativamente baixo
    para a maior parte dos municípios. O RMSE mais elevado reflete a presença de <em>outliers</em> — 
    municípios com PIB per capita extremamente alto, que não podem ser previstos apenas a partir da 
    composição setorial relativa.
    <br><br>
    Isso mostra que o modelo captura bem a dinâmica estrutural da maioria dos municípios, 
    mas tem dificuldade com casos extremos. Nesses casos, variações enormes no PIB per capita dependem 
    de fatores que não estão incluídos explicitamente no modelo (mineração, polos industriais específicos, 
    grandes obras de infraestrutura, choques regionais etc.).
    </p>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# TAB 3 — EFEITOS MUNICIPAIS
# =====================================================
with tab_municipios:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    st.subheader("Efeitos de município (αᵢ) no PIB per capita")

    # coluna amigável para exibição
    df_mun_plot = df_mun.copy()
    df_mun_plot["nome_label"] = df_mun_plot["nome_municipio"].fillna(df_mun_plot["cod_mun"]).astype(str)

    # ordenar por efeito médio
    df_mun_plot = df_mun_plot.sort_values("alpha_mun_mean", ascending=False)

    top_k = st.slider("Top K municípios por efeito αᵢ", 5, 50, 15, 1)

    df_top = df_mun_plot.head(top_k)
    df_bottom = df_mun_plot.tail(top_k).sort_values("alpha_mun_mean", ascending=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Maiores efeitos (αᵢ mais altos)")
        fig_top = px.bar(
            df_top,
            x="alpha_mun_mean",
            y="nome_label",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            labels={
                "alpha_mun_mean": "Efeito municipal médio (αᵢ)",
                "nome_label": "Município",
            },
            title="Top municípios — maior nível estrutural de PIB per capita",
        )
        fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_top, use_container_width=True)

    with c2:
        st.markdown("#### Menores efeitos (αᵢ mais baixos)")
        fig_bot = px.bar(
            df_bottom,
            x="alpha_mun_mean",
            y="nome_label",
            orientation="h",
            template=PLOTLY_TEMPLATE,
            labels={
                "alpha_mun_mean": "Efeito municipal médio (αᵢ)",
                "nome_label": "Município",
            },
            title="Municípios com menor efeito estrutural de PIB per capita",
        )
        fig_bot.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bot, use_container_width=True)

    st.markdown("#### Tabela completa dos efeitos municipais")
    st.dataframe(
        df_mun_plot[["cod_mun", "nome_municipio", "alpha_mun_mean", "alpha_mun_lo", "alpha_mun_hi"]],
        hide_index=True,
        use_container_width=True,
    )
    _download(df_mun_plot, "⬇️ Baixar efeitos municipais (CSV)", "efeitos_municipais_PIBpc_BAYES.csv")

    st.markdown("</div>", unsafe_allow_html=True)

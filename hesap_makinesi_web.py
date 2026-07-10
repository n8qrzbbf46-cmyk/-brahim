import streamlit as st
import sympy as sp
from sympy import symbols, sympify, diff, integrate, solve, factor, roots, simplify, N, latex, Eq, pprint
import numpy as np
from numpy import linalg as la
import scipy
from scipy import integrate as sci_integrate
from scipy.optimize import fsolve, minimize_scalar
import pandas as pd

# Sayfa ayarları
st.set_page_config(
    page_title="🧮 Gelişmiş Web Hesap Makinesi",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Başlık
st.title("🧮 Gelişmiş Web Bilimsel Hesap Makinesi")
st.markdown("**SymPy + NumPy + SciPy + Pandas** ile güçlendirilmiş • Kural yok, her şeyi hesaplar!")
st.markdown("---")

# Session state başlat
if "expr" not in st.session_state:
    st.session_state.expr = ""
if "history" not in st.session_state:
    st.session_state.history = []

def add_to_expr(char):
    st.session_state.expr += str(char)

def clear_expr():
    st.session_state.expr = ""

def calculate_expr():
    if st.session_state.expr.strip() == "":
        return
    try:
        expr = sympify(st.session_state.expr)
        result = simplify(expr)
        numeric = N(result)
        st.session_state.history.append({
            "input": st.session_state.expr,
            "symbolic": str(result),
            "numeric": str(numeric)
        })
        return result, numeric
    except Exception as e:
        st.error(f"Hata: {e}")
        return None, None

# Sidebar - Hızlı Erişim
with st.sidebar:
    st.header("📌 Hızlı Menü")
    st.markdown("""
    - **Temel Hesap Makinesi** → Tuş takımı ile klasik kullanım
    - **Türev & İntegral** → Sembolik calculus
    - **Denklem & Polinom** → Her derece denklem çözümü
    - **NumPy & SciPy** → Sayısal bilimsel hesaplar
    - **Pandas & Veri** → Veri analizi + matris
    - **Serbest İfade** → Her türlü komut
    """)
    st.markdown("---")
    if st.button("🗑️ Tüm Geçmişi Temizle"):
        st.session_state.history = []
        st.rerun()

# Ana Sekmeler
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🧮 Temel Hesap Makinesi (Tuş Takımı)",
    "📐 Türev & İntegral",
    "🔢 Denklem & Polinom",
    "🔬 NumPy & SciPy",
    "📊 Pandas & Veri",
    "✍️ Serbest İfade"
])

# ===================== TAB 1: TEMEL HESAP MAKİNESİ =====================
with tab1:
    st.header("🧮 Temel + Gelişmiş Hesap Makinesi (Tuş Takımı)")
    st.markdown("Butonlara basarak ifade oluştur. Sonra **=** ile hesapla. SymPy destekler: **, sqrt, sin, cos, pi, log, exp, x, y, z")

    # Ekran
    st.text_input("İfade", value=st.session_state.expr, key="display", disabled=True, label_visibility="collapsed")

    # Tuş Takımı - 4 sütun
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("7", use_container_width=True): add_to_expr("7")
        if st.button("4", use_container_width=True): add_to_expr("4")
        if st.button("1", use_container_width=True): add_to_expr("1")
        if st.button("0", use_container_width=True): add_to_expr("0")
        if st.button("(", use_container_width=True): add_to_expr("(")
    with col2:
        if st.button("8", use_container_width=True): add_to_expr("8")
        if st.button("5", use_container_width=True): add_to_expr("5")
        if st.button("2", use_container_width=True): add_to_expr("2")
        if st.button(".", use_container_width=True): add_to_expr(".")
        if st.button(")", use_container_width=True): add_to_expr(")")
    with col3:
        if st.button("9", use_container_width=True): add_to_expr("9")
        if st.button("6", use_container_width=True): add_to_expr("6")
        if st.button("3", use_container_width=True): add_to_expr("3")
        if st.button("=", use_container_width=True):
            result, numeric = calculate_expr()
            if result is not None:
                st.success(f"**Sembolik:** {result}")
                st.info(f"**Sayısal:** {numeric}")
        if st.button("x", use_container_width=True): add_to_expr("x")
    with col4:
        if st.button("÷", use_container_width=True): add_to_expr("/")
        if st.button("×", use_container_width=True): add_to_expr("*")
        if st.button("-", use_container_width=True): add_to_expr("-")
        if st.button("+", use_container_width=True): add_to_expr("+")
        if st.button("**", use_container_width=True): add_to_expr("**")
        if st.button("√", use_container_width=True): add_to_expr("sqrt(")
        if st.button("C", use_container_width=True): clear_expr()
        if st.button("⌫", use_container_width=True):
            st.session_state.expr = st.session_state.expr[:-1]

    st.markdown("---")
    st.subheader("Hızlı Fonksiyonlar")
    colf1, colf2, colf3, colf4 = st.columns(4)
    with colf1:
        if st.button("sin(", use_container_width=True): add_to_expr("sin(")
        if st.button("cos(", use_container_width=True): add_to_expr("cos(")
    with colf2:
        if st.button("log(", use_container_width=True): add_to_expr("log(")
        if st.button("exp(", use_container_width=True): add_to_expr("exp(")
    with colf3:
        if st.button("pi", use_container_width=True): add_to_expr("pi")
        if st.button("E", use_container_width=True): add_to_expr("E")
    with colf4:
        if st.button("sqrt(", use_container_width=True): add_to_expr("sqrt(")

    # Geçmiş
    if st.session_state.history:
        st.subheader("📜 Hesaplama Geçmişi")
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            st.code(f"{item['input']}  →  {item['symbolic']}  |  ≈ {item['numeric']}")

# ===================== TAB 2: TÜREV & İNTEGRAL =====================
with tab2:
    st.header("📐 Sembolik Türev ve İntegral (SymPy)")
    
    col_d, col_i = st.columns(2)
    
    with col_d:
        st.subheader("Türev (Derivative)")
        diff_expr = st.text_input("İfade", value="x**3 + sin(x)**2", key="diff_expr")
        diff_var = st.text_input("Değişken", value="x", key="diff_var")
        if st.button("Türev Al", key="btn_diff"):
            try:
                var = symbols(diff_var)
                expr = sympify(diff_expr)
                result = diff(expr, var)
                st.success("✅ Türev Sonucu:")
                st.latex(latex(result))
                st.code(str(result))
            except Exception as e:
                st.error(str(e))
    
    with col_i:
        st.subheader("İntegral (Indefinite)")
        int_expr = st.text_input("İfade", value="x**2 + 2*x + 1", key="int_expr")
        int_var = st.text_input("Değişken", value="x", key="int_var")
        if st.button("İntegral Al", key="btn_int"):
            try:
                var = symbols(int_var)
                expr = sympify(int_expr)
                result = integrate(expr, var)
                st.success("✅ İntegral Sonucu ( + C )")
                st.latex(latex(result))
                st.code(str(result))
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.subheader("Belirli İntegral (Definite)")
    def_expr = st.text_input("İfade", value="sin(x)", key="def_expr")
    def_var = st.text_input("Değişken", value="x", key="def_var")
    col_a, col_b = st.columns(2)
    with col_a:
        a = st.number_input("Alt sınır (a)", value=0.0, key="int_a")
    with col_b:
        b = st.number_input("Üst sınır (b)", value=3.1416, key="int_b")
    if st.button("Belirli İntegral Hesapla"):
        try:
            var = symbols(def_var)
            expr = sympify(def_expr)
            result = integrate(expr, (var, a, b))
            st.success(f"✅ ∫_{a}^{b} {def_expr} d{def_var} = ")
            st.latex(latex(result))
            st.code(f"{float(N(result)):.10f}  (sayısal)")
        except Exception as e:
            st.error(str(e))

# ===================== TAB 3: DENKLEM & POLİNOM =====================
with tab3:
    st.header("🔢 Denklem Çözücü & Polinom İşlemleri (Her Derece!)")
    
    st.subheader("Denklem Çöz (solve)")
    eq_input = st.text_input("Denklem (örn: x**2 - 5*x + 6 = 0)", value="x**2 - 5*x + 6 = 0", key="eq_input")
    eq_var = st.text_input("Değişken", value="x", key="eq_var")
    if st.button("Denklemi Çöz"):
        try:
            var = symbols(eq_var)
            if "=" in eq_input:
                left, right = eq_input.split("=", 1)
                eq = Eq(sympify(left.strip()), sympify(right.strip()))
            else:
                eq = sympify(eq_input)
            result = solve(eq, var)
            st.success(f"✅ {eq_var} için çözümler:")
            st.latex(latex(result))
            st.code(str(result))
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.subheader("Polinom Kökleri (roots)")
        poly_roots = st.text_input("Polinom", value="x**4 - 5*x**2 + 4", key="poly_roots")
        if st.button("Kökleri Bul"):
            try:
                p = sympify(poly_roots)
                r = roots(p)
                st.success("✅ Kökler (çokluk ile):")
                st.code(str(r))
                for k, v in r.items():
                    st.write(f"Kök: {k}  →  Çokluk: {v}")
            except Exception as e:
                st.error(str(e))
    
    with col_p2:
        st.subheader("Çarpanlara Ayır (factor)")
        poly_factor = st.text_input("Polinom", value="x**3 - 6*x**2 + 11*x - 6", key="poly_factor")
        if st.button("Çarpanlara Ayır"):
            try:
                p = sympify(poly_factor)
                f = factor(p)
                st.success("✅ Çarpanlara Ayrılmış:")
                st.latex(latex(f))
                st.code(str(f))
            except Exception as e:
                st.error(str(e))

# ===================== TAB 4: NUMPY & SCIPY =====================
with tab4:
    st.header("🔬 NumPy & SciPy - Sayısal Bilimsel Hesaplamalar")
    
    st.subheader("NumPy - Matris & Doğrusal Cebir")
    matrix_input = st.text_area("Matris girin (Python listesi formatı, örn: [[1,2],[3,4]])", value="[[1,2],[3,4]]", height=80)
    if st.button("Matris İşlemleri"):
        try:
            mat = np.array(eval(matrix_input))
            st.write("**Matris:**")
            st.dataframe(pd.DataFrame(mat))
            st.write(f"**Determinant:** {la.det(mat):.6f}")
            if mat.shape[0] == mat.shape[1]:
                st.write("**Ters Matris:**")
                st.dataframe(pd.DataFrame(la.inv(mat)))
                eigvals, eigvecs = la.eig(mat)
                st.write(f"**Özdeğerler:** {eigvals}")
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.subheader("SciPy - Sayısal İntegral & Kök Bulma")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Sayısal Belirli İntegral (scipy.integrate.quad)**")
        sci_func = st.text_input("Fonksiyon (lambda x: ...)", value="lambda x: np.sin(x)**2", key="sci_func")
        sci_a = st.number_input("a", value=0.0)
        sci_b = st.number_input("b", value=np.pi)
        if st.button("SciPy ile İntegral"):
            try:
                func = eval(sci_func)
                result, err = sci_integrate.quad(func, sci_a, sci_b)
                st.success(f"Sonuç: {result:.10f}  ± {err:.2e}")
            except Exception as e:
                st.error(str(e))
    
    with col_s2:
        st.markdown("**Denklem Kökü Bulma (scipy.optimize.fsolve)**")
        fsolve_func = st.text_input("Fonksiyon (lambda x: ...)", value="lambda x: x**3 - x - 1", key="fsolve_func")
        fsolve_guess = st.number_input("Başlangıç tahmini", value=1.0)
        if st.button("Kök Bul (fsolve)"):
            try:
                func = eval(fsolve_func)
                root = fsolve(func, fsolve_guess)
                st.success(f"Kök ≈ {root[0]:.10f}")
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.subheader("Optimizasyon (Minimize)")
    min_func = st.text_input("Minimize edilecek fonksiyon", value="lambda x: (x-3)**2 + 2")
    if st.button("Minimum Bul"):
        try:
            func = eval(min_func)
            res = minimize_scalar(func)
            st.success(f"Minimum nokta x ≈ {res.x:.6f}  →  f(x) ≈ {res.fun:.6f}")
        except Exception as e:
            st.error(str(e))

# ===================== TAB 5: PANDAS & VERİ =====================
with tab5:
    st.header("📊 Pandas & Veri Analizi + Matris")
    
    st.subheader("Hızlı İstatistik (Pandas)")
    data_input = st.text_area("Veri girin (virgülle ayrılmış sayılar)", value="12, 15, 18, 22, 19, 25, 30", height=60)
    if st.button("İstatistikleri Hesapla"):
        try:
            nums = [float(x.strip()) for x in data_input.split(",")]
            df = pd.DataFrame(nums, columns=["Değerler"])
            st.dataframe(df.T)
            st.write(f"**Ortalama:** {df.mean().values[0]:.4f}")
            st.write(f"**Medyan:** {df.median().values[0]:.4f}")
            st.write(f"**Std Sapma:** {df.std().values[0]:.4f}")
            st.write(f"**Min / Max:** {df.min().values[0]} / {df.max().values[0]}")
            st.write(f"**Toplam:** {df.sum().values[0]}")
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.subheader("Matris → DataFrame + İşlemler")
    mat_df = st.text_area("Matris (liste)", value="[[1,2,3],[4,5,6],[7,8,9]]", height=80)
    if st.button("DataFrame'e Çevir + Özet"):
        try:
            arr = np.array(eval(mat_df))
            df = pd.DataFrame(arr)
            st.write("**DataFrame:**")
            st.dataframe(df)
            st.write("**Özet İstatistik:**")
            st.dataframe(df.describe())
            st.write(f"**Korelasyon Matrisi:**")
            st.dataframe(df.corr())
        except Exception as e:
            st.error(str(e))

# ===================== TAB 6: SERBEST İFADE =====================
with tab6:
    st.header("✍️ Serbest İfade Modu (Gelişmiş Kullanıcılar)")
    st.warning("Burada doğrudan SymPy, NumPy, SciPy komutları yazabilirsiniz. Dikkatli kullanın.")
    
    free_cmd = st.text_area("Komut / İfade yazın", 
        value='diff(sin(x)**2, x)\n# veya\nintegrate(x*exp(x), x)\n# veya\nsolve(x**5 - x -1, x)\n# veya\nnp.linalg.det(np.array([[1,2],[3,4]]))', 
        height=150, key="free_cmd")
    
    if st.button("Çalıştır"):
        try:
            # Güvenli namespace
            safe_globals = {
                "__builtins__": {},
                "np": np,
                "sp": sp,
                "scipy": scipy,
                "pd": pd,
                "x": symbols('x'), "y": symbols('y'), "z": symbols('z'),
                "diff": diff, "integrate": integrate, "solve": solve,
                "factor": factor, "roots": roots, "simplify": simplify, "N": N,
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
                "pi": sp.pi, "E": sp.E,
                "symbols": symbols, "Eq": Eq,
                "la": la,
                "sci_integrate": sci_integrate,
                "fsolve": fsolve,
            }
            # Çok satırlı destek
            lines = free_cmd.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    result = eval(line, safe_globals)
                    st.success(f"**{line}**")
                    if hasattr(result, "__iter__") and not isinstance(result, str):
                        st.code(str(result))
                    else:
                        try:
                            st.latex(latex(result))
                        except:
                            st.code(str(result))
        except Exception as e:
            st.error(f"Hata: {e}")

# Alt bilgi
st.markdown("---")
st.caption("Made with ❤️ • SymPy 1.14 + NumPy 2.1 + SciPy 1.17 + Pandas 3.0 • Tamamen yerel çalışır • Her türlü matematik ifadesini destekler!")
st.caption("Kullanmak için: `pip install streamlit sympy numpy scipy pandas` sonra `streamlit run hesap_makinesi_web.py`")
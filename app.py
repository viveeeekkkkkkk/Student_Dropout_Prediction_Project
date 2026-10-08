import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# SESSION STATE
# ============================================================

if "started" not in st.session_state:
    st.session_state.started = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ============================================================
   LANDING PAGE
   ============================================================ */

.landing {
    text-align: center;
    padding: 55px 20px 20px 20px;
}

.landing-icon {
    font-size: 2.7rem;
    margin-bottom: 15px;
}

.landing-title {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.landing-subtitle {
    font-size: 1.05rem;
    opacity: 0.65;
    margin-bottom: 40px;
}


/* ============================================================
   TEAM SECTION
   ============================================================ */

.team-card {
    max-width: 500px;
    margin: 0 auto 30px auto;
    padding: 25px 20px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 10px;
    text-align: center;
}

.team-title {
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 12px;
}

.team-names {
    font-size: 1rem;
    line-height: 1.8;
}

.guide-title {
    font-size: 0.88rem;
    opacity: 0.60;
    margin-top: 22px;
    margin-bottom: 5px;
}

.guide-name {
    font-size: 1rem;
    font-weight: 600;
}


/* ============================================================
   BUTTON
   ============================================================ */

div.stButton > button {
    min-height: 45px;
    border-radius: 7px;
    font-weight: 600;
}


/* ============================================================
   MAIN PAGE
   ============================================================ */

.main-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.main-subtitle {
    text-align: center;
    font-size: 1rem;
    opacity: 0.70;
    margin-bottom: 25px;
}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 15px;
}


/* ============================================================
   INPUT LABELS
   ============================================================ */

label {
    font-weight: 500 !important;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    font-size: 0.8rem;
    margin-top: 35px;
    opacity: 0.60;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# STARTING PAGE
# ============================================================

if not st.session_state.started:

    st.markdown(
        """
<div class="landing">

    <div class="landing-icon">🎓</div>

    <div class="landing-title">
        Student Dropout Prediction
    </div>

    <div class="landing-subtitle">
        Machine Learning Based Student Risk Assessment
    </div>

    <div class="team-card">

        <div class="team-title">
            Created by
        </div>

        <div class="team-names">
            Vivek Sharma<br>
            Yash Prabhakar<br>
            Unmesh Guru<br>
            Shashank Gupta<br>
            Souvik Barman
        </div>

        <div class="guide-title">
            Under the Guidance of
        </div>

        <div class="guide-name">
            Sunetra Mukherjee
        </div>

    </div>

</div>
""",
        unsafe_allow_html=True
    )

    if st.button(
        "START PROJECT",
        use_container_width=True
    ):
        st.session_state.started = True
        st.rerun()

    st.stop()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception:
        return None


bundle = load_model()


# ============================================================
# CHECK MODEL
# ============================================================

if bundle is None:
    st.error(
        "Model file 'model.pkl' was not found. "
        "Please run train_model.py first."
    )
    st.stop()


model = bundle["model"]
features = bundle["features"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Student Dropout Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Machine Learning Based Student Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# STUDENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-heading">Student Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=40,
        value=20,
        step=1
    )

    courses_enrolled = st.number_input(
        "Courses Enrolled",
        min_value=1,
        max_value=7,
        value=4,
        step=1
    )

    completion_rate = st.number_input(
        "Completion Rate",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
        format="%.2f"
    )

    last_activity = st.number_input(
        "Last Activity (days ago)",
        min_value=1,
        max_value=70,
        value=7,
        step=1
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    exam_season = st.number_input(
        "Exam Season",
        min_value=0,
        max_value=1,
        value=0,
        step=1
    )

    completed_assignments = st.number_input(
        "Completed Assignments",
        min_value=0,
        max_value=35,
        value=5,
        step=1
    )

    login_frequency = st.number_input(
        "Login Frequency",
        min_value=0.10,
        max_value=14.00,
        value=3.00,
        step=0.10,
        format="%.2f"
    )

    forum_posts = st.number_input(
        "Forum Posts Count",
        min_value=0,
        max_value=24,
        value=3,
        step=1
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "PREDICT DROPOUT",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    sample_df = pd.DataFrame(
        [
            {
                "age": age,
                "exam_season": exam_season,
                "courses_enrolled": courses_enrolled,
                "completed_assignments": completed_assignments,
                "completion_rate": completion_rate,
                "login_frequency": login_frequency,
                "last_activity_days_ago": last_activity,
                "forum_posts_count": forum_posts
            }
        ]
    )[features]


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = int(
        model.predict(sample_df)[0]
    )


    # --------------------------------------------------------
    # DROPOUT PROBABILITY
    # --------------------------------------------------------

    probability = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(sample_df)[0]

        classes = list(model.classes_)

        if 1 in classes:

            dropout_index = classes.index(1)

            probability = probabilities[dropout_index]


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-heading">Prediction Result</div>',
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # ACTIVE STUDENT
    # --------------------------------------------------------

    if prediction == 0:

        st.success(
            "ACTIVE STUDENT\n\n"
            "Prediction Label: 0"
        )


    # --------------------------------------------------------
    # DROPOUT RISK
    # --------------------------------------------------------

    else:

        st.error(
            "DROPOUT RISK\n\n"
            "Prediction Label: 1"
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    if probability is not None:

        st.subheader("Estimated Dropout Probability")

        st.metric(
            "Dropout Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))

        st.caption(
            "This is the model's estimated probability for "
            "the dropout class based on the entered student data."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Student Dropout Prediction | Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)

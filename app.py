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

model_name = bundle.get("model_name", "AdaBoost")
accuracy = bundle.get("accuracy", None)


# ============================================================
# CUSTOM CSS - LIGHT + DARK MODE SUPPORT
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GENERAL
       ======================================================== */

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    .main-title {
        text-align: center;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-subtitle {
        text-align: center;
        font-size: 1rem;
        margin-bottom: 25px;
    }


    /* ========================================================
       MODEL INFORMATION
       ======================================================== */

    .model-box {
        border: 1px solid rgba(128, 128, 128, 0.30);
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 25px;

        /* Uses Streamlit theme automatically */
        background-color: var(--background-color);
    }

    .model-box-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .model-box-text {
        font-size: 0.95rem;
        line-height: 1.7;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-heading {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 15px;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        font-size: 0.8rem;
        margin-top: 35px;
        opacity: 0.65;
    }


    /* ========================================================
       STREAMLIT INPUT LABELS
       ======================================================== */

    label {
        font-weight: 500 !important;
    }


    /* ========================================================
       BUTTON
       ======================================================== */

    div.stButton > button {
        font-weight: 600;
        border-radius: 7px;
        min-height: 42px;
    }


    </style>
    """,
    unsafe_allow_html=True
)


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

    sample_df = pd.DataFrame([
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
    ])[features]


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

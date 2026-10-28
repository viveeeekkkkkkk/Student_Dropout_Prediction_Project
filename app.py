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

/* ============================================================
   GENERAL
   ============================================================ */

.block-container {
    max-width: 1100px;
    padding-top: 4.5rem;
    padding-bottom: 3rem;
}


/* ============================================================
   MAIN HEADER
   ============================================================ */

.main-title {
    text-align: center;
    font-size: 2.35rem;
    font-weight: 750;
    line-height: 1.3;
    margin-top: 0.5rem;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
}

.main-subtitle {
    text-align: center;
    font-size: 1.05rem;
    line-height: 1.4;
    opacity: 0.68;
    margin-bottom: 35px;
}


/* ============================================================
   PAGE INTRO
   ============================================================ */

.page-intro {
    text-align: center;
    font-size: 0.92rem;
    opacity: 0.65;
    margin-bottom: 25px;
}


/* ============================================================
   SECTION HEADINGS
   ============================================================ */

.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 5px;
}

.section-description {
    font-size: 0.88rem;
    opacity: 0.62;
    margin-bottom: 18px;
}


/* ============================================================
   INPUT AREA
   ============================================================ */

div[data-testid="stNumberInput"] {
    margin-bottom: 8px;
}

div[data-testid="stNumberInput"] label {
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

div[data-testid="stNumberInput"] input {
    border-radius: 8px !important;
}


/* ============================================================
   INPUT FOCUS
   ============================================================ */

div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(31, 115, 209, 0.65) !important;
    box-shadow: 0 0 0 1px rgba(31, 115, 209, 0.20) !important;
}


/* ============================================================
   PREDICTION BUTTON
   ============================================================ */

div.stButton > button {
    min-height: 50px;
    border-radius: 9px;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    transition: all 0.15s ease;
}

div.stButton > button:hover {
    transform: translateY(-1px);
}


/* ============================================================
   PREDICTION RESULT
   ============================================================ */

.result-heading {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 15px;
}

.result-description {
    font-size: 0.88rem;
    opacity: 0.65;
    margin-bottom: 15px;
}


/* ============================================================
   PROBABILITY
   ============================================================ */

.probability-title {
    font-size: 1.05rem;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 8px;
}

.probability-note {
    font-size: 0.82rem;
    opacity: 0.60;
    margin-top: 8px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    font-size: 0.78rem;
    margin-top: 45px;
    opacity: 0.55;
}


/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 700px) {

    .block-container {
        padding-top: 3.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .main-title {
        font-size: 1.8rem;
    }

    .main-subtitle {
        font-size: 0.95rem;
        margin-bottom: 28px;
    }

    .section-heading {
        font-size: 1.15rem;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# STARTING / LANDING PAGE
# ============================================================

if not st.session_state.started:

    st.html(
        """
        <style>

        .landing-wrapper {
            width: 100%;
            margin-top: 10px;
        }

        .landing-header {
            position: relative;
            width: 100%;
            height: 155px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .college-logo {
            position: absolute;
            left: 0;
            top: 0;
            width: 125px;
            height: 125px;
            object-fit: contain;
        }

        .title-container {
            width: 100%;
            text-align: center;
        }

        .project-title {
            font-size: 2.4rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
        }

        .project-subtitle {
            font-size: 1.55rem;
            font-weight: 600;
            margin-top: 10px;
            opacity: 0.82;
        }

        .group-number {
            text-align: center;
            font-size: 1.15rem;
            font-weight: 600;
            margin-top: 28px;
            margin-bottom: 28px;
        }

        .students-heading {
            text-align: center;
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 16px;
        }

        .student-list {
            width: 600px;
            max-width: 90%;
            margin: 0 auto;
        }

        .student-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            font-size: 1rem;
        }

        .student-name {
            text-align: left;
        }

        .student-roll {
            text-align: right;
        }

        .pgl {
            font-weight: 700;
        }

        .guidance-title {
            text-align: center;
            font-size: 0.95rem;
            font-weight: 500;
            opacity: 0.65;
            margin-top: 25px;
            margin-bottom: 5px;
        }

        .guidance-name {
            text-align: center;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 5px;
        }

        @media (max-width: 700px) {

            .landing-header {
                height: auto;
                display: block;
                text-align: center;
            }

            .college-logo {
                position: static;
                display: block;
                margin: 0 auto 15px auto;
                width: 100px;
                height: 100px;
            }

            .project-title {
                font-size: 1.8rem;
            }

            .project-subtitle {
                font-size: 1.2rem;
            }

            .student-list {
                width: 100%;
                max-width: 100%;
            }

            .student-row {
                font-size: 0.85rem;
                gap: 10px;
            }
        }

        </style>


        <div class="landing-wrapper">

            <div class="landing-header">

                <img
                    class="college-logo"
                    src="https://cdn-prod.mybharats.in/tasks/17530805401154176.png"
                >

                <div class="title-container">

                    <div class="project-title">
                        Student Dropout Prediction
                    </div>

                    <div class="project-subtitle">
                        using Machine Learning
                    </div>

                </div>

            </div>


            <div class="group-number">
                Project Group Number:- 51/CS591/D/PrjGrp/51
            </div>


            <div class="students-heading">
                Names of the students:-
            </div>


            <div class="student-list">

                <div class="student-row">
                    <span class="student-name">
                        VIVEK SHARMA <span class="pgl">(PGL)</span>
                    </span>

                    <span class="student-roll">
                        (108001240267)
                    </span>
                </div>

                <div class="student-row">
                    <span class="student-name">
                        YASH PRABHAKAR
                    </span>

                    <span class="student-roll">
                        (108001240268)
                    </span>
                </div>

                <div class="student-row">
                    <span class="student-name">
                        UNMESH GURU
                    </span>

                    <span class="student-roll">
                        (108001240263)
                    </span>
                </div>

                <div class="student-row">
                    <span class="student-name">
                        SHASHANK GUPTA
                    </span>

                    <span class="student-roll">
                        (108001240226)
                    </span>
                </div>

                <div class="student-row">
                    <span class="student-name">
                        SOUVIK BARMAN
                    </span>

                    <span class="student-roll">
                        (108001240248)
                    </span>
                </div>

            </div>


            <div class="guidance-title">
                Under the Guidance of
            </div>

            <div class="guidance-name">
                Mr. Sunetra Mukherjee
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # START PROJECT BUTTON
    # --------------------------------------------------------

    st.markdown(
        "<div style='height:35px;'></div>",
        unsafe_allow_html=True
    )

    left, center, right = st.columns([1, 2, 1])

    with center:

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
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Student Dropout Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Machine Learning Based Student Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-intro">'
    'Enter the student information below to assess the likelihood of dropout.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# STUDENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Student Information'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    "<div class='section-description'>"
    "Provide the student's academic and learning-platform activity details."
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# INPUT COLUMNS
# ============================================================

col1, col2 = st.columns(2, gap="large")


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=17,
        max_value=40,
        value=20,
        step=1,
        help="Student age in years."
    )

    courses_enrolled = st.number_input(
        "Courses Enrolled",
        min_value=1,
        max_value=7,
        value=4,
        step=1,
        help="Number of courses the student is currently enrolled in."
    )

    completion_rate = st.number_input(
        "Completion Rate",
        min_value=0.00,
        max_value=1.00,
        value=0.50,
        step=0.01,
        format="%.2f",
        help="Coursework completion rate. Enter a value between 0 and 1."
    )

    last_activity = st.number_input(
        "Last Activity (days ago)",
        min_value=1,
        max_value=70,
        value=7,
        step=1,
        help="Number of days since the student's last activity."
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
        step=1,
        help="0 = Not exam season, 1 = Exam season."
    )

    completed_assignments = st.number_input(
        "Completed Assignments",
        min_value=0,
        max_value=35,
        value=5,
        step=1,
        help="Number of assignments completed by the student."
    )

    login_frequency = st.number_input(
        "Login Frequency",
        min_value=0.10,
        max_value=14.00,
        value=3.00,
        step=0.10,
        format="%.2f",
        help="Student login frequency on the learning platform."
    )

    forum_posts = st.number_input(
        "Forum Posts Count",
        min_value=0,
        max_value=24,
        value=3,
        step=1,
        help="Number of forum or discussion posts made by the student."
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown(
    "<div style='height:15px;'></div>",
    unsafe_allow_html=True
)

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
    # PREDICTION RESULT
    # ========================================================

    st.markdown(
        '<div class="result-heading">'
        'Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="result-description">'
        'The model has evaluated the entered student information.'
        '</div>',
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

        st.markdown(
            '<div class="probability-title">'
            'Estimated Dropout Probability'
            '</div>',
            unsafe_allow_html=True
        )

        st.metric(
            "Dropout Probability",
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )

st.markdown(
    "<div class='probability-note'>"
    "This value represents the model's estimated probability "
    "for the dropout class based on the entered student data."
    "</div>",
    unsafe_allow_html=True
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

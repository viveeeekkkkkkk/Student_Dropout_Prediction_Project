import os
import joblib
import pandas as pd


def clear():
  os.system('cls' if os.name == 'nt' else 'clear')


def run_terminal():
  if not os.path.exists('model.pkl'):
    print(
        "Error: 'model.pkl' not found. Please run 'python train_model.py'"
        ' first.'
    )
    return

  bundle = joblib.load('model.pkl')
  model = bundle['model']
  features = bundle['features']

  clear()
  width = 65

  print('+' + '-' * (width - 2) + '+')
  print('|' + 'STUDENT DROPOUT PREDICTION'.center(width - 2) + '|')
  print(
      '|'
      + 'Machine Learning Based Prediction System'.center(width - 2)
      + '|'
  )
  print('|' + '-' * (width - 2) + '|')
  print('| Student Information'.ljust(width - 1) + '|')
  print('|' + ' ' * (width - 2) + '|')

  # Helper for user input with default fallback
  def get_val(prompt_text, default):
    val = input(f'{prompt_text} [{default}]: ').strip()
    return float(val) if val else float(default)

  print('  Enter Student Values:')
  age = int(get_val('  Age', 20))
  exam_season = int(get_val('  Exam Season (0/1)', 1))
  courses = int(get_val('  Courses Enrolled', 5))
  assignments = int(get_val('  Completed Assignments', 3))
  comp_rate = float(get_val('  Completion Rate (0.00-1.00)', 0.45))
  logins = float(get_val('  Login Frequency', 0.60))
  last_act = int(get_val('  Last Activity (days ago)', 7))
  posts = int(get_val('  Forum Posts Count', 2))

  sample = pd.DataFrame([{
      'age': age,
      'exam_season': exam_season,
      'courses_enrolled': courses,
      'completed_assignments': assignments,
      'completion_rate': comp_rate,
      'login_frequency': logins,
      'last_activity_days_ago': last_act,
      'forum_posts_count': posts,
  }])[features]

  pred = int(model.predict(sample)[0])

  print('|' + '-' * (width - 2) + '|')
  print('|' + 'Prediction Result'.center(width - 2) + '|')
  print('|' + ' ' * (width - 2) + '|')

  if pred == 0:
    print('|' + '🟢  ACTIVE STUDENT'.center(width - 3) + '|')
    print('|' + 'Prediction: 0'.center(width - 2) + '|')
  else:
    print('|' + '🔴  DROPOUT RISK'.center(width - 3) + '|')
    print('|' + 'Prediction: 1'.center(width - 2) + '|')

  print('+' + '-' * (width - 2) + '+')


if __name__ == '__main__':
  run_terminal()
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COMP_ID = os.environ['COMP_ID']
KAGGLE_USERNAME = os.environ['KAGGLE_USERNAME'].strip().lower()
SUBMISSION_FILE = os.environ.get('SUBMISSION_FILE', '').strip()
MAX_ATTEMPTS = 60
POLL_SECONDS = 10
PLOT_DIR = 'docs'
PLOT_PATH = os.path.join(PLOT_DIR, 'kaggle-leaderboard-rank.png')
README_PATH = 'README.md'


def run_kaggle_csv(args):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr.strip() or f'Kaggle command failed: {" ".join(args)}')
        sys.exit(result.returncode)
    if not result.stdout.strip():
        return None
    return pd.read_csv(io.StringIO(result.stdout))


def normalize_columns(frame):
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def find_column(frame, candidates):
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def find_score_column(frame, excluded_columns=None):
    excluded_columns = {str(column).strip().lower() for column in (excluded_columns or [])}

    normalized_columns = [str(column).strip() for column in frame.columns]
    normalized_lookup = {column.lower(): column for column in normalized_columns}

    preferred_candidates = [
        'publicscore',
        'public score',
        'score',
        'public_score',
        'public leaderboard score',
        'public leaderboard score (%)',
    ]

    for candidate in preferred_candidates:
        if candidate in normalized_lookup and candidate not in excluded_columns:
            return normalized_lookup[candidate]

    for column in normalized_columns:
        normalized = column.lower()
        if normalized in excluded_columns:
            continue
        if 'score' in normalized:
            return normalized_lookup[normalized]

    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        if str(column).strip().lower() in excluded_columns:
            continue
        return column

    return None


def submission_is_complete(row, status_column, score_column):
    if status_column is not None:
        status_value = str(row.get(status_column, '')).strip().lower()
        if status_value == 'complete':
            return True

    if score_column is not None:
        score_value = pd.to_numeric(pd.Series([row.get(score_column)]), errors='coerce').iloc[0]
        if pd.notna(score_value):
            return True

    return False


def wait_for_completion():
    latest_submission = None
    target_file_name = Path(SUBMISSION_FILE).name if SUBMISSION_FILE else None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        submissions_df = run_kaggle_csv([
            'kaggle',
            'competitions',
            'submissions',
            '-c',
            COMP_ID,
            '--csv',
        ])
        leaderboard_df = run_kaggle_csv([
            'kaggle',
            'competitions',
            'leaderboard',
            '-c',
            COMP_ID,
            '--show',
            '--csv',
        ])

        if submissions_df is None or submissions_df.empty:
            print('Kaggle returned no submission data yet.')
        else:
            submissions_df = normalize_columns(submissions_df)
            status_column = find_column(submissions_df, ['status', 'state'])
            score_column = find_column(submissions_df, ['publicscore', 'public score'])
            file_column = find_column(submissions_df, ['filename', 'file name', 'file'])
            date_column = find_column(submissions_df, ['date', 'submitted'])

            candidate_rows = submissions_df
            if target_file_name and file_column is not None:
                file_matches = submissions_df[
                    submissions_df[file_column].astype(str).str.strip().str.endswith(target_file_name, na=False)
                ]
                if not file_matches.empty:
                    candidate_rows = file_matches

            if date_column is not None:
                candidate_rows = candidate_rows.sort_values(date_column, ascending=False, na_position='last')

            latest_submission = candidate_rows.iloc[0]
            current_status = str(latest_submission.get(status_column, '') if status_column is not None else '').strip().lower()

            if current_status in {'error', 'failed'}:
                print('Error: Kaggle scoring failed.')
                sys.exit(1)

            if submission_is_complete(latest_submission, status_column, score_column):
                print('Submission scoring complete!')
                return latest_submission

            print(
                f'Status is currently: {current_status or "unknown"}. '
                f'Checking again in {POLL_SECONDS} seconds...'
            )

        if leaderboard_df is not None and not leaderboard_df.empty:
            try:
                leaderboard_df, score_column, my_rank, my_score = find_leaderboard_row(
                    leaderboard_df,
                    latest_submission,
                )
            except Exception:
                leaderboard_df = None
            else:
                if my_rank != 'Pending' and pd.notna(my_score):
                    print('Leaderboard export already shows a completed submission.')
                    return pd.Series({
                        'status': 'complete',
                        'publicScore': my_score,
                    })

        if attempt == MAX_ATTEMPTS:
            print(
                f'Timed out waiting for Kaggle scoring after '
                f'{MAX_ATTEMPTS * POLL_SECONDS} seconds.'
            )
            sys.exit(1)

        time.sleep(POLL_SECONDS)

    return latest_submission


def _submission_identity_terms(latest_submission):
    terms = [KAGGLE_USERNAME]

    if latest_submission is not None:
        for key in ['teamName', 'team name', 'filename', 'fileName', 'file name']:
            value = latest_submission.get(key)
            if value is None:
                continue
            text = str(value).strip().lower()
            if text:
                terms.append(text)

    normalized_terms = []
    for term in terms:
        normalized = str(term).strip().lower()
        if normalized and normalized not in normalized_terms:
            normalized_terms.append(normalized)

    return normalized_terms


def find_leaderboard_row(leaderboard_df, latest_submission):
    leaderboard_df = leaderboard_df.copy()
    leaderboard_df.columns = [str(column).strip() for column in leaderboard_df.columns]

    rank_column = next((c for c in leaderboard_df.columns if 'rank' in c.lower() or 'position' in c.lower()), None)
    name_column = next((c for c in leaderboard_df.columns if any(token in c.lower() for token in ['team', 'user', 'name'])), None)
    score_column = find_score_column(leaderboard_df, excluded_columns=[rank_column] if rank_column is not None else [])
    identity_terms = _submission_identity_terms(latest_submission)

    if score_column is not None:
        leaderboard_df[score_column] = pd.to_numeric(leaderboard_df[score_column], errors='coerce')
        leaderboard_df = leaderboard_df.dropna(subset=[score_column]).reset_index(drop=True)

    my_row = None
    text_columns = list(leaderboard_df.select_dtypes(include=['object', 'string']).columns)
    candidate_text_columns = [name_column] if name_column is not None else []
    for column in text_columns:
        if column not in candidate_text_columns:
            candidate_text_columns.append(column)

    for column in candidate_text_columns:
        if column is None:
            continue
        column_series = leaderboard_df[column].astype(str).str.lower()
        for term in identity_terms:
            matches = leaderboard_df[column_series.str.contains(term, regex=False, na=False)]
            if not matches.empty:
                my_row = matches.iloc[0]
                break
        if my_row is not None:
            break

    if my_row is None and latest_submission is not None and score_column is not None:
        submission_score = pd.to_numeric(
            pd.Series([latest_submission.get('publicScore')]),
            errors='coerce',
        ).iloc[0]
        if pd.notna(submission_score):
            score_matches = leaderboard_df[
                np.isclose(leaderboard_df[score_column], submission_score, rtol=0, atol=1e-6)
            ]
            if not score_matches.empty:
                my_row = score_matches.iloc[0]

    if my_row is None:
        my_rank = 'Pending'
        my_score = float(pd.to_numeric(pd.Series([latest_submission.get('publicScore') if latest_submission is not None else np.nan]), errors='coerce').iloc[0])
    else:
        if rank_column is not None and pd.notna(my_row.get(rank_column)):
            my_rank = str(my_row[rank_column]).strip()
        else:
            my_rank = str(int(my_row.name) + 1)
        if score_column is not None:
            my_score = float(my_row[score_column])
        else:
            my_score = float(pd.to_numeric(pd.Series([latest_submission.get('publicScore') if latest_submission is not None else np.nan]), errors='coerce').iloc[0])

    return leaderboard_df, score_column, my_rank, my_score


def write_plot(leaderboard_df, score_column, my_rank, my_score):
    os.makedirs(PLOT_DIR, exist_ok=True)

    plt.figure(figsize=(10, 5))
    if score_column is not None:
        public_scores = leaderboard_df[score_column].dropna().astype(float)
        plt.hist(public_scores, bins=30, color='#d9d9d9', edgecolor='black')
        plt.axvline(my_score, color='#d62728', linewidth=3, label=f'Your score: {my_score:.4f}')
    else:
        plt.text(
            0.5,
            0.5,
            'Leaderboard export did not include a score column.\nBadge updated from rank information only.',
            ha='center',
            va='center',
            transform=plt.gca().transAxes,
            fontsize=11,
        )
    plt.title('Kaggle leaderboard score distribution')
    plt.xlabel('Public leaderboard score')
    plt.ylabel('Submission count')
    if score_column is not None:
        plt.legend(frameon=False)

    annotation = f'Rank: {my_rank}'
    if pd.notna(my_score):
        annotation = f'{annotation}\nScore: {my_score:.4f}'

    if score_column is not None and pd.notna(my_score):
        plt.text(
            my_score,
            plt.gca().get_ylim()[1] * 0.92,
            annotation,
            color='#d62728',
            fontsize=10,
            ha='left',
            va='top',
            bbox={
                'facecolor': 'white',
                'alpha': 0.85,
                'edgecolor': '#d62728',
            },
        )
    else:
        plt.text(
            0.98,
            0.92,
            annotation,
            transform=plt.gca().transAxes,
            color='#d62728',
            fontsize=10,
            ha='right',
            va='top',
            bbox={
                'facecolor': 'white',
                'alpha': 0.85,
                'edgecolor': '#d62728',
            },
        )

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches='tight')
    plt.close()


def update_readme(my_rank):
    badge_markdown = f'![Kaggle Rank](https://img.shields.io/badge/Kaggle%20rank-{my_rank}-blue?logo=kaggle&logoColor=white)'
    plot_markdown = f'![Kaggle leaderboard score distribution]({PLOT_PATH})'

    with open(README_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(
        r'(<!-- KAGGLE_BADGE_START -->).*?(<!-- KAGGLE_BADGE_END -->)',
        f'\\1\n{badge_markdown}\n\\2',
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r'(<!-- KAGGLE_RANK_PLOT_START -->).*?(<!-- KAGGLE_RANK_PLOT_END -->)',
        f'\\1\n{plot_markdown}\n\\2',
        content,
        flags=re.DOTALL,
    )

    with open(README_PATH, 'w', encoding='utf-8') as f:
        f.write(content)


latest_submission = wait_for_completion()
leaderboard_df = run_kaggle_csv([
    'kaggle',
    'competitions',
    'leaderboard',
    '-c',
    COMP_ID,
    '--show',
    '--csv',
])

if leaderboard_df is None or leaderboard_df.empty:
    print('Kaggle leaderboard export is empty.')
    sys.exit(1)

leaderboard_df, score_column, my_rank, my_score = find_leaderboard_row(leaderboard_df, latest_submission)
write_plot(leaderboard_df, score_column, my_rank, my_score)
update_readme(my_rank)

print(f'Updated README badge to rank {my_rank}')
print(f'Wrote leaderboard plot to {PLOT_PATH}')

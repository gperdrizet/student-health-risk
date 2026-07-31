import io
import os
import random
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
SUBMISSION_FILE = os.environ.get('SUBMISSION_FILE', '').strip()
MAX_ATTEMPTS = 60
POLL_SECONDS = 10
RATE_LIMIT_MAX_RETRIES = int(os.environ.get('KAGGLE_RATE_LIMIT_MAX_RETRIES', '8'))
RATE_LIMIT_BASE_DELAY_SECONDS = int(os.environ.get('KAGGLE_RATE_LIMIT_BASE_DELAY_SECONDS', '5'))
POST_SCORE_COOLDOWN_SECONDS = int(os.environ.get('KAGGLE_POST_SCORE_COOLDOWN_SECONDS', '20'))
PLOT_SCORE_MIN = float(os.environ.get('KAGGLE_PLOT_SCORE_MIN', '0.94'))
ARTIFACT_DIR = os.path.join('data', 'kaggle')
PLOT_PATH = os.path.join(ARTIFACT_DIR, 'kaggle-leaderboard-rank.png')
SUBMISSIONS_EXPORT_PATH = os.path.join(ARTIFACT_DIR, 'kaggle-submissions-export.csv')
LEADERBOARD_EXPORT_PATH = os.path.join(ARTIFACT_DIR, 'kaggle-leaderboard-export.csv')
README_PATH = 'README.md'


class KaggleRateLimitError(RuntimeError):
    """Raised when Kaggle CLI repeatedly returns 429 errors."""


def run_kaggle_command(args):
    delay_seconds = RATE_LIMIT_BASE_DELAY_SECONDS

    for attempt in range(1, RATE_LIMIT_MAX_RETRIES + 1):
        result = subprocess.run(args, capture_output=True, text=True)

        if result.returncode == 0:
            return result.stdout.strip()

        stderr_text = (result.stderr or '').strip()
        stdout_text = (result.stdout or '').strip()
        combined_error_text = f'{stderr_text}\n{stdout_text}'.strip()
        is_rate_limited = '429' in combined_error_text and 'Too Many Requests' in combined_error_text

        if is_rate_limited and attempt < RATE_LIMIT_MAX_RETRIES:
            jitter_seconds = random.uniform(0.0, 1.0)
            sleep_seconds = delay_seconds + jitter_seconds
            print(
                'Kaggle API rate limited (429). '
                f'Retrying command in {sleep_seconds:.1f}s '
                f'(attempt {attempt}/{RATE_LIMIT_MAX_RETRIES})...'
            )
            time.sleep(sleep_seconds)
            delay_seconds = min(delay_seconds * 2, 120)
            continue

        if is_rate_limited:
            raise KaggleRateLimitError(
                f'Kaggle API rate limited after {RATE_LIMIT_MAX_RETRIES} attempts for command: '
                f'{" ".join(args)}'
            )

        print(stderr_text or stdout_text or f'Kaggle command failed: {" ".join(args)}')
        sys.exit(result.returncode)

    raise KaggleRateLimitError(
        f'Kaggle API rate limited after {RATE_LIMIT_MAX_RETRIES} attempts for command: '
        f'{" ".join(args)}'
    )


def run_kaggle_csv(args):
    stdout = run_kaggle_command(args)
    if not stdout:
        return None
    return pd.read_csv(io.StringIO(stdout))


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


def parse_kaggle_paginated_csv(stdout_text):
    lines = stdout_text.splitlines()
    next_page_token = None
    csv_lines = []
    token_pattern = re.compile(r'^\s*Next Page Token\s*=\s*(.+?)\s*$', re.IGNORECASE)

    for line in lines:
        token_match = token_pattern.match(line)
        if token_match and next_page_token is None:
            next_page_token = token_match.group(1).strip()
            continue
        csv_lines.append(line)

    csv_text = '\n'.join(csv_lines).strip()
    if not csv_text:
        return None, next_page_token

    return pd.read_csv(io.StringIO(csv_text)), next_page_token


def fetch_kaggle_submissions(comp_id):
    submissions_df = run_kaggle_csv([
        'kaggle',
        'competitions',
        'submissions',
        '-c',
        comp_id,
        '--csv',
    ])

    if submissions_df is None or submissions_df.empty:
        return None

    return normalize_columns(submissions_df)


def fetch_kaggle_leaderboard_pages(comp_id):
    leaderboard_pages = []
    page_token = None

    while True:
        args = [
            'kaggle',
            'competitions',
            'leaderboard',
            '-c',
            comp_id,
            '--show',
            '--csv',
        ]

        if page_token:
            args.extend(['--page-token', page_token])

        stdout_text = run_kaggle_command(args)
        page_df, next_page_token = parse_kaggle_paginated_csv(stdout_text)

        if page_df is not None and not page_df.empty:
            leaderboard_pages.append(page_df)

        if not next_page_token:
            break

        page_token = next_page_token

    if not leaderboard_pages:
        return None

    return normalize_columns(pd.concat(leaderboard_pages, ignore_index=True))


def select_submission_row(submissions_df):
    file_column = find_column(submissions_df, ['filename', 'file name', 'file'])
    date_column = find_column(submissions_df, ['date', 'submitted'])

    candidate_rows = submissions_df
    if file_column is not None and SUBMISSION_FILE:
        target_file_name = Path(SUBMISSION_FILE).name
        file_matches = submissions_df[
            submissions_df[file_column].astype(str).str.strip().str.endswith(target_file_name, na=False)
        ]
        if not file_matches.empty:
            candidate_rows = file_matches

    if date_column is not None:
        candidate_rows = candidate_rows.sort_values(date_column, ascending=False, na_position='last')

    return candidate_rows.iloc[0]


def wait_for_completion():
    latest_submission = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            submissions_df = fetch_kaggle_submissions(COMP_ID)
        except KaggleRateLimitError:
            print(
                'Kaggle submissions endpoint is currently rate limited (429). '
                f'Will retry in {POLL_SECONDS} seconds...'
            )
            submissions_df = None

        if submissions_df is None or submissions_df.empty:
            print('Kaggle returned no submission data yet.')
        else:
            status_column = find_column(submissions_df, ['status', 'state'])
            score_column = find_column(submissions_df, ['publicscore', 'public score'])
            latest_submission = select_submission_row(submissions_df)

            current_status = str(
                latest_submission.get(status_column, '') if status_column is not None else ''
            ).strip().lower()

            if current_status in {'error', 'failed'}:
                print('Error: Kaggle scoring failed.')
                sys.exit(1)

            score_value = (
                pd.to_numeric(pd.Series([latest_submission.get(score_column)]), errors='coerce').iloc[0]
                if score_column is not None
                else np.nan
            )

            if current_status == 'complete' or pd.notna(score_value):
                print('Submission scoring complete!')
                if POST_SCORE_COOLDOWN_SECONDS > 0:
                    print(
                        'Waiting briefly before leaderboard fetch to reduce immediate rate-limit risk: '
                        f'{POST_SCORE_COOLDOWN_SECONDS}s'
                    )
                    time.sleep(POST_SCORE_COOLDOWN_SECONDS)
                return latest_submission

            print(
                f'Status is currently: {current_status or "unknown"}. '
                f'Checking again in {POLL_SECONDS} seconds...'
            )

        if attempt == MAX_ATTEMPTS:
            print(
                f'Timed out waiting for Kaggle scoring after '
                f'{MAX_ATTEMPTS * POLL_SECONDS} seconds.'
            )
            sys.exit(1)

        time.sleep(POLL_SECONDS)

    return latest_submission


def resolve_best_submission_score(submissions_df):
    score_column = find_column(submissions_df, ['publicscore', 'public score'])
    if score_column is None:
        raise RuntimeError('Could not determine the public score column from Kaggle submissions export.')

    scored_submissions = submissions_df.copy()
    scored_submissions['_public_score_numeric'] = pd.to_numeric(
        scored_submissions[score_column],
        errors='coerce',
    )
    scored_submissions = scored_submissions.dropna(subset=['_public_score_numeric'])

    if scored_submissions.empty:
        raise RuntimeError('Could not find any numeric public score in Kaggle submissions export.')

    best_submission_index = scored_submissions['_public_score_numeric'].idxmax()
    best_submission = scored_submissions.loc[best_submission_index]
    best_score = float(best_submission['_public_score_numeric'])

    file_column = find_column(submissions_df, ['filename', 'file name', 'file'])
    date_column = find_column(submissions_df, ['date', 'submitted'])
    best_file = str(best_submission[file_column]).strip() if file_column is not None else 'unknown'
    best_date = str(best_submission[date_column]).strip() if date_column is not None else 'unknown'
    print(f'Using best submission score: {best_score:.6f} (file={best_file}, date={best_date})')

    return best_score


def calculate_rank_from_scores(leaderboard_df, my_score):
    score_column = find_column(leaderboard_df, ['score', 'publicscore', 'public score'])
    if score_column is None:
        raise RuntimeError('Could not find a score column in the leaderboard export.')

    leaderboard_scores = pd.to_numeric(leaderboard_df[score_column], errors='coerce').dropna().astype(float)
    leaderboard_scores = leaderboard_scores.sort_values(ascending=False).reset_index(drop=True)

    if leaderboard_scores.empty:
        raise RuntimeError(
            'Leaderboard score column was found, but no numeric score values could be parsed.'
        )

    matches = leaderboard_scores[np.isclose(leaderboard_scores, my_score, rtol=0, atol=1e-6)]
    rank = int(matches.index[0] + 1) if not matches.empty else int((leaderboard_scores > my_score).sum() + 1)

    return leaderboard_scores, rank


def save_kaggle_exports(submissions_df, leaderboard_df):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    submissions_df.to_csv(SUBMISSIONS_EXPORT_PATH, index=False)
    leaderboard_df.to_csv(LEADERBOARD_EXPORT_PATH, index=False)
    print(f'Saved Kaggle submissions export to {SUBMISSIONS_EXPORT_PATH}')
    print(f'Saved Kaggle leaderboard export to {LEADERBOARD_EXPORT_PATH}')


def write_plot(leaderboard_scores, my_rank, my_score):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    plot_scores = leaderboard_scores[leaderboard_scores >= PLOT_SCORE_MIN]
    if plot_scores.empty:
        plot_scores = leaderboard_scores

    marker_score = max(my_score, PLOT_SCORE_MIN)
    title_fontsize = 14
    label_fontsize = 12
    tick_fontsize = 12
    legend_fontsize = 12

    plt.figure(figsize=(10, 5))
    plt.hist(plot_scores, bins=100, color='#d9d9d9', edgecolor='black')
    plt.axvline(marker_score, color='#d62728', linewidth=3, label='best submission')
    plt.title('Kaggle leaderboard score distribution', fontsize=title_fontsize)
    plt.xlabel('Public leaderboard score', fontsize=label_fontsize)
    plt.ylabel('Submission count', fontsize=label_fontsize)
    plt.xlim(left=PLOT_SCORE_MIN)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(frameon=False, fontsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200, bbox_inches='tight')
    plt.close()


def update_readme(my_rank, my_score):
    badge_markdown = f'![Kaggle Rank](https://img.shields.io/badge/Kaggle%20rank-{my_rank}-blue?logo=kaggle&logoColor=white)'
    score_badge_markdown = (
        '![Best Leaderboard Balanced Accuracy]('
        f'https://img.shields.io/badge/Best%20leaderboard%20BA-{my_score:.5f}-blue?logo=kaggle&logoColor=white'
        ')'
    )
    plot_markdown = f'![Kaggle leaderboard score distribution]({PLOT_PATH})'

    with open(README_PATH, 'r', encoding='utf-8') as handle:
        content = handle.read()

    content, badge_replacements = re.subn(
        r'(<!-- KAGGLE_BADGE_START -->).*?(<!-- KAGGLE_BADGE_END -->)',
        f'\\1\n{badge_markdown} {score_badge_markdown}\n\\2',
        content,
        flags=re.DOTALL,
    )
    if badge_replacements != 1:
        raise RuntimeError('README badge markers were not found exactly once.')

    content, plot_replacements = re.subn(
        r'(<!-- KAGGLE_RANK_PLOT_START -->).*?(<!-- KAGGLE_RANK_PLOT_END -->)',
        f'\\1\n{plot_markdown}\n\\2',
        content,
        flags=re.DOTALL,
    )
    if plot_replacements != 1:
        raise RuntimeError('README rank plot markers were not found exactly once.')

    with open(README_PATH, 'w', encoding='utf-8') as handle:
        handle.write(content)


def main():
    wait_for_completion()
    submissions_df = fetch_kaggle_submissions(COMP_ID)

    try:
        leaderboard_df = fetch_kaggle_leaderboard_pages(COMP_ID)
    except KaggleRateLimitError as error:
        print(
            'Warning: leaderboard fetch is still rate limited by Kaggle (429). '
            'Skipping README rank/plot update for this run so the workflow can complete. '
            f'Detail: {error}'
        )
        # Keep submission workflow green; rank update will occur on next successful run.
        sys.exit(0)

    if submissions_df is None or submissions_df.empty:
        print('Kaggle submissions export is empty.')
        sys.exit(1)

    if leaderboard_df is None or leaderboard_df.empty:
        print('Kaggle leaderboard export is empty.')
        sys.exit(1)

    save_kaggle_exports(submissions_df, leaderboard_df)
    my_score = resolve_best_submission_score(submissions_df)
    leaderboard_scores, my_rank = calculate_rank_from_scores(leaderboard_df, my_score)
    write_plot(leaderboard_scores, my_rank, my_score)
    update_readme(my_rank, my_score)

    print(f'Updated README badges: rank={my_rank}, best_leaderboard_ba={my_score:.5f}')
    print(f'Wrote leaderboard plot to {PLOT_PATH}')


if __name__ == '__main__':
    main()
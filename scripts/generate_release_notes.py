import argparse
import pathlib
import subprocess
import sys

import yaml


def run_git(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f'Git command failed: {" ".join(command)}')
    return result.stdout.strip()


def get_previous_tag(tag_name):
    try:
        previous_tag = run_git(['git', 'describe', '--tags', '--abbrev=0', f'{tag_name}^'])
    except Exception:
        previous_tag = ''
    return previous_tag


def load_release_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_commits(range_spec, excluded_authors):
    raw_log = run_git([
        'git',
        'log',
        '--reverse',
        '--format=%H%x1f%an%x1f%s%x1f%b%x1e',
        range_spec,
    ])

    commits = []
    for record in raw_log.split('\x1e'):
        record = record.strip()
        if not record:
            continue

        parts = record.split('\x1f')
        if len(parts) < 4:
            continue

        sha, author, subject, body = parts[0], parts[1], parts[2], '\x1f'.join(parts[3:])
        if author in excluded_authors:
            continue

        commits.append({
            'sha': sha,
            'author': author,
            'subject': subject.strip(),
            'body': body.strip(),
        })

    return commits


def classify_commit(subject, categories, fallback_title):
    lowered_subject = subject.lower()

    for category in categories:
        prefixes = category.get('commit-prefixes') or []
        for prefix in prefixes:
            if lowered_subject.startswith(prefix.lower()):
                cleaned = subject[len(prefix):].strip()
                if cleaned.startswith(':'):
                    cleaned = cleaned[1:].strip()
                return category['title'], cleaned or subject

    return fallback_title, subject


def build_notes(tag_name, commits, config):
    changelog = config.get('changelog', {})
    categories = changelog.get('categories', [])
    excluded_authors = set(changelog.get('exclude', {}).get('authors', []))

    if not categories:
        categories = [{'title': 'Changes'}]

    fallback_category = next(
        (category for category in categories if not category.get('commit-prefixes')),
        {'title': 'General maintenance'},
    )

    grouped = {category['title']: [] for category in categories}
    if fallback_category['title'] not in grouped:
        grouped[fallback_category['title']] = []

    for commit in commits:
        title, cleaned_subject = classify_commit(
            commit['subject'],
            categories,
            fallback_category['title'],
        )
        grouped.setdefault(title, []).append(cleaned_subject)

    lines = [f'# Release Notes for {tag_name}', '']

    for category in categories:
        title = category['title']
        entries = grouped.get(title, [])
        if not entries:
            continue

        lines.append(f'## {title}')
        collapse_after = category.get('collapse_after')
        if collapse_after and len(entries) > collapse_after:
            visible = entries[:collapse_after]
            hidden_count = len(entries) - collapse_after
        else:
            visible = entries
            hidden_count = 0

        for entry in visible:
            lines.append(f'- {entry}')

        if hidden_count:
            lines.append(f'- ... and {hidden_count} more commit(s)')

        lines.append('')

    if not any(grouped.values()):
        lines.extend(['## Changes', '- No qualifying commits found.', ''])

    return '\n'.join(lines).rstrip() + '\n'


def main():
    parser = argparse.ArgumentParser(description='Generate release notes from .github/release.yml rules.')
    parser.add_argument('--config', default='.github/release.yml')
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f'Config file not found: {config_path}', file=sys.stderr)
        sys.exit(1)

    config = load_release_config(config_path)
    previous_tag = get_previous_tag(args.tag)
    range_spec = f'{previous_tag}..{args.tag}' if previous_tag else args.tag
    excluded_authors = set(config.get('changelog', {}).get('exclude', {}).get('authors', []))
    commits = load_commits(range_spec, excluded_authors)
    notes = build_notes(args.tag, commits, config)

    output_path = pathlib.Path(args.output)
    output_path.write_text(notes, encoding='utf-8')
    print(notes, end='')


if __name__ == '__main__':
    main()
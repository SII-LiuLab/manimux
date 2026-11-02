#!/usr/bin/env python3
"""Portable, offline ManiMux clip intake. Python standard library + FFmpeg."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile

MODES = {'real_robot_policy', 'teleoperation', 'replay', 'synthetic'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def run(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f'{command[0]} failed: {result.stderr[-2000:]}')
    return result


def probe(path):
    value = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                            '-of', 'json', str(path)]).stdout)
    streams = value.get('streams', [])
    videos = [s for s in streams if s.get('codec_type') == 'video']
    if len(videos) != 1:
        raise ValueError('Expected exactly one video stream')
    stream = videos[0]
    fps = float(Fraction(stream.get('avg_frame_rate', '0/1')))
    return {'width': int(stream['width']), 'height': int(stream['height']),
            'fps': fps, 'codec': stream.get('codec_name'), 'pix_fmt': stream.get('pix_fmt'),
            'duration_seconds': float(stream.get('duration') or value['format']['duration']),
            'has_audio': any(s.get('codec_type') == 'audio' for s in streams)}


def clip_path(root, relative):
    if not isinstance(relative, str):
        raise ValueError('Clip path must be a relative string')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'clips':
        raise ValueError('Clip path must stay under clips/')
    if '\\' in relative or path.suffix.lower() != '.mp4':
        raise ValueError('Use a relative POSIX .mp4 path')
    candidate = root.joinpath(*path.parts)
    if any(p.is_symlink() for p in [candidate, *candidate.parents] if p != root.parent):
        raise ValueError('Symlinks are not accepted in submissions')
    candidate.resolve().relative_to(root.resolve())
    if not candidate.is_file():
        raise ValueError(f'Missing clip: {relative}')
    return candidate


def nonempty(value):
    return isinstance(value, str) and bool(value.strip()) and not value.startswith('<')


def init(args):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{1,63}', args.station):
        raise ValueError('Station ID must be 2–64 lowercase letters, digits or hyphens')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'clips/take-001').mkdir(parents=True)
    template = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'station': {'id': args.station, 'embodiment': None, 'os': None,
                    'manimux_commit': None, 'manimux_dirty': None},
        'content_reviewed_for_sharing': False,
        'takes': [{'id': 'take-001', 'task_prompt': None, 'evidence_mode': None,
                   'policy': {'name': None, 'checkpoint_id': None},
                   'inference': {'name': None, 'settings': {}},
                   'session_id': None, 'config_label': None, 'config_sha256': None,
                   'trajectory_source': None, 'trajectory_display': None,
                   'execution_evidence': None, 'task_outcome': 'unknown',
                   'outcome_basis': None, 'camera_names': [],
                   'gui_physical_camera_visible': False, 'visual_reviewed': False,
                   'model_identity_visible': False, 'live_inference_visible': False,
                   'trajectory_visible': False, 'inference_visibility_evidence': None,
                   'capture_issues': [],
                   'gui': {'path': 'clips/take-001/gui.mp4',
                           'suggested_in_seconds': None, 'suggested_out_seconds': None},
                   'external_robot': None, 'evaluation_clip': None,
                   'evaluation': None}],
        'notes': []}
    write_json(args.out / 'manifest.json', template)
    print(f'Fill {args.out / "manifest.json"} from actual local evidence; no services started.')


def normalize(args):
    if args.output.exists() or args.output.with_suffix('.source.json').exists():
        raise ValueError('Output or source sidecar already exists')
    if args.output.suffix.lower() != '.mp4' or not args.input.is_file():
        raise ValueError('Existing input and a new .mp4 output are required')
    original = probe(args.input)
    original_sha = sha(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # No upscaling; preserve geometry, make dimensions even, normalize cadence only.
    vf = f"scale=w='trunc(min({args.max_width},iw)/2)*2':h=-2,fps=30"
    run(['ffmpeg', '-v', 'error', '-n', '-threads', '1', '-i', str(args.input),
         '-map', '0:v:0', '-an', '-map_metadata', '-1', '-vf', vf, '-filter_threads', '1',
         '-c:v', 'libx264', '-threads', '1', '-preset', 'veryfast', '-crf', '19',
         '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)])
    write_json(args.output.with_suffix('.source.json'),
               {'source_sha256': original_sha, 'source_media': original,
                'operations': ['downscale if necessary; no upscaling', '30 fps cadence conversion',
                               'remove audio/metadata', 'H.264 yuv420p encode; no trim or speed change']})
    print(f'Wrote {args.output}; review original cadence and actual playback.')


def validate(root, decode=False):
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    errors, warnings, media, files = [], [], {}, set()
    previous_path = root / 'validation.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if manifest.get('schema_version') != 1:
        errors.append('schema_version must be 1')
    station = manifest.get('station', {})
    for field in ('id', 'embodiment', 'os'):
        if not nonempty(station.get(field)):
            errors.append(f'station.{field} is required')
    if not station.get('manimux_commit'):
        warnings.append('ManiMux revision unknown; do not infer it from another checkout')
    if manifest.get('content_reviewed_for_sharing') is not True:
        errors.append('Review shareable footage and metadata, then set content_reviewed_for_sharing=true')
    takes = manifest.get('takes')
    if not isinstance(takes, list) or not takes:
        raise ValueError('At least one take is required')
    seen, eligible, gui_hashes = set(), [], set()
    for take in takes:
        before = len(errors)
        take_id = take.get('id')
        if not nonempty(take_id) or take_id in seen:
            errors.append('Every take needs a distinct ID')
        seen.add(take_id)
        label = str(take_id)
        for field in ('task_prompt', 'execution_evidence'):
            if not nonempty(take.get(field)):
                errors.append(f'{label}: {field} is required')
        mode = take.get('evidence_mode')
        if mode not in MODES:
            errors.append(f'{label}: choose an explicit evidence_mode')
        elif mode != 'real_robot_policy':
            warnings.append(f'{label}: {mode}; excluded from real-policy wall eligibility')
        if mode == 'real_robot_policy':
            if take.get('trajectory_source') != 'model_inference':
                errors.append(f'{label}: primary footage requires trajectories generated by actual model inference')
            if take.get('trajectory_display') not in {'predicted_chunks', 'executed_commands', 'measured_motion'}:
                errors.append(f'{label}: primary footage must show predictions, executed commands or measured motion from the real run')
            for field in ('gui_physical_camera_visible', 'model_identity_visible', 'live_inference_visible', 'trajectory_visible'):
                if take.get(field) is not True:
                    errors.append(f'{label}: {field} must be visible in the same nominated shot')
            if not nonempty(take.get('inference_visibility_evidence')):
                errors.append(f'{label}: describe the actual live inference signal and its source')
            for field in ('name', 'checkpoint_id'):
                value = (take.get('policy') or {}).get(field)
                if not nonempty(value) or value.strip().lower() in {'unknown', 'none', 'n/a'}:
                    errors.append(f'{label}: a known public policy.{field} is required for real-model footage')
            for section in ('policy', 'inference'):
                name = (take.get(section) or {}).get('name')
                if not nonempty(name):
                    errors.append(f'{label}: {section}.name required; use unknown with an explanation if unresolved')
                elif name.lower() == 'unknown':
                    warnings.append(f'{label}: {section} identity unknown')
        if take.get('task_outcome') not in {'success', 'failure', 'unknown'}:
            errors.append(f'{label}: task_outcome must be success/failure/unknown')
        elif take.get('task_outcome') != 'unknown' and not nonempty(take.get('outcome_basis')):
            errors.append(f'{label}: a claimed outcome needs outcome_basis')
        if take.get('visual_reviewed') is not True:
            errors.append(f'{label}: watch the nominated interval before visual_reviewed=true')
        if take.get('capture_issues'):
            warnings.append(f'{label}: capture limitations disclosed: {take["capture_issues"]}')
        for role in ('gui', 'external_robot', 'evaluation_clip'):
            item = take.get(role)
            if role != 'gui' and item is None:
                continue
            try:
                if not isinstance(item, dict):
                    raise ValueError(f'{role} requires a path object')
                relative = item.get('path')
                path = clip_path(root, relative)
                if relative in files:
                    raise ValueError('A clip path cannot be counted as multiple takes or roles')
                files.add(relative)
                details = probe(path)
                checksum = sha(path)
                old = previous.get('media', {}).get(relative, {})
                decoded = old.get('sha256') == checksum and old.get('decode_ok') is True
                if decode:
                    run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '1', '-i', str(path),
                         '-map', '0:v:0', '-an', '-threads', '1', '-f', 'null', '-'])
                    decoded = True
                media[relative] = {**details, 'sha256': checksum, 'decode_ok': decoded}
                w, h, fps = details['width'], details['height'], details['fps']
                if w < 1280 or h < 720 or abs(w / h - 16 / 9) > .02:
                    errors.append(f'{label}/{role}: need landscape 16:9 and at least 1280x720')
                if w < 1920:
                    warnings.append(f'{label}/{role}: 720p-class source; use for smaller tiles')
                if not 29 <= fps <= 31:
                    errors.append(f'{label}/{role}: normalize to 30 fps (original {fps:.3f})')
                if details['codec'] != 'h264' or details['pix_fmt'] != 'yuv420p' or details['has_audio']:
                    errors.append(f'{label}/{role}: submit silent H.264 yuv420p MP4')
                sidecar = path.with_suffix('.source.json')
                if sidecar.exists():
                    if sidecar.is_symlink():
                        raise ValueError('Source sidecar cannot be a symlink')
                    source = json.loads(sidecar.read_text())
                    if source.get('source_media', {}).get('fps', 30) < 24:
                        warnings.append(f'{label}/{role}: original source below 24fps; conversion does not repair motion')
                    files.add(sidecar.relative_to(root).as_posix())
                if role == 'gui':
                    if checksum in gui_hashes:
                        errors.append(f'{label}: duplicate GUI media; do not count one clip as different takes')
                    gui_hashes.add(checksum)
                    start, end = item.get('suggested_in_seconds'), item.get('suggested_out_seconds')
                    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)):
                        errors.append(f'{label}: nominate numeric in/out seconds')
                    elif not 0 <= start < end <= details['duration_seconds'] + .05 or end - start < 12:
                        errors.append(f'{label}: nominated interval must be continuous, inside the clip, and at least 12 seconds')
                if role == 'external_robot':
                    offset = item.get('offset_from_gui_seconds')
                    if offset is not None and (isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset)):
                        errors.append(f'{label}: external offset must be numeric or null')
                    if offset is not None and not nonempty(item.get('sync_method')):
                        errors.append(f'{label}: explain how synchronization offset was established')
            except (ValueError, OSError, KeyError, ZeroDivisionError) as exc:
                errors.append(f'{label}/{role}: {exc}')
        if mode == 'real_robot_policy' and len(errors) == before:
            eligible.append(take_id)
    if not eligible:
        warnings.append('No eligible real-robot policy takes: this is not a completed target wall contribution')
    report = {'ok': not errors, 'errors': errors, 'warnings': warnings, 'media': media,
              'real_policy_wall_take_ids': eligible, 'files': sorted(files),
              'manifest_sha256': sha(root / 'manifest.json'),
              'note': 'Technical checks and self-reported provenance; physical execution and visual quality require human review.'}
    write_json(root / 'validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('init'); p.add_argument('--out', required=True, type=Path); p.add_argument('--station', required=True)
    p = sub.add_parser('normalize'); p.add_argument('--input', required=True, type=Path); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--max-width', type=int, choices=[1280, 1920, 2560], default=2560)
    p = sub.add_parser('validate'); p.add_argument('root', type=Path); p.add_argument('--decode', action='store_true')
    p = sub.add_parser('pack'); p.add_argument('root', type=Path); p.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if hasattr(os, 'nice'):
        os.nice(10)
    if args.command == 'init':
        init(args); return
    if not shutil.which('ffprobe') or not shutil.which('ffmpeg'):
        raise ValueError('Install/use local ffmpeg and ffprobe before media processing')
    if args.command == 'normalize':
        normalize(args); return
    report = validate(args.root, getattr(args, 'decode', False))
    print(json.dumps({k: report[k] for k in ('ok', 'errors', 'warnings', 'real_policy_wall_take_ids')}, indent=2))
    if not report['ok']:
        sys.exit(1)
    if args.command == 'pack':
        if any(not item['decode_ok'] for item in report['media'].values()):
            raise ValueError('Run validate --decode first; all media must have a successful decode check')
        if args.output.exists():
            raise ValueError('Choose a new ZIP output path')
        paths = ['manifest.json', 'validation.json', *report['files']]
        sums = ''.join(f'{sha(args.root / name)}  {name}\n' for name in sorted(paths))
        (args.root / 'SHA256SUMS').write_text(sums, encoding='utf-8')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # MP4 is already compressed; store avoids wasting station CPU on ZIP deflate.
        with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_STORED) as archive:
            for name in [*sorted(paths), 'SHA256SUMS']:
                archive.write(args.root / name, name)
        print(f'Package: {args.output.resolve()}\nSHA256: {sha(args.output)}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

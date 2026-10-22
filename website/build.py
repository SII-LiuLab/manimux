"""Build the landing page and canonical root guideline.html for GitHub Pages."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import re
import shutil

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEST = ROOT / 'dist'
GITHUB = 'https://github.com/SII-LiuLab/manimux/blob/main/'


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get('id'):
            if attrs['id'] in self.ids:
                raise ValueError(f"Duplicate HTML id: {attrs['id']}")
            self.ids.add(attrs['id'])
        for key in ('src', 'href', 'poster', 'data-src'):
            if attrs.get(key):
                self.links.append(attrs[key])


def parse(text):
    page = PageLinks()
    page.feed(text)
    return page


def repo_link(match):
    target = match.group(1)
    path = urlsplit(target).path
    if path.startswith(('.agents/', 'manimux/')):
        if not (REPO / path).is_file():
            raise ValueError(f'Missing guide repository reference: {path}')
        return f'href="{GITHUB}{target}"'
    return match.group(0)


if __name__ == '__main__':
    # Clean only this generated directory to avoid publishing obsolete media.
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir()
    for name in ('index.html', 'styles.css', 'app.js'):
        shutil.copy2(ROOT / name, DEST / name)
    shutil.copytree(ROOT / 'assets', DEST / 'assets')

    # Serve the README's original video directly, without GitHub's file-size preview limit.
    readme_video = Path('assets/dream1-task-1440p30.mp4')
    shutil.copy2(REPO / readme_video, DEST / readme_video)

    guide = re.sub(r'href="([^"]+)"', repo_link,
                   (REPO / 'guideline.html').read_text())
    (DEST / 'guideline.html').write_text(guide)
    guide_page = parse(guide)
    for link in guide_page.links:
        if link.startswith('assets/'):
            source = REPO / urlsplit(link).path
            target = DEST / urlsplit(link).path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    home = parse((DEST / 'index.html').read_text())
    # Preserve old documentation deep links except hashes now owned by the home.
    anchors = sorted(guide_page.ids - home.ids)
    redirect = ('const guideAnchors = new Set(' + json.dumps(anchors) + ');\n'
                'function redirectGuide() {\n'
                '  const id = location.hash.slice(1);\n'
                '  if (guideAnchors.has(id) || id === "manimux")\n'
                '    location.replace("guideline.html#" + (id === "manimux" ? "overview" : id));\n'
                '}\nredirectGuide();\naddEventListener("hashchange", redirectGuide);\n')
    (DEST / 'guide-redirect.js').write_text(redirect)
    (DEST / '.nojekyll').touch()

    pages = {'index.html': home, 'guideline.html': guide_page}
    for name, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            path = unquote(url.path)
            target = DEST / path if path else DEST / name
            if not target.exists():
                raise ValueError(f'{name}: missing asset {link}')
            other = pages.get(path or name)
            if url.fragment and other and unquote(url.fragment) not in other.ids:
                raise ValueError(f'{name}: missing anchor {link}')
    print(f'Validated {sum(len(p.links) for p in pages.values())} references; built {DEST}')

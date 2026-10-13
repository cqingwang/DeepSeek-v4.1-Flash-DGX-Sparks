"""Fast TCP check of the Docker registry pull proxy; never print credentials."""
import json
import socket
import sys
from urllib.parse import urlsplit


def check_proxies(info, image=''):
    first = image.split('/', 1)[0]
    registry = (first if '/' in image and ('.' in first or ':' in first or first == 'localhost')
                else 'docker.io')
    host = urlsplit('//' + registry).hostname
    for bypass in (info.get('NoProxy') or info.get('NOProxy') or '').split(','):
        bypass = bypass.strip().lower()
        if bypass == '*' or bypass == host or (bypass.startswith('.') and host.endswith(bypass)):
            return

    # Registry pulls use HTTPS; an HTTP URL here is the proxy transport, not
    # the registry protocol. HttpProxy does not apply to these requests.
    field = 'HttpsProxy' if info.get('HttpsProxy') else 'HTTPSProxy'
    value = info.get(field, '')
    if not value:
        return
    parsed = urlsplit(value if '://' in value else 'http://' + value)
    try:
        endpoint = (parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80))
    except ValueError:
        raise RuntimeError(f'docker {field}: invalid proxy port') from None
    if not endpoint[0]:
        raise RuntimeError(f'docker {field}: missing proxy host')
    try:
        with socket.create_connection(endpoint, timeout=2):
            pass
    except OSError:
        raise RuntimeError(f'docker proxy {endpoint[0]}:{endpoint[1]} unreachable '
                           '(2s TCP timeout); fix daemon proxy or use SKIP_PULL=1') from None


if __name__ == '__main__':
    try:
        check_proxies(json.load(sys.stdin), sys.argv[1])
    except (RuntimeError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)

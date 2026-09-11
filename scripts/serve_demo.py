import http.server
import socketserver
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
PROJECT_ROOT = '/Users/vanch/mlx-AuK'
os.chdir(PROJECT_ROOT)

Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map.update({
    '.wav': 'audio/wav',
    '.mp3': 'audio/mpeg',
    '.json': 'application/json',
})

print()
print('=' * 70)
print('  MLX-AuK 官方 Demo 全量对比与 RTF 评测看板已启动!')
print('  请在浏览器访问: http://localhost:%d/web/index.html' % PORT)
print('=' * 70)
print()

try:
    with socketserver.TCPServer(('', PORT), Handler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print('Server stopped.')
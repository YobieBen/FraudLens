#!/usr/bin/env python3
"""Simple test API server for E2E tests"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading
import time

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'status': 'healthy',
                'timestamp': time.time(),
                'service': 'fraudlens-api'
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress log messages"""
        pass

def start_test_server():
    """Start test server in background"""
    server = HTTPServer(('localhost', 8000), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

if __name__ == '__main__':
    server = start_test_server()
    print("Test API server running on http://localhost:8000")
    print("Health endpoint: http://localhost:8000/health")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
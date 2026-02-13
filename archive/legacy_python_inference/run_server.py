import sys
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import time

# Ensure core module is importable
sys.path.append(".")

try:
    from core.inference import engine
    from core.config import PORT, HOST
except ImportError as e:
    print(f"Error importing core modules: {e}")
    sys.exit(1)

class NPCRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok", 
                "model_loaded": engine.model is not None
            }).encode())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == '/generate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # Extract context
                context = data.get("context", {})
                player_input = data.get("player_input", "")
                
                # Build prompt
                persona = context.get("persona", "You are a helpful NPC.")
                scenario = context.get("scenario", "")
                state = context.get("behavior_state", "idle")
                
                system_msg = f"{persona}\nScenario: {scenario}\nCurrent State: {state}"
                
                # Format and generate
                prompt = engine.format_prompt(system_msg, player_input)
                response = engine.generate(prompt, npc_name=context.get("npc_id", "NPC"))
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "response": response,
                    "npc_id": context.get("npc_id", "NPC"),
                    "success": True
                }).encode())
                
            except Exception as e:
                print(f"Error processing request: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
            return
            
        self.send_response(404)
        self.end_headers()
    
    def log_message(self, format, *args):
        # Use stderr for logging to avoid buffering issues
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.client_address[0],
                          self.log_date_time_string(),
                          format % args))

if __name__ == "__main__":
    print("-" * 50)
    print("NPC AI Server - Clean Architecture")
    print("-" * 50)
    
    # Load model on startup (blocking)
    engine.load_model()
    
    server = HTTPServer((HOST, PORT), NPCRequestHandler)
    print(f"Server running on http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.server_close()

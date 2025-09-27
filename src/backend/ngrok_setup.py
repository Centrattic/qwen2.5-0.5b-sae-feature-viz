import subprocess
import time
import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class NgrokManager:
    def __init__(self, port: int = 8000):
        self.port = port
        self.ngrok_process = None
        self.public_url = None
    
    def start_ngrok(self) -> str:
        """Start ngrok tunnel and return public URL"""
        try:
            # Start ngrok process
            self.ngrok_process = subprocess.Popen([
                "ngrok", "http", str(self.port),
                "--log=stdout"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for ngrok to start
            time.sleep(3)
            
            # Get public URL from ngrok API
            self.public_url = self._get_ngrok_url()
            
            if self.public_url:
                logger.info(f"Ngrok tunnel started: {self.public_url}")
                return self.public_url
            else:
                raise Exception("Failed to get ngrok URL")
                
        except Exception as e:
            logger.error(f"Error starting ngrok: {e}")
            raise
    
    def stop_ngrok(self) -> None:
        """Stop ngrok tunnel"""
        if self.ngrok_process:
            self.ngrok_process.terminate()
            self.ngrok_process.wait()
            logger.info("Ngrok tunnel stopped")
    
    def _get_ngrok_url(self) -> Optional[str]:
        """Get public URL from ngrok API"""
        try:
            response = requests.get("http://localhost:4040/api/tunnels")
            data = response.json()
            
            for tunnel in data.get("tunnels", []):
                if tunnel.get("proto") == "https":
                    return tunnel.get("public_url")
            
            # Fallback to http
            for tunnel in data.get("tunnels", []):
                if tunnel.get("proto") == "http":
                    return tunnel.get("public_url")
                    
        except Exception as e:
            logger.error(f"Error getting ngrok URL: {e}")
        
        return None

def main():
    """Main function to start ngrok and backend server"""
    logging.basicConfig(level=logging.INFO)
    
    # Start ngrok
    ngrok_manager = NgrokManager()
    public_url = ngrok_manager.start_ngrok()
    
    print(f"Backend server available at: {public_url}")
    print("Press Ctrl+C to stop")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ngrok_manager.stop_ngrok()
        print("Server stopped")

if __name__ == "__main__":
    main()

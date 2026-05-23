import os
import sys
import uvicorn

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)

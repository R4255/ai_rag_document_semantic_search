import httpx
import os
import time
import asyncio

BASE_URL = "http://localhost:8000/api"

async def test_backend():
    print("🚀 Starting Backend Tests...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Test Health
        r = await client.get(f"{BASE_URL}/health")
        print("\n[Health Check]:", r.status_code, r.json())
        
        # 2. Test chat (Miss)
        query = "What is the system baseline document?"
        print(f"\n[Test Chat] Query: '{query}'")
        
        t0 = time.perf_counter()
        async with client.stream("POST", f"{BASE_URL}/chat", json={"query": query}) as response:
            answer = ""
            async for chunk in response.aiter_text():
                # parse SSE somewhat
                for line in chunk.split("\n"):
                    if line.startswith("data: "):
                        import json
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content":
                                answer += data.get("content", "")
                        except: pass
        t_miss = time.perf_counter() - t0
        print(f"  First call (Cache Miss): {t_miss:.3f}s")
        print(f"  Answer preview: {answer[:60]}...")
        
        # 3. Test chat (Hit)
        t0 = time.perf_counter()
        async with client.stream("POST", f"{BASE_URL}/chat", json={"query": query}) as response:
            answer_hit = ""
            async for chunk in response.aiter_text():
                for line in chunk.split("\n"):
                    if line.startswith("data: "):
                        import json
                        try:
                            data = json.loads(line[6:])
                            if data.get("type") == "content":
                                answer_hit += data.get("content", "")
                        except: pass
        t_hit = time.perf_counter() - t0
        print(f"  Second call (Cache Hit): {t_hit:.3f}s")
        print(f"  Answer preview: {answer_hit[:60]}...")
        
        # 4. Check Cache Stats
        r = await client.get(f"{BASE_URL}/cache/stats")
        print("\n[Cache Stats]:", r.status_code, r.json())

if __name__ == "__main__":
    asyncio.run(test_backend())


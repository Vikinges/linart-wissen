import json, sys, time, urllib.request
qs = ["Как подключить устройство к Wi-Fi? Перечисли шаги.",
      "Welche App wird für die WLAN-Verbindung benötigt und welches WLAN-Band wird unterstützt?",
      "What is the Modbus register for the flow temperature?"]
for q in qs:
    t = time.time()
    req = urllib.request.Request("http://127.0.0.1:7890/api/ask", data=json.dumps({"q": q}).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    print("=" * 80); print("Q:", q); print(r["answer"])
    print("SOURCES:", [(s["n"], s["file"], s["page"]) for s in r["sources"][:5]], f"{time.time()-t:.1f}s")

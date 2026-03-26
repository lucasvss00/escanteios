"""Script de debug — mostra o formato real retornado pela API."""
import argparse
import json
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--token", required=True)
parser.add_argument("--event-id", default="11538610")
args = parser.parse_args()

print(f"\nConsultando stats_trend para event_id={args.event_id}...\n")

r = requests.get(
    "https://api.b365api.com/v1/event/stats_trend",
    params={"event_id": args.event_id, "token": args.token},
)
data = r.json()
results = data.get("results", [])

print(f"Tipo de results  : {type(results)}")
print(f"Total de snapshots: {len(results)}")

if results:
    print(f"\nTipo do 1º snapshot: {type(results[0])}")
    print(f"\n--- Primeiro snapshot ---")
    print(json.dumps(results[0], indent=2))
    if len(results) > 1:
        print(f"\n--- Segundo snapshot ---")
        print(json.dumps(results[1], indent=2))
else:
    print("\nSem dados — response completa:")
    print(json.dumps(data, indent=2))

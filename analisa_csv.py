import pandas as pd
import sys

df = pd.read_csv(r'C:\Users\Lucas\Desktop\FIFA\escanteios\dados_escanteios\panorama_jogos.csv')

print(f"Total de jogos no arquivo: {len(df)}")
print()

d15 = df[df['kickoff_dt'].str.startswith('2026-03-15', na=False)].copy()
print(f"=== DIA 15/03/2026 ===")
print(f"Total jogos: {len(d15)}")
print(f"has_stats True: {(d15['has_stats']==True).sum()}")
print(f"corners_total preenchido: {d15['corners_total'].notna().sum()}")
print(f"corners_total NaN: {d15['corners_total'].isna().sum()}")
if d15['corners_total'].notna().any():
    print(f"Media corners: {d15['corners_total'].mean():.2f}")
    print(f"Min/Max: {d15['corners_total'].min()} / {d15['corners_total'].max()}")

print()
print(f"--- Pag 1 (primeiros 50) ---")
p1 = d15.head(50)
print(f"  corners preenchidos: {p1['corners_total'].notna().sum()}/{len(p1)}")
print(f"  has_stats True: {(p1['has_stats']==True).sum()}/{len(p1)}")

if len(d15) > 50:
    p2 = d15.iloc[50:]
    print(f"\n--- Pag 2+ (jogos 51 em diante) ---")
    print(f"  quantidade: {len(p2)}")
    print(f"  corners preenchidos: {p2['corners_total'].notna().sum()}/{len(p2)}")
    print(f"  has_stats True: {(p2['has_stats']==True).sum()}/{len(p2)}")
    if p2['corners_total'].notna().any():
        print(f"  media corners: {p2['corners_total'].mean():.2f}")
    # Mostrar alguns exemplos
    print(f"\nExemplos pag 2+ (primeiros 5):")
    cols = ['event_id','home_team','away_team','corners_total','has_stats']
    print(p2[cols].head(5).to_string(index=False))
else:
    print(f"\n(Apenas {len(d15)} jogos no dia 15 - paginacao nao atingida)")

sys.stdout.flush()

$d15 = Import-Csv 'C:\Users\Lucas\Desktop\FIFA\escanteios\dados_escanteios\panorama_jogos.csv' | Where-Object { $_.kickoff_dt -like '2026-03-15*' }
$p2  = $d15 | Select-Object -Skip 50
Write-Host "Dia 15 total: $($d15.Count)"
Write-Host "Com corners: $(($d15 | Where-Object { $_.corners_total -ne '' }).Count)"
Write-Host "Sem corners: $(($d15 | Where-Object { $_.corners_total -eq '' }).Count)"
Write-Host "P2+ (skip 50): $($p2.Count)"
Write-Host "P2+ com corners: $(($p2 | Where-Object { $_.corners_total -ne '' }).Count)"
Write-Host "P2+ sem corners: $(($p2 | Where-Object { $_.corners_total -eq '' }).Count)"
Write-Host "P2+ has_stats True: $(($p2 | Where-Object { $_.has_stats -eq 'True' }).Count)"

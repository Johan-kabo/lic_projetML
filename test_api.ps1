# Script de test pour l'API Fraude Detection
# Lance avec: powershell -ExecutionPolicy Bypass -File test_api.ps1

# Variables de la transaction
$Time      = 12345.0
$V1        = 0.1
$V2        = -0.2
$V3        = 0.05
$V4        = -0.01
$V5        = 0.03
$V6        = 0.0
$V7        = -0.02
$V8        = 0.04
$V9        = 0.01
$V10       = -0.03
$V11       = 0.02
$V12       = -0.01
$V13       = 0.0
$V14       = 0.005
$V15       = -0.005
$V16       = 0.002
$V17       = -0.002
$V18       = 0.001
$V19       = -0.001
$V20       = 0.0
$V21       = 0.0
$V22       = 0.0
$V23       = 0.0
$V24       = 0.0
$V25       = 0.0
$V26       = 0.0
$V27       = 0.0
$V28       = 0.0
$Amount    = 100.0

# Construire le JSON (sans parse errors)
$body = @{
    Time   = $Time
    V1     = $V1
    V2     = $V2
    V3     = $V3
    V4     = $V4
    V5     = $V5
    V6     = $V6
    V7     = $V7
    V8     = $V8
    V9     = $V9
    V10    = $V10
    V11    = $V11
    V12    = $V12
    V13    = $V13
    V14    = $V14
    V15    = $V15
    V16    = $V16
    V17    = $V17
    V18    = $V18
    V19    = $V19
    V20    = $V20
    V21    = $V21
    V22    = $V22
    V23    = $V23
    V24    = $V24
    V25    = $V25
    V26    = $V26
    V27    = $V27
    V28    = $V28
    Amount = $Amount
} | ConvertTo-Json

Write-Host "=== TEST API FRAUDE DETECTION ===" -ForegroundColor Green
Write-Host "Envoi d'une requête POST vers /predict-fraud..." -ForegroundColor Cyan

try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/predict-fraud" `
                                  -Method POST `
                                  -ContentType "application/json" `
                                  -Body $body

    Write-Host "✅ Réponse reçue (Status: $($response.StatusCode))" -ForegroundColor Green
    $jsonResponse = $response.Content | ConvertFrom-Json
    Write-Host ($jsonResponse | ConvertTo-Json -Depth 10) -ForegroundColor White
}
catch {
    Write-Host "❌ Erreur : $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== RÃ‰CUPÃ‰RATION DE L'HISTORIQUE ===" -ForegroundColor Green
Write-Host "Requête GET vers /get-history..." -ForegroundColor Cyan

try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/get-history?limit=10" `
                                  -Method GET

    Write-Host "✅ Historique reçu (Status: $($response.StatusCode))" -ForegroundColor Green
    $jsonResponse = $response.Content | ConvertFrom-Json
    Write-Host ($jsonResponse | ConvertTo-Json -Depth 10) -ForegroundColor White
}
catch {
    Write-Host "❌ Erreur : $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== MONITORING - GUICHET CENTRAL ===" -ForegroundColor Green
Write-Host "Requête GET vers /get-monitoring (Serveur de Mémoire)..." -ForegroundColor Cyan

try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:8000/get-monitoring" `
                                  -Method GET

    Write-Host "✅ Monitoring reçu (Status: $($response.StatusCode))" -ForegroundColor Green
    $jsonResponse = $response.Content | ConvertFrom-Json
    Write-Host ($jsonResponse | ConvertTo-Json -Depth 10) -ForegroundColor White
}
catch {
    Write-Host "❌ Erreur : $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== FIN DU TEST ===" -ForegroundColor Green

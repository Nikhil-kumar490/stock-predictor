# Stock Predictor — Docker Setup

## Run dashboard

`ash
docker-compose up --build
`

Open http://localhost:5000

## Train a ticker

`ash
docker run --rm -v \C:\Users\927ni\my\stock-predictor/model:/app/model stock-predictor python model/train.py --ticker AAPL --epochs 50
`

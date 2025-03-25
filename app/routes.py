import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

from flask import Blueprint, render_template, request, jsonify
from predict import predict_next

main = Blueprint('main', __name__)


@main.route('/')
def index():
    return render_template('dashboard.html')


@main.route('/api/predict')
def api_predict():
    ticker = request.args.get('ticker', 'AAPL').upper()
    days = int(request.args.get('days', 30))
    try:
        data = predict_next(ticker, days)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': f'Model for {ticker} not found. Run model/train.py --ticker {ticker} first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

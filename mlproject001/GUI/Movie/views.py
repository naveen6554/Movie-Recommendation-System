from django.shortcuts import render
from django.contrib.auth.decorators import login_required
import pandas as pd
import joblib
import os

def predict_movie(request):
    # Load dataset to get unique names for dropdowns
    df = pd.read_csv("C:\\Users\\nk568\\OneDrive\\Desktop\\mlproject001\\Letterbox Movie Classification\\Letterbox Movie Classification Dataset.csv")
    
    # Extract unique values for text-based columns from the screenshot
    context = {
        'directors': sorted(df['Director'].unique()),
        'languages': sorted(df['Original_language'].unique()),
        'genres': sorted(df['Genres'].unique()),
        'studios': sorted(df['Studios'].unique()),
        'titles': sorted(df['Film_title'].unique()),
        'descriptions': sorted(df['Description'].unique()),
    }

    if request.method == "POST":
        # 1. Capture inputs (Mapping text to their LabelEncoded integer index)
        # We find the index of the selected text to match what the model learned
        input_features = [
            int(request.POST.get('index', 0)),
            list(context['titles']).index(request.POST.get('title')),
            list(context['directors']).index(request.POST.get('director')),
            list(context['genres']).index(request.POST.get('genres_input')),
            float(request.POST.get('runtime', 0)),
            list(context['languages']).index(request.POST.get('lang')),
            list(context['descriptions']).index(request.POST.get('desc')),
            list(context['studios']).index(request.POST.get('studios_input')),
            int(request.POST.get('watches', 0)),
            int(request.POST.get('lists', 0)),
            int(request.POST.get('likes', 0)),
            int(request.POST.get('fans', 0)),
            int(request.POST.get('lowest', 0)),
            int(request.POST.get('medium', 0)),
            int(request.POST.get('highest', 0)),
            int(request.POST.get('total', 0)),
        ]

        # 2. Load the best model and accuracies
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'random_forest_model.pkl')
        accuracy_path = os.path.join(os.path.dirname(__file__), 'models', 'accuracies.pkl')

        # Load models properly
        model = joblib.load(model_path)
        accuracies = joblib.load(accuracy_path)
        
        try:
            prediction = model.predict([input_features])[0]
        except ValueError as e:
            context.update({'error': f"Prediction error: {str(e)}"})
            return render(request, 'predict.html', context)
        prediction = model.predict([input_features])[0]
        best_algo = max(accuracies, key=accuracies.get)

        context.update({
            'prediction': prediction,
            'accuracies': accuracies,
            'best_algo': best_algo.replace('_', ' ').title()
        })
        return render(request, 'result.html', context)

    return render(request, 'predict.html', context)
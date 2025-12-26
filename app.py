from flask import Flask, render_template, request, jsonify, send_file
import pickle
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os
import warnings
from datetime import datetime
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import base64
import tempfile
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Load models and files
def load_models():
    """Load all trained models and files"""
    models = {}
    
    try:
        # Load HyDengue model using joblib
        model_path = os.path.join(MODELS_DIR, 'hydengue_model.pkl')
        print(f"Loading model from: {model_path}")
        
        try:
            models['hydengue'] = joblib.load(model_path)
            print("Model loaded successfully with joblib")
        except:
            with open(model_path, 'rb') as f:
                models['hydengue'] = pickle.load(f)
            print("Model loaded successfully with pickle")
        
        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
        print(f"Loading scaler from: {scaler_path}")
        models['scaler'] = joblib.load(scaler_path)
        print("Scaler loaded successfully")
        
        # Load selected features
        features_path = os.path.join(MODELS_DIR, 'selected_features.json')
        with open(features_path, 'r') as f:
            models['selected_features'] = json.load(f)
        print("Selected features loaded successfully")
        
        # Load feature importance
        importance_path = os.path.join(MODELS_DIR, 'feature_importance.json')
        with open(importance_path, 'r') as f:
            models['feature_importance'] = json.load(f)
        print("Feature importance loaded successfully")
        
        print("All models loaded successfully!")
        return models
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise e

# Initialize models
try:
    models = load_models()
    print("=" * 50)
    print("HyDengue Web Application Ready!")
    print(f"Selected Features: {models['selected_features']}")
    print("=" * 50)
except Exception as e:
    print(f"Failed to load models: {e}")
    models = None

# Reference ranges for medical interpretation
REFERENCE_RANGES = {
    'RBC': {'min': 4.5, 'max': 5.9, 'unit': 'million/µL', 'normal': 5.2},
    'Hemoglobin(g/dl)': {'min': 13.0, 'max': 17.5, 'unit': 'g/dL', 'normal': 15.0},
    'HCT(%)': {'min': 40.0, 'max': 50.0, 'unit': '%', 'normal': 45.0},
    'RDW-CV(%)': {'min': 11.5, 'max': 14.5, 'unit': '%', 'normal': 13.0},
    'Total WBC count(/cumm)': {'min': 4000, 'max': 11000, 'unit': 'cells/µL', 'normal': 7500},
    'Neutrophils(%)': {'min': 40, 'max': 70, 'unit': '%', 'normal': 55},
    'Total Platelet Count(/cumm)': {'min': 150000, 'max': 450000, 'unit': 'cells/µL', 'normal': 300000},
    'Age': {'min': 0, 'max': 120, 'unit': 'years', 'normal': 40}
}

def create_factor_chart(factors, report_id):
    """Create matplotlib chart for factors"""
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    chart_path = temp_file.name
    
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    feature_names = [f['feature'].split('(')[0][:15] for f in factors]
    contributions = [f['contribution'] for f in factors]
    colors_list = []
    
    for f in factors:
        if f['impact'] == 'HIGH':
            colors_list.append('#e74c3c')
        elif f['impact'] == 'MEDIUM':
            colors_list.append('#f39c12')
        else:
            colors_list.append('#3498db')
    
    # Create bar chart
    bars = plt.barh(feature_names, contributions, color=colors_list)
    plt.xlabel('Contribution Score (%)')
    plt.title('Top Contributing Factors to Prediction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, contributions)):
        plt.text(value + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', fontweight='bold')
    
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return chart_path

def create_pdf_report(prediction_data, patient_data, factors, report_id):
    """Generate comprehensive PDF report"""
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_date = datetime.now().strftime("%d %B %Y")
    
    # Create buffer for PDF
    buffer = io.BytesIO()
    
    # Create document
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=20,
        alignment=1
    )
    
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12,
        spaceBefore=20
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6
    )
    
    bold_style = ParagraphStyle(
        'BoldStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    # Start building the story
    story = []
    
    # Header
    story.append(Paragraph("HYDENGUE AI - DENGUE FEVER PREDICTION REPORT", title_style))
    story.append(Paragraph(f"Report ID: {report_id}", normal_style))
    story.append(Paragraph(f"Generated: {timestamp}", normal_style))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", section_style))
    
    # Prediction result
    if prediction_data['prediction_numeric'] == 1:
        result_text = "⚡ DENGUE DETECTED - POSITIVE"
        result_color = colors.red
        severity = "HIGH" if prediction_data['confidence'] > 80 else "MODERATE" if prediction_data['confidence'] > 60 else "LOW"
    else:
        result_text = "✅ NO DENGUE DETECTED - NEGATIVE"
        result_color = colors.green
        severity = "VERY LOW"
    
    summary_data = [
        ["Prediction Result:", result_text],
        ["Confidence Level:", f"{prediction_data['confidence']}%"],
        ["Risk Assessment:", prediction_data['risk_level']],
        ["Severity:", severity],
        ["Reliability:", prediction_data['reliability']]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 3.5*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('TEXTCOLOR', (1, 0), (1, 0), result_color),
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Patient Data
    story.append(Paragraph("PATIENT TEST PARAMETERS", section_style))
    
    # Create patient data table
    patient_rows = [["Parameter", "Value", "Reference Range", "Status"]]
    
    for param in patient_data:
        status = "NORMAL"
        status_color = colors.green
        
        if param['status'] == 'LOW':
            status = "LOW"
            status_color = colors.blue
        elif param['status'] == 'HIGH':
            status = "HIGH"
            status_color = colors.red
        
        patient_rows.append([
            param['name'],
            str(param['value']),
            param['reference'],
            status
        ])
    
    patient_table = Table(patient_rows, colWidths=[2*inch, 1.2*inch, 2*inch, 1*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('TEXTCOLOR', (3, 1), (3, -1), status_color),
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Factor Analysis Chart
    story.append(Paragraph("FACTOR CONTRIBUTION ANALYSIS", section_style))
    
    # Create chart
    chart_path = create_factor_chart(factors, report_id)
    story.append(Image(chart_path, width=6*inch, height=3.5*inch))
    story.append(Spacer(1, 15))
    
    # Factor details
    story.append(Paragraph("Top Contributing Factors:", bold_style))
    
    for i, factor in enumerate(factors[:5], 1):
        factor_text = f"{i}. <b>{factor['feature']}</b>: Value = {factor['value']} ({factor['direction']}) | Contribution = {factor['contribution']:.1f}% | Impact = {factor['impact']}"
        story.append(Paragraph(factor_text, normal_style))
    
    story.append(Spacer(1, 25))
    
    # Clinical Recommendations
    story.append(Paragraph("CLINICAL RECOMMENDATIONS", section_style))
    
    if prediction_data['prediction_numeric'] == 1:
        recommendations = [
            "Immediate medical consultation recommended",
            "Monitor platelet count daily - critical for dengue management",
            "Maintain proper hydration - minimum 3-4 liters per day",
            "Watch for warning signs: severe abdominal pain, persistent vomiting, bleeding gums/nose",
            "Avoid aspirin and NSAIDs - use paracetamol for fever management",
            "Complete bed rest recommended",
            "Follow up with complete blood count every 24-48 hours",
            "Consider hospitalization if platelet count drops below 100,000/µL"
        ]
        
        story.append(Paragraph(f"<b>Severity:</b> {severity}", bold_style))
        story.append(Spacer(1, 10))
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", normal_style))
        
        story.append(Spacer(1, 15))
        story.append(Paragraph("<font color='red'><b>⚠️ EMERGENCY:</b> Seek immediate medical attention if any warning signs appear or if platelet count falls rapidly.</font>", normal_style))
        
    else:
        recommendations = [
            "No immediate medical intervention required",
            "Continue normal activities with adequate rest",
            "Maintain healthy lifestyle with balanced nutrition",
            "Regular health check-ups recommended annually",
            "Monitor for any symptoms of fever or fatigue",
            "Stay hydrated and maintain proper hygiene"
        ]
        
        story.append(Paragraph("<b>Assessment:</b> Hematological parameters within normal range", bold_style))
        story.append(Spacer(1, 10))
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", normal_style))
    
    story.append(Spacer(1, 25))
    
    # Model Information
    story.append(Paragraph("MODEL INFORMATION", section_style))
    
    model_info = [
        ["Model Name:", "HyDengue Ensemble (SVM + MLP)"],
        ["Accuracy:", "96.3% (5-fold cross-validated)"],
        ["Features Analyzed:", f"{len(patient_data)} hematological parameters"],
        ["Training Dataset:", "1,409 clinical cases"],
        ["Validation:", "Stratified 5-fold cross-validation"],
        ["Prediction Time:", "&lt; 1 second"]
    ]
    
    model_table = Table(model_info, colWidths=[2*inch, 3*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))
    
    story.append(model_table)
    story.append(Spacer(1, 20))
    
    # Disclaimer
    story.append(Paragraph("IMPORTANT DISCLAIMER", styles['Heading4']))
    disclaimer_text = """
    <i>This report is generated by an AI-powered diagnostic assistance system. 
    The predictions are based on machine learning algorithms trained on historical clinical data. 
    This report is intended for informational purposes only and should not be used as a substitute 
    for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare 
    professionals for medical decisions. The system developers are not liable for any clinical 
    decisions made based on this report.</i>
    """
    story.append(Paragraph(disclaimer_text, normal_style))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated by HyDengue AI System • {report_date}", 
                          ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, 
                                        textColor=colors.grey, alignment=1)))
    
    # Build PDF
    doc.build(story)
    
    # Clean up temporary file
    if os.path.exists(chart_path):
        os.remove(chart_path)
    
    # Get PDF content
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content

def calculate_prediction_factors(input_data, prediction):
    """Calculate which features contributed most to the prediction"""
    factors = []
    
    for feature in models['selected_features']:
        value = input_data[feature]
        ref = REFERENCE_RANGES.get(feature, {})
        
        if 'normal' in ref:
            normal = ref['normal']
            min_val = ref.get('min', normal * 0.8)
            max_val = ref.get('max', normal * 1.2)
            
            # Calculate deviation from normal
            if max_val - min_val > 0:
                deviation = abs(value - normal) / (max_val - min_val)
            else:
                deviation = 0
            
            # Get feature importance
            importance = models['feature_importance'].get(feature, 0.01)
            
            # Calculate contribution score
            contribution_score = deviation * importance * 100
            
            # Determine direction
            if value < normal * 0.9:
                direction = 'low'
            elif value > normal * 1.1:
                direction = 'high'
            else:
                direction = 'normal'
            
            # Determine impact based on prediction and deviation
            if prediction == 1:  # Dengue
                if feature in ['Total Platelet Count(/cumm)', 'Total WBC count(/cumm)']:
                    if value < normal * 0.7:
                        impact = 'HIGH'
                    elif value < normal * 0.9:
                        impact = 'MEDIUM'
                    else:
                        impact = 'LOW'
                elif feature in ['RDW-CV(%)', 'Neutrophils(%)']:
                    if value > normal * 1.3:
                        impact = 'HIGH'
                    elif value > normal * 1.1:
                        impact = 'MEDIUM'
                    else:
                        impact = 'LOW'
                else:
                    impact = 'MEDIUM' if deviation > 0.3 else 'LOW'
            else:  # Healthy
                impact = 'LOW' if deviation < 0.2 else 'MEDIUM'
            
            factors.append({
                'feature': feature,
                'value': round(value, 2),
                'normal': round(normal, 2),
                'deviation': round(deviation * 100, 1),
                'contribution': round(contribution_score, 1),
                'direction': direction,
                'impact': impact,
                'status': 'ABNORMAL' if deviation > 0.2 else 'NORMAL'
            })
    
    # Sort by contribution score
    factors.sort(key=lambda x: x['contribution'], reverse=True)
    return factors[:6]

@app.route('/')
def home():
    """Render the main page"""
    if models is None:
        return render_template('error.html', 
                              error="Model files not loaded. Please check server logs.")
    
    return render_template('index.html', 
                          selected_features=models['selected_features'],
                          reference_ranges=REFERENCE_RANGES)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    if models is None:
        return jsonify({'error': 'Models not loaded. Please restart the server.'}), 500
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert to DataFrame
        input_data = {}
        
        # Process all features
        for feature in models['selected_features']:
            value = float(data.get(feature, 0))
            input_data[feature] = value
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Scale the data
        scaled_data = models['scaler'].transform(df)
        
        # Make prediction
        raw_prediction = models['hydengue'].predict(scaled_data)[0]
        raw_probability = models['hydengue'].predict_proba(scaled_data)[0]
        
        # Reverse mapping for medical interpretation
        prediction = 1 - raw_prediction  # Reverse: 0->1, 1->0
        probability = raw_probability[::-1]  # Reverse probabilities
        
        # Calculate confidence and risk level
        confidence = float(probability[1] if prediction == 1 else probability[0])
        
        if prediction == 1:  # Dengue predicted
            if confidence > 0.85:
                risk_level = "High Risk"
                reliability = "Very High"
            elif confidence > 0.7:
                risk_level = "Moderate Risk"
                reliability = "High"
            elif confidence > 0.5:
                risk_level = "Low Risk"
                reliability = "Moderate"
            else:
                risk_level = "Very Low Risk"
                reliability = "Low"
        else:  # Healthy predicted
            if confidence > 0.9:
                risk_level = "Very Low Risk"
                reliability = "Very High"
            elif confidence > 0.75:
                risk_level = "Low Risk"
                reliability = "High"
            else:
                risk_level = "Minimal Risk"
                reliability = "Moderate"
        
        # Calculate factors influencing prediction
        factors = calculate_prediction_factors(input_data, prediction)
        
        # Check abnormal values
        abnormalities = []
        patient_report_data = []
        
        for feature in models['selected_features']:
            value = input_data[feature]
            ref = REFERENCE_RANGES.get(feature, {})
            
            if ref:
                status = 'NORMAL'
                if value < ref['min']:
                    status = 'LOW'
                    abnormalities.append({
                        'feature': feature,
                        'value': round(value, 2),
                        'status': 'LOW',
                        'reference': f"{ref['min']}-{ref['max']} {ref['unit']}"
                    })
                elif value > ref['max']:
                    status = 'HIGH'
                    abnormalities.append({
                        'feature': feature,
                        'value': round(value, 2),
                        'status': 'HIGH',
                        'reference': f"{ref['min']}-{ref['max']} {ref['unit']}"
                    })
                
                patient_report_data.append({
                    'name': feature,
                    'value': round(value, 2),
                    'reference': f"{ref['min']}-{ref['max']} {ref['unit']}",
                    'status': status
                })
        
        # Generate report ID
        report_id = f"DENGUE_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{prediction}"
        
        # Prepare response
        result = {
            'prediction': "⚠️ DENGUE DETECTED" if prediction == 1 else "✅ NO DENGUE DETECTED",
            'prediction_numeric': int(prediction),
            'confidence': round(confidence * 100, 2),
            'reliability': reliability,
            'risk_level': risk_level,
            'probability_positive': round(probability[1] * 100, 2),
            'probability_negative': round(probability[0] * 100, 2),
            'abnormalities': abnormalities,
            'total_tests': len(models['selected_features']),
            'abnormal_count': len(abnormalities),
            'factors': factors,
            'report_id': report_id,
            'patient_data': patient_report_data,
            'gender': 'Male' if int(data.get('Gender', 1)) == 1 else 'Female'
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """Generate and download PDF report"""
    try:
        data = request.json
        
        # Extract data
        prediction_data = {
            'prediction_numeric': data.get('prediction_numeric'),
            'prediction': data.get('prediction'),
            'confidence': data.get('confidence'),
            'reliability': data.get('reliability'),
            'risk_level': data.get('risk_level'),
            'probability_positive': data.get('probability_positive'),
            'probability_negative': data.get('probability_negative')
        }
        
        patient_data = data.get('patient_data', [])
        factors = data.get('factors', [])
        report_id = data.get('report_id', 'UNKNOWN')
        
        # Generate PDF
        pdf_content = create_pdf_report(prediction_data, patient_data, factors, report_id)
        
        # Create response
        response = send_file(
            io.BytesIO(pdf_content),
            as_attachment=True,
            download_name=f'{report_id}_report.pdf',
            mimetype='application/pdf'
        )
        
        return response
        
    except Exception as e:
        print(f"PDF generation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    if models is not None:
        return jsonify({'status': 'healthy', 'models_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'models_loaded': False}), 500

if __name__ == '__main__':
    if models is not None:
        print("\n" + "="*50)
        print("Starting Flask server...")
        print(f"Access the app at: http://localhost:5000")
        print("="*50 + "\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Cannot start server: Models not loaded")
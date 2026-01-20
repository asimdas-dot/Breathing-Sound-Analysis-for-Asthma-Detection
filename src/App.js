import React, { useState, useRef } from 'react';
import './App.css';

function App() {
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const [formData, setFormData] = useState({
    name: '',
    age: '',
    gender: '',
    weight: '',
    height: '',
    allergies: '',
    genetics: '',
    aqi: '',
    location: '',
    smoking: '',
    profession: '',
    workTime: '',
    duration: '',
    breathingDistress: '',
    symptoms: {
      tiredness: false,
      dryCough: false,
      difficultyBreathing: false,
      soreThroat: false,
      pains: false,
      nasalCongestion: false,
      runnyNose: false,
    },
    xrayImage: null,
  });

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    
    if (type === 'checkbox') {
      setFormData({
        ...formData,
        symptoms: {
          ...formData.symptoms,
          [name]: checked,
        },
      });
    } else {
      setFormData({
        ...formData,
        [name]: value,
      });
    }
  };

  const handleFileChange = (e) => {
    setFormData({
      ...formData,
      xrayImage: e.target.files[0],
    });
  };

  const nextStep = () => {
    if (step < 5) setStep(step + 1);
  };

  const prevStep = () => {
    if (step > 1) setStep(step - 1);
  };

  const submitData = async () => {
    setLoading(true);
    
    try {
      // Spirometry prediction
      const formDataToSend = new FormData();
      formDataToSend.append('name', formData.name);
      formDataToSend.append('age', formData.age);
      formDataToSend.append('Tiredness', formData.symptoms.tiredness ? 1 : 0);
      formDataToSend.append('Dry-Cough', formData.symptoms.dryCough ? 1 : 0);
      formDataToSend.append('Difficulty-in-Breathing', formData.symptoms.difficultyBreathing ? 1 : 0);
      formDataToSend.append('Sore-Throat', formData.symptoms.soreThroat ? 1 : 0);
      formDataToSend.append('Pains', formData.symptoms.pains ? 1 : 0);
      formDataToSend.append('Nasal-Congestion', formData.symptoms.nasalCongestion ? 1 : 0);
      formDataToSend.append('Runny-Nose', formData.symptoms.runnyNose ? 1 : 0);
      formDataToSend.append('None_Sympton', 0);
      formDataToSend.append('None_Experiencing', 0);
      formDataToSend.append('Age_0-9', formData.age < 10 ? 1 : 0);
      formDataToSend.append('Age_10-19', (formData.age >= 10 && formData.age < 20) ? 1 : 0);
      formDataToSend.append('Age_20-24', (formData.age >= 20 && formData.age <= 24) ? 1 : 0);
      formDataToSend.append('Age_25-59', (formData.age >= 25 && formData.age <= 59) ? 1 : 0);
      formDataToSend.append('Age_60+', formData.age >= 60 ? 1 : 0);
      formDataToSend.append('Gender_Female', formData.gender === 'female' ? 1 : 0);
      formDataToSend.append('Gender_Male', formData.gender === 'male' ? 1 : 0);

      const spirometryResponse = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formDataToSend,
      });

      const spirometryData = await spirometryResponse.json();

      // X-ray prediction (if image provided)
      let xrayData = null;
      if (formData.xrayImage) {
        const xrayFormData = new FormData();
        xrayFormData.append('xray_image', formData.xrayImage);
        xrayFormData.append('patient_id', formData.name);

        const xrayResponse = await fetch('http://localhost:5000/analyze-xray', {
          method: 'POST',
          body: xrayFormData,
        });

        xrayData = await xrayResponse.json();
      }

      setResult({
        spirometry: spirometryData,
        xray: xrayData,
      });

      setStep(5);
    } catch (error) {
      console.error('Error:', error);
      alert('API Error: Make sure backend is running at http://localhost:5000');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8">
      {/* Header */}
      <div className="mb-8 text-center">
        <div className="flex items-center justify-center mb-4">
          <svg className="w-12 h-12 text-blue-600 mr-3" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5.951-1.429 5.951 1.429a1 1 0 001.169-1.409l-7-14z" />
          </svg>
          <h1 className="text-4xl font-bold text-blue-900">🫁 Asthma AI Detector</h1>
        </div>
        <p className="text-blue-600 text-lg">Advanced Respiratory Health Analysis System</p>
      </div>

      {/* Main Container */}
      <div className="max-w-2xl mx-auto px-4">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between mb-2">
            {[1, 2, 3, 4, 5].map((num) => (
              <div key={num} className={`flex-1 mx-1 h-2 rounded-full transition-colors ${
                num <= step ? 'bg-blue-600' : 'bg-blue-200'
              }`} />
            ))}
          </div>
          <p className="text-center text-blue-600 font-semibold">Step {step} of 5</p>
        </div>

        {/* Card Container */}
        <div className="bg-white rounded-xl shadow-2xl overflow-hidden border-2 border-blue-200">
          <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white p-6">
            <h2 className="text-2xl font-bold">
              {step === 1 && '👤 Personal Information'}
              {step === 2 && '⚕️ Medical History'}
              {step === 3 && '🌍 Environmental Factors'}
              {step === 4 && '🖼️ X-ray Analysis'}
              {step === 5 && '📊 Results & Diagnosis'}
            </h2>
          </div>

          <div className="p-8">
            {/* Step 1: Personal Information */}
            {step === 1 && (
              <div className="space-y-6">
                <div>
                  <label className="block text-blue-900 font-semibold mb-2">Full Name *</label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    placeholder="Enter your full name"
                    className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-blue-900 font-semibold mb-2">Age *</label>
                    <input
                      type="number"
                      name="age"
                      value={formData.age}
                      onChange={handleChange}
                      placeholder="Years"
                      className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                    />
                  </div>

                  <div>
                    <label className="block text-blue-900 font-semibold mb-2">Gender *</label>
                    <select
                      name="gender"
                      value={formData.gender}
                      onChange={handleChange}
                      className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                    >
                      <option value="">Select Gender</option>
                      <option value="male">Male</option>
                      <option value="female">Female</option>
                      <option value="other">Other</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-blue-900 font-semibold mb-2">Weight (kg) *</label>
                    <input
                      type="number"
                      name="weight"
                      value={formData.weight}
                      onChange={handleChange}
                      placeholder="kg"
                      className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                    />
                  </div>

                  <div>
                    <label className="block text-blue-900 font-semibold mb-2">Height (cm) *</label>
                    <input
                      type="number"
                      name="height"
                      value={formData.height}
                      onChange={handleChange}
                      placeholder="cm"
                      className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Step 2: Medical History */}
            {step === 2 && (
              <div className="space-y-6">
                <div>
                  <label className="block text-blue-900 font-semibold mb-4">Current Symptoms</label>
                  <div className="space-y-3">
                    {[
                      { key: 'tiredness', label: 'Tiredness / Fatigue' },
                      { key: 'dryCough', label: 'Dry Cough' },
                      { key: 'difficultyBreathing', label: 'Difficulty in Breathing' },
                      { key: 'soreThroat', label: 'Sore Throat' },
                      { key: 'pains', label: 'Body Pains' },
                      { key: 'nasalCongestion', label: 'Nasal Congestion' },
                      { key: 'runnyNose', label: 'Runny Nose' },
                    ].map((symptom) => (
                      <label key={symptom.key} className="flex items-center space-x-3 cursor-pointer">
                        <input
                          type="checkbox"
                          name={symptom.key}
                          checked={formData.symptoms[symptom.key]}
                          onChange={handleChange}
                          className="w-5 h-5 accent-blue-600 cursor-pointer"
                        />
                        <span className="text-blue-900">{symptom.label}</span>
                      </label>
                    ))}
                  </div>
                </div>

                <div>
                  <label className="block text-blue-900 font-semibold mb-2">Duration of Symptoms</label>
                  <input
                    type="text"
                    name="duration"
                    value={formData.duration}
                    onChange={handleChange}
                    placeholder="e.g., 2 weeks, 1 month"
                    className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                  />
                </div>

                <div>
                  <label className="block text-blue-900 font-semibold mb-2">Any Family History of Asthma?</label>
                  <select
                    name="genetics"
                    value={formData.genetics}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                  >
                    <option value="">Select Option</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                    <option value="maybe">Not Sure</option>
                  </select>
                </div>

                <div>
                  <label className="block text-blue-900 font-semibold mb-2">Do you smoke?</label>
                  <select
                    name="smoking"
                    value={formData.smoking}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                  >
                    <option value="">Select Option</option>
                    <option value="yes">Yes</option>
                    <option value="no">No</option>
                    <option value="former">Former Smoker</option>
                  </select>
                </div>
              </div>
            )}

            {/* Step 3: Environmental Factors */}
            {step === 3 && (
              <div className="space-y-6">
                <div>
                  <label className="block text-blue-900 font-semibold mb-2">Current Location</label>
                  <input
                    type="text"
                    name="location"
                    value={formData.location}
                    onChange={handleChange}
                    placeholder="City/Region"
                    className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                  />
                </div>

                <div>
                  <label className="block text-blue-900 font-semibold mb-2">Air Quality Index (AQI)</label>
                  <select
                    name="aqi"
                    value={formData.aqi}
                    onChange={handleChange}
                    className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                  >
                    <option value="">Select AQI Level</option>
                    <option value="good">Good (0-50)</option>
                    <option value="moderate">Moderate (51-100)</option>
                    <option value="poor">Poor (101-150)</option>
                    <option value="verybad">Very Bad (151-200)</option>
                    <option value="severe">Severe (200+)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-blue-900 font-semibold mb-2">Profession / Work Type</label>
                  <input
                    type="text"
                    name="profession"
                    value={formData.profession}
                    onChange={handleChange}
                    placeholder="e.g., Software Engineer, Construction Worker"
                    className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                  />
                </div>

                <div>
                  <label className="block text-blue-900 font-semibold mb-2">Work Hours per Day</label>
                  <input
                    type="number"
                    name="workTime"
                    value={formData.workTime}
                    onChange={handleChange}
                    placeholder="Hours"
                    className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                  />
                </div>

                <div>
                  <label className="block text-blue-900 font-semibold mb-2">Any Known Allergies?</label>
                  <textarea
                    name="allergies"
                    value={formData.allergies}
                    onChange={handleChange}
                    placeholder="List any allergies (dust, pollen, pets, etc.)"
                    rows="3"
                    className="w-full px-4 py-3 border-2 border-blue-300 rounded-lg focus:outline-none focus:border-blue-600 transition"
                  />
                </div>
              </div>
            )}

            {/* Step 4: X-ray Analysis */}
            {step === 4 && (
              <div className="space-y-6">
                <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-6">
                  <h3 className="text-blue-900 font-semibold mb-4">📤 Upload Chest X-ray (Optional)</h3>
                  <p className="text-blue-700 mb-4 text-sm">
                    Upload a chest X-ray image for AI analysis. Supported formats: PNG, JPG, JPEG
                  </p>

                  <div className="border-2 border-dashed border-blue-400 rounded-lg p-8 text-center cursor-pointer hover:bg-blue-100 transition"
                    onClick={() => fileInputRef.current?.click()}>
                    {formData.xrayImage ? (
                      <div>
                        <p className="text-green-600 font-semibold">✅ File Selected</p>
                        <p className="text-blue-600">{formData.xrayImage.name}</p>
                      </div>
                    ) : (
                      <div>
                        <svg className="w-12 h-12 text-blue-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                        </svg>
                        <p className="text-blue-900 font-semibold">Click to upload or drag file</p>
                        <p className="text-blue-600 text-sm">PNG, JPG, JPEG (max 50MB)</p>
                      </div>
                    )}
                  </div>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </div>

                <div className="bg-blue-50 border-l-4 border-blue-600 p-4 rounded">
                  <p className="text-blue-900 text-sm">
                    <strong>Note:</strong> X-ray upload is optional. Our system can perform analysis based on your symptoms alone.
                  </p>
                </div>
              </div>
            )}

            {/* Step 5: Results */}
            {step === 5 && result && (
              <div className="space-y-6">
                {/* Spirometry Results */}
                {result.spirometry && (
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border-2 border-blue-300 rounded-lg p-6">
                    <h3 className="text-xl font-bold text-blue-900 mb-4">📋 Spirometry Analysis Results</h3>

                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-blue-700 font-semibold">Patient Name:</span>
                        <span className="text-blue-900">{result.spirometry.patient_name}</span>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-blue-700 font-semibold">Severity Level:</span>
                        <span className="px-4 py-2 bg-white border-2 border-blue-300 rounded-lg font-bold text-blue-900">
                          {result.spirometry.severity_label}
                        </span>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-blue-700 font-semibold">Confidence Score:</span>
                        <span className="text-blue-900 font-bold">{result.spirometry.confidence?.toFixed(1)}%</span>
                      </div>

                      <div className="w-full bg-blue-200 rounded-full h-3 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-green-500 to-blue-600 h-full rounded-full transition-all"
                          style={{ width: `${result.spirometry.confidence || 0}%` }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* X-ray Results */}
                {result.xray && (
                  <div className="bg-gradient-to-r from-indigo-50 to-blue-50 border-2 border-indigo-300 rounded-lg p-6">
                    <h3 className="text-xl font-bold text-blue-900 mb-4">🖼️ X-ray Analysis Results</h3>

                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-blue-700 font-semibold">Diagnosis:</span>
                        <span className="px-4 py-2 bg-white border-2 border-indigo-300 rounded-lg font-bold text-blue-900">
                          {result.xray.prediction_label}
                        </span>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-blue-700 font-semibold">Risk Level:</span>
                        <span className={`px-4 py-2 rounded-lg font-bold ${
                          result.xray.risk_level === 'Low' ? 'bg-green-100 text-green-900' :
                          result.xray.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-900' :
                          'bg-red-100 text-red-900'
                        }`}>
                          {result.xray.risk_level}
                        </span>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-blue-700 font-semibold">Confidence:</span>
                        <span className="text-blue-900 font-bold">{result.xray.confidence?.toFixed(1)}%</span>
                      </div>

                      <div className="w-full bg-blue-200 rounded-full h-3 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-indigo-600 h-full rounded-full transition-all"
                          style={{ width: `${result.xray.confidence || 0}%` }}
                        />
                      </div>
                    </div>
                  </div>
                )}

                {/* Final Recommendation */}
                <div className="bg-blue-100 border-2 border-blue-600 rounded-lg p-6">
                  <h3 className="text-lg font-bold text-blue-900 mb-3">💊 Recommendations</h3>
                  <ul className="space-y-2 text-blue-900">
                    <li>✓ Consult with a respiratory specialist</li>
                    <li>✓ Keep your rescue inhaler handy</li>
                    <li>✓ Avoid air pollutants and allergens</li>
                    <li>✓ Maintain regular exercise routine</li>
                    <li>✓ Track your symptoms daily</li>
                  </ul>
                </div>
              </div>
            )}

            {/* Error State */}
            {step === 5 && !result && (
              <div className="text-center py-12">
                <svg className="w-16 h-16 text-red-500 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4v.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-red-600 text-lg font-semibold">Error loading results</p>
                <p className="text-blue-600">Please check backend connection</p>
              </div>
            )}
          </div>

          {/* Navigation Buttons */}
          <div className="bg-blue-50 border-t-2 border-blue-200 p-6 flex justify-between">
            {step > 1 && (
              <button
                onClick={prevStep}
                className="px-6 py-3 bg-blue-200 hover:bg-blue-300 text-blue-900 font-bold rounded-lg transition"
              >
                ← Back
              </button>
            )}

            <div className="flex-1" />

            {step < 4 && (
              <button
                onClick={nextStep}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition"
              >
                Next →
              </button>
            )}

            {step === 4 && (
              <button
                onClick={submitData}
                disabled={loading || !formData.name || !formData.age || !formData.gender}
                className={`px-8 py-3 font-bold rounded-lg transition text-white ${
                  loading || !formData.name || !formData.age || !formData.gender
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-green-600 hover:bg-green-700'
                }`}
              >
                {loading ? '🔄 Analyzing...' : '✓ Analyze & Get Results'}
              </button>
            )}

            {step === 5 && (
              <button
                onClick={() => {
                  setStep(1);
                  setFormData({
                    name: '', age: '', gender: '', weight: '', height: '',
                    allergies: '', genetics: '', aqi: '', location: '', smoking: '',
                    profession: '', workTime: '', duration: '', breathingDistress: '',
                    symptoms: {
                      tiredness: false, dryCough: false, difficultyBreathing: false,
                      soreThroat: false, pains: false, nasalCongestion: false, runnyNose: false,
                    },
                    xrayImage: null,
                  });
                  setResult(null);
                }}
                className="px-8 py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition"
              >
                🔄 Start New Analysis
              </button>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-blue-700 text-sm">
            ⚕️ This is an AI-assisted tool. Always consult with a qualified healthcare professional for proper diagnosis.
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;

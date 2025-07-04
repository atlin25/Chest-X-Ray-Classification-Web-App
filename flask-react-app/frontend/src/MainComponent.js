import React, { useState, useCallback } from 'react';
import { useUpload } from './hooks/useUpload';
import { useHandleStreamResponse } from './hooks/useHandleStreamResponse';

function MainComponent() {
  const [image, setImage] = useState(null);
  const [messages, setMessages] = useState([]);
  const [streamingMessage, setStreamingMessage] = useState("");
  const [error, setError] = useState(null);
  const [upload, { loading: uploading }] = useUpload();
  const [analyzing, setAnalyzing] = useState(false);
  const [cnnResults, setCnnResults] = useState(null);
  const [aiInsight, setAiInsight] = useState(null);

  console.log('MainComponent rendered'); // Debug: Confirm component renders

  const handleFinish = useCallback((message) => {
    setMessages((prev) => [...prev, { role: "assistant", content: message }]);
    setStreamingMessage("");
    setAnalyzing(false);
  }, []);

  const handleStreamResponse = useHandleStreamResponse({
    onChunk: setStreamingMessage,
    onFinish: handleFinish,
  });
  
  const fileToBase64 = (file) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });  
  const handleImageUpload = useCallback(async (e) => {
  if (e.target.files) {
    const file = e.target.files[0];
    try {
      const base64Image = await fileToBase64(file);
      setImage(base64Image); // Optional: for preview or state

      setAnalyzing(true);
      setCnnResults(null);
      setAiInsight(null);

      // Call backend with base64 image only
      const cnnResponse = await fetch("http://localhost:5000/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64Image }),
      });

      if (!cnnResponse.ok) {
        throw new Error(`CNN API failed with status ${cnnResponse.status}`);
      }

      const cnnData = await cnnResponse.json();
      setCnnResults(cnnData);

      const geminiResponse = await fetch("http://localhost:5000/api/gemini_insights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prediction: cnnData.prediction,
          labels: cnnData.labels,
          summary: cnnData.summary,
          // Or send base64Image or other data your Gemini expects
        }),
      });
      if (!geminiResponse.ok) throw new Error(`Gemini API failed: ${geminiResponse.status}`);
      const geminiData = await geminiResponse.json();
      setAiInsight(geminiData.insight || geminiData);

      setAnalyzing(false);
    } catch (error) {
      console.error("Error:", error);
      setError(error.message || "Something went wrong");
      setAnalyzing(false);
    }
  }
}, []);

  
  return (
    <div>
      <style>
        {`
          .container {
            min-height: 100vh;
            background-color: #282828;
            color: #ebdbb2;
            padding: 16px;
            font-family: 'Roboto', sans-serif;
          }
          @media (min-width: 768px) {
            .container {
              padding: 32px;
            }
          }
          .main {
            max-width: 1280px;
            margin: 0 auto;
          }
          .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 48px;
          }
          .header-title-container {
            display: flex;
            align-items: center;
          }
          .header-icon {
            color: #b8bb26;
            font-size: 32px;
            margin-right: 16px;
          }
          .header-title {
            font-size: 32px;
            font-weight: bold;
            color: #b8bb26;
          }
          @media (min-width: 768px) {
            .header-title {
              font-size: 40px;
            }
          }
          .social-links {
            display: flex;
            gap: 24px;
            align-items: center;
            border: 1px dashed #928374; /* Debug: Visible border to confirm position */
            padding: 8px;
          }
          .social-link {
            color: #8ec07c;
            font-size: 32px; /* Larger icons */
            transition: color 0.2s, transform 0.2s;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 8px;
          }
          .social-link:hover {
            color: #b8bb26;
            transform: scale(1.1);
          }
          .social-link-text {
            font-size: 16px;
            color: #ebdbb2;
          }
          @media (max-width: 767px) {
            .social-link-text {
              display: none; /* Hide text on small screens */
            }
          }
          .grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 32px;
          }
          @media (min-width: 1024px) {
            .grid {
              grid-template-columns: repeat(3, 1fr);
            }
          }
          .card {
            background-color: #3c3836;
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 2px solid #928374;
          }
          .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 24px;
          }
          .card-icon {
            color: #8ec07c;
            font-size: 20px;
            margin-right: 12px;
          }
          .card-title {
            font-size: 24px;
            font-weight: 600;
            color: #8ec07c;
          }
          .upload-area {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 24px;
          }
          .upload-label {
            width: 100%;
          }
          .upload-box {
            background-color: #504945;
            text-align: center;
            padding: 24px;
            border-radius: 12px;
            border: 2px dashed #928374;
            cursor: pointer;
            transition: all 0.2s;
          }
          .upload-box:hover {
            background-color: #665c54;
            border-color: #8ec07c;
          }
          .upload-icon {
            font-size: 32px;
            color: #8ec07c;
            margin-bottom: 12px;
          }
          .upload-text {
            color: #ebdbb2;
          }
          .upload-hint {
            color: #928374;
            font-size: 14px;
            margin-top: 8px;
          }
          .uploading-box {
            background-color: #504945;
            padding: 16px;
            border-radius: 8px;
            width: 100%;
            text-align: center;
            color: #fabd2f;
          }
          .uploading-icon {
            font-size: 24px;
            margin-right: 8px;
            animation: spin 1s linear infinite;
          }
          .error-box {
            background-color: #3d1f1f;
            color: #fb4934;
            padding: 16px;
            border-radius: 8px;
            width: 100%;
            display: flex;
            align-items: center;
          }
          .error-icon {
            margin-right: 8px;
          }
          .image-preview {
            width: 100%;
            background-color: #504945;
            padding: 16px;
            border-radius: 12px;
          }
          .image-preview img {
            width: 100%;
            height: auto;
            border-radius: 8px;
          }
          .analysis-content {
            display: flex;
            flex-direction: column;
            gap: 24px;
          }
          .loading {
            text-align: center;
            padding: 24px;
          }
          .loading-icon {
            color: #fabd2f;
            font-size: 32px;
            margin-bottom: 12px;
            animation: spin 1s linear infinite;
          }
          .loading-text {
            color: #fabd2f;
          }
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          .results-box {
            background-color: #504945;
            padding: 24px;
            border-radius: 12px;
          }
          .results-title {
            font-size: 20px;
            font-weight: 500;
            color: #83a598;
            margin-bottom: 16px;
          }
          .result-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background-color: #3c3836;
            border-radius: 8px;
            margin-bottom: 16px;
          }
          .result-label {
            color: #ebdbb2;
          }
          .result-value {
            color: #b8bb26;
            font-weight: 600;
            padding: 4px 12px;
            background-color: #4c4f26;
            border-radius: 9999px;
          }
          .result-value-confidence {
            color: #8ec07c;
            background-color: #3b4c3b;
          }
          .findings-section {
            margin-top: 24px;
          }
          .findings-title {
            color: #83a598;
            font-weight: 500;
            margin-bottom: 12px;
          }
          .findings-list {
            display: flex;
            flex-direction: column;
            gap: 8px;
          }
          .finding-item {
            display: flex;
            align-items: flex-start;
          }
          .finding-icon {
            color: #8ec07c;
            margin-top: 4px;
            margin-right: 8px;
          }
          .placeholder {
            text-align: center;
            padding: 48px 0;
          }
          .placeholder-icon {
            color: #928374;
            font-size: 32px;
            margin-bottom: 12px;
          }
          .placeholder-text {
            color: #928374;
          }
          .insights-content {
            display: flex;
            flex-direction: column;
            gap: 24px;
          }
          .insight-box {
            background-color: #504945;
            padding: 24px;
            border-radius: 12px;
          }
          .insight-text {
            color: #ebdbb2;
            white-space: pre-wrap;
            line-height: 1.5;
          }
        `}
      </style>
      <div className="container">
        <div className="main">
          <div className="header">
            <div className="header-title-container">
              <i className="fas fa-x-ray header-icon"></i>
              <h1 className="header-title">X-ray Analysis Platform</h1>
            </div>
            <div className="social-links">
              <a href="https://www.linkedin.com/in/andrew-lin-469043261/" target="_blank" rel="noopener noreferrer" className="social-link">
                <i className="fab fa-linkedin"></i>
                <span className="social-link-text">My LinkedIn</span>
              </a>
              <a href="https://github.com/atlin25" target="_blank" rel="noopener noreferrer" className="social-link">
                <i className="fab fa-github"></i>
                <span className="social-link-text">My GitHub</span>
              </a>
            </div>
          </div>

          <div className="grid">
            <div className="card">
              <div className="card-header">
                <i className="fas fa-upload card-icon"></i>
                <h2 className="card-title">Upload X-ray</h2>
              </div>
              <div className="upload-area">
                <label className="upload-label">
                  <div className="upload-box">
                    <i className="fas fa-cloud-upload-alt upload-icon"></i>
                    <div className="upload-text">Click to upload or drag and drop</div>
                    <div className="upload-hint">Supported formats: JPEG, PNG</div>
                  </div>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </label>

                {uploading && (
                  <div className="uploading-box">
                    <i className="fas fa-spinner uploading-icon"></i>
                    Uploading image...
                  </div>
                )}

                {error && (
                  <div className="error-box">
                    <i className="fas fa-exclamation-circle error-icon"></i>
                    {error}
                  </div>
                )}

                {image && (
                  <div className="image-preview">
                    <img src={image} alt="Uploaded X-ray" />
                  </div>
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <i className="fas fa-brain card-icon"></i>
                <h2 className="card-title">CNN Analysis</h2>
              </div>
              <div className="analysis-content">
                {analyzing && !cnnResults && (
                  <div className="loading">
                    <i className="fas fa-spinner loading-icon"></i>
                    <div className="loading-text">Processing image...</div>
                  </div>
                )}

                {cnnResults && (
                  <div className="results-box">
                    <h3 className="results-title">Classification Results</h3>
                    <div>
                      <div className="result-item">
                        <span className="result-label">Primary Classification</span>
                        <span className="result-value">{cnnResults.primary_class}</span>
                      </div>
                      <div className="result-item">
                        <span className="result-label">Confidence</span>
                        <span className="result-value result-value-confidence">
                        {typeof cnnResults.confidence === 'number' && !isNaN(cnnResults.confidence)
                            ? `${cnnResults.confidence.toFixed(2)}%`
                            : 'N/A'}
                        </span>
                      </div>
                      {cnnResults.additional_findings && (
                        <div className="findings-section">
                          <h4 className="findings-title">Additional Findings</h4>
                          <ul className="findings-list">
                            {cnnResults.additional_findings.map((finding, index) => (
                              <li key={index} className="finding-item">
                                <i className="fas fa-circle-check finding-icon"></i>
                                <span>{finding}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {!image && !analyzing && (
                  <div className="placeholder">
                    <i className="fas fa-arrow-up-from-bracket placeholder-icon"></i>
                    <div className="placeholder-text">Upload an X-ray to begin analysis</div>
                  </div>
                )}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <i className="fas fa-robot card-icon"></i>
                <h2 className="card-title">AI Insights</h2>
              </div>
              <div className="insights-content">
                {messages.map((msg, index) => (
                  <div key={index} className="insight-box">
                    <p className="insight-text">{msg.content}</p>
                  </div>
                ))}

                {streamingMessage && (
                  <div className="insight-box">
                    <p className="insight-text">{streamingMessage}</p>
                  </div>
                )}
                
                {aiInsight && (
                    <div className="insight-box">
                        <p className="insight-text">{aiInsight}</p>
                    </div>
                )}

                {analyzing && !streamingMessage && (
                  <div className="loading">
                    <i className="fas fa-spinner loading-icon"></i>
                    <div className="loading-text">Generating insights...</div>
                  </div>
                )}

                {!image && !analyzing && (
                  <div className="placeholder">
                    <i className="fas fa-arrow-up-from-bracket placeholder-icon"></i>
                    <div className="placeholder-text">Upload an X-ray to begin analysis</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default MainComponent;

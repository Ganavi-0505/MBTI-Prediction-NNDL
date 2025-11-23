import React, { useState } from 'react';
import { Upload, Play, CheckCircle, AlertCircle, Loader, TrendingUp } from 'lucide-react';

export default function MBTIPredictionApp() {
  const [video, setVideo] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const mbtiDescriptions = {
    ISTJ: { title: "The Logistician", traits: "Practical, fact-oriented, reliable", description: "ISTJs are responsible, logical, and methodical. They excel in organized environments and value tradition." },
    ISFJ: { title: "The Defender", traits: "Warm, conscientious, dedicated", description: "ISFJs are loyal protectors who value harmony and are deeply committed to helping others." },
    INFJ: { title: "The Advocate", traits: "Insightful, principled, idealistic", description: "INFJs are intuitive leaders with a strong sense of purpose and deep concern for humanity." },
    INTJ: { title: "The Architect", traits: "Strategic, logical, independent", description: "INTJs are visionary planners who think long-term and develop innovative strategies." },
    ISTP: { title: "The Virtuoso", traits: "Bold, practical, experimental", description: "ISTPs are logical troubleshooters who enjoy hands-on problem-solving and technical challenges." },
    ISFP: { title: "The Adventurer", traits: "Flexible, charming, sensitive", description: "ISFPs are artistic and sensitive, living in the moment with a love for beauty and aesthetics." },
    INFP: { title: "The Mediator", traits: "Poetic, idealistic, open-minded", description: "INFPs are idealistic dreamers who seek authenticity and personal growth." },
    INTP: { title: "The Logician", traits: "Innovative, analytical, curious", description: "INTPs are logical innovators who love exploring ideas and understanding complex systems." },
    ESTP: { title: "The Entrepreneur", traits: "Energetic, perceptive, pragmatic", description: "ESTPsare risk-takers who thrive in dynamic environments and adapt quickly to change." },
    ESFP: { title: "The Entertainer", traits: "Outgoing, spontaneous, enjoyable", description: "ESFPs are enthusiastic performers who love being the center of attention." },
    ENFP: { title: "The Campaigner", traits: "Enthusiastic, creative, sociable", description: "ENFPs are inspiring explorers who bring energy and creativity to every interaction." },
    ENTP: { title: "The Debater", traits: "Smart, curious, versatile", description: "ENTPs are quick-witted debaters who enjoy intellectual challenges and unconventional ideas." },
    ESTJ: { title: "The Executive", traits: "Efficient, organized, commanding", description: "ESTJs are natural leaders who organize and coordinate effectively toward clear goals." },
    ESFJ: { title: "The Consul", traits: "Caring, dutiful, people-focused", description: "ESFJs are caring facilitators who prioritize harmony and serving others." },
    ENFJ: { title: "The Protagonist", traits: "Charismatic, inspiring, responsible", description: "ENFJs are charismatic leaders who inspire and motivate others toward meaningful goals." },
    ENTJ: { title: "The Commander", traits: "Decisive, ambitious, strategic", description: "ENTJs are strategic commanders who lead with confidence and ambition." },
  };

  const handleVideoUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith('video/')) {
      setError('Please upload a valid video file');
      return;
    }

    if (file.size > 500 * 1024 * 1024) {
      setError('Video file must be less than 500MB');
      return;
    }

    setVideo(file);
    setError(null);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setVideoPreview(e.target?.result);
    };
    reader.readAsDataURL(file);
  };

  const handlePrediction = async () => {
    if (!video) {
      setError('Please upload a video first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('video_file', video);

      const BACKEND_URL = 'http://localhost:8000';
      
      const response = await fetch(`${BACKEND_URL}/predict_mbti`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Prediction failed. Please try again.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred during prediction');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setVideo(null);
    setVideoPreview(null);
    setResult(null);
    setError(null);
  };

  const resultData = result ? mbtiDescriptions[result.mbti_type] : null;

  return (
    <div style={{ minHeight: '100vh', background: 'linear-gradient(to bottom right, #0f172a, #1e293b, #0f172a)', color: 'white' }}>
      {/* Header */}
      <div style={{ borderBottom: '1px solid rgba(148, 163, 184, 0.2)', background: 'rgba(15, 23, 42, 0.5)', backdropFilter: 'blur(10px)', paddingBottom: '24px', marginBottom: '48px' }}>
        <div style={{ maxWidth: '96rem', margin: '0 auto', padding: '24px 16px' }}>
          <h1 style={{ fontSize: '2.25rem', fontWeight: 'bold', marginBottom: '8px' }}>MBTI Personality Prediction</h1>
          <p style={{ color: '#cbd5e1' }}>Discover your MBTI type through advanced facial & vocal analysis</p>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: '96rem', margin: '0 auto', padding: '0 16px', marginBottom: '48px' }}>
        {!result ? (
          <div>
            {/* Upload Section */}
            <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', padding: '32px', marginBottom: '32px' }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Upload size={24} /> Upload Your Video
              </h2>
              <p style={{ color: '#94a3b8', marginBottom: '24px' }}>Submit a ~2 minute video of yourself for personality analysis</p>

              <div>
                {/* File Input */}
                <label style={{ display: 'block', cursor: 'pointer' }}>
                  <div style={{ border: '2px dashed #475569', borderRadius: '8px', padding: '48px', textAlign: 'center', cursor: 'pointer', transition: 'all 0.3s' }}>
                    <input
                      type="file"
                      accept="video/*"
                      onChange={handleVideoUpload}
                      disabled={loading}
                      style={{ display: 'none' }}
                    />
                    <Upload size={48} style={{ margin: '0 auto 12px', color: '#64748b' }} />
                    <p style={{ fontSize: '1.125rem', fontWeight: '500', marginBottom: '8px' }}>Click to upload or drag and drop</p>
                    <p style={{ color: '#94a3b8', fontSize: '0.875rem' }}>MP4, MOV, or other video formats • Max 500MB • ~2 minutes recommended</p>
                  </div>
                </label>

                {/* Video Preview */}
                {videoPreview && (
                  <div style={{ marginTop: '16px' }}>
                    <p style={{ fontSize: '0.875rem', fontWeight: '500', color: '#cbd5e1', marginBottom: '8px' }}>Video Preview</p>
                    <video
                      src={videoPreview}
                      controls
                      style={{ width: '100%', maxHeight: '320px', borderRadius: '8px', background: '#0f172a', objectFit: 'contain' }}
                    />
                    <p style={{ fontSize: '0.875rem', color: '#94a3b8', marginTop: '8px' }}>
                      File: {video?.name} • {(video?.size ? (video.size / 1024 / 1024).toFixed(2) : 0)}MB
                    </p>
                  </div>
                )}

                {/* Error Message */}
                {error && (
                  <div style={{ background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.5)', borderRadius: '8px', padding: '16px', display: 'flex', gap: '12px', marginTop: '16px' }}>
                    <AlertCircle size={20} style={{ color: '#ef4444', marginTop: '4px' }} />
                    <p style={{ color: '#fca5a5' }}>{error}</p>
                  </div>
                )}

                {/* Loading Message */}
                {loading && (
                  <div style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.5)', borderRadius: '8px', padding: '16px', marginTop: '16px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
                      <Loader size={20} style={{ color: '#3b82f6', animation: 'spin 2s linear infinite' }} />
                      <p style={{ fontWeight: '500', color: '#3b82f6' }}>Processing your video...</p>
                    </div>
                    <p style={{ fontSize: '0.875rem', color: '#93c5fd' }}>This may take a few minutes depending on video length and server load</p>
                  </div>
                )}

                {/* Action Buttons */}
                <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
                  <button
                    onClick={handlePrediction}
                    disabled={!videoPreview || loading}
                    style={{
                      flex: 1,
                      background: loading || !videoPreview ? 'linear-gradient(to right, #475569, #334155)' : 'linear-gradient(to right, #3b82f6, #1d4ed8)',
                      color: 'white',
                      fontWeight: '600',
                      padding: '12px 24px',
                      borderRadius: '8px',
                      border: 'none',
                      cursor: loading || !videoPreview ? 'not-allowed' : 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px',
                      opacity: loading || !videoPreview ? 0.6 : 1,
                      transition: 'all 0.3s'
                    }}
                  >
                    {loading ? (
                      <>
                        <Loader size={20} style={{ animation: 'spin 2s linear infinite' }} />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Play size={20} />
                        Get MBTI Prediction
                      </>
                    )}
                  </button>
                  {videoPreview && !loading && (
                    <button
                      onClick={() => {
                        setVideo(null);
                        setVideoPreview(null);
                      }}
                      style={{
                        padding: '12px 24px',
                        background: '#475569',
                        color: 'white',
                        fontWeight: '600',
                        borderRadius: '8px',
                        border: 'none',
                        cursor: 'pointer',
                        transition: 'all 0.3s'
                      }}
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* Info Section */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px' }}>
              <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', padding: '16px' }}>
                <p style={{ fontSize: '0.875rem', fontWeight: '600', color: '#60a5fa', marginBottom: '8px' }}>📹 Video Requirements</p>
                <ul style={{ listStyle: 'none' }}>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• Duration: ~2 minutes</li>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• Clear face visibility throughout</li>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• Good lighting conditions</li>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• Natural expressions and speech</li>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• Audio must be clear</li>
                </ul>
              </div>
              <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', padding: '16px' }}>
                <p style={{ fontSize: '0.875rem', fontWeight: '600', color: '#a78bfa', marginBottom: '8px' }}>✨ Analysis Method</p>
                <ul style={{ listStyle: 'none' }}>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• Facial expressions (FER)</li>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• Visual features (CLIP)</li>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• Voice analysis (Wav2Vec2)</li>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• Audio prosody (VGGish)</li>
                  <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '4px 0' }}>• BiLSTM + Transformer fusion</li>
                </ul>
              </div>
            </div>
          </div>
        ) : (
          /* Result Section */
          <div>
            <div style={{ background: 'linear-gradient(to bottom right, #3b82f6, #1d4ed8)', borderRadius: '8px', padding: '48px', textAlign: 'center', marginBottom: '32px', boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.3)' }}>
              <CheckCircle size={64} style={{ margin: '0 auto 16px' }} />
              <h2 style={{ fontSize: '3rem', fontWeight: 'bold', marginBottom: '8px' }}>{result.mbti_type}</h2>
              <p style={{ fontSize: '1.5rem', fontWeight: '600', marginBottom: '8px' }}>{resultData?.title}</p>
              <p style={{ fontSize: '1.125rem', opacity: 0.9, marginBottom: '16px' }}>{resultData?.traits}</p>
              <p style={{ fontSize: '1rem', opacity: 0.8 }}>{resultData?.description}</p>
            </div>

            {/* MBTI Dimensions */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', marginBottom: '32px' }}>
              <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', padding: '24px' }}>
                <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <TrendingUp size={20} style={{ color: '#60a5fa' }} />
                  MBTI Probabilities
                </h3>
                <div>
                  {Object.entries(result.probabilities || {}).map(([key, prob]) => (
                    <div key={key} style={{ marginBottom: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <span style={{ fontSize: '0.875rem', fontWeight: '500', color: '#cbd5e1' }}>{key}</span>
                        <span style={{ fontSize: '0.875rem', fontWeight: '600', color: '#60a5fa' }}>{(prob * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ width: '100%', height: '8px', background: '#475569', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ background: 'linear-gradient(to right, #3b82f6, #1d4ed8)', height: '100%', width: `${prob * 100}%`, transition: 'width 0.5s ease' }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', padding: '24px' }}>
                <h3 style={{ fontSize: '1.125rem', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <TrendingUp size={20} style={{ color: '#a78bfa' }} />
                  OCEAN Personality Traits
                </h3>
                <div>
                  {Object.entries(result.ocean_scores || {}).map(([key, score]) => (
                    <div key={key} style={{ marginBottom: '16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                        <span style={{ fontSize: '0.875rem', fontWeight: '500', color: '#cbd5e1' }}>{key}</span>
                        <span style={{ fontSize: '0.875rem', fontWeight: '600', color: '#a78bfa' }}>{score.toFixed(2)}</span>
                      </div>
                      <div style={{ width: '100%', height: '8px', background: '#475569', borderRadius: '4px', overflow: 'hidden' }}>
                        <div style={{ background: 'linear-gradient(to right, #a78bfa, #7c3aed)', height: '100%', width: `${Math.min(score / 2 * 100, 100)}%`, transition: 'width 0.5s ease' }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Details */}
            <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: '8px', padding: '32px', marginBottom: '32px' }}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '24px' }}>About Your Personality Type</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '32px' }}>
                <div>
                  <p style={{ fontSize: '0.875rem', fontWeight: '600', color: '#94a3b8', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Key Characteristics</p>
                  <ul style={{ listStyle: 'none' }}>
                    <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '8px 0', display: 'flex', gap: '8px' }}>
                      <span style={{ color: '#22c55e', fontWeight: 'bold' }}>✓</span> Unique personality profile
                    </li>
                    <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '8px 0', display: 'flex', gap: '8px' }}>
                      <span style={{ color: '#22c55e', fontWeight: 'bold' }}>✓</span> Predicted from multimodal analysis
                    </li>
                    <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '8px 0', display: 'flex', gap: '8px' }}>
                      <span style={{ color: '#22c55e', fontWeight: 'bold' }}>✓</span> Based on facial & vocal patterns
                    </li>
                    <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '8px 0', display: 'flex', gap: '8px' }}>
                      <span style={{ color: '#22c55e', fontWeight: 'bold' }}>✓</span> Deep learning model predictions
                    </li>
                  </ul>
                </div>
                <div>
                  <p style={{ fontSize: '0.875rem', fontWeight: '600', color: '#94a3b8', marginBottom: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Analysis Details</p>
                  <ul style={{ listStyle: 'none' }}>
                    <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '8px 0' }}>• Facial Expression Recognition (FER)</li>
                    <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '8px 0' }}>• Vision Transformer (CLIP)</li>
                    <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '8px 0' }}>• Speech Representation (Wav2Vec2)</li>
                    <li style={{ fontSize: '0.875rem', color: '#cbd5e1', padding: '8px 0' }}>• Audio Features (VGGish)</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* Action Button */}
            <div style={{ textAlign: 'center' }}>
              <button
                onClick={resetForm}
                style={{
                  background: 'linear-gradient(to right, #3b82f6, #1d4ed8)',
                  color: 'white',
                  fontWeight: '600',
                  padding: '12px 32px',
                  borderRadius: '8px',
                  border: 'none',
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                  fontSize: '1rem'
                }}
              >
                Analyze Another Video
              </button>
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
